"""
app.py
======
Local Flask web UI for the webcam eye tracker.

Run with:  python app.py
It picks a free local port, prints the address and opens your browser.
(Port 5000 is deliberately avoided: macOS uses it for AirPlay Receiver.)

Flow:
  1) (Recommended) Click "Run Calibration" - a fullscreen window with 9
     dots appears on your PRIMARY monitor. Look at each dot as it appears.
  2) Pick a duration (0.5 / 1 / 1.5 / 2 min) and click "Start Tracking".
  3) Go look at whatever you want on screen (e.g. switch to your browser
     tab / the webpage you want to study) until the timer ends.
  4) Come back to this tab - your heatmap + CSV download links appear.

Everything runs locally. No video or images ever leave your machine.
"""

from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
import zipfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory

import eye_tracker as et

# Templates/static come from et.APP_DIR so this also works inside a
# PyInstaller bundle (where resources live in sys._MEIPASS).
app = Flask(
    __name__,
    template_folder=str(et.APP_DIR / "templates"),
    static_folder=str(et.APP_DIR / "static"),
)

DATA_DIR = et.DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

SCREEN_W, SCREEN_H = et.get_screen_size()

# ---------------------------------------------------------------------
# Global job state (single-user local tool -> one job at a time is fine)
# ---------------------------------------------------------------------
_lock = threading.Lock()
state = {
    "mode": "idle",         # idle | calibrating | starting | tracking | done | error
    "progress": 0.0,
    "message": "",
    "calibrated": et.load_calibration() is not None,
    "last_csv": None,       # filename only
    "last_heatmap": None,   # filename only
    "last_mouse_heatmap": None,
    "last_comparison": None,
    "error": None,
}


def _set_state(**kwargs):
    with _lock:
        state.update(kwargs)


def _get_state():
    with _lock:
        return dict(state)


def _progress_cb(p, msg):
    _set_state(progress=p, message=msg)


def _calibration_command() -> list[str]:
    """Command that runs calibration in its OWN process.

    The fullscreen calibration window is an OpenCV/Cocoa GUI window, and
    macOS only allows GUI windows on a process's main thread - so it can't
    run on a Flask worker thread. A child process has its own main thread.
    """
    if et.FROZEN:
        return [sys.executable, "--worker-calibrate"]  # same bundled executable
    return [sys.executable, str(et.APP_DIR / "eye_tracker.py"), "--calibrate"]


def _calibration_job():
    try:
        _set_state(
            mode="calibrating", progress=0.0, error=None,
            message="A fullscreen window is opening — look at each dot until it moves (ESC cancels).",
        )
        proc = subprocess.Popen(
            _calibration_command(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            output, _ = proc.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate()
            raise RuntimeError("Calibration timed out after 5 minutes.")

        if proc.returncode == 0:
            _set_state(mode="idle", progress=1.0, message="Calibration complete.", calibrated=True)
        elif proc.returncode == 2:
            _set_state(mode="idle", progress=0.0, message="Calibration cancelled.")
        else:
            tail = "\n".join((output or "").strip().splitlines()[-6:])
            raise RuntimeError(f"Calibration process failed (exit {proc.returncode}).\n{tail}")
    except Exception as e:
        traceback.print_exc()
        _set_state(mode="error", error=str(e), message="Calibration failed.")


def _tracking_job(duration_s: float, delay_s: float = 2.0, open_url: str | None = None,
                   track_mouse: bool = False):
    try:
        _set_state(mode="starting", progress=0.0, message="Starting webcam...", error=None)
        engine = et.GazeEngine()
        coeffs = et.load_calibration()

        # Open the target page (if given) right away, then count down so the
        # user has time to switch to / arrive on the page before recording
        # actually starts.
        if open_url:
            try:
                webbrowser.open(open_url, new=2)
            except Exception:
                pass  # non-fatal - user can still switch manually

        wait_start = time.time()
        while True:
            remaining = delay_s - (time.time() - wait_start)
            if remaining <= 0:
                break
            _set_state(
                mode="starting",
                progress=0.0,
                message=f"Switch to the page you want to look at — starting in {remaining:.1f}s",
            )
            engine.read_frame()  # keep the camera buffer fresh during the wait
            time.sleep(0.05)

        _set_state(mode="tracking", progress=0.0, message="Tracking...")
        try:
            df = et.run_tracking_session(
                engine, duration_s, SCREEN_W, SCREEN_H, coeffs,
                progress_cb=_progress_cb, track_mouse=track_mouse,
            )
        finally:
            engine.close()

        _set_state(message="Saving results...")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_name = f"session_{stamp}.csv"
        heatmap_name = f"heatmap_{stamp}.png"
        et.save_session_csv(df, DATA_DIR / csv_name)

        bg = et.grab_screenshot(SCREEN_W, SCREEN_H)
        heat = et.build_heatmap(df, SCREEN_W, SCREEN_H, background=bg)
        import cv2
        cv2.imwrite(str(DATA_DIR / heatmap_name), heat)

        mouse_heatmap_name = None
        comparison_name = None
        mouse_tracked = track_mouse and "mouse_x" in df.columns and df["mouse_x"].notna().any()
        if mouse_tracked:
            mouse_heat = et.build_heatmap(
                df, SCREEN_W, SCREEN_H, background=bg,
                x_col="mouse_x", y_col="mouse_y", colormap=cv2.COLORMAP_OCEAN,
            )
            mouse_heatmap_name = f"mouse_heatmap_{stamp}.png"
            cv2.imwrite(str(DATA_DIR / mouse_heatmap_name), mouse_heat)

            combo = et.build_comparison_heatmap(df, SCREEN_W, SCREEN_H, background=bg)
            comparison_name = f"comparison_{stamp}.png"
            cv2.imwrite(str(DATA_DIR / comparison_name), combo)
        elif track_mouse:
            # User asked for mouse tracking but it wasn't available on this
            # system (see _make_mouse_reader) - don't fail the whole session,
            # just note it in the final message.
            pass

        found_pct = 100.0 * df["face_found"].mean() if len(df) else 0.0
        msg = f"Done. Face detected in {found_pct:.0f}% of frames."
        if track_mouse and not mouse_tracked:
            msg += " (Mouse tracking was requested but unavailable on this system.)"
        _set_state(
            mode="done",
            progress=1.0,
            message=msg,
            last_csv=csv_name,
            last_heatmap=heatmap_name,
            last_mouse_heatmap=mouse_heatmap_name,
            last_comparison=comparison_name,
        )
    except Exception as e:
        traceback.print_exc()
        _set_state(mode="error", error=str(e), message="Tracking failed.")


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        calibrated=_get_state()["calibrated"],
        screen_w=SCREEN_W,
        screen_h=SCREEN_H,
    )


RUNNING_MODES = ("calibrating", "tracking", "starting")


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    s = _get_state()
    if s["mode"] in RUNNING_MODES:
        return jsonify({"ok": False, "error": "A job is already running."}), 409
    t = threading.Thread(target=_calibration_job, daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/track", methods=["POST"])
def api_track():
    s = _get_state()
    if s["mode"] in RUNNING_MODES:
        return jsonify({"ok": False, "error": "A job is already running."}), 409
    payload = request.get_json(force=True, silent=True) or {}

    try:
        duration_s = float(payload.get("duration_s", 60))
    except (TypeError, ValueError):
        duration_s = 60.0
    duration_s = max(5.0, min(duration_s, 600.0))  # sane bounds: 5s - 10min

    try:
        delay_s = float(payload.get("delay_s", 2))
    except (TypeError, ValueError):
        delay_s = 2.0
    delay_s = max(0.0, min(delay_s, 30.0))  # sane bounds: 0-30s

    open_url = (payload.get("open_url") or "").strip() or None
    if open_url and not open_url.startswith(("http://", "https://")):
        open_url = "https://" + open_url

    track_mouse = bool(payload.get("track_mouse", False))

    t = threading.Thread(
        target=_tracking_job, args=(duration_s, delay_s, open_url, track_mouse), daemon=True
    )
    t.start()
    return jsonify({
        "ok": True, "duration_s": duration_s, "delay_s": delay_s,
        "open_url": open_url, "track_mouse": track_mouse,
    })


@app.route("/api/status")
def api_status():
    return jsonify(_get_state())


@app.route("/results")
def results():
    s = _get_state()
    return render_template(
        "results.html",
        csv_name=s.get("last_csv"),
        heatmap_name=s.get("last_heatmap"),
        mouse_heatmap_name=s.get("last_mouse_heatmap"),
        comparison_name=s.get("last_comparison"),
    )


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


@app.route("/download_all")
def download_all():
    """Zips up every file from the most recent session and sends it as one download."""
    s = _get_state()
    filenames = [s.get("last_csv"), s.get("last_heatmap"), s.get("last_mouse_heatmap"), s.get("last_comparison")]
    filenames = [f for f in filenames if f]
    if not filenames:
        return jsonify({"ok": False, "error": "No session results available yet."}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in filenames:
            fpath = DATA_DIR / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
    buf.seek(0)

    # Reuse the session's own timestamp (from "session_<stamp>.csv") for a
    # friendly, matching zip filename.
    stamp = None
    if s.get("last_csv") and s["last_csv"].startswith("session_"):
        stamp = s["last_csv"][len("session_"):-len(".csv")]
    zip_name = f"eye_tracking_{stamp}.zip" if stamp else "eye_tracking_session.zip"

    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=zip_name)


@app.route("/data/<path:filename>")
def data_file(filename):
    # Non-attachment version, so the heatmap <img> tag can display it inline.
    return send_from_directory(DATA_DIR, filename)


@app.route("/api/quit", methods=["POST"])
def api_quit():
    """Lets the user shut the app down from the page (a bundled .app has no terminal)."""
    if _get_state()["mode"] in RUNNING_MODES:
        return jsonify({"ok": False, "error": "A job is still running - wait for it to finish."}), 409
    threading.Timer(0.6, lambda: os._exit(0)).start()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------

def _find_free_port(start: int = 5057, tries: int = 50) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free local port found.")


def _probe_camera_permission():
    """On macOS the camera-permission prompt only appears reliably when the
    camera is first opened from the MAIN thread, so do that once at launch
    (before Flask worker threads are involved)."""
    if sys.platform != "darwin":
        return
    try:
        import cv2

        cap = cv2.VideoCapture(et.CAMERA_INDEX)
        cap.read()
        cap.release()
    except Exception:
        traceback.print_exc()


def _redirect_output_to_log():
    """A windowed .app has no terminal: keep a log file for troubleshooting."""
    try:
        et.USER_DIR.mkdir(parents=True, exist_ok=True)
        log = open(et.USER_DIR / "app.log", "a", buffering=1)
        sys.stdout = log
        sys.stderr = log
        print(f"\n--- launched {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    except Exception:
        pass


def main():
    # Child-process entry points (see _calibration_command). They must run
    # before anything starts a server.
    if "--worker-calibrate" in sys.argv:
        sys.argv = [sys.argv[0], "--calibrate"]
        et._cli()
        return

    if et.FROZEN:
        _redirect_output_to_log()

    port = int(os.environ.get("EYE_TRACKER_PORT", "0")) or _find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Detected screen size: {SCREEN_W}x{SCREEN_H}")
    print(f"Webcam Eye Tracker running at {url}")
    print(f"Data folder: {DATA_DIR}")

    _probe_camera_permission()

    if not os.environ.get("EYE_TRACKER_NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
