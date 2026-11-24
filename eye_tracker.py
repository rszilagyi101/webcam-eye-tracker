"""
eye_tracker.py
==============
Core webcam eye-tracking engine. No extra hardware required — just a
regular webcam. Uses MediaPipe's FaceLandmarker task (478-point face mesh,
including 10 iris landmarks) to estimate where on the screen you're
looking, based on the position of your iris relative to your eye corners.

This module has NO Flask dependency, so it can also be run standalone:

    python eye_tracker.py --preview            # see the tracking overlay
    python eye_tracker.py --calibrate          # run/save a calibration
    python eye_tracker.py --track 60           # record a 60s session

Honesty note on accuracy
-------------------------
This is appearance-based gaze estimation from a single 2D webcam — the
same category of technique used by libraries like WebGazer.js. It is NOT
as accurate as infrared/corneal-reflection eye trackers. Typical accuracy
after calibration is roughly 1-3cm on screen for someone sitting still in
decent lighting. It's great for "which general area of the screen did I
look at" heatmaps, not for precision fixation research. Head movement,
glasses glare, and poor lighting will degrade it further.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

from mediapipe.tasks.python import vision, BaseOptions
import mediapipe as mp

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------

FROZEN = bool(getattr(sys, "frozen", False))  # True inside a PyInstaller .app

if FROZEN:
    # Inside a .app bundle the code/resources are read-only, so user data
    # (model, calibration, sessions, logs) lives in the user's Documents.
    APP_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))  # bundled resources
    USER_DIR = Path(os.environ.get("EYE_TRACKER_HOME", Path.home() / "Documents" / "WebcamEyeTracker"))
else:
    APP_DIR = Path(__file__).resolve().parent
    USER_DIR = APP_DIR

MODEL_DIR = USER_DIR / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

DATA_DIR = USER_DIR / "data"
CALIBRATION_PATH = DATA_DIR / "calibration.json"

# Landmark indices, taken directly from MediaPipe's own
# FaceLandmarksConnections constants (not guessed):
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_EYE_CORNERS = (33, 133)   # outer, inner
LEFT_EYE_CORNERS = (362, 263)  # inner, outer
RIGHT_EYE_TOP_BOTTOM = (159, 145)
LEFT_EYE_TOP_BOTTOM = (386, 374)

CAMERA_INDEX = int(os.environ.get("EYE_TRACKER_CAMERA_INDEX", "0"))


# --------------------------------------------------------------------------
# Model download
# --------------------------------------------------------------------------

def ensure_model() -> str:
    """Download MediaPipe's face landmark model on first run (~4MB)."""
    if MODEL_PATH.exists():
        return str(MODEL_PATH)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[eye_tracker] Downloading face landmark model to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "Could not download the MediaPipe face landmark model "
            f"from {MODEL_URL}. Check your internet connection, or "
            "download it manually and place it at "
            f"{MODEL_PATH}. Original error: {e}"
        ) from e
    print("[eye_tracker] Model downloaded.")
    return str(MODEL_PATH)


def _screen_grabber():
    """Return a new mss screen-capture object.

    Newer mss versions renamed the class to `mss.MSS` and deprecate the old
    `mss.mss()` factory; support both so older installs keep working.
    """
    import mss

    cls = getattr(mss, "MSS", None)
    return cls() if cls is not None else mss.mss()


def get_screen_size() -> tuple[int, int]:
    """Best-effort screen resolution, used to size the heatmap/calibration."""
    # mss first: it needs no GUI toolkit (tkinter can misbehave inside a
    # bundled macOS app), and reports the primary monitor in the same
    # coordinate space (points) that the mouse pointer uses.
    try:
        with _screen_grabber() as sct:
            mon = sct.monitors[1]
            w, h = int(mon["width"]), int(mon["height"])
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1920, 1080  # sane fallback


# --------------------------------------------------------------------------
# Gaze estimation
# --------------------------------------------------------------------------

@dataclass
class GazeSample:
    t: float
    h_ratio: Optional[float]
    v_ratio: Optional[float]
    found: bool


def _landmark_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def _iris_center(landmarks, iris_idxs, w, h):
    pts = np.array([_landmark_xy(landmarks, i, w, h) for i in iris_idxs])
    return pts.mean(axis=0)


def compute_gaze_ratios(landmarks, frame_w: int, frame_h: int):
    """
    Given the 478 face landmarks for one frame, compute a normalized
    (h_ratio, v_ratio) describing where the iris sits within the eye
    socket, averaged over both eyes. ~0.5/0.5 is "looking straight at
    the camera"; values move away from that as gaze shifts.
    """
    try:
        r_iris = _iris_center(landmarks, RIGHT_IRIS, frame_w, frame_h)
        l_iris = _iris_center(landmarks, LEFT_IRIS, frame_w, frame_h)

        r_out_x, _ = _landmark_xy(landmarks, RIGHT_EYE_CORNERS[0], frame_w, frame_h)
        r_in_x, _ = _landmark_xy(landmarks, RIGHT_EYE_CORNERS[1], frame_w, frame_h)
        l_in_x, _ = _landmark_xy(landmarks, LEFT_EYE_CORNERS[0], frame_w, frame_h)
        l_out_x, _ = _landmark_xy(landmarks, LEFT_EYE_CORNERS[1], frame_w, frame_h)

        _, r_top_y = _landmark_xy(landmarks, RIGHT_EYE_TOP_BOTTOM[0], frame_w, frame_h)
        _, r_bot_y = _landmark_xy(landmarks, RIGHT_EYE_TOP_BOTTOM[1], frame_w, frame_h)
        _, l_top_y = _landmark_xy(landmarks, LEFT_EYE_TOP_BOTTOM[0], frame_w, frame_h)
        _, l_bot_y = _landmark_xy(landmarks, LEFT_EYE_TOP_BOTTOM[1], frame_w, frame_h)

        r_xmin, r_xmax = sorted([r_out_x, r_in_x])
        l_xmin, l_xmax = sorted([l_in_x, l_out_x])

        def safe_ratio(val, lo, hi):
            if hi - lo < 1e-3:
                return 0.5
            return (val - lo) / (hi - lo)

        r_h = safe_ratio(r_iris[0], r_xmin, r_xmax)
        l_h = safe_ratio(l_iris[0], l_xmin, l_xmax)
        r_v = safe_ratio(r_iris[1], min(r_top_y, r_bot_y), max(r_top_y, r_bot_y))
        l_v = safe_ratio(l_iris[1], min(l_top_y, l_bot_y), max(l_top_y, l_bot_y))

        h_ratio = float(np.clip((r_h + l_h) / 2.0, -0.5, 1.5))
        v_ratio = float(np.clip((r_v + l_v) / 2.0, -0.5, 1.5))
        return h_ratio, v_ratio
    except Exception:
        return None, None


class GazeEngine:
    """Wraps the webcam + MediaPipe FaceLandmarker."""

    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.camera_index = camera_index
        model_path = ensure_model()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self.camera_index}. "
                "Make sure no other app is using it, and that this app has "
                "camera permission."
            )
        self._t0 = time.time()

    def read_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def process(self, frame_bgr) -> tuple[Optional[float], Optional[float], object]:
        """Returns (h_ratio, v_ratio, landmarks_or_None) for one frame."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.time() - self._t0) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None, None, None
        landmarks = result.face_landmarks[0]
        h, w = frame_bgr.shape[:2]
        h_ratio, v_ratio = compute_gaze_ratios(landmarks, w, h)
        return h_ratio, v_ratio, landmarks

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.landmarker.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Calibration: map (h_ratio, v_ratio) -> screen (x, y)
# --------------------------------------------------------------------------

def _poly_features(h, v):
    return np.array([1.0, h, v, h * h, v * v, h * v])


def fit_calibration(samples: list[tuple[float, float, float, float]]):
    """samples: list of (h_ratio, v_ratio, screen_x, screen_y)."""
    X = np.array([_poly_features(h, v) for h, v, _, _ in samples])
    ys_x = np.array([sx for _, _, sx, _ in samples])
    ys_y = np.array([sy for _, _, _, sy in samples])
    coef_x, *_ = np.linalg.lstsq(X, ys_x, rcond=None)
    coef_y, *_ = np.linalg.lstsq(X, ys_y, rcond=None)
    return {"coef_x": coef_x.tolist(), "coef_y": coef_y.tolist()}


def apply_calibration(coeffs: dict, h: float, v: float, screen_w: int, screen_h: int):
    feats = _poly_features(h, v)
    x = float(np.dot(feats, np.array(coeffs["coef_x"])))
    y = float(np.dot(feats, np.array(coeffs["coef_y"])))
    x = float(np.clip(x, 0, screen_w - 1))
    y = float(np.clip(y, 0, screen_h - 1))
    return x, y


def naive_map(h: float, v: float, screen_w: int, screen_h: int):
    """Rough fallback mapping used only if the user skips calibration."""
    # Empirically, comfortable eye movement tends to swing h_ratio over
    # roughly [0.25, 0.75] and v_ratio over [0.3, 0.7] -- stretch that
    # range to fill the screen.
    x = np.interp(h, [0.25, 0.75], [0, screen_w - 1])
    y = np.interp(v, [0.3, 0.7], [0, screen_h - 1])
    return float(np.clip(x, 0, screen_w - 1)), float(np.clip(y, 0, screen_h - 1))


def save_calibration(coeffs: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(coeffs, f)


def load_calibration() -> Optional[dict]:
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    return None


CALIBRATION_POINTS_NORM = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]


def run_calibration(engine: GazeEngine, screen_w: int, screen_h: int,
                     progress_cb: Optional[Callable[[float, str], None]] = None) -> Optional[dict]:
    """
    Opens a fullscreen window with 9 dots in sequence. The user looks at
    each dot; we collect gaze ratio samples and fit a mapping. Returns the
    fitted coefficients, or None if the window was closed early (Esc).
    """
    window = "Eye Tracker Calibration - look at each dot, press ESC to cancel"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    samples = []
    aborted = False
    n_points = len(CALIBRATION_POINTS_NORM)

    for i, (nx, ny) in enumerate(CALIBRATION_POINTS_NORM):
        px, py = int(nx * screen_w), int(ny * screen_h)
        point_ratios = []
        # Give the user a brief moment to move their eyes to the new dot,
        # then collect samples for ~1 second.
        settle_end = time.time() + 0.6
        collect_end = None
        while True:
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            collecting = collect_end is not None
            color = (0, 255, 0) if collecting else (0, 165, 255)
            cv2.circle(canvas, (px, py), 18, color, -1)
            cv2.circle(canvas, (px, py), 18, (255, 255, 255), 2)
            cv2.putText(canvas, f"Point {i + 1}/{n_points} - look at the dot",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                aborted = True
                break

            frame = engine.read_frame()
            now = time.time()
            if frame is not None and now >= settle_end:
                if collect_end is None:
                    collect_end = now + 1.0
                h_ratio, v_ratio, _ = engine.process(frame)
                if h_ratio is not None:
                    point_ratios.append((h_ratio, v_ratio))
                if now >= collect_end:
                    break
            if progress_cb:
                progress_cb((i) / n_points, f"Calibrating point {i + 1}/{n_points}")
        if aborted:
            break
        if point_ratios:
            arr = np.array(point_ratios)
            med_h, med_v = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
            samples.append((med_h, med_v, float(px), float(py)))

    cv2.destroyWindow(window)
    cv2.waitKey(1)

    if aborted or len(samples) < 6:
        return None

    coeffs = fit_calibration(samples)
    save_calibration(coeffs)
    if progress_cb:
        progress_cb(1.0, "Calibration complete")
    return coeffs


# --------------------------------------------------------------------------
# Tracking session
# --------------------------------------------------------------------------

def _make_mouse_reader(progress_cb: Optional[Callable[[float, str], None]] = None):
    """
    Returns a zero-arg function that gives the current (x, y) global mouse
    position, or None if mouse tracking isn't available on this system
    (e.g. pynput isn't installed, or there's no display/permission).
    """
    try:
        from pynput import mouse as pynput_mouse

        controller = pynput_mouse.Controller()
        _ = controller.position  # sanity check it actually works here
    except Exception as e:
        if progress_cb:
            progress_cb(0.0, f"Mouse tracking unavailable ({e}); continuing without it.")
        return None

    def _read():
        try:
            x, y = controller.position
            return float(x), float(y)
        except Exception:
            return None, None

    return _read


def run_tracking_session(
    engine: GazeEngine,
    duration_s: float,
    screen_w: int,
    screen_h: int,
    coeffs: Optional[dict],
    progress_cb: Optional[Callable[[float, str], None]] = None,
    track_mouse: bool = False,
) -> pd.DataFrame:
    """Records gaze samples for duration_s seconds and returns a DataFrame.
    If track_mouse is True, also records the global mouse pointer position
    (mouse_x, mouse_y columns) at the same cadence, when available."""
    mouse_reader = _make_mouse_reader(progress_cb) if track_mouse else None

    rows = []
    start = time.time()
    end = start + duration_s
    while True:
        now = time.time()
        if now >= end:
            break
        frame = engine.read_frame()
        if frame is None:
            continue
        h_ratio, v_ratio, _ = engine.process(frame)
        found = h_ratio is not None
        if found:
            if coeffs:
                x, y = apply_calibration(coeffs, h_ratio, v_ratio, screen_w, screen_h)
            else:
                x, y = naive_map(h_ratio, v_ratio, screen_w, screen_h)
        else:
            x, y = None, None

        row = {
            "timestamp_s": round(now - start, 3),
            "screen_x": x,
            "screen_y": y,
            "h_ratio": h_ratio,
            "v_ratio": v_ratio,
            "face_found": found,
        }
        if mouse_reader is not None:
            mx, my = mouse_reader()
            row["mouse_x"] = mx
            row["mouse_y"] = my
        rows.append(row)
        if progress_cb:
            progress_cb(min(1.0, (now - start) / duration_s), "Tracking")

    df = pd.DataFrame(rows)
    return df


def save_session_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------
# Heatmap
# --------------------------------------------------------------------------

def grab_screenshot(screen_w: int, screen_h: int) -> Optional[np.ndarray]:
    """Best-effort screenshot of the primary monitor as a BGR numpy array."""
    try:
        with _screen_grabber() as sct:
            monitor = sct.monitors[1]  # primary monitor
            shot = np.array(sct.grab(monitor))  # BGRA
            bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
            bgr = cv2.resize(bgr, (screen_w, screen_h))
            return bgr
    except Exception:
        return None


def build_heatmap(df: pd.DataFrame, screen_w: int, screen_h: int,
                   background: Optional[np.ndarray] = None,
                   max_alpha: float = 0.7,
                   x_col: str = "screen_x", y_col: str = "screen_y",
                   colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Turns recorded (x_col, y_col) points into a colored heatmap image (BGR)."""
    valid = df.dropna(subset=[x_col, y_col])

    down = 8  # downsample factor for the density grid, then upscale+blur
    grid_w = max(1, screen_w // down)
    grid_h = max(1, screen_h // down)
    density = np.zeros((grid_h, grid_w), dtype=np.float32)

    if not valid.empty:
        xs = (valid[x_col].to_numpy() / down).astype(int)
        ys = (valid[y_col].to_numpy() / down).astype(int)
        xs = np.clip(xs, 0, grid_w - 1)
        ys = np.clip(ys, 0, grid_h - 1)
        for x, y in zip(xs, ys):
            density[y, x] += 1.0

    density = cv2.resize(density, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
    ksize = max(31, (min(screen_w, screen_h) // 10) | 1)  # odd kernel size
    density = cv2.GaussianBlur(density, (ksize, ksize), 0)

    if density.max() > 0:
        norm = density / density.max()
    else:
        norm = density

    heat_u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, colormap)

    if background is None:
        background = np.full((screen_h, screen_w, 3), 30, dtype=np.uint8)

    alpha = (norm * max_alpha)[:, :, None]
    out = (background.astype(np.float32) * (1 - alpha) +
           colored.astype(np.float32) * alpha).astype(np.uint8)
    return out


def build_comparison_heatmap(df: pd.DataFrame, screen_w: int, screen_h: int,
                              background: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Overlays eye-gaze density (warm, red/yellow) and mouse-pointer density
    (cool, cyan/blue) on the same image so you can compare where you looked
    vs. where you pointed. Requires both screen_x/y and mouse_x/y columns.
    """
    down = 8
    grid_w = max(1, screen_w // down)
    grid_h = max(1, screen_h // down)

    def density_for(x_col, y_col):
        d = np.zeros((grid_h, grid_w), dtype=np.float32)
        valid = df.dropna(subset=[x_col, y_col])
        if not valid.empty:
            xs = (valid[x_col].to_numpy() / down).astype(int)
            ys = (valid[y_col].to_numpy() / down).astype(int)
            xs = np.clip(xs, 0, grid_w - 1)
            ys = np.clip(ys, 0, grid_h - 1)
            for x, y in zip(xs, ys):
                d[y, x] += 1.0
        d = cv2.resize(d, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
        ksize = max(31, (min(screen_w, screen_h) // 10) | 1)
        d = cv2.GaussianBlur(d, (ksize, ksize), 0)
        return d / d.max() if d.max() > 0 else d

    eye_norm = density_for("screen_x", "screen_y")
    mouse_norm = density_for("mouse_x", "mouse_y")

    if background is None:
        out = np.full((screen_h, screen_w, 3), 30, dtype=np.uint8).astype(np.float32)
    else:
        out = background.astype(np.float32).copy()

    # Eye gaze: warm red/yellow tint. Mouse: cool cyan/blue tint. Blended
    # additively so overlapping regions read as bright white-ish.
    eye_color = np.zeros((screen_h, screen_w, 3), dtype=np.float32)
    eye_color[:, :, 2] = 255  # red channel (BGR)
    eye_color[:, :, 1] = 140  # a little green -> orange/yellow warmth

    mouse_color = np.zeros((screen_h, screen_w, 3), dtype=np.float32)
    mouse_color[:, :, 0] = 255  # blue channel
    mouse_color[:, :, 1] = 200  # a little green -> cyan

    eye_alpha = (eye_norm * 0.6)[:, :, None]
    mouse_alpha = (mouse_norm * 0.6)[:, :, None]

    out = out * (1 - eye_alpha) + eye_color * eye_alpha
    out = out * (1 - mouse_alpha) + mouse_color * mouse_alpha

    legend_y = 30
    cv2.putText(out.astype(np.uint8), "red/yellow = eye gaze", (20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out = out.astype(np.uint8)
    cv2.putText(out, "blue/cyan = mouse pointer", (20, legend_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Webcam eye tracker (standalone CLI)")
    parser.add_argument("--preview", action="store_true", help="Show live tracking overlay")
    parser.add_argument("--calibrate", action="store_true", help="Run 9-point calibration")
    parser.add_argument("--track", type=float, default=None, help="Record a session for N seconds")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--delay", type=float, default=2.0,
                         help="Seconds to wait before recording starts, e.g. to switch pages (default: 2)")
    parser.add_argument("--open-url", type=str, default=None,
                         help="Open this URL in your default browser right before the countdown")
    parser.add_argument("--track-mouse", action="store_true",
                         help="Also record the global mouse pointer position")
    args = parser.parse_args()

    screen_w, screen_h = get_screen_size()

    if args.preview:
        engine = GazeEngine(args.camera)
        print("Press ESC to quit preview.")
        try:
            while True:
                frame = engine.read_frame()
                if frame is None:
                    continue
                h_ratio, v_ratio, landmarks = engine.process(frame)
                if landmarks is not None:
                    fh, fw = frame.shape[:2]
                    for idx in RIGHT_IRIS + LEFT_IRIS:
                        x, y = _landmark_xy(landmarks, idx, fw, fh)
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    cv2.putText(frame, f"h={h_ratio:.2f} v={v_ratio:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Eye Tracker Preview", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
        finally:
            engine.close()
            cv2.destroyAllWindows()
        return

    if args.calibrate:
        engine = GazeEngine(args.camera)
        try:
            coeffs = run_calibration(engine, screen_w, screen_h,
                                      progress_cb=lambda p, m: print(f"{m} ({p*100:.0f}%)"))
        finally:
            engine.close()
            cv2.destroyAllWindows()
        print("Calibration saved." if coeffs else "Calibration cancelled/failed.")
        sys.exit(0 if coeffs else 2)  # exit code lets the web app tell saved vs cancelled

    if args.track is not None:
        engine = GazeEngine(args.camera)
        coeffs = load_calibration()
        if not coeffs:
            print("No calibration found - using rough uncalibrated mapping. "
                  "Run --calibrate first for much better accuracy.")
        if args.open_url:
            webbrowser.open(args.open_url, new=2)
        if args.delay > 0:
            print(f"Switch to the page you want to look at - recording starts in {args.delay:.1f}s...")
            wait_start = time.time()
            while time.time() - wait_start < args.delay:
                engine.read_frame()  # keep camera buffer fresh
                time.sleep(0.05)
        try:
            df = run_tracking_session(
                engine, args.track, screen_w, screen_h, coeffs,
                progress_cb=lambda p, m: print(f"\r{m}: {p*100:.0f}%", end=""),
                track_mouse=args.track_mouse,
            )
        finally:
            engine.close()
        print()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = DATA_DIR / f"session_{stamp}.csv"
        save_session_csv(df, csv_path)
        bg = grab_screenshot(screen_w, screen_h)
        heat = build_heatmap(df, screen_w, screen_h, background=bg)
        png_path = DATA_DIR / f"heatmap_{stamp}.png"
        cv2.imwrite(str(png_path), heat)
        print(f"Saved {csv_path}")
        print(f"Saved {png_path}")
        if args.track_mouse and "mouse_x" in df.columns:
            mouse_heat = build_heatmap(df, screen_w, screen_h, background=bg,
                                        x_col="mouse_x", y_col="mouse_y",
                                        colormap=cv2.COLORMAP_OCEAN)
            mouse_png_path = DATA_DIR / f"mouse_heatmap_{stamp}.png"
            cv2.imwrite(str(mouse_png_path), mouse_heat)
            print(f"Saved {mouse_png_path}")

            combo = build_comparison_heatmap(df, screen_w, screen_h, background=bg)
            combo_path = DATA_DIR / f"comparison_{stamp}.png"
            cv2.imwrite(str(combo_path), combo)
            print(f"Saved {combo_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
