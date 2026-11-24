#!/bin/bash
# ---------------------------------------------------------------------------
# Build Mac App.command  -  double-click (or run in Terminal) ON A MAC to
# build a standalone "Webcam Eye Tracker.app".
#
# Result:  dist/Webcam Eye Tracker.app   (drag it to /Applications)
#          dist/WebcamEyeTracker-mac.zip (same app, zipped for sharing)
#
# The app is built for the CPU of the Mac you build it on (Apple Silicon or
# Intel). Takes ~5-10 minutes the first time (downloads dependencies).
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")" || exit 1

finish() {
  echo
  if [ -t 0 ]; then read -r -p "Press Enter to close this window..." _; fi
  exit "${1:-0}"
}
fail() { echo; echo "BUILD FAILED: $1"; finish 1; }

[ "$(uname)" = "Darwin" ] || fail "This must be run on macOS (PyInstaller can't cross-compile)."

PY=""
for p in python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v "$p" >/dev/null 2>&1; then PY="$p"; break; fi
done
[ -n "$PY" ] || fail "Python 3 not found. Install Python 3.12 from https://www.python.org/downloads/macos/"
echo "Using $($PY --version) ($PY)"

echo "==> Preparing build environment (.build-venv)"
if [ ! -x .build-venv/bin/python ]; then
  "$PY" -m venv .build-venv || fail "could not create virtualenv"
fi
.build-venv/bin/python -m pip install --upgrade pip >/dev/null
.build-venv/bin/python -m pip install -r requirements.txt pyinstaller || fail "dependency install failed"

echo "==> Making app icon (optional)"
rm -f icon.icns
if .build-venv/bin/python make_icon.py && [ -f icon.png ]; then
  mkdir -p build/icon.iconset
  for s in 16 32 128 256 512; do
    sips -z "$s" "$s" icon.png --out "build/icon.iconset/icon_${s}x${s}.png" >/dev/null 2>&1
    sips -z "$((s*2))" "$((s*2))" icon.png --out "build/icon.iconset/icon_${s}x${s}@2x.png" >/dev/null 2>&1
  done
  iconutil -c icns build/icon.iconset -o icon.icns 2>/dev/null || echo "   (icon conversion skipped - the app will use the default icon)"
else
  echo "   (icon generation skipped - the app will use the default icon)"
fi

echo "==> Building the app with PyInstaller"
.build-venv/bin/python -m PyInstaller --noconfirm --clean EyeTracker.spec || fail "PyInstaller failed"

APP="dist/Webcam Eye Tracker.app"
[ -d "$APP" ] || fail "expected $APP was not produced"

echo "==> Signing (ad-hoc, local use) and clearing quarantine flags"
codesign --force --deep --sign - "$APP" 2>/dev/null || echo "   (codesign skipped)"
xattr -cr "$APP" 2>/dev/null || true

echo "==> Zipping for sharing"
rm -f dist/WebcamEyeTracker-mac.zip
ditto -c -k --keepParent "$APP" dist/WebcamEyeTracker-mac.zip

echo
echo "Done!  ->  $(pwd)/$APP"
echo "Drag it into /Applications. First launch: right-click -> Open (once),"
echo "then allow Camera access. Results are saved to ~/Documents/WebcamEyeTracker/"
open -R "$APP" 2>/dev/null || true
finish 0
