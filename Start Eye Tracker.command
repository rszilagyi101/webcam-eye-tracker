#!/bin/bash
# ---------------------------------------------------------------------------
# Start Eye Tracker.command  -  double-click to run on macOS.
#
# First run: creates a private Python environment in ./.venv and installs the
# dependencies (a few minutes, needs internet). Every run after that starts
# in a couple of seconds. Your browser opens automatically.
#
# To stop: click "Quit app" on the page, or close this Terminal window.
# ---------------------------------------------------------------------------

cd "$(dirname "$0")" || exit 1

pause_and_exit() {
  echo
  read -r -p "Press Enter to close this window..." _
  exit "${1:-1}"
}

# Prefer a Python version that MediaPipe publishes wheels for (3.10-3.12).
PY=""
for p in python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v "$p" >/dev/null 2>&1; then PY="$p"; break; fi
done

if [ -z "$PY" ]; then
  echo "Python 3 was not found."
  echo "Install Python 3.12 from https://www.python.org/downloads/macos/"
  echo "then double-click this file again."
  pause_and_exit 1
fi

if "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)'; then
  echo "Note: $("$PY" --version) is very new; if the install below fails with a"
  echo "MediaPipe error, install Python 3.12 from python.org and try again."
  echo
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "First run - setting up (this takes a few minutes)..."
  "$PY" -m venv .venv || { echo "Could not create the virtual environment."; pause_and_exit 1; }
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt || {
    echo
    echo "Dependency install failed - see the error above."
    rm -rf .venv   # so the next run retries cleanly
    pause_and_exit 1
  }
  echo "Setup complete."
fi

echo "Starting Webcam Eye Tracker... (your browser will open)"
echo "macOS will ask for Camera access the first time - click Allow."
echo
.venv/bin/python app.py
status=$?
if [ $status -ne 0 ]; then
  echo
  echo "The app exited with an error (code $status)."
  pause_and_exit $status
fi
