# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Webcam Eye Tracker.
# Normally run via 'Build Mac App.command' (macOS) - it produces
#   dist/Webcam Eye Tracker.app
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

APP_NAME = "Webcam Eye Tracker"

datas = [("templates", "templates"), ("static", "static")]
binaries = []
hiddenimports = []

# MediaPipe ships native libs + data files that PyInstaller can't see by itself.
for pkg in ("mediapipe",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Global mouse position (pynput) and screenshots/screen size (mss).
hiddenimports += collect_submodules("pynput") + collect_submodules("mss")

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter"],  # not needed (mss gives the screen size)
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WebcamEyeTracker",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # no Terminal window; logs go to ~/Documents/WebcamEyeTracker/app.log
    argv_emulation=False,
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="WebcamEyeTracker")

if sys.platform == "darwin":
    icon = "icon.icns" if os.path.exists("icon.icns") else None
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=icon,
        bundle_identifier="com.local.webcam-eye-tracker",
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": "1.0",
            "LSMinimumSystemVersion": "11.0",
            "NSHighResolutionCapable": True,
            # Shown in the macOS camera-permission dialog. Without this key
            # macOS would silently refuse camera access to the app.
            "NSCameraUsageDescription":
                "Webcam Eye Tracker uses the camera to estimate where on the screen "
                "you are looking. Video is processed locally and never leaves this Mac.",
        },
    )
