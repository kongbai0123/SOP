# -*- mode: python ; coding: utf-8 -*-
"""Windows x64 onedir payload. Run through tools/build_payload.py."""
from importlib import metadata
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

ROOT = Path(SPECPATH).resolve().parent
BUILD = ROOT / "installer" / ".build"
GENERATED = BUILD / "generated"

datas = [
    (str(GENERATED / "example_sop.json"), "sops"),
    (str(ROOT / "sop_app" / "ui" / "assets"), "sop_app/ui/assets"),
    (str(GENERATED / "notices"), "notices"),
]
for package in ("torch", "torchvision", "ultralytics", "ultralytics-thop", "ultralytics-platform",
                "numpy", "opencv-python-headless", "PySide6", "PySide6_Essentials", "PySide6_Addons",
                "shiboken6", "matplotlib", "polars", "psutil", "comtypes", "pygrabber", "cloudpickle",
                "nvidia-ml-py", "httpx", "requests"):
    try:
        datas += copy_metadata(package)
    except metadata.PackageNotFoundError:
        raise SystemExit(f"Required runtime distribution missing: {package}")

# Qt's existing hooks collect only imported modules and their required plugins.
# Matplotlib must use Agg; otherwise its hook pulls GUI backends into an AI helper.
qt_modules = {"QtCore", "QtGui", "QtWidgets", "QtNetwork", "QtSvg", "QtOpenGL", "QtOpenGLWidgets"}
qt_root = Path(metadata.distribution("PySide6_Essentials").locate_file("PySide6"))
excluded_qt = ["PySide6." + item.stem for item in qt_root.glob("Qt*.pyd") if item.stem not in qt_modules]
excluded_qt += ["PySide6." + item.stem for item in Path(metadata.distribution("PySide6_Addons").locate_file("PySide6")).glob("Qt*.pyd") if item.stem not in qt_modules]

a = Analysis(
    [str(ROOT / "installer" / "frozen_entry.py")],
    pathex=[str(ROOT)], binaries=[], datas=datas,
    hiddenimports=["sop_app.ui.startup_window", "sop_app.installer_smoke", "cv2", "torchvision", "torchvision.models.detection",
                   "torchvision.ops", "ultralytics.models", "ultralytics.models.yolo.model",
                   "ultralytics.models.rtdetr.model", "PySide6.QtSvg", "polars", "pynvml"],
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=[],
    excludes=excluded_qt + ["tkinter", "_tkinter", "PyQt5", "PyQt6", "PySide2", "tensorflow", "keras",
                            "tensorboard", "IPython", "jupyter", "notebook", "pytest"],
    noarchive=False, optimize=0,
)
# The Codex host's PATH contains document-rendering tools (including a different
# ICU build). None are application dependencies or valid deployment sources.
for destination, source, kind in a.binaries:
    if "codex-runtimes" in str(source).lower():
        raise SystemExit(f"Unrelated host-tool native library leaked into payload: {source}")
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="SOP", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False, disable_windowed_traceback=False,
    icon=str(GENERATED / "SOP.ico"), version=str(GENERATED / "version-info.txt"),
    uac_admin=False, contents_directory="_internal",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SOP")
