"""PyInstaller entry; shared launcher also dispatches spawned child processes."""
import os

# Inference imports must not invoke pip for optional training/tracker packages.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from sop_app.launcher import start


if __name__ == "__main__":
    raise SystemExit(start())
