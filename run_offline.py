#!/usr/bin/env python3
"""Run the offline video pipeline with either motion or YOLO detection."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from boom.config import get_config
from boom.pipelines.main import main as run_pipeline


def main():
    args = list(sys.argv[1:])
    detector = get_config()["detector"]["backend"]
    if args and args[0].lower() in ("motion", "yolo"):
        detector = args.pop(0).lower()
    elif args and args[0].lower() in ("optical_flow", "flow"):
        detector = "motion"
        args.pop(0)
    elif args and args[0].lower() in ("yolo-seg", "yolov8", "yolo26", "yolov11"):
        detector = "yolo"
        args.pop(0)

    if not any(arg == "--detector" or arg.startswith("--detector=") for arg in args):
        args = ["--detector", detector, *args]
    return run_pipeline(default_detector=detector, allow_detector_selection=True, argv=args)


if __name__ == "__main__":
    sys.exit(main() or 0)
