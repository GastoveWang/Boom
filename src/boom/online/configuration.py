# 檔案功能：即時 CLI、相機及偵測設定校驗。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""Online CLI and detector/camera configuration validation."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any
from boom.detection.factory import build_detector_config, build_yolo_detector_config, create_detector
from boom.config.defaults import YOLO_MODEL_PATH
from boom.detection.optical_flow import DetectorConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ONLINE_PIPELINES = ("motion", "yolo")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Boom real-time wilderness monitoring using a "
            "FLIR/Teledyne SpinView (Spinnaker) camera."
        )
    )
    parser.add_argument("pipeline", nargs="?", choices=ONLINE_PIPELINES,
                        help="Detection pipeline: optical-flow motion or YOLO (default: motion).")
    camera = parser.add_argument_group("Spinnaker camera")
    camera.add_argument(
        "--serial", help="Camera serial; uses the first camera by default.")
    camera.add_argument("--width", type=int, help="Optional camera ROI width.")
    camera.add_argument("--height", type=int,
                        help="Optional camera ROI height.")
    camera.add_argument("--offset-x", type=int)
    camera.add_argument("--offset-y", type=int)
    camera.add_argument("--camera-fps", type=float, default=30.0)
    camera.add_argument("--exposure-us", type=float)
    camera.add_argument("--gain-db", type=float)
    camera.add_argument("--camera-timeout-ms", type=int, default=1000)

    model = parser.add_argument_group("Detector / GPU inference")
    model.add_argument("--detector", choices=ONLINE_PIPELINES,
                       help="Compatibility alias for the pipeline name.")
    model.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                       help="auto uses CUDA for YOLO when available; motion uses CPU.")
    model.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    model.add_argument("--inference-stride", type=int, default=1)
    model.add_argument("--processing-width", type=int, default=0,
                       help="Resize full camera frame to this width before detection; 0 keeps native resolution.")
    model.add_argument("--yolo-model-path", type=Path, default=YOLO_MODEL_PATH)
    model.add_argument("--yolo-imgsz", type=int, default=640)
    model.add_argument("--yolo-target-classes", nargs="+", metavar="CLASS",
                       help="Model class names to detect, e.g. person car; defaults to the smoke event class.")

    detector = parser.add_argument_group("Optional detector overrides")
    detector.add_argument("--downscale", type=float)
    detector.add_argument("--stride", type=int)
    detector.add_argument("--diff-threshold", type=int)
    detector.add_argument("--min-blob-area", type=float)
    detector.add_argument("--min-growth-ratio", type=float)
    detector.add_argument(
        "--disable-flow-check",
        action="store_true",
        help="Disable the original radial optical-flow validation.",
    )
    detector.add_argument("--debug-tiles", action="store_true")

    runtime = parser.add_argument_group("Runtime/output")
    runtime.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "output" / "realtime"
    )
    runtime.add_argument(
        "--display",
        action="store_true",
        help="Show the live window. Headless logging/output remains active without it.",
    )
    runtime.add_argument("--display-width", type=int, default=1280)
    runtime.add_argument("--max-frames", type=int, default=-1)
    runtime.add_argument(
        "--report-every",
        type=int,
        default=30,
        help="Write a heartbeat log every N processed frames.",
    )

    geo = parser.add_argument_group(
        "Static pose (until live telemetry is connected)")
    geo.add_argument("--drone-lon", type=float)
    geo.add_argument("--drone-lat", type=float)
    geo.add_argument("--drone-alt", type=float)
    geo.add_argument("--drone-yaw", type=float, default=0.0)
    geo.add_argument("--drone-pitch", type=float, default=-45.0)
    geo.add_argument("--drone-roll", type=float, default=0.0)
    return parser


def detector_config_from_args(args: argparse.Namespace) -> DetectorConfig:
    config = build_detector_config()
    overrides: dict[str, Any] = {}
    if args.downscale is not None:
        if not 0 < args.downscale <= 1:
            raise ValueError("--downscale must be in the range (0, 1].")
        overrides["downscale"] = args.downscale
    if args.stride is not None:
        overrides["stride_frames"] = max(args.stride, 1)
    if args.diff_threshold is not None:
        overrides["diff_threshold"] = args.diff_threshold
    if args.min_blob_area is not None:
        overrides["min_blob_area"] = args.min_blob_area
    if args.min_growth_ratio is not None:
        overrides["min_growth_ratio"] = args.min_growth_ratio
    if args.disable_flow_check:
        overrides["enable_radial_flow_check"] = False
    if args.debug_tiles:
        overrides.update(
            draw_debug_tiles=True,
            show_candidate_boxes=True,
            show_status_overlay=True,
        )
    return replace(config, **overrides)


def selected_pipeline(args: argparse.Namespace) -> str:
    pipeline = getattr(args, "pipeline", None)
    detector = getattr(args, "detector", None)
    if pipeline and detector and pipeline != detector:
        raise ValueError("Positional pipeline and --detector must match when both are supplied")
    return pipeline or detector or "motion"


def build_runtime_detector(args: argparse.Namespace):
    """Build one of the two independent real-time detectors."""
    if args.camera_fps <= 0 or args.inference_stride <= 0 or args.yolo_imgsz <= 0:
        raise ValueError("Camera FPS, inference stride and YOLO image size must be positive")
    if args.display_width < 0 or args.processing_width < 0:
        raise ValueError("display-width and processing-width must be nonnegative")
    pipeline = selected_pipeline(args)
    if pipeline == "motion":
        return create_detector("motion", args.camera_fps, config=detector_config_from_args(args))
    if pipeline == "yolo":
        config = replace(build_yolo_detector_config(args), motion_compensation=False)
        return create_detector("yolo", args.camera_fps, config=config)
    raise ValueError(f"Unsupported real-time pipeline: {pipeline}")
