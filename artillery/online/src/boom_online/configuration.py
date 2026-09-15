"""Online CLI and detector/camera configuration validation."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any
from artillery.common.detector_factory import build_detector_config
from artillery.offline.src.detectors.optical_flow_smoke_detector import DetectorConfig

ONLINE_ROOT = Path(__file__).resolve().parents[2]

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Boom real-time artillery impact detection using a "
            "FLIR/Teledyne SpinView (Spinnaker) camera."
        )
    )
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
        "--output-dir", type=Path, default=ONLINE_ROOT / "output"
    )
    runtime.add_argument(
        "--display",
        action="store_true",
        help="Show the live window. Headless logging/output remains active without it.",
    )
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
