"""Offline pipeline argument parsing and validation."""
from __future__ import annotations

from artillery.common.defaults import CONFIRMED_BOX_HOLD_SEC
from artillery.common.defaults import DETECTOR_BACKEND
from artillery.common.defaults import DISPLAY_SECONDS
from artillery.common.defaults import FUSION_CONFIRMATION_WINDOW_SEC
from artillery.common.defaults import FUSION_MATCH_DISTANCE_PX
from artillery.common.defaults import MAX_FRAMES
from artillery.common.defaults import NEW_SMOKE_WINDOW_SEC
from artillery.common.defaults import PIDNET_DEVICE
from artillery.common.defaults import PIDNET_EARLY_CANDIDATE_THRESHOLD
from artillery.common.defaults import PIDNET_INPUT_HEIGHT
from artillery.common.defaults import PIDNET_INPUT_WIDTH
from artillery.common.defaults import PIDNET_MODEL_PATH
from artillery.common.defaults import PIDNET_THRESHOLD
from artillery.common.defaults import PIDNET_WARMUP_SEC
from artillery.common.defaults import MAP_ROOT
from artillery.common.defaults import DRONE_ICON_PATH
from artillery.common.defaults import START_FRAME
from pathlib import Path
import argparse


def build_arg_parser(
    default_detector: str = DETECTOR_BACKEND,
    *,
    allow_detector_selection: bool = True,
) -> argparse.ArgumentParser:
    algorithm_names = {
        "pidnet": "PIDNet-S",
        "motion": "optical flow",
        "fusion": "optical-flow onset plus PIDNet confirmation",
    }
    algorithm_name = algorithm_names.get(default_detector, default_detector)
    parser = argparse.ArgumentParser(
        description=f"Offline artillery-impact detection using {algorithm_name}."
    )
    parser.add_argument("--video", type=Path,
                        help="Input video; defaults to the first video under data/.")
    parser.add_argument("--reference-image", type=Path,
                        help="DJI image containing GPS/XMP metadata; required unless --no-map is used.")
    map_options = parser.add_mutually_exclusive_group()
    map_options.add_argument("--map-dir", type=Path,
                             help=f"WGS84 GeoTIFF directory (default: {MAP_ROOT}).")
    map_options.add_argument("--no-map", action="store_const", const=None, dest="map_dir",
                             help="Disable map initialization and VO; use the legacy coordinate panel.")
    parser.set_defaults(map_dir=MAP_ROOT)
    parser.add_argument("--map-mpp", type=float, default=0.75,
                        help="Map preview metres per pixel (default: 0.75).")
    parser.add_argument("--ground-height", type=float,
                        help="Camera height above local ground in metres; default: photo RelativeAltitude approximation.")
    parser.add_argument("--allow-gps-seed", action="store_true",
                        help="Allow explicitly provisional GPS-photo VO if photo/map visual matching fails.")
    parser.add_argument("--drone-icon", type=Path, default=DRONE_ICON_PATH)
    parser.add_argument("--display", action="store_true", help="Show the processed video and map window.")
    parser.add_argument("--start-frame", type=int, default=START_FRAME)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--report-every", type=int, default=60,
                        help="Print progress every N frames (default: 60).")
    if allow_detector_selection:
        parser.add_argument(
            "--detector",
            choices=("pidnet", "motion", "fusion"),
            default=default_detector,
            help="Select PIDNet-S, optical flow, or their staged fusion.",
        )
    else:
        parser.set_defaults(detector=default_detector)
    if allow_detector_selection or default_detector in ("pidnet", "fusion"):
        parser.add_argument("--model-path", type=Path, default=PIDNET_MODEL_PATH)
        parser.add_argument(
            "--device", choices=("auto", "cpu", "cuda"), default=PIDNET_DEVICE
        )
        parser.add_argument("--seg-threshold", type=float, default=PIDNET_THRESHOLD)
        parser.add_argument(
            "--early-seg-threshold",
            type=float,
            default=PIDNET_EARLY_CANDIDATE_THRESHOLD,
            help="PIDNet-only weak-smoke threshold used to preserve onset time.",
        )
        parser.add_argument("--model-width", type=int, default=PIDNET_INPUT_WIDTH)
        parser.add_argument("--model-height", type=int, default=PIDNET_INPUT_HEIGHT)
        parser.add_argument("--inference-stride", type=int, default=1)
        parser.add_argument(
            "--new-smoke-window-sec", type=float, default=NEW_SMOKE_WINDOW_SEC
        )
        parser.add_argument("--warmup-sec", type=float, default=PIDNET_WARMUP_SEC)
    if allow_detector_selection or default_detector == "fusion":
        parser.add_argument(
            "--fusion-window-sec",
            type=float,
            default=FUSION_CONFIRMATION_WINDOW_SEC,
            help="Maximum delay from optical onset to PIDNet confirmation.",
        )
        parser.add_argument(
            "--fusion-match-distance",
            type=float,
            default=FUSION_MATCH_DISTANCE_PX,
            help="Maximum optical/PIDNet impact-point distance in source pixels.",
        )
    parser.add_argument("--box-hold-sec", type=float, default=CONFIRMED_BOX_HOLD_SEC)
    parser.add_argument("--panel-hold-sec", type=float, default=DISPLAY_SECONDS)
    return parser


def validate_args(args):
    if args.box_hold_sec < 0 or args.panel_hold_sec < 0:
        raise ValueError("hold durations cannot be negative")
    if args.detector in ("pidnet", "fusion"):
        if not 0.0 < args.seg_threshold < 1.0:
            raise ValueError("--seg-threshold must be between 0 and 1")
        if not 0.0 < args.early_seg_threshold <= args.seg_threshold:
            raise ValueError(
                "--early-seg-threshold must be positive and no greater than --seg-threshold"
            )
        if args.new_smoke_window_sec <= 0:
            raise ValueError("--new-smoke-window-sec must be positive")
        if args.warmup_sec < 0:
            raise ValueError("--warmup-sec cannot be negative")
        if args.model_width <= 0 or args.model_height <= 0 or args.inference_stride <= 0:
            raise ValueError(
                "model dimensions and --inference-stride must be positive")
    if args.detector == "fusion":
        if args.fusion_window_sec <= 0:
            raise ValueError("--fusion-window-sec must be positive")
        if args.fusion_match_distance <= 0:
            raise ValueError("--fusion-match-distance must be positive")
