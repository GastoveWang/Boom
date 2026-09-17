"""
==============================================================================
Boom 離線管線 - 命令列參數解析與合法性校驗 (cli.py)
==============================================================================

【檔案定位】
本檔案為離線處理流程的命令列規格定義中心。負責建立 argparse 解析器，定義
輸入媒體、演算法選擇、模型設定、地圖匹配門檻與畫面呈現參數，並在執行前進行
參數約束檢查。

【核心功能】
1. 完整 CLI 參數解析器構建 (`build_arg_parser`)：
   - 支援 `--video`, `--reference-image`, `--map-dir`, `--no-map`, `--display`。
   - 支援演算法切換 (`--detector [pidnet|motion|fusion]`) 及各模型專屬超參數。
   - 支援航拍圖資匹配參數（搜尋半徑、特徵數量、RANSAC 門檻等）。
2. 參數合法性校驗 (`validate_args`)：
   - 檢查數值範圍（如各項門檻必須在 (0, 1) 間、尺寸與時窗不得為負數等）。
3. 地圖匹配設定組裝 (`map_matching_config`)：
   - 將 CLI 中的 `--map-*` 參數轉換為 `MapMatchingConfig` dataclass 實例。

【相依模組】
- 上游：被 `pipelines/main.py`, `run_offline.py`, `offline/__main__.py` 調用。
- 下游：使用 `artillery.common.defaults` 中的常數與設定類別。
==============================================================================
"""
from __future__ import annotations

from boom.config.defaults import CONFIRMED_BOX_HOLD_SEC
from boom.config.defaults import DETECTOR_BACKEND
from boom.config.defaults import DISPLAY_SECONDS
from boom.config.defaults import FUSION_CONFIRMATION_WINDOW_SEC
from boom.config.defaults import FUSION_MATCH_DISTANCE_PX
from boom.config.defaults import MAX_FRAMES
from boom.config.defaults import NEW_SMOKE_WINDOW_SEC
from boom.config.defaults import PIDNET_DEVICE
from boom.config.defaults import PIDNET_EARLY_CANDIDATE_THRESHOLD
from boom.config.defaults import PIDNET_INPUT_HEIGHT
from boom.config.defaults import PIDNET_INPUT_WIDTH
from boom.config.defaults import PIDNET_MODEL_PATH
from boom.config.defaults import PIDNET_THRESHOLD
from boom.config.defaults import PIDNET_WARMUP_SEC
from boom.config.defaults import MAP_ROOT
from boom.config.defaults import DRONE_ICON_PATH
from boom.config.defaults import START_FRAME
from boom.config.defaults import MapMatchingConfig
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
                             help=f"Georeferenced RGB/grayscale GeoTIFF directory (default: {MAP_ROOT}).")
    map_options.add_argument("--no-map", action="store_const", const=None, dest="map_dir",
                             help="Disable map initialization and VO; use the legacy coordinate panel.")
    parser.set_defaults(map_dir=MAP_ROOT)
    parser.add_argument("--map-mpp", type=float, default=0.75,
                        help="Map preview metres per pixel (default: 0.75).")
    defaults = MapMatchingConfig()
    parser.add_argument("--map-matcher-backend", choices=("superpoint_lightglue", "sift"),
                        default=defaults.matcher_backend)
    parser.add_argument("--map-device", choices=("auto", "cpu", "cuda"), default=defaults.device)
    parser.add_argument("--map-lightglue-filter-threshold", type=float,
                        default=defaults.lightglue_filter_threshold)
    for name, value_type in (("search_radius_m", float), ("max_candidates", int),
                             ("roi_size_m", float), ("max_image_size", int),
                             ("max_features", int), ("ratio_threshold", float),
                             ("ransac_threshold_px", float), ("min_inliers", int),
                             ("min_inlier_ratio", float), ("max_reprojection_error_px", float),
                             ("min_coverage", float), ("cache_entries", int)):
        parser.add_argument("--map-" + name.replace("_", "-"), type=value_type,
                            default=getattr(defaults, name))
    parser.add_argument("--map-debug", action="store_true",
                        help="Save each map-match call's ROI/matches/inliers/footprint diagnostics.")
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
    map_matching_config(args)
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


def map_matching_config(args):
    return MapMatchingConfig(**{
        name: getattr(args, "map_" + name, field.default)
        for name, field in MapMatchingConfig.__dataclass_fields__.items()
    })
