# 檔案功能：共用預設常數與地圖匹配設定結構。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""Shared defaults. Existing detector thresholds are preserved."""
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import math
from .loader import get_config


@dataclass(frozen=True)
class MapMatchingConfig:
    """Bounded nearby search and geometric validation; distances are metres."""

    search_radius_m: float = 1000.0
    max_candidates: int = 9
    roi_size_m: float = 1000.0
    max_image_size: int = 1600
    max_features: int = 2048
    matcher_backend: str = "edm"
    device: str = "auto"
    lightglue_filter_threshold: float = 0.1
    edm_model_path: str = "models/pretrained/edm_w640_h480_topk1680.onnx"
    edm_conf_threshold: float = 0.2
    edm_sigma_threshold: float = 1e-6
    ratio_threshold: float = 0.70
    ransac_threshold_px: float = 3.0
    min_inliers: int = 16
    min_inlier_ratio: float = 0.35
    max_reprojection_error_px: float = 2.5
    min_coverage: float = 0.02
    cache_entries: int = 9
    debug: bool = False

    def __post_init__(self):
        if self.matcher_backend not in ("superpoint_lightglue", "edm", "sift"):
            raise ValueError("Unsupported map matcher_backend")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Map matcher device must be auto, cpu or cuda")
        if not 0 <= self.lightglue_filter_threshold <= 1:
            raise ValueError("lightglue_filter_threshold must be in [0, 1]")
        if not 0 <= self.edm_conf_threshold <= 1:
            raise ValueError("edm_conf_threshold must be in [0, 1]")
        if not 0 <= self.edm_sigma_threshold <= 1:
            raise ValueError("edm_sigma_threshold must be in [0, 1]")
        if self.matcher_backend == "edm" and not self.edm_model_path:
            raise ValueError("edm_model_path is required")
        for name in ("search_radius_m", "roi_size_m", "ransac_threshold_px",
                     "max_reprojection_error_px"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_candidates", "max_image_size", "max_features", "cache_entries"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.min_inliers, int) or self.min_inliers < 4:
            raise ValueError("min_inliers must be at least 4")
        for name in ("ratio_threshold", "min_inlier_ratio", "min_coverage"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAP_ROOT = PROJECT_ROOT / "asset" / "maps"
DRONE_ICON_PATH = PROJECT_ROOT / "asset" / "pictures" / "drone.png"

IMAGE_PATH: Optional[str] = None
VIDEO_PATH: Optional[str] = None
OUTPUT_ROOT = PROJECT_ROOT / "output" / "offline"

UI_MODE = "client"
DISPLAY_SCALE = 0.5
DISPLAY_SECONDS = 10.0
START_FRAME = 0
MAX_FRAMES = -1
NO_DISPLAY = True
PANEL_WIDTH = 500

UI_FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msjhbd.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
]


DOWNSCALE = 0.7
STRIDE_FRAMES = 2
DIFF_INTERVAL_SEC = 0.12
DIFF_THRESHOLD = 18
MIN_BLOB_AREA = 15.0
MIN_CONFIRM_AREA = 35.0
MIN_GROWTH_RATIO = 1.3
OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 7
CLOSE_ITERATIONS = 1
MIN_RESIDUAL_POLARITY_RATIO = 0.60
MIN_RESIDUAL_SIGNED_COHERENCE = 0.24
MIN_CANDIDATE_FILL_RATIO = 0.16
MAX_INITIAL_MOTION_HISTORY_OVERLAP = 0.55
SMALL_MIN_BLOB_AREA = 4.0
SMALL_DIFF_FLOOR = 8.0
SMALL_NOISE_WINDOW_SIZE = 31
SMALL_NOISE_SIGMA = 2.2
SMALL_MIN_SIGNAL_TO_NOISE = 2.2
SMALL_MIN_GROWTH_RATIO = 1.35
SMALL_MAX_CENTER_SHIFT_RATIO = 1.0
SMALL_CONFIRMATION_HITS_REQUIRED = 3
SMALL_MIN_CUMULATIVE_SIGNAL = 700.0
MAX_DISTRIBUTED_CANDIDATES = 8
MIN_CANDIDATE_FIELD_SPAN_RATIO = 0.30
CONFIRMATION_HITS_REQUIRED = 2
STOP_GROWTH_RATIO = 1.08
MAX_EVENT_AGE_SEC = 3.0
CONFIRMED_BOX_HOLD_SEC = 1.5
TRACK_MATCH_DISTANCE = 200.0
TRAJECTORY_MATCH_DISTANCE = 270.0
RECENT_CONFIRMED_MATCH_SEC = 10.0
POST_EVENT_MEMORY_MAX_SEC = 600.0
POST_EVENT_MEMORY_IDLE_SEC = 600.0
POST_EVENT_SPATIAL_HISTORY_SEC = 600.0
POST_EVENT_MATCH_DISTANCE = 110.0
POST_EVENT_TRAJECTORY_DISTANCE = 80.0
POST_EVENT_RETRIGGER_MIN_AREA = 70.0
POST_EVENT_RETRIGGER_MIN_GROWTH_RATIO = 4.5
MAX_MISSED_FRAMES = 6
MIN_DIMENSION_GROWTH_RATIO = 1.08
MAX_CENTER_SHIFT_RATIO = 0.65
SPLIT_BLOB_AREA = 2500.0
ENABLE_RADIAL_FLOW_CHECK = True
DRAW_DEBUG_TILES = False
SHOW_CANDIDATE_BOXES = True

DETECTOR_BACKEND = "motion"

# YOLO instance segmentation / object detection for boom events
YOLO_MODEL_PATH = PROJECT_ROOT / get_config()["detector"]["yolo"]["model_path"]
YOLO_DEVICE = "auto"
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESHOLD = 0.50
YOLO_IOU_THRESHOLD = 0.45
YOLO_TARGET_CLASSES = tuple(get_config()["detector"]["yolo"]["target_classes"])
YOLO_CONFIRMATION_HITS = 2
YOLO_TRACK_MAX_MISSED_SEC = 2.0
YOLO_TRACK_MATCH_DISTANCE = 140.0


DRONE_LON = None
DRONE_LAT = None
DRONE_ALT = None
DRONE_YAW = 0.0
DRONE_PITCH = 20.0
DRONE_ROLL = 0.0
