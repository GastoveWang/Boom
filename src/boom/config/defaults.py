"""Shared defaults. Existing detector thresholds are preserved."""
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MapMatchingConfig:
    """Bounded nearby search and geometric validation; distances are metres."""

    search_radius_m: float = 1000.0
    max_candidates: int = 9
    roi_size_m: float = 1000.0
    max_image_size: int = 1600
    max_features: int = 2048
    matcher_backend: str = "superpoint_lightglue"
    device: str = "auto"
    lightglue_filter_threshold: float = 0.1
    ratio_threshold: float = 0.70
    ransac_threshold_px: float = 3.0
    min_inliers: int = 16
    min_inlier_ratio: float = 0.35
    max_reprojection_error_px: float = 2.5
    min_coverage: float = 0.02
    cache_entries: int = 9
    debug: bool = False

    def __post_init__(self):
        if self.matcher_backend not in ("superpoint_lightglue", "sift"):
            raise ValueError("Unsupported map matcher_backend")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Map matcher device must be auto, cpu or cuda")
        if not 0 <= self.lightglue_filter_threshold <= 1:
            raise ValueError("lightglue_filter_threshold must be in [0, 1]")
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

# PIDNet-S semantic smoke segmentation + new-smoke tracking.  The first second
# is the only interval in which a newly created track may become an impact event.
DETECTOR_BACKEND = "pidnet"
PIDNET_MODEL_PATH = PROJECT_ROOT / "models" / "pretrained" / "sam_sup_pidnet_s.pt"
PIDNET_DEVICE = "auto"
PIDNET_INPUT_WIDTH = 960
PIDNET_INPUT_HEIGHT = 544
PIDNET_THRESHOLD = 0.80
PIDNET_EARLY_CANDIDATE_THRESHOLD = 0.45
PIDNET_MIN_COMPONENT_AREA = 30
PIDNET_EARLY_CANDIDATE_MIN_AREA = 10
PIDNET_WARMUP_SEC = 0.50
NEW_SMOKE_WINDOW_SEC = 1.0
PIDNET_CONFIRMATION_HITS = 2
PIDNET_MIN_CONFIRMATION_CONFIDENCE = 0.72
FUSION_CONFIRMATION_WINDOW_SEC = 1.0
FUSION_MATCH_DISTANCE_PX = 220.0


DRONE_LON = None
DRONE_LAT = None
DRONE_ALT = None
DRONE_YAW = 0.0
DRONE_PITCH = 20.0
DRONE_ROLL = 0.0
