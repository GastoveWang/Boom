from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import csv
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    """
    Central place for field-tuning parameters.

    The defaults are intentionally conservative so the script can run as a
    proof-of-concept on 4K drone videos after downsampling.
    """

    # Phase 1: preprocessing + ego-motion compensation.
    downscale: float = 0.5
    stride_frames: int = 5
    # When set, use a time-normalized frame gap so 30/60 FPS sources have the
    # same temporal baseline. ``stride_frames`` remains the compatibility
    # fallback for existing callers.
    diff_interval_sec: Optional[float] = None
    gaussian_blur_ksize: int = 5

    orb_features: int = 1500
    orb_ratio_test: float = 0.75
    homography_ransac_thresh: float = 4.0
    homography_min_matches: int = 20
    homography_min_inliers: int = 15
    homography_min_inlier_ratio: float = 0.35

    # Reject short periods where camera motion/focus makes frame differencing unreliable.
    camera_guard_enabled: bool = True
    max_camera_translation_px: float = 6.0
    max_camera_rotation_deg: float = 0.20
    max_camera_scale_change: float = 0.008
    min_focus_sharpness_ratio: float = 0.65
    max_focus_sharpness_ratio: float = 1.55
    max_global_residual_ratio: float = 0.10
    camera_guard_cooldown_frames: int = 4

    shi_tomasi_max_corners: int = 800
    shi_tomasi_quality_level: float = 0.01
    shi_tomasi_min_distance: float = 8.0

    # Phase 2: strided frame differencing.
    diff_threshold: int = 40
    open_kernel_size: int = 3
    close_kernel_size: int = 7
    open_iterations: int = 1
    close_iterations: int = 1
    min_blob_area: float = 220.0
    max_blob_area_ratio: float = 0.35
    max_motion_ratio: float = 0.20
    # A newly appearing dust/smoke patch normally changes local intensity in
    # one dominant direction. Translation/parallax tends to create paired
    # bright and dark edges and therefore a much lower polarity score.
    min_residual_polarity_ratio: float = 0.68
    min_residual_signed_coherence: float = 0.24
    min_candidate_fill_ratio: float = 0.16
    max_initial_motion_history_overlap: float = 0.45
    # Small/distant impact branch.  Small blobs may enter tracking below the
    # normal area threshold, but only when they are strong relative to their
    # local noise and accumulate consistent temporal evidence.
    small_min_blob_area: float = 4.0
    small_diff_floor: float = 8.0
    small_noise_window_size: int = 31
    small_noise_sigma: float = 2.2
    small_min_signal_to_noise: float = 2.2
    small_min_polarity_ratio: float = 0.62
    small_min_signed_coherence: float = 0.18
    small_min_fill_ratio: float = 0.18
    small_min_growth_ratio: float = 1.35
    small_max_center_shift_ratio: float = 1.0
    small_confirmation_hits_required: int = 3
    small_confirmation_min_area_retention: float = 0.40
    small_track_match_distance: float = 12.0
    small_min_cumulative_signal: float = 700.0
    small_edge_guard_ratio: float = 0.12
    small_edge_min_fill_ratio: float = 0.40
    motion_history_decay_sec: float = 2.0
    motion_history_active_threshold: float = 0.20
    motion_history_dilate_px: int = 7
    # A bad perspective fit often produces many disconnected residual blobs
    # spread across a 1440x1080 drone frame at once.  A real impact remains
    # spatially local even when its dust cloud fragments into several blobs.
    max_distributed_candidates: int = 8
    min_candidate_field_span_ratio: float = 0.30
    candidate_cluster_radius_ratio: float = 0.12
    candidate_cluster_min_dominance: float = 0.45
    border_ignore_px: int = 12
    border_ignore_ratio: float = 0.04
    feature_border_margin_px: int = 24
    valid_mask_erode_px: int = 10
    rotation_mask_expand_per_deg: float = 1.5
    max_rotation_deg: float = 12.0

    # Phase 3: radial expansion check.
    track_match_distance: float = 180.0
    growth_window_frames: int = 3
    min_growth_ratio: float = 2.2
    min_confirm_area: float = 220.0
    min_dimension_growth_ratio: float = 1.08
    min_expansion_balance: float = 0.55
    max_center_shift_ratio: float = 0.65
    strong_growth_override_ratio: float = 4.5
    confirmation_hits_required: int = 2
    confirmation_window_frames: int = 3
    confirmation_min_area_retention: float = 0.75
    confirmation_min_short_growth_ratio: float = 1.20

    enable_radial_flow_check: bool = True
    flow_min_blob_area: float = 180.0
    flow_edge_dilate_px: int = 4
    flow_max_corners: int = 120
    flow_quality_level: float = 0.01
    flow_min_distance: int = 4
    flow_win_size: int = 21
    flow_max_level: int = 3
    flow_min_vectors: int = 8
    flow_min_magnitude: float = 0.7
    radial_cosine_threshold: float = 0.35
    min_radial_outward_ratio: float = 0.60

    # Phase 4: lifecycle / silence logic.
    stop_growth_ratio: float = 1.08
    stop_patience_frames: int = 4
    max_tentative_age_frames: int = 8
    max_missed_frames: int = 6
    max_event_age_sec: float = 3.0
    confirmed_box_hold_sec: float = 0.0

    # Re-association rules to avoid repeatedly creating new IDs for the same target.
    trajectory_match_distance: float = 140.0
    trajectory_extrapolation_scale: float = 2.0
    recent_confirmed_match_sec: float = 10.0
    # A confirmed impact leaves a drifting dust/smoke plume.  Keep a separate
    # spatio-temporal memory of that plume so its moving front is not emitted
    # as a new impact every time the short-lived tracker expires.  The memory
    # follows the latest plume position instead of freezing the original box,
    # which still permits a spatially independent impact behind the plume.
    post_event_memory_enabled: bool = True
    post_event_memory_max_events: int = 12
    post_event_memory_max_sec: float = 600.0
    post_event_memory_idle_sec: float = 600.0
    post_event_spatial_history_sec: float = 600.0
    post_event_match_distance: float = 110.0
    post_event_anchor_match_distance: float = 140.0
    post_event_trajectory_distance: float = 80.0
    post_event_follow_base_distance: float = 22.0
    post_event_follow_speed_px_sec: float = 90.0
    post_event_follow_max_distance: float = 140.0
    post_event_follow_frontier_sec: float = 0.75
    post_event_follow_max_branches: int = 5
    post_event_bbox_margin_ratio: float = 0.65
    post_event_min_bbox_margin_px: float = 18.0
    post_event_max_bbox_margin_px: float = 60.0
    post_event_retrigger_min_area: float = 70.0
    post_event_retrigger_min_growth_ratio: float = 4.5
    post_event_small_retrigger_min_area: float = 35.0
    post_event_small_retrigger_growth_ratio: float = 8.0
    post_event_small_retrigger_signal_scale: float = 4.0

    # Candidate splitting for nearby simultaneous bursts.
    enable_blob_splitting: bool = True
    split_blob_area: float = 2500.0
    split_peak_threshold: float = 0.45
    split_min_peak_area: int = 6
    split_kernel_size: int = 3

    # Visualization.
    draw_debug_tiles: bool = True
    display_scale: float = 1.0
    show_candidate_boxes: bool = False
    show_status_overlay: bool = True


@dataclass
class FrameBundle:
    index: int
    bgr: np.ndarray
    gray: np.ndarray
    diff_gray: np.ndarray
    sharpness: float


@dataclass
class HomographyResult:
    H: Optional[np.ndarray]
    method: str
    matches: int
    inliers: int
    inlier_ratio: float
    transform_kind: str = "unknown"
    dx: Optional[float] = None
    dy: Optional[float] = None
    scale: Optional[float] = None
    rotation_deg: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.H is not None


@dataclass
class RadialFlowResult:
    outward_ratio: Optional[float]
    mean_cosine: Optional[float]
    valid_vectors: int


@dataclass
class BlobCandidate:
    contour: np.ndarray
    area: float
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    residual_polarity_ratio: float = 0.0
    residual_signed_coherence: float = 0.0
    fill_ratio: float = 0.0
    mean_abs_delta: float = 0.0
    prior_motion_overlap: float = 0.0
    local_noise: float = 1.0
    signal_to_noise: float = 0.0
    small_scale: bool = False
    edge_risk: bool = False
    radial_flow: Optional[RadialFlowResult] = None


@dataclass
class EventTrack:
    event_id: Optional[int]
    confirmed: bool
    first_seen_frame: int
    last_seen_frame: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    confirm_frame: Optional[int] = None
    missed_frames: int = 0
    non_growth_frames: int = 0
    confirmation_hits: int = 0
    last_signature_frame: Optional[int] = None
    initial_motion_overlap: float = 0.0
    suppressed: bool = False
    suppression_source_id: Optional[int] = None
    small_scale: bool = False
    observation_count: int = 0
    cumulative_signal: float = 0.0
    peak_growth: float = 1.0
    area_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=16))
    growth_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=16))
    radial_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=16))
    centroid_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=16))
    bbox_history: Deque[Tuple[int, int, int, int]] = field(
        default_factory=lambda: deque(maxlen=16))

    @property
    def current_area(self) -> float:
        return float(self.area_history[-1]) if self.area_history else 0.0


@dataclass
class ConfirmedEvent:
    event_id: int
    frame_idx: int
    timestamp_sec: float
    bbox: Tuple[int, int, int, int]
    area: float
    growth_ratio: float
    radial_ratio: Optional[float]


@dataclass
class OpticalCandidateEvent:
    """First local-change observation exposed to the fusion pipeline only."""

    candidate_id: int
    frame_idx: int
    timestamp_sec: float
    bbox: Tuple[int, int, int, int]
    area: float
    signal_to_noise: float
    residual_polarity_ratio: float


@dataclass
class ConfirmedTrackSnapshot:
    event_id: int
    confirm_frame: int
    last_seen_frame: int
    centroid_history: List[Tuple[float, float]]
    bbox_history: List[Tuple[int, int, int, int]]


@dataclass
class EventMemory:
    """Recent spatio-temporal tube for one confirmed impact plume."""

    event_id: int
    confirm_frame: int
    last_update_frame: int
    centroid_history: Deque[Tuple[int, Tuple[float, float]]] = field(
        default_factory=lambda: deque(maxlen=256)
    )
    bbox_history: Deque[Tuple[int, Tuple[int, int, int, int]]] = field(
        default_factory=lambda: deque(maxlen=256)
    )
    # A verified plume observation and its frame provide a scene-aligned
    # anchor.  When the drone translates or zooms, the anchor is projected
    # into the current view before duplicate-event suppression is evaluated.
    anchor_frame: Optional[int] = None
    anchor_gray: Optional[np.ndarray] = None
    anchor_centroid: Optional[Tuple[float, float]] = None
    anchor_bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class VideoRunSummary:
    video_name: str
    processed_frames: int
    confirmed_events: int
    snapshots_saved: int
    output_video_path: Optional[Path]


@dataclass
class DetectionArtifact:
    video_name: str
    event_id: int
    frame_idx: int
    timestamp_sec: float
    area: float
    growth_ratio: float
    radial_ratio: Optional[float]
    full_frame_path: Path
    crop_path: Path


