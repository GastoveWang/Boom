from __future__ import annotations

import argparse
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


class InstantSmokeDustDetector:
    """
    Proof-of-concept detector for instantaneous abnormal smoke/dust events
    under drone ego-motion.

    Pipeline:
      Phase 1. Downsample + estimate homography to compensate ego-motion.
      Phase 2. Perform strided frame differencing on aligned frames.
      Phase 3. Filter candidates by short-term area explosion and optional
               radial outward optical flow.
      Phase 4. Maintain event IDs only while the blob is still in its early,
               rapidly expanding lifecycle.
    """

    def __init__(self, config: DetectorConfig, fps: float) -> None:
        self.config = config
        self.fps = fps if fps > 0 else 30.0
        self.max_event_age_frames = max(
            int(round(self.config.max_event_age_sec * self.fps)), 1)
        self.confirmed_box_hold_frames = max(
            int(round(self.config.confirmed_box_hold_sec * self.fps)), 0)
        self.recent_confirmed_match_frames = max(
            int(round(self.config.recent_confirmed_match_sec * self.fps)), 1)
        if self.config.diff_interval_sec is not None and self.config.diff_interval_sec > 0:
            self.stride_frames = max(
                int(round(self.config.diff_interval_sec * self.fps)), 1)
        else:
            self.stride_frames = max(int(self.config.stride_frames), 1)
        self.history: Deque[FrameBundle] = deque(
            maxlen=max(self.stride_frames + 1, 2))
        self.tracks: Dict[int, EventTrack] = {}
        self.recent_confirmed_tracks: Deque[ConfirmedTrackSnapshot] = deque(maxlen=64)
        self.event_memories: Dict[int, EventMemory] = {}
        self.pending_confirmations: List[ConfirmedEvent] = []
        self.pending_candidates: List[OpticalCandidateEvent] = []
        self.next_track_id = 1
        self.next_event_id = 1
        self.guard_cooldown = 0
        self.active_guard_reason: Optional[str] = None
        self.guard_counts: Dict[str, int] = {}
        self.motion_history: Optional[np.ndarray] = None
        self.orb = cv2.ORB_create(
            nfeatures=self.config.orb_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=15,
        )

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        self.pending_confirmations = []
        self.pending_candidates = []
        bundle = self._build_frame_bundle(frame_bgr, frame_idx)
        self.history.append(bundle)

        vis = bundle.bgr.copy()
        zero_debug = np.zeros(bundle.gray.shape, dtype=np.uint8)

        if len(self.history) <= self.stride_frames:
            self._expire_tracks_without_measurement(frame_idx)
            return self._draw_visualization(
                vis,
                frame_idx,
                zero_debug,
                zero_debug,
                None,
                measurement_ok=False,
                candidates=[],
            )

        current = self.history[-1]
        reference = self.history[0]
        self._decay_motion_history(current.gray.shape)

        # Align T-N onto T so that most of the residual motion comes from the
        # event itself instead of the drone platform motion.
        align_result = self.estimate_homography(reference.gray, current.gray)
        candidates: List[BlobCandidate] = []
        diff_img = zero_debug
        motion_mask = zero_debug
        measurement_ok = False

        if align_result.ok:
            aligned_ref, valid_mask = self._warp_reference(
                reference.diff_gray, current.gray.shape, align_result)
            guard_reason = self._camera_guard_reason(
                reference, current, align_result, aligned_ref, valid_mask)
            guard_blocked = self._update_camera_guard(guard_reason, frame_idx)
            if not guard_blocked:
                diff_img, motion_mask, candidates, measurement_ok = self._extract_candidates(
                    current.diff_gray,
                    aligned_ref,
                    valid_mask,
                )

            if measurement_ok and candidates and self.config.enable_radial_flow_check and len(self.history) >= 2:
                prev_frame = self.history[-2]
                prev_align = self.estimate_homography(
                    prev_frame.gray, current.gray)
                if prev_align.ok:
                    # Optical flow is evaluated after compensating the global
                    # background motion between T-1 and T.
                    aligned_prev, _ = self._warp_reference(
                        prev_frame.gray, current.gray.shape, prev_align)
                    self._populate_radial_flow(
                        current.gray, aligned_prev, candidates)

        self._update_tracks(candidates, frame_idx, measurement_ok)
        if measurement_ok:
            self._update_motion_history(motion_mask)

        return self._draw_visualization(
            vis,
            frame_idx,
            diff_img,
            motion_mask,
            align_result,
            measurement_ok,
            candidates,
        )

    def consume_pending_confirmations(self) -> List[ConfirmedEvent]:
        confirmations = self.pending_confirmations
        self.pending_confirmations = []
        return confirmations

    def consume_pending_candidates(self) -> List[OpticalCandidateEvent]:
        candidates = self.pending_candidates
        self.pending_candidates = []
        return candidates

    def camera_guard_summary(self) -> Dict[str, int]:
        return dict(self.guard_counts)

    def _decay_motion_history(self, shape: Tuple[int, int]) -> None:
        if self.motion_history is None or self.motion_history.shape != shape:
            self.motion_history = np.zeros(shape, dtype=np.float32)
            return
        decay_frames = max(self.config.motion_history_decay_sec * self.fps, 1.0)
        self.motion_history *= float(np.exp(-1.0 / decay_frames))

    def _update_motion_history(self, motion_mask: np.ndarray) -> None:
        if self.motion_history is None or self.motion_history.shape != motion_mask.shape:
            self.motion_history = np.zeros(motion_mask.shape, dtype=np.float32)
        history_mask = motion_mask
        dilate_k = _odd_ksize(self.config.motion_history_dilate_px)
        if dilate_k > 1:
            history_mask = cv2.dilate(
                history_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)),
                iterations=1,
            )
        np.maximum(self.motion_history, history_mask.astype(np.float32) / 255.0,
                   out=self.motion_history)

    def _build_frame_bundle(self, frame_bgr: np.ndarray, frame_idx: int) -> FrameBundle:
        if self.config.downscale != 1.0:
            frame_bgr = cv2.resize(
                frame_bgr,
                None,
                fx=self.config.downscale,
                fy=self.config.downscale,
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur_k = _odd_ksize(self.config.gaussian_blur_ksize)
        if blur_k > 1:
            diff_gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        else:
            diff_gray = gray

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return FrameBundle(
            index=frame_idx,
            bgr=frame_bgr,
            gray=gray,
            diff_gray=diff_gray,
            sharpness=sharpness,
        )

    def _camera_guard_reason(
        self,
        reference: FrameBundle,
        current: FrameBundle,
        align_result: HomographyResult,
        aligned_reference: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Optional[str]:
        if not self.config.camera_guard_enabled:
            return None

        if align_result.dx is not None and align_result.dy is not None:
            translation = float(np.hypot(align_result.dx, align_result.dy))
            if translation > self.config.max_camera_translation_px:
                return f"camera_translation({translation:.1f}px)"
        if (
            align_result.rotation_deg is not None
            and abs(align_result.rotation_deg) > self.config.max_camera_rotation_deg
        ):
            return f"camera_rotation({align_result.rotation_deg:.2f}deg)"
        if (
            align_result.scale is not None
            and abs(align_result.scale - 1.0) > self.config.max_camera_scale_change
        ):
            return f"camera_zoom(scale={align_result.scale:.4f})"

        focus_ratio = current.sharpness / max(reference.sharpness, 1e-6)
        if (
            focus_ratio < self.config.min_focus_sharpness_ratio
            or focus_ratio > self.config.max_focus_sharpness_ratio
        ):
            return f"focus_change(ratio={focus_ratio:.2f})"

        residual = cv2.absdiff(current.diff_gray, aligned_reference)
        residual = cv2.bitwise_and(residual, valid_mask)
        valid_pixels = max(int(cv2.countNonZero(valid_mask)), 1)
        residual_pixels = int(np.count_nonzero(
            (residual > self.config.diff_threshold) & (valid_mask > 0)))
        residual_ratio = residual_pixels / valid_pixels
        if residual_ratio > self.config.max_global_residual_ratio:
            return f"global_residual({residual_ratio:.1%})"
        return None

    def _update_camera_guard(self, reason: Optional[str], frame_idx: int) -> bool:
        if reason is not None:
            reason_key = reason.split("(", 1)[0]
            self.guard_counts[reason_key] = self.guard_counts.get(reason_key, 0) + 1
            self.guard_cooldown = max(self.config.camera_guard_cooldown_frames, 0)
            if reason_key != self.active_guard_reason:
                print(f"[GUARD] Pause detection at frame {frame_idx}: {reason}", flush=True)
            self.active_guard_reason = reason_key
            return True

        if self.guard_cooldown > 0:
            self.guard_cooldown -= 1
            return True

        if self.active_guard_reason is not None:
            print(f"[GUARD] Detection resumed at frame {frame_idx}", flush=True)
            self.active_guard_reason = None
        return False

    def estimate_homography(self, src_gray: np.ndarray, dst_gray: np.ndarray) -> HomographyResult:
        # The ground occupies most of these fixed 1440x1080 drone frames, so
        # perspective change is significant even over a short frame interval.
        # Prefer a projective model; the similarity transforms remain robust
        # fallbacks for low-texture frames where a homography cannot be fit.
        orb_h_result = self._estimate_homography_orb(src_gray, dst_gray)
        if orb_h_result.ok:
            return orb_h_result
        lk_h_result = self._estimate_homography_shi_tomasi(src_gray, dst_gray)
        if lk_h_result.ok:
            return lk_h_result

        orb_result = self._estimate_similarity_orb(src_gray, dst_gray)
        if orb_result.ok:
            return orb_result
        return self._estimate_similarity_shi_tomasi(src_gray, dst_gray)

    def _estimate_similarity_orb(self, src_gray: np.ndarray, dst_gray: np.ndarray) -> HomographyResult:
        feature_mask = self._build_feature_mask(src_gray.shape)
        keypoints_a, desc_a = self.orb.detectAndCompute(src_gray, feature_mask)
        keypoints_b, desc_b = self.orb.detectAndCompute(dst_gray, feature_mask)
        if desc_a is None or desc_b is None or len(keypoints_a) < 8 or len(keypoints_b) < 8:
            return HomographyResult(None, "orb_similarity", 0, 0, 0.0)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)

        good_matches = []
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.config.orb_ratio_test * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.config.homography_min_matches:
            return HomographyResult(None, "orb_similarity", len(good_matches), 0, 0.0)

        src_pts = np.float32(
            [keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        return self._estimate_similarity_from_points(
            src_pts,
            dst_pts,
            method="orb_similarity",
            matches=len(good_matches),
        )

    def _estimate_similarity_shi_tomasi(self, src_gray: np.ndarray, dst_gray: np.ndarray) -> HomographyResult:
        corners = cv2.goodFeaturesToTrack(
            src_gray,
            maxCorners=self.config.shi_tomasi_max_corners,
            qualityLevel=self.config.shi_tomasi_quality_level,
            minDistance=self.config.shi_tomasi_min_distance,
            blockSize=7,
            mask=self._build_feature_mask(src_gray.shape),
        )
        if corners is None or len(corners) < self.config.homography_min_matches:
            return HomographyResult(None, "shi_tomasi_similarity", 0, 0, 0.0)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            src_gray,
            dst_gray,
            corners,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None or status is None:
            return HomographyResult(None, "shi_tomasi_similarity", 0, 0, 0.0)

        valid = status.ravel() == 1
        src_valid = corners[valid]
        dst_valid = next_pts[valid]
        matches = int(len(src_valid))
        if matches < self.config.homography_min_matches:
            return HomographyResult(None, "shi_tomasi_similarity", matches, 0, 0.0)

        return self._estimate_similarity_from_points(
            src_valid,
            dst_valid,
            method="shi_tomasi_similarity",
            matches=matches,
        )

    def _estimate_similarity_from_points(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        *,
        method: str,
        matches: int,
    ) -> HomographyResult:
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.homography_ransac_thresh,
        )
        if affine is None or inlier_mask is None:
            return HomographyResult(None, method, matches, 0, 0.0)

        inliers = int(inlier_mask.ravel().sum())
        inlier_ratio = inliers / max(matches, 1)
        if (
            inliers < self.config.homography_min_inliers
            or inlier_ratio < self.config.homography_min_inlier_ratio
        ):
            return HomographyResult(None, method, matches, inliers, inlier_ratio)

        H = np.eye(3, dtype=np.float32)
        H[:2, :] = affine
        dx = float(affine[0, 2])
        dy = float(affine[1, 2])
        scale = float(np.hypot(affine[0, 0], affine[1, 0]))
        rotation_deg = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
        if abs(rotation_deg) > self.config.max_rotation_deg:
            return HomographyResult(None, method, matches, inliers, inlier_ratio)
        return HomographyResult(
            H,
            method,
            matches,
            inliers,
            inlier_ratio,
            transform_kind="similarity",
            dx=dx,
            dy=dy,
            scale=scale,
            rotation_deg=rotation_deg,
        )

    def _estimate_homography_orb(self, src_gray: np.ndarray, dst_gray: np.ndarray) -> HomographyResult:
        keypoints_a, desc_a = self.orb.detectAndCompute(src_gray, None)
        keypoints_b, desc_b = self.orb.detectAndCompute(dst_gray, None)
        if desc_a is None or desc_b is None or len(keypoints_a) < 8 or len(keypoints_b) < 8:
            return HomographyResult(None, "orb", 0, 0, 0.0)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)

        good_matches = []
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.config.orb_ratio_test * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.config.homography_min_matches:
            return HomographyResult(None, "orb", len(good_matches), 0, 0.0)

        src_pts = np.float32(
            [keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, inlier_mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=self.config.homography_ransac_thresh,
        )
        if H is None or inlier_mask is None:
            return HomographyResult(None, "orb", len(good_matches), 0, 0.0)

        inliers = int(inlier_mask.ravel().sum())
        inlier_ratio = inliers / max(len(good_matches), 1)
        if (
            inliers < self.config.homography_min_inliers
            or inlier_ratio < self.config.homography_min_inlier_ratio
        ):
            return HomographyResult(None, "orb", len(good_matches), inliers, inlier_ratio)

        return HomographyResult(
            H,
            "orb",
            len(good_matches),
            inliers,
            inlier_ratio,
            transform_kind="homography",
        )

    def _estimate_homography_shi_tomasi(self, src_gray: np.ndarray, dst_gray: np.ndarray) -> HomographyResult:
        corners = cv2.goodFeaturesToTrack(
            src_gray,
            maxCorners=self.config.shi_tomasi_max_corners,
            qualityLevel=self.config.shi_tomasi_quality_level,
            minDistance=self.config.shi_tomasi_min_distance,
            blockSize=7,
            mask=self._build_feature_mask(src_gray.shape),
        )
        if corners is None or len(corners) < self.config.homography_min_matches:
            return HomographyResult(None, "shi_tomasi_lk", 0, 0, 0.0)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            src_gray,
            dst_gray,
            corners,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None or status is None:
            return HomographyResult(None, "shi_tomasi_lk", 0, 0, 0.0)

        valid = status.ravel() == 1
        src_valid = corners[valid]
        dst_valid = next_pts[valid]
        matches = int(len(src_valid))
        if matches < self.config.homography_min_matches:
            return HomographyResult(None, "shi_tomasi_lk", matches, 0, 0.0)

        H, inlier_mask = cv2.findHomography(
            src_valid,
            dst_valid,
            cv2.RANSAC,
            ransacReprojThreshold=self.config.homography_ransac_thresh,
        )
        if H is None or inlier_mask is None:
            return HomographyResult(None, "shi_tomasi_lk", matches, 0, 0.0)

        inliers = int(inlier_mask.ravel().sum())
        inlier_ratio = inliers / max(matches, 1)
        if (
            inliers < self.config.homography_min_inliers
            or inlier_ratio < self.config.homography_min_inlier_ratio
        ):
            return HomographyResult(None, "shi_tomasi_lk", matches, inliers, inlier_ratio)

        return HomographyResult(
            H,
            "shi_tomasi_lk",
            matches,
            inliers,
            inlier_ratio,
            transform_kind="homography",
        )

    def _warp_reference(
        self,
        src_img: np.ndarray,
        dst_shape: Tuple[int, int],
        align_result: HomographyResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        height, width = dst_shape
        if align_result.transform_kind == "similarity":
            affine = np.asarray(align_result.H[:2, :], dtype=np.float32)
            aligned = cv2.warpAffine(
                src_img,
                affine,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            valid_mask = cv2.warpAffine(
                np.full((src_img.shape[0], src_img.shape[1]), 255, dtype=np.uint8),
                affine,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            aligned = cv2.warpPerspective(
                src_img,
                align_result.H,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            valid_mask = cv2.warpPerspective(
                np.full((src_img.shape[0], src_img.shape[1]), 255, dtype=np.uint8),
                align_result.H,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        valid_mask = self._stabilize_valid_mask(valid_mask, align_result)
        return aligned, valid_mask


    def _build_feature_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        height, width = image_shape
        mask = np.full((height, width), 255, dtype=np.uint8)
        margin = max(int(self.config.feature_border_margin_px), 0)
        if margin <= 0:
            return mask
        margin = min(margin, max(min(height, width) // 3, 1))
        mask[:margin, :] = 0
        mask[-margin:, :] = 0
        mask[:, :margin] = 0
        mask[:, -margin:] = 0
        return mask

    def _stabilize_valid_mask(
        self,
        valid_mask: np.ndarray,
        align_result: HomographyResult,
    ) -> np.ndarray:
        margin = max(int(self.config.valid_mask_erode_px), 0)
        if align_result.rotation_deg is not None:
            margin += int(np.ceil(abs(align_result.rotation_deg) * self.config.rotation_mask_expand_per_deg))

        if margin > 0:
            kernel_size = max(1, margin * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            valid_mask = cv2.erode(valid_mask, kernel, iterations=1)

        valid_mask = cv2.threshold(valid_mask, 254, 255, cv2.THRESH_BINARY)[1]
        return valid_mask

    def _extract_candidates(
        self,
        current_gray: np.ndarray,
        aligned_reference: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[BlobCandidate], bool]:
        signed_diff = current_gray.astype(np.int16) - aligned_reference.astype(np.int16)
        diff = cv2.absdiff(current_gray, aligned_reference)
        diff = cv2.bitwise_and(diff, valid_mask)
        _, motion_mask = cv2.threshold(
            diff, self.config.diff_threshold, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.bitwise_and(motion_mask, valid_mask)

        b = max(
            int(self.config.border_ignore_px),
            int(np.ceil(min(motion_mask.shape) * self.config.border_ignore_ratio)),
        )
        b = min(b, max(min(motion_mask.shape) // 4, 0))
        if b > 0:
            motion_mask[:b, :] = 0
            motion_mask[-b:, :] = 0
            motion_mask[:, :b] = 0
            motion_mask[:, -b:] = 0
            diff[:b, :] = 0
            diff[-b:, :] = 0
            diff[:, :b] = 0
            diff[:, -b:] = 0

        open_k = _odd_ksize(self.config.open_kernel_size)
        close_k = _odd_ksize(self.config.close_kernel_size)
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_k, open_k))
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_k, close_k))

        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=self.config.open_iterations,
        )
        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=self.config.close_iterations,
        )

        # A second, locally normalized branch preserves small distant onsets.
        # It lowers the threshold only inside quiet image regions, so a weak
        # impact can survive without globally admitting compression noise.
        diff_float = diff.astype(np.float32)
        noise_k = _odd_ksize(self.config.small_noise_window_size)
        local_mean = cv2.boxFilter(
            diff_float, cv2.CV_32F, (noise_k, noise_k), normalize=True
        )
        local_square_mean = cv2.boxFilter(
            diff_float * diff_float,
            cv2.CV_32F,
            (noise_k, noise_k),
            normalize=True,
        )
        local_std = np.sqrt(np.maximum(
            local_square_mean - local_mean * local_mean, 0.0
        ))
        adaptive_threshold = np.maximum(
            float(self.config.small_diff_floor),
            local_mean + float(self.config.small_noise_sigma) * local_std,
        )
        small_mask = np.uint8(
            (diff_float >= adaptive_threshold) & (valid_mask > 0)
        ) * 255
        small_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        small_mask = cv2.morphologyEx(
            small_mask, cv2.MORPH_CLOSE, small_close, iterations=1
        )

        frame_area = motion_mask.shape[0] * motion_mask.shape[1]
        valid_area = int(cv2.countNonZero(valid_mask))
        motion_ratio = float(cv2.countNonZero(
            motion_mask)) / max(valid_area, 1)
        if motion_ratio > self.config.max_motion_ratio:
            # When too much of the frame is foreground, it is often caused by
            # a bad alignment estimate rather than a real explosion.
            return diff, motion_mask, [], False

        standard_contours = self._find_candidate_contours(motion_mask)
        small_contours, _ = cv2.findContours(
            small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        standard_boxes = [cv2.boundingRect(contour) for contour in standard_contours]
        contour_entries: List[Tuple[np.ndarray, bool]] = [
            (contour, False) for contour in standard_contours
        ]
        for contour in small_contours:
            small_box = cv2.boundingRect(contour)
            if any(_bboxes_overlap(small_box, box) for box in standard_boxes):
                continue
            contour_entries.append((contour, True))
        candidates: List[BlobCandidate] = []

        for contour, from_small_branch in contour_entries:
            area = float(cv2.contourArea(contour))
            min_area = (
                self.config.small_min_blob_area
                if from_small_branch
                else self.config.min_blob_area
            )
            if area < min_area:
                continue
            if area > self.config.max_blob_area_ratio * frame_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if (
                x <= b
                or y <= b
                or x + w >= motion_mask.shape[1] - b
                or y + h >= motion_mask.shape[0] - b
            ):
                continue
            centroid = _contour_centroid(contour, (x + w / 2.0, y + h / 2.0))
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            shifted_contour = contour.copy().astype(np.int32)
            shifted_contour[:, 0, 0] -= x
            shifted_contour[:, 0, 1] -= y
            cv2.drawContours(contour_mask, [shifted_contour], -1, 255, thickness=-1)

            roi_signed = signed_diff[y: y + h, x: x + w]
            roi_diff = diff[y: y + h, x: x + w]
            active = (
                (contour_mask > 0)
                & (
                    roi_diff
                    >= (
                        self.config.small_diff_floor
                        if from_small_branch
                        else self.config.diff_threshold
                    )
                )
            )
            values = roi_signed[active]
            if values.size:
                positive = int(np.count_nonzero(values > 0))
                negative = int(np.count_nonzero(values < 0))
                changed = max(positive + negative, 1)
                residual_polarity_ratio = max(positive, negative) / changed
                abs_values = np.abs(values.astype(np.float32))
                residual_signed_coherence = float(
                    abs(float(np.sum(values))) / max(float(np.sum(abs_values)), 1e-6)
                )
                mean_abs_delta = float(np.mean(abs_values))
            else:
                residual_polarity_ratio = 0.0
                residual_signed_coherence = 0.0
                mean_abs_delta = 0.0
            fill_ratio = float(area / max(w * h, 1))
            edge_guard = int(np.ceil(
                min(motion_mask.shape) * self.config.small_edge_guard_ratio
            ))
            edge_risk = (
                x <= edge_guard
                or y <= edge_guard
                or x + w >= motion_mask.shape[1] - edge_guard
                or y + h >= motion_mask.shape[0] - edge_guard
            )
            padding = max(max(w, h), 8)
            x0 = max(x - padding, 0)
            y0 = max(y - padding, 0)
            x1 = min(x + w + padding, diff.shape[1])
            y1 = min(y + h + padding, diff.shape[0])
            context = diff[y0:y1, x0:x1]
            context_valid = valid_mask[y0:y1, x0:x1] > 0
            context_values = context[context_valid]
            local_noise = float(np.percentile(context_values, 75)) \
                if context_values.size else 1.0
            local_noise = max(local_noise, 1.0)
            signal_to_noise = mean_abs_delta / local_noise
            small_scale = from_small_branch

            if from_small_branch and (
                signal_to_noise < self.config.small_min_signal_to_noise
                or residual_polarity_ratio < self.config.small_min_polarity_ratio
                or residual_signed_coherence
                < self.config.small_min_signed_coherence
                or fill_ratio < self.config.small_min_fill_ratio
            ):
                continue

            prior_motion_overlap = 0.0
            if self.motion_history is not None:
                history_values = self.motion_history[y: y + h, x: x + w][
                    contour_mask > 0
                ]
                if history_values.size:
                    prior_motion_overlap = float(np.mean(
                        history_values >= self.config.motion_history_active_threshold
                    ))
            candidates.append(
                BlobCandidate(
                    contour=contour,
                    area=area,
                    bbox=(x, y, w, h),
                    centroid=centroid,
                    residual_polarity_ratio=float(residual_polarity_ratio),
                    residual_signed_coherence=residual_signed_coherence,
                    fill_ratio=fill_ratio,
                    mean_abs_delta=mean_abs_delta,
                    prior_motion_overlap=prior_motion_overlap,
                    local_noise=local_noise,
                    signal_to_noise=signal_to_noise,
                    small_scale=small_scale,
                    edge_risk=edge_risk,
                )
            )

        candidates, field_ok = self._filter_distributed_candidate_field(
            candidates, motion_mask.shape
        )
        if not field_ok:
            return diff, motion_mask, [], False

        return diff, motion_mask, candidates, True

    def _filter_distributed_candidate_field(
        self,
        candidates: List[BlobCandidate],
        shape: Tuple[int, int],
    ) -> Tuple[List[BlobCandidate], bool]:
        if len(candidates) < max(self.config.max_distributed_candidates, 1):
            return candidates, True

        height, width = shape
        xs = [candidate.centroid[0] for candidate in candidates]
        ys = [candidate.centroid[1] for candidate in candidates]
        x_span_ratio = (max(xs) - min(xs)) / max(width, 1)
        y_span_ratio = (max(ys) - min(ys)) / max(height, 1)
        min_span = max(self.config.min_candidate_field_span_ratio, 0.0)
        if x_span_ratio < min_span and y_span_ratio < min_span:
            return candidates, True

        radius = max(
            min(height, width) * self.config.candidate_cluster_radius_ratio,
            1.0,
        )
        weights = np.asarray([
            candidate.area * max(
                candidate.mean_abs_delta - candidate.local_noise, 1.0
            )
            for candidate in candidates
        ], dtype=np.float32)
        total_weight = max(float(np.sum(weights)), 1e-6)

        best_index = 0
        best_cluster_indices: List[int] = []
        best_weight = 0.0
        for center_index, center in enumerate(candidates):
            cluster_indices = [
                index
                for index, candidate in enumerate(candidates)
                if _euclidean(center.centroid, candidate.centroid) <= radius
            ]
            cluster_weight = float(np.sum(weights[cluster_indices]))
            if cluster_weight > best_weight:
                best_index = center_index
                best_cluster_indices = cluster_indices
                best_weight = cluster_weight

        dominance = best_weight / total_weight
        if dominance < self.config.candidate_cluster_min_dominance:
            return [], False

        dominant_center = candidates[best_index].centroid
        kept = [
            candidate
            for candidate in candidates
            if _euclidean(dominant_center, candidate.centroid) <= radius * 1.5
        ]
        return kept, True

    def _find_candidate_contours(self, motion_mask: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not self.config.enable_blob_splitting:
            return contours

        refined: List[np.ndarray] = []
        for contour in contours:
            refined.extend(self._split_contour_if_needed(motion_mask, contour))
        return refined

    def _split_contour_if_needed(self, motion_mask: np.ndarray, contour: np.ndarray) -> List[np.ndarray]:
        area = float(cv2.contourArea(contour))
        if area < self.config.split_blob_area:
            return [contour]

        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12:
            return [contour]

        roi = motion_mask[y: y + h, x: x + w]
        if roi.size == 0:
            return [contour]

        fg = np.uint8(roi > 0)
        dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
        if float(dist.max()) < 1.5:
            return [contour]

        _, sure_fg = cv2.threshold(
            dist,
            self.config.split_peak_threshold * float(dist.max()),
            255,
            cv2.THRESH_BINARY,
        )
        sure_fg = np.uint8(sure_fg)

        num_labels, peak_markers, stats, _ = cv2.connectedComponentsWithStats(
            sure_fg)
        valid_peak_labels = [
            label
            for label in range(1, num_labels)
            if stats[label, cv2.CC_STAT_AREA] >= self.config.split_min_peak_area
        ]
        if len(valid_peak_labels) < 2:
            return [contour]

        filtered_fg = np.zeros_like(sure_fg)
        for label in valid_peak_labels:
            filtered_fg[peak_markers == label] = 255

        split_k = _odd_ksize(self.config.split_kernel_size)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (split_k, split_k))
        sure_bg = cv2.dilate(np.uint8(fg * 255), kernel, iterations=1)
        unknown = cv2.subtract(sure_bg, filtered_fg)

        _, markers = cv2.connectedComponents(filtered_fg)
        markers = markers + 1
        markers[unknown > 0] = 0

        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        watershed = cv2.watershed(roi_color, markers)

        split_contours: List[np.ndarray] = []
        for label in range(2, int(watershed.max()) + 1):
            seg = np.uint8(watershed == label) * 255
            seg = cv2.bitwise_and(seg, np.uint8(fg * 255))
            sub_contours, _ = cv2.findContours(
                seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for sub_contour in sub_contours:
                if cv2.contourArea(sub_contour) < self.config.min_blob_area * 0.5:
                    continue
                sub_contour = sub_contour.copy()
                sub_contour[:, 0, 0] += x
                sub_contour[:, 0, 1] += y
                split_contours.append(sub_contour)

        return split_contours if len(split_contours) >= 2 else [contour]

    def _populate_radial_flow(
        self,
        current_gray: np.ndarray,
        aligned_prev_gray: np.ndarray,
        candidates: List[BlobCandidate],
    ) -> None:
        for candidate in candidates:
            if candidate.area < self.config.flow_min_blob_area:
                continue
            candidate.radial_flow = self._compute_radial_flow(
                current_gray, aligned_prev_gray, candidate)

    def _compute_radial_flow(
        self,
        current_gray: np.ndarray,
        aligned_prev_gray: np.ndarray,
        candidate: BlobCandidate,
    ) -> RadialFlowResult:
        x, y, w, h = candidate.bbox
        pad = max(8, self.config.flow_win_size)
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, current_gray.shape[1])
        y1 = min(y + h + pad, current_gray.shape[0])

        curr_roi = current_gray[y0:y1, x0:x1]
        prev_roi = aligned_prev_gray[y0:y1, x0:x1]
        if curr_roi.size == 0 or prev_roi.size == 0:
            return RadialFlowResult(None, None, 0)

        edge_mask = np.zeros(curr_roi.shape, dtype=np.uint8)
        shifted_contour = candidate.contour.copy().astype(np.int32)
        shifted_contour[:, 0, 0] -= x0
        shifted_contour[:, 0, 1] -= y0
        cv2.drawContours(edge_mask, [shifted_contour], -1, 255, thickness=1)

        dilate_k = _odd_ksize(self.config.flow_edge_dilate_px)
        edge_mask = cv2.dilate(
            edge_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)),
            iterations=1,
        )

        points_prev = cv2.goodFeaturesToTrack(
            prev_roi,
            maxCorners=self.config.flow_max_corners,
            qualityLevel=self.config.flow_quality_level,
            minDistance=self.config.flow_min_distance,
            mask=edge_mask,
            blockSize=5,
        )
        if points_prev is None or len(points_prev) < self.config.flow_min_vectors:
            return RadialFlowResult(None, None, 0)

        points_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_roi,
            curr_roi,
            points_prev,
            None,
            winSize=(self.config.flow_win_size, self.config.flow_win_size),
            maxLevel=self.config.flow_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if points_curr is None or status is None:
            return RadialFlowResult(None, None, 0)

        valid = status.ravel() == 1
        points_prev = points_prev[valid].reshape(-1, 2)
        points_curr = points_curr[valid].reshape(-1, 2)
        if len(points_curr) < self.config.flow_min_vectors:
            return RadialFlowResult(None, None, int(len(points_curr)))

        displacements = points_curr - points_prev
        magnitudes = np.linalg.norm(displacements, axis=1)
        strong = magnitudes >= self.config.flow_min_magnitude
        if int(np.count_nonzero(strong)) < self.config.flow_min_vectors:
            return RadialFlowResult(None, None, int(np.count_nonzero(strong)))

        points_curr = points_curr[strong]
        displacements = displacements[strong]

        # A true burst should have edge motion pointing away from the blob
        # center, unlike a vehicle that mostly translates in one direction.
        centroid_local = np.array(
            [candidate.centroid[0] - x0, candidate.centroid[1] - y0], dtype=np.float32)
        radial_vectors = points_curr - centroid_local
        radial_norm = np.linalg.norm(radial_vectors, axis=1) + 1e-6
        flow_norm = np.linalg.norm(displacements, axis=1) + 1e-6
        cosines = np.sum(displacements * radial_vectors,
                         axis=1) / (radial_norm * flow_norm)

        outward_ratio = float(
            np.mean(cosines > self.config.radial_cosine_threshold))
        mean_cosine = float(np.mean(cosines))
        return RadialFlowResult(outward_ratio, mean_cosine, int(len(cosines)))

    def _update_tracks(
        self,
        candidates: List[BlobCandidate],
        frame_idx: int,
        measurement_ok: bool,
    ) -> None:
        self._prune_event_memories(frame_idx)
        if not measurement_ok:
            self._expire_tracks_without_measurement(frame_idx)
            return

        self._follow_event_memories(candidates, frame_idx)
        matches, unmatched_track_ids, unmatched_candidate_ids = self._match_tracks(
            candidates)

        for track_id, candidate_id in matches:
            self._update_single_track(
                self.tracks[track_id], candidates[candidate_id], frame_idx)

        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            track.missed_frames += 1

        for candidate_id in unmatched_candidate_ids:
            self._create_track(candidates[candidate_id], frame_idx)

        self._prune_finished_tracks(frame_idx)

    def _match_tracks(
        self, candidates: List[BlobCandidate]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.tracks or not candidates:
            return [], list(self.tracks.keys()), list(range(len(candidates)))

        pairings: List[Tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for candidate_id, candidate in enumerate(candidates):
                score = self._track_candidate_match_score(track, candidate)
                if score is not None:
                    pairings.append((score, track_id, candidate_id))

        pairings.sort(key=lambda item: item[0])

        matches: List[Tuple[int, int]] = []
        used_tracks = set()
        used_candidates = set()

        for _, track_id, candidate_id in pairings:
            if track_id in used_tracks or candidate_id in used_candidates:
                continue
            used_tracks.add(track_id)
            used_candidates.add(candidate_id)
            matches.append((track_id, candidate_id))

        unmatched_track_ids = [
            track_id for track_id in self.tracks if track_id not in used_tracks]
        unmatched_candidate_ids = [cid for cid in range(
            len(candidates)) if cid not in used_candidates]
        return matches, unmatched_track_ids, unmatched_candidate_ids

    def _track_candidate_match_score(
        self,
        track: EventTrack,
        candidate: BlobCandidate,
    ) -> Optional[float]:
        dist = _euclidean(track.centroid, candidate.centroid)
        if _bboxes_overlap(track.bbox, candidate.bbox):
            return dist * 0.25

        # Do not let a tiny tentative residual steal a newly appearing large
        # plume simply because both fall inside the legacy 200 px association
        # radius.  Genuine scale growth normally keeps the boxes overlapping;
        # without overlap, a small track may only move a short distance.
        if (
            track.small_scale
            and track.current_area < self.config.min_confirm_area
            and dist > self.config.small_track_match_distance
        ):
            return None

        if dist <= self.config.track_match_distance:
            return dist

        trajectory_dist = self._trajectory_distance(track, candidate.centroid)
        if trajectory_dist is not None and trajectory_dist <= self.config.trajectory_match_distance:
            return self.config.track_match_distance + trajectory_dist

        return None

    def _trajectory_distance(
        self,
        track: EventTrack,
        point: Tuple[float, float],
    ) -> Optional[float]:
        return self._trajectory_distance_from_history(list(track.centroid_history), point)

    def _trajectory_distance_from_history(
        self,
        history: List[Tuple[float, float]],
        point: Tuple[float, float],
    ) -> Optional[float]:
        if not history:
            return None
        if len(history) == 1:
            return _euclidean(history[0], point)

        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for start, end in zip(history[:-1], history[1:]):
            segments.append((start, end))

        prev_pt = np.array(history[-2], dtype=np.float32)
        last_pt = np.array(history[-1], dtype=np.float32)
        direction = last_pt - prev_pt
        if float(np.linalg.norm(direction)) > 1e-6:
            extended_end = tuple(
                (last_pt + direction * self.config.trajectory_extrapolation_scale).tolist())
            segments.append((history[-1], extended_end))

        return min(_point_to_segment_distance(point, start, end) for start, end in segments)

    def _create_track(self, candidate: BlobCandidate, frame_idx: int) -> None:
        track = EventTrack(
            event_id=None,
            confirmed=False,
            first_seen_frame=frame_idx,
            last_seen_frame=frame_idx,
            centroid=candidate.centroid,
            bbox=candidate.bbox,
            initial_motion_overlap=candidate.prior_motion_overlap,
            small_scale=candidate.small_scale,
            observation_count=1,
            cumulative_signal=(
                candidate.area
                * max(candidate.mean_abs_delta - candidate.local_noise, 0.0)
            ),
        )
        track.area_history.append(candidate.area)
        track.centroid_history.append(candidate.centroid)
        track.bbox_history.append(candidate.bbox)
        self.tracks[self.next_track_id] = track
        self.pending_candidates.append(
            OpticalCandidateEvent(
                candidate_id=self.next_track_id,
                frame_idx=frame_idx,
                timestamp_sec=frame_idx / self.fps,
                bbox=candidate.bbox,
                area=candidate.area,
                signal_to_noise=candidate.signal_to_noise,
                residual_polarity_ratio=candidate.residual_polarity_ratio,
            )
        )
        self.next_track_id += 1

    def _update_single_track(self, track: EventTrack, candidate: BlobCandidate, frame_idx: int) -> None:
        prev_area = track.current_area if track.area_history else max(
            candidate.area, 1.0)
        instant_growth = candidate.area / max(prev_area, 1.0)
        prev_centroid = track.centroid
        prev_bbox = track.bbox

        track.area_history.append(candidate.area)
        short_growth = self._compute_short_growth(track)
        track.growth_history.append(short_growth)

        track.centroid = candidate.centroid
        track.bbox = candidate.bbox
        track.centroid_history.append(candidate.centroid)
        track.bbox_history.append(candidate.bbox)
        track.last_seen_frame = frame_idx
        track.missed_frames = 0

        radial_ratio = None
        if candidate.radial_flow is not None and candidate.radial_flow.outward_ratio is not None:
            radial_ratio = candidate.radial_flow.outward_ratio
            track.radial_history.append(radial_ratio)

        effective_growth = max(instant_growth, short_growth)
        track.observation_count += 1
        track.small_scale = track.small_scale or candidate.small_scale
        track.cumulative_signal += (
            candidate.area
            * max(candidate.mean_abs_delta - candidate.local_noise, 0.0)
        )
        track.peak_growth = max(track.peak_growth, effective_growth)
        expansion_pass, expansion_metrics = self._check_expansion_motion(
            prev_bbox, candidate.bbox, prev_centroid, candidate.centroid)
        if effective_growth < self.config.stop_growth_ratio:
            # Once the area stops expanding rapidly, we start counting down to
            # silence the event instead of tracking diffuse smoke forever.
            track.non_growth_frames += 1
        else:
            track.non_growth_frames = 0

        if not track.confirmed:
            if track.suppressed:
                return
            radial_pass = False
            if (
                candidate.radial_flow is not None
                and candidate.radial_flow.outward_ratio is not None
                and candidate.radial_flow.valid_vectors >= self.config.flow_min_vectors
            ):
                radial_pass = candidate.radial_flow.outward_ratio >= self.config.min_radial_outward_ratio

            strong_growth_override = (
                effective_growth >= self.config.strong_growth_override_ratio
                and expansion_metrics["center_shift_ratio"] <= self.config.max_center_shift_ratio
            )
            appearance_pass = (
                candidate.residual_polarity_ratio >= self.config.min_residual_polarity_ratio
                and candidate.residual_signed_coherence
                >= self.config.min_residual_signed_coherence
                and candidate.fill_ratio >= self.config.min_candidate_fill_ratio
            )
            small_appearance_pass = (
                candidate.signal_to_noise
                >= self.config.small_min_signal_to_noise
                and candidate.residual_polarity_ratio
                >= self.config.small_min_polarity_ratio
                and candidate.residual_signed_coherence
                >= self.config.small_min_signed_coherence
                and candidate.fill_ratio >= self.config.small_min_fill_ratio
            )
            motion_signature_pass = expansion_pass or radial_pass or strong_growth_override
            onset_pass = (
                candidate.area >= self.config.min_confirm_area
                and effective_growth >= self.config.min_growth_ratio
                and appearance_pass
                and motion_signature_pass
                and track.initial_motion_overlap
                <= self.config.max_initial_motion_history_overlap
            )
            continuation_pass = (
                track.confirmation_hits > 0
                and track.last_signature_frame is not None
                and frame_idx - track.last_signature_frame
                <= self.config.confirmation_window_frames
                and candidate.area
                >= prev_area * self.config.confirmation_min_area_retention
                and short_growth
                >= self.config.confirmation_min_short_growth_ratio
                and appearance_pass
                and expansion_metrics["center_shift_ratio"]
                <= self.config.max_center_shift_ratio
            )
            small_motion_signature_pass = (
                expansion_pass
                or radial_pass
                or effective_growth >= self.config.small_min_growth_ratio
                or track.peak_growth >= self.config.small_min_growth_ratio
            )
            small_onset_pass = (
                track.small_scale
                and candidate.area >= self.config.small_min_blob_area
                and effective_growth >= self.config.small_min_growth_ratio
                and small_appearance_pass
                and small_motion_signature_pass
                and expansion_metrics["center_shift_ratio"]
                <= self.config.small_max_center_shift_ratio
                and track.initial_motion_overlap
                <= self.config.max_initial_motion_history_overlap
            )
            small_continuation_pass = (
                track.small_scale
                and track.confirmation_hits > 0
                and track.last_signature_frame is not None
                and frame_idx - track.last_signature_frame
                <= self.config.confirmation_window_frames
                and candidate.area
                >= prev_area * self.config.small_confirmation_min_area_retention
                and small_appearance_pass
                and expansion_metrics["center_shift_ratio"]
                <= self.config.small_max_center_shift_ratio
            )

            if (
                onset_pass
                or continuation_pass
                or small_onset_pass
                or small_continuation_pass
            ):
                if (
                    track.last_signature_frame is None
                    or frame_idx - track.last_signature_frame
                    > self.config.confirmation_window_frames
                ):
                    track.confirmation_hits = 1
                else:
                    track.confirmation_hits += 1
                track.last_signature_frame = frame_idx
            elif (
                track.last_signature_frame is not None
                and frame_idx - track.last_signature_frame
                > self.config.confirmation_window_frames
            ):
                track.confirmation_hits = 0
                track.last_signature_frame = None

            edge_small_shape_pass = (
                not (track.small_scale and candidate.edge_risk)
                or candidate.fill_ratio >= self.config.small_edge_min_fill_ratio
            )
            normal_evidence_ready = (
                candidate.area >= self.config.min_confirm_area
                and edge_small_shape_pass
            )
            small_evidence_ready = (
                track.small_scale
                and track.cumulative_signal
                >= self.config.small_min_cumulative_signal
                and track.peak_growth >= self.config.small_min_growth_ratio
                and edge_small_shape_pass
            )
            required_hits = (
                self.config.confirmation_hits_required
                if normal_evidence_ready
                else self.config.small_confirmation_hits_required
            )
            if (
                track.confirmation_hits >= max(required_hits, 1)
                and (normal_evidence_ready or small_evidence_ready)
            ):
                suppression_source_id = self._matches_recent_confirmed_event(
                    track, candidate, frame_idx
                )
                if suppression_source_id is not None:
                    track.suppressed = True
                    track.suppression_source_id = suppression_source_id
                    track.confirmation_hits = 0
                    # Feed only fully verified duplicate proposals back into
                    # the long-term tube.  This bridges intermittent plume
                    # observations without letting every raw residual expand
                    # the exclusion region.
                    self._remember_event_observation(
                        suppression_source_id,
                        candidate,
                        frame_idx,
                        update_anchor=True,
                    )
                    return
                # Only confirmed, non-duplicate events consume public IDs.
                # Candidate tracking keeps its own stable dictionary keys.
                track.event_id = self.next_event_id
                self.next_event_id += 1
                track.confirmed = True
                track.confirm_frame = frame_idx
                self._remember_event_observation(
                    track.event_id,
                    candidate,
                    frame_idx,
                    frame_idx,
                    update_anchor=True,
                )
                self.pending_confirmations.append(
                    ConfirmedEvent(
                        event_id=track.event_id,
                        frame_idx=frame_idx,
                        timestamp_sec=frame_idx / self.fps,
                        bbox=candidate.bbox,
                        area=candidate.area,
                        growth_ratio=effective_growth,
                        radial_ratio=radial_ratio,
                    )
                )
                print(
                    "[INFO] Confirm event "
                    f"{track.event_id} at frame {frame_idx}: "
                    f"area={candidate.area:.1f}, growth={effective_growth:.2f}, "
                    f"radial={radial_ratio if radial_ratio is not None else 'NA'}, "
                    f"expand={expansion_metrics['width_growth']:.2f}/{expansion_metrics['height_growth']:.2f}, "
                    f"shift={expansion_metrics['center_shift_ratio']:.2f}"
                )

    def _follow_event_memories(
        self,
        candidates: List[BlobCandidate],
        frame_idx: int,
    ) -> None:
        """Advance each plume tube with a bounded set of connected branches."""
        if not self.config.post_event_memory_enabled or not candidates:
            return

        used_candidates = set()
        ordered_memories = sorted(
            self.event_memories.values(),
            key=lambda memory: memory.last_update_frame,
            reverse=True,
        )
        for memory in ordered_memories:
            if not memory.centroid_history:
                continue
            frontier_frames = max(
                int(round(self.config.post_event_follow_frontier_sec * self.fps)),
                1,
            )
            frontier_min_frame = memory.last_update_frame - frontier_frames
            frontier = [
                (observed_frame, centroid)
                for observed_frame, centroid in memory.centroid_history
                if observed_frame >= frontier_min_frame
            ]
            if not frontier:
                frontier = [memory.centroid_history[-1]]

            options: List[Tuple[float, int]] = []
            for candidate_id, candidate in enumerate(candidates):
                if candidate_id in used_candidates:
                    continue
                best_distance = float("inf")
                for observed_frame, centroid in frontier:
                    elapsed_sec = max(
                        (frame_idx - observed_frame) / self.fps,
                        1.0 / self.fps,
                    )
                    follow_distance = min(
                        self.config.post_event_follow_base_distance
                        + self.config.post_event_follow_speed_px_sec * elapsed_sec,
                        self.config.post_event_follow_max_distance,
                    )
                    distance = _euclidean(centroid, candidate.centroid)
                    if distance <= follow_distance:
                        best_distance = min(best_distance, distance)
                if not np.isfinite(best_distance):
                    continue
                # Favor coherent, substantial plume pieces.  Distance remains
                # dominant so isolated large objects cannot hijack the tube.
                score = best_distance - min(
                    np.sqrt(max(candidate.area, 0.0)), 25.0
                ) * 0.30
                options.append((float(score), candidate_id))

            if not options:
                continue
            max_branches = max(self.config.post_event_follow_max_branches, 1)
            for _, candidate_id in sorted(options)[:max_branches]:
                used_candidates.add(candidate_id)
                self._remember_event_observation(
                    memory.event_id,
                    candidates[candidate_id],
                    frame_idx,
                )

    def _remember_event_observation(
        self,
        event_id: int,
        candidate: BlobCandidate,
        frame_idx: int,
        confirm_frame: Optional[int] = None,
        update_anchor: bool = False,
    ) -> None:
        if not self.config.post_event_memory_enabled:
            return

        memory = self.event_memories.get(event_id)
        if memory is None:
            if confirm_frame is None:
                return
            max_memories = max(self.config.post_event_memory_max_events, 1)
            if len(self.event_memories) >= max_memories:
                oldest_event_id = min(
                    self.event_memories,
                    key=lambda key: (
                        self.event_memories[key].last_update_frame,
                        self.event_memories[key].confirm_frame,
                    ),
                )
                self.event_memories.pop(oldest_event_id, None)
            memory = EventMemory(
                event_id=event_id,
                confirm_frame=confirm_frame,
                last_update_frame=frame_idx,
            )
            self.event_memories[event_id] = memory

        memory.last_update_frame = frame_idx
        memory.centroid_history.append((frame_idx, candidate.centroid))
        memory.bbox_history.append((frame_idx, candidate.bbox))
        if update_anchor and self.history:
            current_gray = self.history[-1].gray
            memory.anchor_frame = frame_idx
            memory.anchor_gray = current_gray.copy()
            memory.anchor_centroid = candidate.centroid
            memory.anchor_bbox = candidate.bbox

    def _prune_event_memories(self, frame_idx: int) -> None:
        if not self.event_memories:
            return

        max_age_frames = max(
            int(round(self.config.post_event_memory_max_sec * self.fps)), 1
        )
        idle_frames = max(
            int(round(self.config.post_event_memory_idle_sec * self.fps)), 1
        )
        expired = [
            event_id
            for event_id, memory in self.event_memories.items()
            if frame_idx - memory.confirm_frame > max_age_frames
            or frame_idx - memory.last_update_frame > idle_frames
        ]
        for event_id in expired:
            self.event_memories.pop(event_id, None)

    def _event_memory_matches(
        self,
        memory: EventMemory,
        candidate: BlobCandidate,
        frame_idx: int,
    ) -> bool:
        max_age_frames = max(
            int(round(self.config.post_event_memory_max_sec * self.fps)), 1
        )
        idle_frames = max(
            int(round(self.config.post_event_memory_idle_sec * self.fps)), 1
        )
        if (
            frame_idx - memory.confirm_frame > max_age_frames
            or frame_idx - memory.last_update_frame > idle_frames
        ):
            return False

        spatial_history_frames = max(
            int(round(self.config.post_event_spatial_history_sec * self.fps)), 1
        )
        min_frame = frame_idx - spatial_history_frames
        recent_boxes = [
            bbox for observed_frame, bbox in memory.bbox_history
            if observed_frame >= min_frame
        ]
        recent_centroids = [
            centroid for observed_frame, centroid in memory.centroid_history
            if observed_frame >= min_frame
        ]
        if not recent_boxes or not recent_centroids:
            return False

        for recent_box in recent_boxes:
            margin = float(np.clip(
                max(recent_box[2], recent_box[3])
                * self.config.post_event_bbox_margin_ratio,
                self.config.post_event_min_bbox_margin_px,
                self.config.post_event_max_bbox_margin_px,
            ))
            if _bboxes_overlap(_expand_bbox(recent_box, margin), candidate.bbox):
                return True

        nearest_distance = min(
            _euclidean(centroid, candidate.centroid)
            for centroid in recent_centroids
        )
        if nearest_distance <= self.config.post_event_match_distance:
            return True
        return self._event_anchor_matches(memory, candidate)

    def _event_anchor_matches(
        self,
        memory: EventMemory,
        candidate: BlobCandidate,
    ) -> bool:
        """Match a plume after compensating long-baseline drone motion."""
        if (
            memory.anchor_gray is None
            or memory.anchor_centroid is None
            or memory.anchor_bbox is None
            or not self.history
        ):
            return False

        current_gray = self.history[-1].gray
        if memory.anchor_gray.shape != current_gray.shape:
            return False
        align_result = self.estimate_homography(memory.anchor_gray, current_gray)
        if not align_result.ok or align_result.H is None:
            return False

        H = np.asarray(align_result.H, dtype=np.float32)
        centroid = np.asarray(
            [[[memory.anchor_centroid[0], memory.anchor_centroid[1]]]],
            dtype=np.float32,
        )
        projected_centroid = cv2.perspectiveTransform(centroid, H)[0, 0]

        x, y, w, h = memory.anchor_bbox
        corners = np.asarray(
            [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]],
            dtype=np.float32,
        )
        projected_corners = cv2.perspectiveTransform(corners, H)[0]
        if (
            not np.all(np.isfinite(projected_centroid))
            or not np.all(np.isfinite(projected_corners))
        ):
            return False

        min_xy = np.min(projected_corners, axis=0)
        max_xy = np.max(projected_corners, axis=0)
        projected_bbox = (
            int(round(float(min_xy[0]))),
            int(round(float(min_xy[1]))),
            max(int(round(float(max_xy[0] - min_xy[0]))), 1),
            max(int(round(float(max_xy[1] - min_xy[1]))), 1),
        )
        margin = float(np.clip(
            max(projected_bbox[2], projected_bbox[3])
            * self.config.post_event_bbox_margin_ratio,
            self.config.post_event_min_bbox_margin_px,
            self.config.post_event_max_bbox_margin_px,
        ))
        if _bboxes_overlap(_expand_bbox(projected_bbox, margin), candidate.bbox):
            return True
        return _euclidean(
            (float(projected_centroid[0]), float(projected_centroid[1])),
            candidate.centroid,
        ) <= self.config.post_event_anchor_match_distance

    def _matches_recent_confirmed_event(
        self,
        track: EventTrack,
        candidate: BlobCandidate,
        frame_idx: int,
    ) -> Optional[int]:
        if self.config.post_event_memory_enabled:
            for memory in self.event_memories.values():
                if memory.event_id == track.event_id:
                    continue
                if self._event_memory_matches(memory, candidate, frame_idx):
                    return memory.event_id

        # Strong change-point evidence may bypass the short legacy tracker,
        # but never a scene-aligned plume match.  Mature smoke can exhibit a
        # large apparent growth when the drone changes viewpoint.
        if self._is_independent_post_event_retrigger(track, candidate):
            return None

        for other in self.tracks.values():
            if not other.confirmed or other.confirm_frame is None or other.event_id == track.event_id:
                continue
            if self._is_duplicate_confirmation(
                frame_idx,
                other.last_seen_frame,
                other.bbox,
                other.centroid,
                list(other.centroid_history),
                candidate,
            ):
                return other.event_id

        for snapshot in self.recent_confirmed_tracks:
            if snapshot.event_id == track.event_id:
                continue
            if self._is_duplicate_confirmation(
                frame_idx,
                snapshot.last_seen_frame,
                snapshot.bbox_history[-1],
                snapshot.centroid_history[-1],
                snapshot.centroid_history,
                candidate,
            ):
                return snapshot.event_id
        return None

    def _is_independent_post_event_retrigger(
        self,
        track: EventTrack,
        candidate: BlobCandidate,
    ) -> bool:
        normal_retrigger = (
            candidate.area >= self.config.post_event_retrigger_min_area
            and track.peak_growth
            >= self.config.post_event_retrigger_min_growth_ratio
        )
        small_retrigger = (
            track.small_scale
            and candidate.area >= self.config.post_event_small_retrigger_min_area
            and track.peak_growth
            >= self.config.post_event_small_retrigger_growth_ratio
            and track.cumulative_signal
            >= self.config.small_min_cumulative_signal
            * self.config.post_event_small_retrigger_signal_scale
            and track.initial_motion_overlap
            <= self.config.max_initial_motion_history_overlap
        )
        return normal_retrigger or small_retrigger

    def _is_duplicate_confirmation(
        self,
        frame_idx: int,
        last_seen_frame: int,
        bbox: Tuple[int, int, int, int],
        centroid: Tuple[float, float],
        centroid_history: List[Tuple[float, float]],
        candidate: BlobCandidate,
    ) -> bool:
        if frame_idx - last_seen_frame > self.recent_confirmed_match_frames:
            return False
        if _bboxes_overlap(bbox, candidate.bbox):
            return True
        if _euclidean(centroid, candidate.centroid) <= self.config.track_match_distance:
            return True
        trajectory_dist = self._trajectory_distance_from_history(centroid_history, candidate.centroid)
        return trajectory_dist is not None and trajectory_dist <= self.config.trajectory_match_distance

    def _check_expansion_motion(
        self,
        prev_bbox: Tuple[int, int, int, int],
        curr_bbox: Tuple[int, int, int, int],
        prev_centroid: Tuple[float, float],
        curr_centroid: Tuple[float, float],
    ) -> Tuple[bool, Dict[str, float]]:
        prev_w = max(prev_bbox[2], 1)
        prev_h = max(prev_bbox[3], 1)
        curr_w = max(curr_bbox[2], 1)
        curr_h = max(curr_bbox[3], 1)

        width_growth = curr_w / prev_w
        height_growth = curr_h / prev_h
        prev_diag = float(np.hypot(prev_w, prev_h))
        curr_diag = float(np.hypot(curr_w, curr_h))
        center_shift = _euclidean(prev_centroid, curr_centroid)
        center_shift_ratio = center_shift / \
            max(0.5 * (prev_diag + curr_diag), 1.0)
        expansion_balance = min(width_growth, height_growth) / \
            max(max(width_growth, height_growth), 1e-6)

        passes = (
            min(width_growth, height_growth) >= self.config.min_dimension_growth_ratio
            and expansion_balance >= self.config.min_expansion_balance
            and center_shift_ratio <= self.config.max_center_shift_ratio
        )
        return passes, {
            "width_growth": float(width_growth),
            "height_growth": float(height_growth),
            "center_shift_ratio": float(center_shift_ratio),
            "expansion_balance": float(expansion_balance),
        }

    def _compute_short_growth(self, track: EventTrack) -> float:
        if len(track.area_history) < 2:
            return 1.0
        history = list(track.area_history)
        baseline_idx = max(0, len(history) - 1 -
                           self.config.growth_window_frames)
        baseline_area = history[baseline_idx]
        return float(history[-1] / max(baseline_area, 1.0))

    def _expire_tracks_without_measurement(self, frame_idx: int) -> None:
        self._prune_finished_tracks(frame_idx)

    def _prune_finished_tracks(self, frame_idx: int) -> None:
        expired: List[Tuple[int, str]] = []
        for track_id, track in self.tracks.items():
            if track.confirmed and track.confirm_frame is not None:
                if frame_idx - track.confirm_frame >= self.max_event_age_frames:
                    expired.append((track_id, "timeout"))
                    continue
                if track.non_growth_frames >= self.config.stop_patience_frames:
                    expired.append((track_id, "growth_stalled"))
                    continue

            if track.missed_frames > self.config.max_missed_frames:
                expired.append((track_id, "missed"))
                continue

            if not track.confirmed and frame_idx - track.first_seen_frame >= self.config.max_tentative_age_frames:
                expired.append((track_id, "tentative_timeout"))

        for track_id, reason in expired:
            track = self.tracks.pop(track_id, None)
            if track is None or not track.confirmed:
                continue
            if track.confirm_frame is not None:
                self.recent_confirmed_tracks.append(
                    ConfirmedTrackSnapshot(
                        event_id=track.event_id,
                        confirm_frame=track.confirm_frame,
                        last_seen_frame=track.last_seen_frame,
                        centroid_history=list(track.centroid_history),
                        bbox_history=list(track.bbox_history),
                    )
                )
            print(
                f"[INFO] Drop event {track.event_id} at frame {frame_idx}: "
                f"reason={reason}, age_frames={frame_idx - (track.confirm_frame or frame_idx)}"
            )

    def _draw_visualization(
        self,
        vis: np.ndarray,
        frame_idx: int,
        diff_img: np.ndarray,
        motion_mask: np.ndarray,
        align_result: Optional[HomographyResult],
        measurement_ok: bool,
        candidates: List[BlobCandidate],
    ) -> np.ndarray:
        if self.config.show_candidate_boxes:
            for candidate in candidates:
                x, y, w, h = candidate.bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 170, 255), 1)

        visible_event_count = 0
        for track in self.tracks.values():
            if not track.confirmed:
                continue
            if track.confirm_frame is None:
                continue
            if frame_idx - track.confirm_frame > self.confirmed_box_hold_frames:
                continue
            visible_event_count += 1
            x, y, w, h = track.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"EVENT {track.event_id}"
            if track.growth_history:
                label += f" g={track.growth_history[-1]:.2f}"
            if track.radial_history:
                label += f" r={track.radial_history[-1]:.2f}"
            cv2.putText(
                vis,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if self.config.show_status_overlay:
            info_lines = [
                f"frame={frame_idx}",
                f"visible_events={visible_event_count}",
                f"phase2_ok={int(measurement_ok)}",
            ]
            if align_result is None:
                info_lines.append("align=warmup")
            elif align_result.ok:
                info_lines.append(
                    f"align={align_result.method} {align_result.inliers}/{align_result.matches} "
                    f"({align_result.inlier_ratio:.2f})"
                )
                if align_result.dx is not None and align_result.dy is not None:
                    scale_text = "?"
                    if align_result.scale is not None:
                        scale_text = f"{align_result.scale:.4f}"
                    rotation_text = "?"
                    if align_result.rotation_deg is not None:
                        rotation_text = f"{align_result.rotation_deg:.2f}deg"
                    info_lines.append(
                        f"global_shift=({align_result.dx:.1f}, {align_result.dy:.1f}) scale={scale_text} rot={rotation_text}"
                    )
            else:
                info_lines.append(f"align=failed({align_result.method})")

            y0 = 24
            for line in info_lines:
                cv2.putText(
                    vis,
                    line,
                    (12, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y0 += 24

        if self.config.draw_debug_tiles:
            vis = self._overlay_debug_tiles(vis, diff_img, motion_mask)
        return vis

    def _overlay_debug_tiles(
        self,
        canvas: np.ndarray,
        diff_img: np.ndarray,
        motion_mask: np.ndarray,
    ) -> np.ndarray:
        tile_w = max(canvas.shape[1] // 5, 1)
        tile_h = max(canvas.shape[0] // 5, 1)

        diff_tile = cv2.resize(cv2.cvtColor(
            diff_img, cv2.COLOR_GRAY2BGR), (tile_w, tile_h))
        mask_tile = cv2.resize(cv2.cvtColor(
            motion_mask, cv2.COLOR_GRAY2BGR), (tile_w, tile_h))

        x0 = canvas.shape[1] - tile_w - 8
        canvas[8: 8 + tile_h, x0: x0 + tile_w] = diff_tile
        canvas[16 + tile_h: 16 + 2 * tile_h, x0: x0 + tile_w] = mask_tile

        cv2.putText(canvas, "absdiff", (x0, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "motion_mask",
            (x0, 32 + tile_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas


def _odd_ksize(value: int) -> int:
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1


def _contour_centroid(contour: np.ndarray, fallback: Tuple[float, float]) -> Tuple[float, float]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        return fallback
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _bboxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1 = ax0 + aw
    ay1 = ay0 + ah
    bx1 = bx0 + bw
    by1 = by0 + bh
    return min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0)


def _expand_bbox(
    bbox: Tuple[int, int, int, int], margin: float
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    pad = max(int(round(margin)), 0)
    return x - pad, y - pad, w + 2 * pad, h + 2 * pad


def _point_to_segment_distance(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    point_vec = np.array(point, dtype=np.float32)
    start_vec = np.array(start, dtype=np.float32)
    end_vec = np.array(end, dtype=np.float32)
    segment = end_vec - start_vec
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-6:
        return float(np.linalg.norm(point_vec - start_vec))

    t = float(np.dot(point_vec - start_vec, segment) / length_sq)
    t = min(max(t, 0.0), 1.0)
    projection = start_vec + t * segment
    return float(np.linalg.norm(point_vec - projection))


def _default_video_from_cwd() -> Optional[Path]:
    mp4_files = sorted(Path.cwd().glob("*.mp4"))
    return mp4_files[0] if mp4_files else None


def _default_batch_output_dir(input_dir: Path) -> Path:
    return input_dir / "batch_results"


def _ensure_batch_subdirs(output_root: Path) -> Tuple[Path, Path, Path]:
    annotated_dir = output_root / "annotated_videos"
    detection_dir = output_root / "detection_frames"
    logs_dir = output_root / "logs"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    detection_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return annotated_dir, detection_dir, logs_dir


def _save_detection_snapshot(
    video_path: Path,
    detection_root: Path,
    vis_frame: np.ndarray,
    event: ConfirmedEvent,
) -> DetectionArtifact:
    video_dir = detection_root / video_path.stem
    video_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"{video_path.stem}__event{event.event_id:03d}"
        f"__frame{event.frame_idx:06d}__t{event.timestamp_sec:08.2f}"
    )
    full_frame_path = video_dir / f"{stem}__full.jpg"
    crop_path = video_dir / f"{stem}__crop.jpg"

    snapshot = vis_frame.copy()
    x, y, w, h = event.bbox
    cv2.rectangle(snapshot, (x, y), (x + w, y + h), (0, 255, 0), 3)
    overlay_text = f"EVENT {event.event_id} frame={event.frame_idx} t={event.timestamp_sec:.2f}s"
    cv2.putText(
        snapshot,
        overlay_text,
        (max(12, x), max(28, y - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(full_frame_path), snapshot)

    pad = 40
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, snapshot.shape[1])
    y1 = min(y + h + pad, snapshot.shape[0])
    crop = snapshot[y0:y1, x0:x1]
    cv2.imwrite(str(crop_path), crop)

    return DetectionArtifact(
        video_name=video_path.name,
        event_id=event.event_id,
        frame_idx=event.frame_idx,
        timestamp_sec=event.timestamp_sec,
        area=event.area,
        growth_ratio=event.growth_ratio,
        radial_ratio=event.radial_ratio,
        full_frame_path=full_frame_path,
        crop_path=crop_path,
    )


def _write_batch_csvs(
    logs_dir: Path,
    video_summaries: List[VideoRunSummary],
    detections: List[DetectionArtifact],
) -> None:
    summary_csv = logs_dir / "video_summary.csv"
    detection_csv = logs_dir / "detections.csv"

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "processed_frames",
                        "confirmed_events", "snapshots_saved", "output_video_path"])
        for row in video_summaries:
            writer.writerow(
                [
                    row.video_name,
                    row.processed_frames,
                    row.confirmed_events,
                    row.snapshots_saved,
                    str(row.output_video_path) if row.output_video_path else "",
                ]
            )

    with detection_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video_name",
                "event_id",
                "frame_idx",
                "timestamp_sec",
                "area",
                "growth_ratio",
                "radial_ratio",
                "full_frame_path",
                "crop_path",
            ]
        )
        for row in detections:
            writer.writerow(
                [
                    row.video_name,
                    row.event_id,
                    row.frame_idx,
                    f"{row.timestamp_sec:.3f}",
                    f"{row.area:.3f}",
                    f"{row.growth_ratio:.3f}",
                    "" if row.radial_ratio is None else f"{row.radial_ratio:.3f}",
                    str(row.full_frame_path),
                    str(row.crop_path),
                ]
            )


def run_video(
    video_path: Path,
    config: DetectorConfig,
    start_frame: int = 0,
    max_frames: int = -1,
    display_enabled: bool = False,
    output_video_path: Optional[Path] = None,
    detection_root: Optional[Path] = None,
    report_every: int = 300,
) -> Tuple[VideoRunSummary, List[DetectionArtifact]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = InstantSmokeDustDetector(config, fps)

    writer = None
    frame_idx = start_frame
    processed = 0
    confirmed_count = 0
    detections: List[DetectionArtifact] = []

    if output_video_path is not None:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            vis = detector.process_frame(frame, frame_idx)

            if writer is None and output_video_path is not None:
                height, width = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_video_path), fourcc, detector.fps, (width, height))

            if writer is not None:
                writer.write(vis)

            for event in detector.consume_pending_confirmations():
                confirmed_count += 1
                if detection_root is not None:
                    detections.append(_save_detection_snapshot(
                        video_path, detection_root, vis, event))

            if display_enabled:
                show_frame = vis
                if config.display_scale != 1.0:
                    show_frame = cv2.resize(
                        vis,
                        None,
                        fx=config.display_scale,
                        fy=config.display_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow("Instant Smoke / Dust POC", show_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_idx += 1
            processed += 1
            if report_every > 0 and processed % report_every == 0:
                print(
                    f"[INFO] {video_path.name}: processed {processed} frames")
            if max_frames > 0 and processed >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display_enabled:
            cv2.destroyAllWindows()

    summary = VideoRunSummary(
        video_name=video_path.name,
        processed_frames=processed,
        confirmed_events=confirmed_count,
        snapshots_saved=len(detections) * 2,
        output_video_path=output_video_path,
    )
    return summary, detections


def run_batch(
    input_dir: Path,
    output_root: Path,
    config: DetectorConfig,
    video_glob: str,
    start_frame: int = 0,
    max_frames: int = -1,
    report_every: int = 300,
) -> Tuple[List[VideoRunSummary], List[DetectionArtifact]]:
    videos = sorted(input_dir.glob(video_glob))
    if not videos:
        raise FileNotFoundError(
            f"No videos found in {input_dir} matching {video_glob}")

    annotated_dir, detection_dir, logs_dir = _ensure_batch_subdirs(output_root)
    all_summaries: List[VideoRunSummary] = []
    all_detections: List[DetectionArtifact] = []

    print(f"[INFO] Batch input dir: {input_dir}")
    print(f"[INFO] Batch output dir: {output_root}")
    print(f"[INFO] Found {len(videos)} videos")

    for idx, video_path in enumerate(videos, start=1):
        print(f"[INFO] [{idx}/{len(videos)}] Processing {video_path.name}")
        output_video_path = annotated_dir / f"{video_path.stem}__annotated.mp4"
        summary, detections = run_video(
            video_path=video_path,
            config=config,
            start_frame=start_frame,
            max_frames=max_frames,
            display_enabled=False,
            output_video_path=output_video_path,
            detection_root=detection_dir,
            report_every=report_every,
        )
        all_summaries.append(summary)
        all_detections.extend(detections)
        print(
            f"[INFO] Finished {video_path.name}: "
            f"frames={summary.processed_frames}, events={summary.confirmed_events}, "
            f"snapshots={summary.snapshots_saved}"
        )

    _write_batch_csvs(logs_dir, all_summaries, all_detections)
    return all_summaries, all_detections


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="POC detector for instantaneous dust/smoke anomalies under drone ego-motion."
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Input video path. Defaults to first *.mp4 in CWD.")
    parser.add_argument("--batch-dir", type=str, default=None,
                        help="Process all videos in this directory.")
    parser.add_argument("--video-glob", type=str,
                        default="*.mp4", help="Glob used in batch mode.")
    parser.add_argument("--output", type=str, default=None,
                        help="Single-video annotated output path.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Batch output root or single-video artifact root.")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start processing from a specific frame.")
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="Process only the first N frames after start-frame.")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable cv2.imshow window.")
    parser.add_argument("--report-every", type=int, default=300,
                        help="Print progress every N processed frames.")

    parser.add_argument("--downscale", type=float, default=0.5,
                        help="Spatial downsampling factor.")
    parser.add_argument("--stride", type=int, default=5,
                        help="Frame gap N used for T vs T-N differencing.")
    parser.add_argument("--diff-threshold", type=int, default=40,
                        help="Threshold for absolute difference image.")
    parser.add_argument("--min-blob-area", type=float,
                        default=220.0, help="Minimum candidate contour area.")
    parser.add_argument("--min-growth-ratio", type=float,
                        default=2.2, help="Short-term area explosion threshold.")
    parser.add_argument("--stop-growth-ratio", type=float, default=1.08,
                        help="Silence event when growth falls below this.")
    parser.add_argument("--max-age-sec", type=float, default=3.0,
                        help="Maximum active lifetime per confirmed event.")
    parser.add_argument("--track-match-distance", type=float,
                        default=120.0, help="Track association radius in pixels.")
    parser.add_argument("--max-missed-frames", type=int, default=6,
                        help="How many missing frames an event can survive.")
    parser.add_argument(
        "--min-dimension-growth-ratio",
        type=float,
        default=1.08,
        help="Minimum width/height growth used to distinguish expansion from translation.",
    )
    parser.add_argument(
        "--max-center-shift-ratio",
        type=float,
        default=0.65,
        help="Reject translation-dominant motion when centroid shifts too much relative to blob size.",
    )
    parser.add_argument(
        "--split-blob-area",
        type=float,
        default=2500.0,
        help="Try watershed splitting when a motion blob becomes larger than this area.",
    )
    parser.add_argument("--disable-flow-check", action="store_true",
                        help="Disable radial optical-flow validation.")
    parser.add_argument("--hide-debug-tiles", action="store_true",
                        help="Hide absdiff / mask thumbnails.")
    parser.add_argument("--show-candidates", action="store_true",
                        help="Draw orange candidate boxes for debugging.")
    return parser


def config_from_args(args: argparse.Namespace) -> DetectorConfig:
    return DetectorConfig(
        downscale=args.downscale,
        stride_frames=max(args.stride, 1),
        diff_threshold=args.diff_threshold,
        min_blob_area=args.min_blob_area,
        min_growth_ratio=args.min_growth_ratio,
        stop_growth_ratio=args.stop_growth_ratio,
        max_event_age_sec=args.max_age_sec,
        track_match_distance=args.track_match_distance,
        max_missed_frames=args.max_missed_frames,
        min_dimension_growth_ratio=args.min_dimension_growth_ratio,
        max_center_shift_ratio=args.max_center_shift_ratio,
        split_blob_area=args.split_blob_area,
        enable_radial_flow_check=not args.disable_flow_check,
        draw_debug_tiles=not args.hide_debug_tiles,
        show_candidate_boxes=args.show_candidates,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = config_from_args(args)

    if args.batch_dir:
        input_dir = Path(args.batch_dir)
        if not input_dir.exists():
            raise FileNotFoundError(
                f"Batch input directory does not exist: {input_dir}")
        output_root = Path(
            args.output_dir) if args.output_dir else _default_batch_output_dir(input_dir)
        summaries, detections = run_batch(
            input_dir=input_dir,
            output_root=output_root,
            config=config,
            video_glob=args.video_glob,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            report_every=args.report_every,
        )
        print(
            f"[INFO] Batch complete: videos={len(summaries)}, "
            f"confirmed_events={sum(s.confirmed_events for s in summaries)}, "
            f"saved_images={sum(s.snapshots_saved for s in summaries)}"
        )
        print(f"[INFO] Output root: {output_root}")
        print(
            f"[INFO] Detection log: {output_root / 'logs' / 'detections.csv'}")
        return

    video_path = Path(args.video) if args.video else _default_video_from_cwd()
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(
            "No input video found. Use --video or place an .mp4 in the current directory.")

    output_video_path = Path(args.output) if args.output else None
    detection_root = None
    if args.output_dir:
        output_root = Path(args.output_dir)
        annotated_dir, detection_dir, _ = _ensure_batch_subdirs(output_root)
        detection_root = detection_dir
        if output_video_path is None:
            output_video_path = annotated_dir / \
                f"{video_path.stem}__annotated.mp4"

    summary, detections = run_video(
        video_path=video_path,
        config=config,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        display_enabled=not args.no_display,
        output_video_path=output_video_path,
        detection_root=detection_root,
        report_every=args.report_every,
    )
    print(
        f"[INFO] Finished {summary.video_name}: "
        f"frames={summary.processed_frames}, events={summary.confirmed_events}, "
        f"snapshots={summary.snapshots_saved}"
    )
    if detections:
        print(f"[INFO] Detection frames saved under: {detection_root}")


if __name__ == "__main__":
    main()



