"""Detector construction shared by offline and online runtimes; no UI or runner imports."""
from __future__ import annotations
import argparse
from artillery.common.defaults import CLOSE_ITERATIONS
from artillery.common.defaults import CLOSE_KERNEL_SIZE
from artillery.common.defaults import CONFIRMATION_HITS_REQUIRED
from artillery.common.defaults import CONFIRMED_BOX_HOLD_SEC
from artillery.common.defaults import DIFF_INTERVAL_SEC
from artillery.common.defaults import DIFF_THRESHOLD
from artillery.common.defaults import DOWNSCALE
from artillery.common.defaults import DRAW_DEBUG_TILES
from artillery.common.defaults import ENABLE_RADIAL_FLOW_CHECK
from artillery.common.defaults import MAX_CENTER_SHIFT_RATIO
from artillery.common.defaults import MAX_DISTRIBUTED_CANDIDATES
from artillery.common.defaults import MAX_EVENT_AGE_SEC
from artillery.common.defaults import MAX_INITIAL_MOTION_HISTORY_OVERLAP
from artillery.common.defaults import MAX_MISSED_FRAMES
from artillery.common.defaults import MIN_BLOB_AREA
from artillery.common.defaults import MIN_CANDIDATE_FIELD_SPAN_RATIO
from artillery.common.defaults import MIN_CANDIDATE_FILL_RATIO
from artillery.common.defaults import MIN_CONFIRM_AREA
from artillery.common.defaults import MIN_DIMENSION_GROWTH_RATIO
from artillery.common.defaults import MIN_GROWTH_RATIO
from artillery.common.defaults import MIN_RESIDUAL_POLARITY_RATIO
from artillery.common.defaults import MIN_RESIDUAL_SIGNED_COHERENCE
from artillery.common.defaults import OPEN_KERNEL_SIZE
from artillery.common.defaults import PIDNET_CONFIRMATION_HITS
from artillery.common.defaults import PIDNET_EARLY_CANDIDATE_MIN_AREA
from artillery.common.defaults import PIDNET_MIN_COMPONENT_AREA
from artillery.common.defaults import PIDNET_MIN_CONFIRMATION_CONFIDENCE
from artillery.common.defaults import POST_EVENT_MATCH_DISTANCE
from artillery.common.defaults import POST_EVENT_MEMORY_IDLE_SEC
from artillery.common.defaults import POST_EVENT_MEMORY_MAX_SEC
from artillery.common.defaults import POST_EVENT_RETRIGGER_MIN_AREA
from artillery.common.defaults import POST_EVENT_RETRIGGER_MIN_GROWTH_RATIO
from artillery.common.defaults import POST_EVENT_SPATIAL_HISTORY_SEC
from artillery.common.defaults import POST_EVENT_TRAJECTORY_DISTANCE
from artillery.common.defaults import RECENT_CONFIRMED_MATCH_SEC
from artillery.common.defaults import SHOW_CANDIDATE_BOXES
from artillery.common.defaults import SMALL_CONFIRMATION_HITS_REQUIRED
from artillery.common.defaults import SMALL_DIFF_FLOOR
from artillery.common.defaults import SMALL_MAX_CENTER_SHIFT_RATIO
from artillery.common.defaults import SMALL_MIN_BLOB_AREA
from artillery.common.defaults import SMALL_MIN_CUMULATIVE_SIGNAL
from artillery.common.defaults import SMALL_MIN_GROWTH_RATIO
from artillery.common.defaults import SMALL_MIN_SIGNAL_TO_NOISE
from artillery.common.defaults import SMALL_NOISE_SIGMA
from artillery.common.defaults import SMALL_NOISE_WINDOW_SIZE
from artillery.common.defaults import SPLIT_BLOB_AREA
from artillery.common.defaults import STOP_GROWTH_RATIO
from artillery.common.defaults import STRIDE_FRAMES
from artillery.common.defaults import TRACK_MATCH_DISTANCE
from artillery.common.defaults import TRAJECTORY_MATCH_DISTANCE
from artillery.common.defaults import UI_MODE

def build_detector_config() -> DetectorConfig:
    from artillery.offline.src.detectors.optical_flow_smoke_detector import DetectorConfig
    debug_mode = UI_MODE.lower() == "debug"
    return DetectorConfig(
        downscale=DOWNSCALE,
        stride_frames=max(STRIDE_FRAMES, 1),
        diff_interval_sec=DIFF_INTERVAL_SEC,
        diff_threshold=DIFF_THRESHOLD,
        open_kernel_size=OPEN_KERNEL_SIZE,
        close_kernel_size=CLOSE_KERNEL_SIZE,
        close_iterations=CLOSE_ITERATIONS,
        min_blob_area=MIN_BLOB_AREA,
        min_confirm_area=MIN_CONFIRM_AREA,
        min_growth_ratio=MIN_GROWTH_RATIO,
        min_residual_polarity_ratio=MIN_RESIDUAL_POLARITY_RATIO,
        min_residual_signed_coherence=MIN_RESIDUAL_SIGNED_COHERENCE,
        min_candidate_fill_ratio=MIN_CANDIDATE_FILL_RATIO,
        max_initial_motion_history_overlap=MAX_INITIAL_MOTION_HISTORY_OVERLAP,
        small_min_blob_area=SMALL_MIN_BLOB_AREA,
        small_diff_floor=SMALL_DIFF_FLOOR,
        small_noise_window_size=SMALL_NOISE_WINDOW_SIZE,
        small_noise_sigma=SMALL_NOISE_SIGMA,
        small_min_signal_to_noise=SMALL_MIN_SIGNAL_TO_NOISE,
        small_min_growth_ratio=SMALL_MIN_GROWTH_RATIO,
        small_max_center_shift_ratio=SMALL_MAX_CENTER_SHIFT_RATIO,
        small_confirmation_hits_required=SMALL_CONFIRMATION_HITS_REQUIRED,
        small_min_cumulative_signal=SMALL_MIN_CUMULATIVE_SIGNAL,
        max_distributed_candidates=MAX_DISTRIBUTED_CANDIDATES,
        min_candidate_field_span_ratio=MIN_CANDIDATE_FIELD_SPAN_RATIO,
        confirmation_hits_required=CONFIRMATION_HITS_REQUIRED,
        stop_growth_ratio=STOP_GROWTH_RATIO,
        max_event_age_sec=MAX_EVENT_AGE_SEC,
        confirmed_box_hold_sec=CONFIRMED_BOX_HOLD_SEC,
        track_match_distance=TRACK_MATCH_DISTANCE,
        trajectory_match_distance=TRAJECTORY_MATCH_DISTANCE,
        recent_confirmed_match_sec=RECENT_CONFIRMED_MATCH_SEC,
        post_event_memory_max_sec=POST_EVENT_MEMORY_MAX_SEC,
        post_event_memory_idle_sec=POST_EVENT_MEMORY_IDLE_SEC,
        post_event_spatial_history_sec=POST_EVENT_SPATIAL_HISTORY_SEC,
        post_event_match_distance=POST_EVENT_MATCH_DISTANCE,
        post_event_trajectory_distance=POST_EVENT_TRAJECTORY_DISTANCE,
        post_event_retrigger_min_area=POST_EVENT_RETRIGGER_MIN_AREA,
        post_event_retrigger_min_growth_ratio=POST_EVENT_RETRIGGER_MIN_GROWTH_RATIO,
        max_missed_frames=MAX_MISSED_FRAMES,
        min_dimension_growth_ratio=MIN_DIMENSION_GROWTH_RATIO,
        max_center_shift_ratio=MAX_CENTER_SHIFT_RATIO,
        split_blob_area=SPLIT_BLOB_AREA,
        enable_radial_flow_check=ENABLE_RADIAL_FLOW_CHECK,
        draw_debug_tiles=DRAW_DEBUG_TILES if debug_mode else False,
        show_candidate_boxes=SHOW_CANDIDATE_BOXES if debug_mode else False,
        show_status_overlay=debug_mode,
    )


def build_pidnet_detector_config(args: argparse.Namespace) -> PIDNetSmokeTrackerConfig:
    from artillery.offline.src.detectors.pidnet_smoke_detector import PIDNetSmokeTrackerConfig
    debug_mode = UI_MODE.lower() == "debug"
    return PIDNetSmokeTrackerConfig(
        model_path=args.model_path,
        device=args.device,
        input_width=args.model_width,
        input_height=args.model_height,
        inference_stride=max(args.inference_stride, 1),
        segmentation_threshold=args.seg_threshold,
        early_candidate_threshold=args.early_seg_threshold,
        min_component_area=PIDNET_MIN_COMPONENT_AREA,
        early_candidate_min_area=PIDNET_EARLY_CANDIDATE_MIN_AREA,
        warmup_sec=args.warmup_sec,
        new_smoke_window_sec=args.new_smoke_window_sec,
        confirmation_hits=PIDNET_CONFIRMATION_HITS,
        min_confirmation_confidence=PIDNET_MIN_CONFIRMATION_CONFIDENCE,
        track_match_distance=TRACK_MATCH_DISTANCE,
        post_event_memory_sec=POST_EVENT_MEMORY_MAX_SEC,
        show_status_overlay=debug_mode,
        show_candidate_boxes=SHOW_CANDIDATE_BOXES if debug_mode else False,
    )


def build_fusion_config(args: argparse.Namespace) -> OpticalPIDNetFusionConfig:
    from artillery.offline.src.detectors.optical_pidnet_fusion_detector import OpticalPIDNetFusionConfig
    return OpticalPIDNetFusionConfig(
        confirmation_window_sec=max(args.fusion_window_sec, 0.01),
        match_distance_px=max(args.fusion_match_distance, 1.0),
        draw_pending_candidates=True,
    )


def create_detector(args, fps):
    """Select an implementation without importing unused neural backends."""
    if args.detector == "pidnet":
        from artillery.offline.src.detectors.pidnet_smoke_detector import PIDNetSmokeImpactDetector
        return PIDNetSmokeImpactDetector(build_pidnet_detector_config(args), fps)
    if args.detector == "fusion":
        from artillery.offline.src.detectors.optical_pidnet_fusion_detector import OpticalPIDNetFusionDetector
        return OpticalPIDNetFusionDetector(build_detector_config(), build_pidnet_detector_config(args), fps, build_fusion_config(args))
    if args.detector == "motion":
        from artillery.offline.src.detectors.optical_flow_smoke_detector import InstantSmokeDustDetector
        return InstantSmokeDustDetector(build_detector_config(), fps)
    raise ValueError(f"Unknown detector: {args.detector}")
