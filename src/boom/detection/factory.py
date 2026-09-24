"""
==============================================================================
Boom 煙霧辨識器工廠 (factory.py)
==============================================================================

【模組職責】
根據指定的演算法類型（`motion`, `yolo`）與設定參數，
動態建立並初始化對應的煙霧偵測器實例。
支援傳入字串、設定物件或命令列解析之 Namespace。
所有回傳之偵測器均遵循 `BaseSmokeDetector` 介面契約。
完全獨立於 artillery，只依賴 boom.config.defaults。
==============================================================================
"""

from typing import Any, Optional
import argparse
from boom.interfaces.detector import BaseSmokeDetector
from boom.detection.optical_flow import DetectorConfig, InstantSmokeDustDetector
from boom.detection.yolo import YOLOSmokeDetectorConfig, YOLOSmokeImpactDetector
from boom.config.defaults import (
    CLOSE_ITERATIONS, CLOSE_KERNEL_SIZE, CONFIRMATION_HITS_REQUIRED,
    CONFIRMED_BOX_HOLD_SEC, DIFF_INTERVAL_SEC, DIFF_THRESHOLD, DOWNSCALE,
    DRAW_DEBUG_TILES, ENABLE_RADIAL_FLOW_CHECK, MAX_CENTER_SHIFT_RATIO,
    MAX_DISTRIBUTED_CANDIDATES, MAX_EVENT_AGE_SEC, MAX_INITIAL_MOTION_HISTORY_OVERLAP,
    MAX_MISSED_FRAMES, MIN_BLOB_AREA, MIN_CANDIDATE_FIELD_SPAN_RATIO,
    MIN_CANDIDATE_FILL_RATIO, MIN_CONFIRM_AREA, MIN_DIMENSION_GROWTH_RATIO,
    MIN_GROWTH_RATIO, MIN_RESIDUAL_POLARITY_RATIO, MIN_RESIDUAL_SIGNED_COHERENCE,
    OPEN_KERNEL_SIZE,
    POST_EVENT_MATCH_DISTANCE, POST_EVENT_MEMORY_IDLE_SEC, POST_EVENT_MEMORY_MAX_SEC,
    POST_EVENT_RETRIGGER_MIN_AREA, POST_EVENT_RETRIGGER_MIN_GROWTH_RATIO,
    POST_EVENT_SPATIAL_HISTORY_SEC, POST_EVENT_TRAJECTORY_DISTANCE,
    RECENT_CONFIRMED_MATCH_SEC, SHOW_CANDIDATE_BOXES, SMALL_CONFIRMATION_HITS_REQUIRED,
    SMALL_DIFF_FLOOR, SMALL_MAX_CENTER_SHIFT_RATIO, SMALL_MIN_BLOB_AREA,
    SMALL_MIN_CUMULATIVE_SIGNAL, SMALL_MIN_GROWTH_RATIO, SMALL_MIN_SIGNAL_TO_NOISE,
    SMALL_NOISE_SIGMA, SMALL_NOISE_WINDOW_SIZE, SPLIT_BLOB_AREA, STOP_GROWTH_RATIO,
    STRIDE_FRAMES, TRACK_MATCH_DISTANCE, TRAJECTORY_MATCH_DISTANCE, UI_MODE,
    YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_TARGET_CLASSES,
    YOLO_CONFIRMATION_HITS, YOLO_TRACK_MAX_MISSED_SEC, YOLO_TRACK_MATCH_DISTANCE,
)


def build_detector_config() -> DetectorConfig:
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


def build_yolo_detector_config(args: argparse.Namespace) -> YOLOSmokeDetectorConfig:
    model_path = getattr(args, "yolo_model_path", None) or getattr(args, "model_path", YOLO_MODEL_PATH)
    conf = getattr(args, "yolo_conf", YOLO_CONF_THRESHOLD)
    iou = getattr(args, "yolo_iou", YOLO_IOU_THRESHOLD)
    imgsz = getattr(args, "yolo_imgsz", 640)
    stride = getattr(args, "inference_stride", 1)
    device = getattr(args, "device", "auto")
    hits = getattr(args, "confirmation_hits", YOLO_CONFIRMATION_HITS)
    match_dist = getattr(args, "yolo_track_match_distance", YOLO_TRACK_MATCH_DISTANCE)
    max_missed = getattr(args, "yolo_max_missed_sec", YOLO_TRACK_MAX_MISSED_SEC)
    return YOLOSmokeDetectorConfig(
        model_path=model_path,
        device=device,
        precision=getattr(args, "precision", "fp32"),
        input_size=imgsz,
        conf_threshold=conf,
        iou_threshold=iou,
        inference_stride=max(stride, 1),
        target_classes=tuple(getattr(args, "yolo_target_classes", None) or YOLO_TARGET_CLASSES),
        confirmation_hits=hits,
        track_match_distance=match_dist,
        max_missed_sec=max_missed,
        post_event_memory_sec=POST_EVENT_MEMORY_MAX_SEC,
        show_status_overlay=True,
    )


def create_detector(
    detector_or_args: Any,
    fps: float = 30.0,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> BaseSmokeDetector:
    """
    建立煙霧偵測器。

    :param detector_or_args: 偵測器型別字串 ("motion", "yolo") 或含 .detector 的 Namespace 物件
    :param fps: 影像來源之影格率
    :param config: 對應演算法之設定物件 (可選)
    :param kwargs: 額外參數覆寫
    :return: 實作 BaseSmokeDetector 之偵測器物件
    """
    if hasattr(detector_or_args, "detector"):
        args = detector_or_args
        d_type = str(args.detector).lower().strip()
        if d_type == "motion":
            return InstantSmokeDustDetector(build_detector_config(), fps)
        elif d_type in ("yolo", "yolo-seg", "yolov8", "yolov11", "yolo26"):
            cfg = build_yolo_detector_config(args)
            return YOLOSmokeImpactDetector(cfg, fps)
        else:
            raise ValueError(f"Unknown detector in args: {args.detector}")

    d_type = str(detector_or_args).lower().strip()

    if d_type == "motion":
        if config is None or not isinstance(config, DetectorConfig):
            cfg = DetectorConfig(**kwargs) if kwargs else DetectorConfig()
        else:
            cfg = config
        return InstantSmokeDustDetector(cfg, fps)

    elif d_type in ("yolo", "yolo-seg", "yolov8", "yolov11", "yolo26"):
        if config is None or not isinstance(config, YOLOSmokeDetectorConfig):
            cfg = YOLOSmokeDetectorConfig(**kwargs) if kwargs else YOLOSmokeDetectorConfig()
        else:
            cfg = config
        return YOLOSmokeImpactDetector(cfg, fps)

    else:
        raise ValueError(
            f"Unsupported detector type: '{detector_or_args}'. Supported: 'motion', 'yolo'"
        )
