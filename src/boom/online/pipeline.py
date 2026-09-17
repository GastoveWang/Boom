"""Live frame acquisition, detection and evidence orchestration."""
from __future__ import annotations
import argparse
import signal
import time
from typing import Any, Optional
import cv2
from boom.config.defaults import CONFIRMED_BOX_HOLD_SEC
from boom.detection.optical_flow import ConfirmedEvent, InstantSmokeDustDetector
from .spinnaker_camera import LatestFrameGrabber, SpinCamera, SpinCameraConfig
from .configuration import detector_config_from_args
from .positioning import static_pose_from_args, estimate_position
from .storage import EventStore
from .logging_setup import configure_logging
from .display import _draw_confirmed_box

def run(args: argparse.Namespace) -> int:
    detector_config = detector_config_from_args(args)
    static_pose = static_pose_from_args(args)
    camera_config = SpinCameraConfig(
        serial=args.serial,
        width=args.width,
        height=args.height,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        fps=args.camera_fps,
        exposure_us=args.exposure_us,
        gain_db=args.gain_db,
        timeout_ms=args.camera_timeout_ms,
    )
    store = EventStore(args.output_dir)
    logger = configure_logging(store.runtime_log_path)
    detector = InstantSmokeDustDetector(detector_config, args.camera_fps)

    stop_requested = False
    grabber: Optional[LatestFrameGrabber] = None
    recent_events: list[tuple[int, ConfirmedEvent]] = []
    processed = 0
    event_count = 0
    last_sequence = 0
    first_camera_frame_id: Optional[int] = None
    last_camera_frame_id: Optional[int] = None
    started_at = time.monotonic()

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        logger.info("Stop requested by signal %s", signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    logger.info("Boom online runtime starting")
    logger.info("Display enabled: %s", args.display)
    logger.info("Event JSONL: %s", store.jsonl_path)
    logger.info("Detection CSV: %s", store.csv_path)
    logger.info("Coordinate log: %s", store.coordinate_log_path)
    logger.info("Snapshot directory: %s", store.snapshot_dir)
    logger.info(
        "Detector config: downscale=%s stride=%s diff_threshold=%s "
        "min_blob_area=%s min_growth_ratio=%s radial_flow=%s",
        detector_config.downscale,
        detector_config.stride_frames,
        detector_config.diff_threshold,
        detector_config.min_blob_area,
        detector_config.min_growth_ratio,
        detector_config.enable_radial_flow_check,
    )
    if static_pose is None:
        logger.warning(
            "No GPS/pose supplied; events will be saved with coordinate=unavailable"
        )
    else:
        logger.warning(
            "Using static GPS/pose; coordinates are approximate until live telemetry is connected"
        )

    try:
        logger.info(
            "CAMERA CHECK starting: requested_serial=%s target_fps=%.3f "
            "requested_roi=%sx%s offset=%s,%s",
            args.serial or "first-available",
            args.camera_fps,
            args.width or "camera-default",
            args.height or "camera-default",
            args.offset_x or 0,
            args.offset_y or 0,
        )
        with SpinCamera(camera_config) as camera:
            logger.info(
                "CAMERA CHECK open=OK serial=%s target_fps=%.3f",
                camera.serial,
                args.camera_fps,
            )
            grabber = LatestFrameGrabber(camera)
            grabber.start()
            first_frame_logged = False

            while not stop_requested:
                try:
                    last_sequence, captured = grabber.get_latest(
                        last_sequence, timeout=2.0
                    )
                except TimeoutError:
                    logger.warning(
                        "CAMERA CHECK camera_status=TIMEOUT "
                        "no frame received for 2 seconds"
                    )
                    continue

                if first_camera_frame_id is None:
                    first_camera_frame_id = captured.camera_frame_id
                last_camera_frame_id = captured.camera_frame_id
                if not first_frame_logged:
                    height, width = captured.bgr.shape[:2]
                    channels = 1 if captured.bgr.ndim == 2 else captured.bgr.shape[2]
                    logger.info(
                        "CAMERA CHECK first_frame=OK serial=%s "
                        "width=%s height=%s channels=%s camera_frame=%s "
                        "camera_timestamp_ns=%s",
                        camera.serial,
                        width,
                        height,
                        channels,
                        captured.camera_frame_id,
                        captured.camera_timestamp_ns,
                    )
                    first_frame_logged = True

                visualization = detector.process_frame(captured.bgr, processed)
                confirmations = detector.consume_pending_confirmations()
                hold_frames = max(
                    int(round(CONFIRMED_BOX_HOLD_SEC * args.camera_fps)), 1
                )

                for event in confirmations:
                    _draw_confirmed_box(visualization, event)
                    recent_events.append((processed, event))
                    position = estimate_position(
                        event,
                        visualization.shape[1],
                        visualization.shape[0],
                        static_pose,
                    )
                    latency_ms = (
                        time.monotonic() - captured.captured_monotonic
                    ) * 1000.0
                    record = store.save(
                        event,
                        captured,
                        visualization,
                        processed_frame_idx=processed,
                        processing_latency_ms=latency_ms,
                        position=position,
                    )
                    event_count += 1
                    logger.warning(
                        "EVENT id=%s processed_frame=%s camera_frame=%s "
                        "bbox=%s area=%.2f growth=%.3f radial=%s "
                        "latency_ms=%.2f position=%s screenshot=%s",
                        event.event_id,
                        processed,
                        captured.camera_frame_id,
                        event.bbox,
                        event.area,
                        event.growth_ratio,
                        event.radial_ratio,
                        record["processing_latency_ms"],
                        position,
                        record["full_frame_path"],
                    )

                recent_events = [
                    item
                    for item in recent_events
                    if processed - item[0] <= hold_frames
                ]
                for _, event in recent_events:
                    _draw_confirmed_box(visualization, event)

                elapsed = max(time.monotonic() - started_at, 1e-6)
                processing_fps = (processed + 1) / elapsed
                processing_latency_ms = (
                    time.monotonic() - captured.captured_monotonic
                ) * 1000.0
                cv2.putText(
                    visualization,
                    f"ONLINE {processing_fps:.1f} FPS",
                    (12, visualization.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (80, 255, 80),
                    2,
                    cv2.LINE_AA,
                )

                if args.display:
                    cv2.imshow("Boom - Impact Point Detection", visualization)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        logger.info("Display quit key pressed")
                        break

                processed += 1
                if args.report_every > 0 and processed % args.report_every == 0:
                    captured_count = (
                        last_camera_frame_id - first_camera_frame_id + 1
                        if first_camera_frame_id is not None
                        and last_camera_frame_id is not None
                        else processed
                    )
                    skipped = max(captured_count - processed, 0)
                    logger.info(
                        "HEARTBEAT camera_status=OK processed=%s camera_frame=%s "
                        "processing_fps=%.2f latency_ms=%.2f "
                        "camera_frames_skipped=%s confirmed_events=%s",
                        processed,
                        captured.camera_frame_id,
                        processing_fps,
                        processing_latency_ms,
                        skipped,
                        event_count,
                    )
                if args.max_frames > 0 and processed >= args.max_frames:
                    logger.info("Reached --max-frames=%s", args.max_frames)
                    break
    except Exception:
        logger.exception("Online runtime failed")
        raise
    finally:
        if grabber is not None:
            grabber.stop()
        if args.display:
            cv2.destroyAllWindows()

        guard = detector.camera_guard_summary()
        elapsed = max(time.monotonic() - started_at, 0.0)
        logger.info(
            "Runtime stopped: processed=%s events=%s elapsed_sec=%.3f camera_guard=%s",
            processed,
            event_count,
            elapsed,
            guard,
        )
        logger.info("Runtime log: %s", store.runtime_log_path)
        store.close()

    return 0
