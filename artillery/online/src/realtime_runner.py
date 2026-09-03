"""Run the existing offline detector live from a SpinView/Spinnaker camera."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2

ONLINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artillery.offline.src.pipelines.shared_pipeline import (  # noqa: E402
    CONFIRMED_BOX_HOLD_SEC,
    build_detector_config,
    coordinate,
    twd97_to_wgs84,
)
from artillery.offline.src.detectors.optical_flow_smoke_detector import (  # noqa: E402
    ConfirmedEvent,
    DetectorConfig,
    InstantSmokeDustDetector,
)
from spinnaker_camera import (  # noqa: E402
    CapturedFrame,
    LatestFrameGrabber,
    SpinCamera,
    SpinCameraConfig,
)


LOGGER_NAME = "boom.online"


@dataclass(frozen=True)
class StaticPose:
    lon: float
    lat: float
    alt: float
    yaw: float
    pitch: float
    roll: float


class EventStore:
    """Keep the same evidence categories as the offline output."""

    def __init__(self, output_root: Path) -> None:
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = output_root / session_name
        self.snapshot_dir = self.session_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.runtime_log_path = self.session_dir / "runtime.log"
        self.jsonl_path = self.session_dir / "events.jsonl"
        self.csv_path = self.session_dir / "detections.csv"
        self.coordinate_log_path = self.session_dir / "impact_coordinates.txt"

        self._jsonl = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
        self._csv_file = self.csv_path.open(
            "a", newline="", encoding="utf-8", buffering=1
        )
        self._coordinate_log = self.coordinate_log_path.open(
            "a", encoding="utf-8", buffering=1
        )
        self._csv_writer = csv.writer(self._csv_file)
        if self.csv_path.stat().st_size == 0:
            self._csv_writer.writerow(
                [
                    "event_id",
                    "logged_at_utc",
                    "processed_frame_idx",
                    "camera_frame_id",
                    "camera_timestamp_ns",
                    "processing_latency_ms",
                    "bbox_x",
                    "bbox_y",
                    "bbox_w",
                    "bbox_h",
                    "area",
                    "growth_ratio",
                    "radial_outward_ratio",
                    "twd97_easting",
                    "twd97_northing",
                    "wgs84_lon",
                    "wgs84_lat",
                    "full_frame_path",
                    "crop_path",
                ]
            )

    def close(self) -> None:
        self._jsonl.close()
        self._csv_file.close()
        self._coordinate_log.close()

    def save(
        self,
        event: ConfirmedEvent,
        frame: CapturedFrame,
        visualization: Any,
        *,
        processed_frame_idx: int,
        processing_latency_ms: float,
        position: Optional[dict[str, float]],
    ) -> dict[str, Any]:
        stem = (
            f"event_{event.event_id:04d}"
            f"_frame_{processed_frame_idx:08d}"
            f"_camera_{frame.camera_frame_id:010d}"
        )
        full_path = self.snapshot_dir / f"{stem}_full.jpg"
        crop_path = self.snapshot_dir / f"{stem}_crop.jpg"
        if not cv2.imwrite(str(full_path), visualization):
            raise RuntimeError(f"Failed to save event screenshot: {full_path}")

        x, y, w, h = event.bbox
        pad = 32
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1 = min(x + w + pad, visualization.shape[1])
        y1 = min(y + h + pad, visualization.shape[0])
        crop = visualization[y0:y1, x0:x1]
        saved_crop_path: Optional[Path] = None
        if crop.size:
            if not cv2.imwrite(str(crop_path), crop):
                raise RuntimeError(f"Failed to save event crop: {crop_path}")
            saved_crop_path = crop_path

        logged_at = datetime.now(timezone.utc).isoformat()
        record: dict[str, Any] = {
            "event_id": event.event_id,
            "logged_at_utc": logged_at,
            "processed_frame_idx": processed_frame_idx,
            "camera_frame_id": frame.camera_frame_id,
            "camera_timestamp_ns": frame.camera_timestamp_ns,
            "processing_latency_ms": round(processing_latency_ms, 3),
            "bbox_xywh": list(event.bbox),
            "area": event.area,
            "growth_ratio": event.growth_ratio,
            "radial_outward_ratio": event.radial_ratio,
            "full_frame_path": str(full_path),
            "crop_path": str(saved_crop_path) if saved_crop_path else None,
            "position_is_approximate": position is not None,
            "position": position,
        }
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

        values = position or {}
        self._csv_writer.writerow(
            [
                event.event_id,
                logged_at,
                processed_frame_idx,
                frame.camera_frame_id,
                frame.camera_timestamp_ns,
                f"{processing_latency_ms:.3f}",
                x,
                y,
                w,
                h,
                f"{event.area:.3f}",
                f"{event.growth_ratio:.3f}",
                "" if event.radial_ratio is None else f"{event.radial_ratio:.3f}",
                _format_optional(values.get("twd97_easting")),
                _format_optional(values.get("twd97_northing")),
                _format_optional(values.get("wgs84_lon"), 7),
                _format_optional(values.get("wgs84_lat"), 7),
                str(full_path),
                str(saved_crop_path) if saved_crop_path else "",
            ]
        )

        if position:
            self._coordinate_log.write(
                f"ID {event.event_id}: "
                f"t={event.timestamp_sec:.2f}s, "
                f"TWD97(E={position['twd97_easting']:.2f}, "
                f"N={position['twd97_northing']:.2f}), "
                f"WGS84(lat={position['wgs84_lat']:.6f}, "
                f"lon={position['wgs84_lon']:.6f})\n"
            )
        else:
            self._coordinate_log.write(
                f"ID {event.event_id}: "
                f"t={event.timestamp_sec:.2f}s, coordinate=unavailable\n"
            )
        return record


def _format_optional(value: Optional[float], precision: int = 3) -> str:
    return "" if value is None else f"{value:.{precision}f}"


def configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


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


def static_pose_from_args(args: argparse.Namespace) -> Optional[StaticPose]:
    supplied = (args.drone_lon, args.drone_lat, args.drone_alt)
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied):
        raise ValueError(
            "--drone-lon, --drone-lat and --drone-alt must be supplied together."
        )
    return StaticPose(
        lon=args.drone_lon,
        lat=args.drone_lat,
        alt=args.drone_alt,
        yaw=args.drone_yaw,
        pitch=args.drone_pitch,
        roll=args.drone_roll,
    )


def estimate_position(
    event: ConfirmedEvent,
    image_width: int,
    image_height: int,
    pose: Optional[StaticPose],
) -> Optional[dict[str, float]]:
    if pose is None:
        return None
    x, y, w, h = event.bbox
    cx = (x + w / 2.0) * 2840.0 / max(image_width, 1)
    cy = (y + h / 2.0) * 2840.0 / max(image_height, 1)
    easting, northing = coordinate(
        pose.lon, pose.lat, pose.alt, pose.yaw, pose.pitch, pose.roll, cx, cy
    )
    lon, lat = twd97_to_wgs84(easting, northing)
    return {
        "twd97_easting": easting,
        "twd97_northing": northing,
        "wgs84_lon": lon,
        "wgs84_lat": lat,
    }


def _draw_confirmed_box(canvas: Any, event: ConfirmedEvent) -> None:
    x, y, w, h = event.bbox
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.putText(
        canvas,
        f"IMPACT CANDIDATE {event.event_id}",
        (x, max(y - 10, 26)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


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


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        raise SystemExit(run(args))
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
