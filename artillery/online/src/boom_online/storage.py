"""Online event evidence persistence (JSONL, CSV, coordinates and screenshots)."""
from __future__ import annotations
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import cv2
from artillery.offline.src.detectors.optical_flow_smoke_detector import ConfirmedEvent
from artillery.online.src.spinnaker_camera import CapturedFrame

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
