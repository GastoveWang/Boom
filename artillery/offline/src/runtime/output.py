"""Non-overwriting offline output paths and event coordinate logs."""
from __future__ import annotations
import cv2

from artillery.common.defaults import OUTPUT_ROOT
from artillery.common.events import LoggedEvent
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
import math


def resolve_output_paths(
    video_stem: str,
    detector_name: str,
    output_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Return paths in a method-labelled folder without overwriting prior runs."""
    root = OUTPUT_ROOT if output_root is None else output_root
    method_names = {
        "pidnet": "pidnet",
        "motion": "optical_flow",
        "fusion": "optical_pidnet",
    }
    method_name = method_names.get(detector_name, detector_name)
    folder_stem = f"{video_stem}_{method_name}"
    output_dir = root / folder_stem
    sequence = 2
    while output_dir.exists():
        output_dir = root / f"{folder_stem}_{sequence:02d}"
        sequence += 1
    output_dir.mkdir(parents=True, exist_ok=False)

    return (
        output_dir / f"output_{video_stem}.mp4",
        output_dir / f"{video_stem}_impact_coordinates.txt",
    )


def write_coordinate_log(log_path: Path, events: List[LoggedEvent]) -> Path:
    with log_path.open("w", encoding="utf-8") as fh:
        for event in events:
            if not math.isfinite(event.lon) or not math.isfinite(event.lat):
                fh.write(f"ID {event.event_id}: t={event.timestamp_sec:.2f}s, "
                         f"confidence={event.confidence:.3f}, coordinate=unavailable, "
                         f"impact_px={event.impact_point}\n")
                continue
            fh.write(
                f"ID {event.event_id}: "
                f"t={event.timestamp_sec:.2f}s, "
                f"confidence={event.confidence:.3f}, "
                f"coordinate_status={event.coordinate_status}, "
                f"confirmation_delay={event.confirmation_delay_sec:.3f}s, "
                f"impact_px=({event.impact_point[0]:.1f}, {event.impact_point[1]:.1f}), "
                f"TWD97(E={event.easting:.2f}, N={event.northing:.2f}), "
                f"WGS84(lat={event.lat:.6f}, lon={event.lon:.6f})\n"
            )
    return log_path


class VideoOutput:
    """Encode output after the composed frame dimensions become known."""

    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.writer = None

    def write(self, frame):
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot write output video: {self.path}")
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
