"""Runtime-facing static pose and logged event records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GeoReference:
    lon: float = 121.54060381534254
    lat: float = 25.013457612687223
    alt: float = 100.0
    yaw: float = 0.0
    pitch: float = -45.0
    roll: float = 0.0


@dataclass
class LoggedEvent:
    event_id: int
    frame_idx: int
    timestamp_sec: float
    bbox: Tuple[int, int, int, int]
    easting: float
    northing: float
    lon: float
    lat: float
    confidence: float
    impact_point: Tuple[float, float]
    confirmed_frame_idx: int
    confirmation_delay_sec: float
    coordinate_status: str = "static_pose_estimate"
