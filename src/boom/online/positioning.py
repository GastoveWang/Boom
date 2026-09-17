"""Optional static-pose localization for the live runtime."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional
from boom.core.coordinates import coordinate, twd97_to_wgs84
from boom.core.events import ConfirmedEvent

@dataclass(frozen=True)
class StaticPose:
    lon: float
    lat: float
    alt: float
    yaw: float
    pitch: float
    roll: float


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
