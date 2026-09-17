"""Impact-point geolocation and presentation panel shared by MOD-IR output."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boom.core.events import GeoReference
from boom.core.coordinates import coordinate, twd97_to_wgs84
from boom.runtime.reference import read_geo_reference_from_image

PANEL_WIDTH = 500
UI_FONT_PATHS = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msjhbd.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
)


@dataclass
class LocatedEvent:
    event_id: int
    frame_idx: int
    timestamp_sec: float
    bbox: tuple[int, int, int, int]
    easting: Optional[float] = None
    northing: Optional[float] = None
    lon: Optional[float] = None
    lat: Optional[float] = None

    @property
    def has_coordinate(self) -> bool:
        return self.easting is not None and self.northing is not None


def resolve_geo_reference(
    reference_image: Optional[Path],
    lon: Optional[float],
    lat: Optional[float],
    alt: Optional[float],
    yaw: Optional[float],
    pitch: Optional[float],
    roll: Optional[float],
) -> Optional[GeoReference]:
    if reference_image is not None:
        if not reference_image.exists():
            raise FileNotFoundError(f"找不到定位參考照片：{reference_image}")
        return read_geo_reference_from_image(reference_image)
    values = (lon, lat, alt, yaw, pitch, roll)
    if all(value is not None for value in values):
        return GeoReference(
            lon=float(lon), lat=float(lat), alt=float(alt), yaw=float(yaw),
            pitch=float(pitch), roll=float(roll),
        )
    if any(value is not None for value in values):
        raise ValueError("手動定位必須同時提供 lon/lat/alt/yaw/pitch/roll")
    return None


def locate_event(
    event_id: int,
    frame_idx: int,
    fps: float,
    bbox: tuple[int, int, int, int],
    process_scale: float,
    source_width: int,
    source_height: int,
    geo: Optional[GeoReference],
) -> LocatedEvent:
    event = LocatedEvent(event_id, frame_idx, frame_idx / fps, bbox)
    if geo is None:
        return event

    x, y, width, height = bbox
    source_x = (x + width / 2.0) / process_scale
    source_y = (y + height / 2.0) / process_scale

    # The original artillery projection is calibrated for a 2840 x 2840 image.
    # Normalize arbitrary video frames into that same calibration plane.
    calibrated_x = source_x / max(source_width, 1) * 2840.0
    calibrated_y = source_y / max(source_height, 1) * 2840.0
    easting, northing = coordinate(
        geo.lon, geo.lat, geo.alt, geo.yaw, geo.pitch, geo.roll,
        calibrated_x, calibrated_y,
    )
    lon, lat = twd97_to_wgs84(easting, northing)
    event.easting = easting
    event.northing = northing
    event.lon = lon
    event.lat = lat
    return event


def build_impact_panel(
    height: int,
    events: list[LocatedEvent],
    video_name: str,
    geo_available: bool,
    width: int = PANEL_WIDTH,
) -> np.ndarray:
    panel = np.full((height, width, 3), 246, dtype=np.uint8)
    _draw_header(panel, video_name)

    top = 176
    if not events:
        _draw_empty_card(panel, top, width, geo_available)
        return panel

    for event in events[-4:]:
        top = _draw_event_card(panel, top, width, event)
        if top >= height - 150:
            break
    return panel


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in UI_FONT_PATHS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_text(
    canvas: np.ndarray,
    text: str,
    position: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(image)
    drawer.text(position, text, font=_get_font(size),
                fill=(color[2], color[1], color[0]))
    canvas[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _draw_header(panel: np.ndarray, video_name: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 150), (18, 34, 62), -1)
    cv2.rectangle(panel, (0, 150), (panel.shape[1], 158), (108, 176, 255), -1)
    _draw_text(panel, "煙霧偵測", (18, 12), 36, (247, 250, 255))
    _draw_text(panel, "MOD-IR Smoke Detection", (20, 58), 23, (214, 228, 250))
    _draw_text(panel, video_name, (20, 102), 18, (214, 228, 250))


def _draw_empty_card(panel: np.ndarray, top: int, width: int, geo_available: bool) -> None:
    left, right = 20, width - 20
    bottom = min(top + 134, panel.shape[0] - 20)
    cv2.rectangle(panel, (left, top), (right, bottom), (244, 247, 252), -1)
    cv2.rectangle(panel, (left, top), (left + 8, bottom), (83, 148, 255), -1)
    cv2.rectangle(panel, (left, top), (right, bottom), (208, 217, 230), 1)
    _draw_text(panel, "尚無事件", (left + 18, top + 14), 28, (34, 46, 64))
    _draw_text(panel, "No Event", (left + 20, top + 55), 20, (88, 100, 119))
    status = "定位資料已就緒" if geo_available else "未提供定位照片"
    _draw_text(panel, status, (left + 20, top + 88), 18, (88, 100, 119))


def _draw_event_card(
    panel: np.ndarray,
    top: int,
    width: int,
    event: LocatedEvent,
) -> int:
    left, right = 20, width - 20
    card_height = 282 if event.has_coordinate else 196
    bottom = min(top + card_height, panel.shape[0] - 20)
    if bottom <= top:
        return panel.shape[0]
    cv2.rectangle(panel, (left, top), (right, bottom), (250, 252, 254), -1)
    cv2.rectangle(panel, (left, top), (left + 10, bottom), (83, 148, 255), -1)
    cv2.rectangle(panel, (left, top), (right, bottom), (204, 212, 224), 1)
    _draw_text(
        panel, f"事件 {event.event_id:02d}  Event {event.event_id:02d}",
        (left + 18, top + 14), 34, (30, 43, 61),
    )
    _draw_text(panel, f"時間  {event.timestamp_sec:.2f} 秒",
               (left + 18, top + 58), 21, (70, 82, 101))

    if not event.has_coordinate:
        _draw_text(panel, "座標無法取得", (left + 18, top + 103), 25, (40, 52, 72))
        _draw_text(panel, "請提供 DJI 定位參考照片",
                   (left + 18, top + 143), 20, (88, 100, 119))
        return bottom + 14

    labels = ("TWD97 E", "TWD97 N", "WGS84 Lat", "WGS84 Lon")
    values = (
        f"{event.easting:.2f}", f"{event.northing:.2f}",
        f"{event.lat:.6f}", f"{event.lon:.6f}",
    )
    for index, (label, value) in enumerate(zip(labels, values)):
        row_y = top + 96 + index * 42
        _draw_text(panel, label, (left + 16, row_y), 20, (40, 52, 72))
        _draw_text(panel, value, (left + 170, row_y), 24, (40, 52, 72))
    return bottom + 14
