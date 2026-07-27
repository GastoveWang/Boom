from __future__ import annotations
import argparse
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import GPSTAGS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mine.offline.smoke_dust_detector import DetectorConfig, InstantSmokeDustDetector


IMAGE_PATH: Optional[str] = None
VIDEO_PATH: Optional[str] = None
OUTPUT_ROOT = str(PROJECT_ROOT / "output" / "offline")

UI_MODE = "client"
DISPLAY_SCALE = 0.5
DISPLAY_SECONDS = 10.0
START_FRAME = 0
MAX_FRAMES = -1
NO_DISPLAY = True
PANEL_WIDTH = 500

UI_FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msjhbd.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
]


DOWNSCALE = 0.7
STRIDE_FRAMES = 2
DIFF_THRESHOLD = 25
MIN_BLOB_AREA = 30.0
MIN_GROWTH_RATIO = 1.2
STOP_GROWTH_RATIO = 1.08
MAX_EVENT_AGE_SEC = 3.0
CONFIRMED_BOX_HOLD_SEC = 1.5
TRACK_MATCH_DISTANCE = 200.0
TRAJECTORY_MATCH_DISTANCE = 270.0
RECENT_CONFIRMED_MATCH_SEC = 10.0
MAX_MISSED_FRAMES = 6
MIN_DIMENSION_GROWTH_RATIO = 1.08
MAX_CENTER_SHIFT_RATIO = 0.65
SPLIT_BLOB_AREA = 2500.0
ENABLE_RADIAL_FLOW_CHECK = True
DRAW_DEBUG_TILES = False
SHOW_CANDIDATE_BOXES = True


DRONE_LON = None
DRONE_LAT = None
DRONE_ALT = None
DRONE_YAW = 0.0
DRONE_PITCH = 20.0
DRONE_ROLL = 0.0


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


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in UI_FONT_PATHS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_text_inplace(
    canvas: np.ndarray,
    text: str,
    position: Tuple[int, int],
    size: int,
    color: Tuple[int, int, int],
) -> None:
    pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(pil_image)
    drawer.text(position, text, font=_get_font(size),
                fill=(color[2], color[1], color[0]))
    canvas[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_confirmed_event_overlays(frame: np.ndarray, active_events: List[LoggedEvent]) -> np.ndarray:
    canvas = frame.copy()
    for event in active_events:
        x, y, w, h = event.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"EVENT {event.event_id}",
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _decode_gps_coordinate(values: Tuple[float, float, float], ref: str) -> float:
    degrees, minutes, seconds = values
    decimal = float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0
    if ref in {"S", "W"}:
        decimal *= -1.0
    return decimal


def _parse_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _read_image_xmp_fields(image_path: Path) -> Dict[str, str]:
    payload = image_path.read_bytes()
    text = payload.decode("utf-8", errors="ignore")
    fields: Dict[str, str] = {}
    for key in (
        "AbsoluteAltitude",
        "RelativeAltitude",
        "GimbalYawDegree",
        "GimbalPitchDegree",
        "GimbalRollDegree",
        "FlightYawDegree",
    ):
        match = re.search(rf'{key}="([^"]+)"', text)
        if match:
            fields[key] = match.group(1)
    return fields


def read_geo_reference_from_image(image_path: Path) -> GeoReference:
    image = Image.open(image_path)
    exif = image.getexif()
    gps_ifd = exif.get_ifd(0x8825) if exif else {}
    gps = {GPSTAGS.get(key, key): value for key, value in gps_ifd.items()}
    xmp = _read_image_xmp_fields(image_path)

    lat_values = gps.get("GPSLatitude")
    lat_ref = gps.get("GPSLatitudeRef")
    lon_values = gps.get("GPSLongitude")
    lon_ref = gps.get("GPSLongitudeRef")
    if not lat_values or not lat_ref or not lon_values or not lon_ref:
        raise ValueError(
            f"Image does not contain complete GPS EXIF info: {image_path}")

    lon = _decode_gps_coordinate(lon_values, lon_ref)
    lat = _decode_gps_coordinate(lat_values, lat_ref)

    exif_alt = gps.get("GPSAltitude")
    alt = (
        _parse_float(xmp.get("AbsoluteAltitude"))
        or _parse_float(xmp.get("RelativeAltitude"))
        or _parse_float(exif_alt)
        or 0.0
    )

    yaw = _parse_float(xmp.get("GimbalYawDegree"))
    if yaw is None:
        yaw = _parse_float(xmp.get("FlightYawDegree"))
    pitch = _parse_float(xmp.get("GimbalPitchDegree"))
    roll = _parse_float(xmp.get("GimbalRollDegree"))

    return GeoReference(
        lon=lon if DRONE_LON is None else float(DRONE_LON),
        lat=lat if DRONE_LAT is None else float(DRONE_LAT),
        alt=alt if DRONE_ALT is None else float(DRONE_ALT),
        yaw=(yaw if yaw is not None else 0.0) if DRONE_YAW is None else float(
            DRONE_YAW),
        pitch=(pitch if pitch is not None else -
               45.0) if DRONE_PITCH is None else float(DRONE_PITCH),
        roll=(roll if roll is not None else 0.0) if DRONE_ROLL is None else float(
            DRONE_ROLL),
    )


def resolve_video_path(image_path: Path) -> Path:
    if VIDEO_PATH:
        return Path(VIDEO_PATH)

    image_stem = image_path.stem
    if not image_stem.endswith("_P"):
        raise ValueError(
            f"Image filename must end with '_P': {image_path.name}")

    video_stem = f"{image_stem[:-2]}_V"
    for suffix in (".MP4", ".mp4", ".MOV", ".mov", ".AVI", ".avi"):
        candidate = image_path.with_name(f"{video_stem}{suffix}")
        if candidate.exists():
            return candidate

    fuzzy_candidates = sorted(image_path.parent.glob(f"{video_stem}.*"))
    if fuzzy_candidates:
        return fuzzy_candidates[0]

    raise FileNotFoundError(
        f"Could not find matching video for {image_path.name}. Expected something like {video_stem}.MP4"
    )


def _default_video_path() -> Optional[Path]:
    data_dir = PROJECT_ROOT / "data"
    for pattern in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"):
        matches = sorted(data_dir.rglob(pattern)) if data_dir.exists() else []
        if matches:
            return matches[0]
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline smoke/dust detection with Terminal progress."
    )
    parser.add_argument("--video", type=Path,
                        help="Input video; defaults to the first video under data/.")
    parser.add_argument("--reference-image", type=Path,
                        help="Optional DJI image containing GPS/XMP metadata.")
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_ROOT))
    parser.add_argument("--start-frame", type=int, default=START_FRAME)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--report-every", type=int, default=60,
                        help="Print progress every N frames (default: 60).")
    return parser


def _format_duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def build_detector_config() -> DetectorConfig:
    debug_mode = UI_MODE.lower() == "debug"
    return DetectorConfig(
        downscale=DOWNSCALE,
        stride_frames=max(STRIDE_FRAMES, 1),
        diff_threshold=DIFF_THRESHOLD,
        min_blob_area=MIN_BLOB_AREA,
        min_growth_ratio=MIN_GROWTH_RATIO,
        stop_growth_ratio=STOP_GROWTH_RATIO,
        max_event_age_sec=MAX_EVENT_AGE_SEC,
        confirmed_box_hold_sec=CONFIRMED_BOX_HOLD_SEC,
        track_match_distance=TRACK_MATCH_DISTANCE,
        trajectory_match_distance=TRAJECTORY_MATCH_DISTANCE,
        recent_confirmed_match_sec=RECENT_CONFIRMED_MATCH_SEC,
        max_missed_frames=MAX_MISSED_FRAMES,
        min_dimension_growth_ratio=MIN_DIMENSION_GROWTH_RATIO,
        max_center_shift_ratio=MAX_CENTER_SHIFT_RATIO,
        split_blob_area=SPLIT_BLOB_AREA,
        enable_radial_flow_check=ENABLE_RADIAL_FLOW_CHECK,
        draw_debug_tiles=DRAW_DEBUG_TILES if debug_mode else False,
        show_candidate_boxes=SHOW_CANDIDATE_BOXES if debug_mode else False,
        show_status_overlay=debug_mode,
    )


def _draw_event_card(
    panel: np.ndarray,
    top: int,
    width: int,
    event: LoggedEvent,
    *,
    compact: bool,
) -> int:
    left = 20
    right = width - 20
    card_height = 282 if compact else 336
    bottom = min(top + card_height, panel.shape[0] - 20)
    if bottom <= top:
        return panel.shape[0]

    cv2.rectangle(panel, (left, top), (right, bottom), (250, 252, 254), -1)
    cv2.rectangle(panel, (left, top), (left + 10, bottom), (83, 148, 255), -1)
    cv2.rectangle(panel, (left, top), (right, bottom), (204, 212, 224), 1)

    _draw_text_inplace(
        panel,
        f"\u4e8b\u4ef6 {event.event_id:02d}  Event {event.event_id:02d}",
        (left + 18, top + 14),
        34 if compact else 42,
        (30, 43, 61),
    )

    label_x = left + 16
    value_x = left + (170 if compact else 218)
    row_y = top + 72
    row_gap = 49 if compact else 60
    label_size = 24 if compact else 31
    value_size = 28 if compact else 37

    _draw_text_inplace(panel, "TWD97 E", (label_x, row_y),
                       label_size, (40, 52, 72))
    _draw_text_inplace(panel, f"{event.easting:.2f}",
                       (value_x, row_y), value_size, (40, 52, 72))

    _draw_text_inplace(panel, "TWD97 N", (label_x, row_y +
                       row_gap), label_size, (40, 52, 72))
    _draw_text_inplace(panel, f"{event.northing:.2f}",
                       (value_x, row_y + row_gap), value_size, (40, 52, 72))

    _draw_text_inplace(panel, "WGS84 Lat", (label_x, row_y +
                       row_gap * 2), label_size, (40, 52, 72))
    _draw_text_inplace(panel, f"{event.lat:.6f}", (value_x,
                       row_y + row_gap * 2), value_size, (40, 52, 72))

    _draw_text_inplace(panel, "WGS84 Lon", (label_x, row_y +
                       row_gap * 3), label_size, (40, 52, 72))
    _draw_text_inplace(panel, f"{event.lon:.6f}", (value_x,
                       row_y + row_gap * 3), value_size, (40, 52, 72))
    return bottom + 14


def build_debug_side_panel(height: int, width: int, active_events: List[LoggedEvent]) -> np.ndarray:
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width, 92), (234, 240, 248), -1)
    _draw_text_inplace(panel, "\u7832\u64ca\u9ede\u5075\u6e2c",
                       (18, 10), 35, (28, 41, 58))
    _draw_text_inplace(panel, "Impact Point Detection",
                       (18, 50), 22, (78, 92, 112))

    top = 112
    for event in active_events[-4:]:
        top = _draw_event_card(panel, top, width, event, compact=False)
        if top >= height - 150:
            break
    return panel


def _draw_client_header(panel: np.ndarray, video_name: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 150), (18, 34, 62), -1)
    cv2.rectangle(panel, (0, 150), (panel.shape[1], 158), (108, 176, 255), -1)
    _draw_text_inplace(panel, "\u7832\u64ca\u9ede\u5075\u6e2c",
                       (18, 12), 36, (247, 250, 255))
    _draw_text_inplace(panel, "Impact Point Detection",
                       (20, 58), 23, (214, 228, 250))
    _draw_text_inplace(panel, video_name, (20, 102), 18, (214, 228, 250))


def _draw_empty_client_card(panel: np.ndarray, top: int, width: int) -> None:
    left = 20
    right = width - 20
    bottom = min(top + 108, panel.shape[0] - 20)
    cv2.rectangle(panel, (left, top), (right, bottom), (244, 247, 252), -1)
    cv2.rectangle(panel, (left, top), (left + 8, bottom), (83, 148, 255), -1)
    cv2.rectangle(panel, (left, top), (right, bottom), (208, 217, 230), 1)
    _draw_text_inplace(panel, "\u5c1a\u7121\u4e8b\u4ef6",
                       (left + 18, top + 14), 28, (34, 46, 64))
    _draw_text_inplace(panel, "No Event", (left + 20,
                       top + 58), 20, (88, 100, 119))


def build_client_side_panel(
    height: int,
    width: int,
    active_events: List[LoggedEvent],
    video_name: str,
) -> np.ndarray:
    panel = np.full((height, width, 3), 246, dtype=np.uint8)
    _draw_client_header(panel, video_name)

    top = 176
    if not active_events:
        _draw_empty_client_card(panel, top, width)
        return panel

    for event in active_events[-4:]:
        top = _draw_event_card(panel, top, width, event, compact=True)
        if top >= height - 150:
            break
    return panel


def build_side_panel(
    height: int,
    width: int,
    active_events: List[LoggedEvent],
    mode: str,
    video_name: str,
) -> np.ndarray:
    if mode.lower() == "client":
        return build_client_side_panel(height, width, active_events, video_name)
    return build_debug_side_panel(height, width, active_events)


def get_panel_width(mode: str) -> int:
    if mode.lower() == "client":
        return PANEL_WIDTH
    return max(PANEL_WIDTH, 760)


def write_coordinate_log(output_dir: Path, video_stem: str, events: List[LoggedEvent]) -> Path:
    log_path = output_dir / f"{video_stem}_impact_coordinates.txt"
    with log_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(
                f"ID {event.event_id}: "
                f"t={event.timestamp_sec:.2f}s, "
                f"TWD97(E={event.easting:.2f}, N={event.northing:.2f}), "
                f"WGS84(lat={event.lat:.6f}, lon={event.lon:.6f})\n"
            )
    return log_path


def coordinate(
    lon: float,
    lat: float,
    alt: float,
    yaw: float,
    pitch: float,
    roll: float,
    orix: float,
    oriy: float,
) -> Tuple[float, float]:
    a = 6378137.0
    b = 6356752.3142451
    lon0 = 121 * math.pi / 180
    k0 = 0.9999
    dx = 250000
    dy = 0
    e = 1 - b**2 / a**2
    e2 = e / (1 - e)

    lon = (lon - math.floor((lon + 180) / 360) * 360) * math.pi / 180
    lat = lat * math.pi / 180
    v = a / math.sqrt(1 - e * math.sin(lat) ** 2)
    t = math.tan(lat) ** 2
    c = e2 * math.cos(lat) ** 2
    a_term = math.cos(lat) * (lon - lon0)
    m = a * (
        (1 - e / 4 - 3 * e**2 / 64 - 5 * e**3 / 256) * lat
        - (3 * e / 8 + 3 * e**2 / 32 + 45 * e**3 / 1024) * math.sin(2 * lat)
        + (15 * e**2 / 256 + 45 * e**3 / 1024) * math.sin(4 * lat)
        - (35 * e**3 / 3072) * math.sin(6 * lat)
    )

    twd_e = dx + k0 * v * (
        a_term
        + (1 - t + c) * a_term**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * e2) * a_term**5 / 120
    )
    twd_n = dy + k0 * (
        m
        + v
        * math.tan(lat)
        * (
            a_term**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a_term**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * e2) * a_term**6 / 720
        )
    )

    yaw = yaw * math.pi / 180
    pitch = pitch * math.pi / 180
    roll = roll * math.pi / 180

    m11 = math.sin(yaw) * math.sin(pitch) * math.sin(roll) + \
        math.cos(yaw) * math.cos(roll)
    m12 = math.cos(yaw) * math.sin(pitch) * math.sin(roll) - \
        math.sin(yaw) * math.cos(roll)
    m13 = -math.cos(pitch) * math.sin(roll)
    m21 = math.sin(yaw) * math.cos(pitch)
    m22 = math.cos(yaw) * math.cos(pitch)
    m23 = math.sin(pitch)
    m31 = -(math.sin(yaw) * math.sin(pitch) *
            math.cos(roll) - math.cos(yaw) * math.sin(roll))
    m32 = -(math.cos(yaw) * math.sin(pitch) *
            math.cos(roll) + math.sin(yaw) * math.sin(roll))
    m33 = math.cos(pitch) * math.cos(roll)

    f = 4.5 / 0.00274
    f2 = 4.5
    pixel_size = 0.00274
    image_width = 2840
    image_height = 2840

    dx_pixel = orix - image_width / 2
    dy_pixel = -oriy + image_height / 2
    dist = math.sqrt(dx_pixel**2 + dy_pixel**2)
    if dist == 0:
        dist = 1e-6

    theta = dist / f
    radius = f * math.tan(theta)
    x = dx_pixel * radius / dist * pixel_size
    y = dy_pixel * radius / dist * pixel_size

    easting = -alt * (m11 * x + m21 * y - m31 * f2) / \
        (m13 * x + m23 * y - m33 * f2) + twd_e
    northing = -alt * (m12 * x + m22 * y - m32 * f2) / \
        (m13 * x + m23 * y - m33 * f2) + twd_n
    return easting, northing


def twd97_to_wgs84(easting: float, northing: float) -> Tuple[float, float]:
    a = 6378137.0
    b = 6356752.3142451
    lon0 = 121 * math.pi / 180
    k0 = 0.9999
    dx = 250000
    e = math.sqrt(1 - (b**2 / a**2))
    x = easting - dx
    y = northing

    m = y / k0
    mu = m / (a * (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256))
    e1 = (1 - math.sqrt(1 - e**2)) / (1 + math.sqrt(1 - e**2))

    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512

    fp = (
        mu
        + j1 * math.sin(2 * mu)
        + j2 * math.sin(4 * mu)
        + j3 * math.sin(6 * mu)
        + j4 * math.sin(8 * mu)
    )

    e2 = e**2 / (1 - e**2)
    c1 = e2 * math.cos(fp) ** 2
    t1 = math.tan(fp) ** 2
    r1 = a * (1 - e**2) / ((1 - e**2 * math.sin(fp) ** 2) ** 1.5)
    n1 = a / math.sqrt(1 - e**2 * math.sin(fp) ** 2)
    d = x / (n1 * k0)

    q1 = n1 * math.tan(fp) / r1
    q2 = d**2 / 2
    q3 = (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e2) * d**4 / 24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * t1 **
          2 - 252 * e2 - 3 * c1**2) * d**6 / 720
    lat = fp - q1 * (q2 - q3 + q4)

    q5 = d
    q6 = (1 + 2 * t1 + c1) * d**3 / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e2 + 24 * t1**2) * d**5 / 120
    lon = lon0 + (q5 - q6 + q7) / math.cos(fp)

    return math.degrees(lon), math.degrees(lat)


def main() -> None:
    args = build_arg_parser().parse_args()
    config = build_detector_config()
    image_path = args.reference_image or (Path(IMAGE_PATH) if IMAGE_PATH else None)
    if image_path is not None and not image_path.exists():
        raise FileNotFoundError(f"Reference image does not exist: {image_path}")
    if image_path is not None:
        geo = read_geo_reference_from_image(image_path)
    else:
        geo = GeoReference()
        print("[WARN] No reference image supplied; using default geo reference.")

    video_path = args.video or (Path(VIDEO_PATH) if VIDEO_PATH else None)
    if video_path is None and image_path is not None:
        video_path = resolve_video_path(image_path)
    if video_path is None:
        video_path = _default_video_path()
    if video_path is None:
        raise FileNotFoundError("No input video found. Use --video or place a video under data/.")
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    detector = InstantSmokeDustDetector(config, fps)

    base_name = video_path.stem
    output_dir = args.output_dir / base_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"output_{base_name}.mp4"

    writer = None
    frame_idx = args.start_frame
    processed = 0
    recent_events: List[LoggedEvent] = []
    recent_red_events: List[LoggedEvent] = []
    all_events: List[LoggedEvent] = []
    keep_frames = max(int(round(DISPLAY_SECONDS * detector.fps)), 1)
    red_hold_frames = max(int(round(CONFIRMED_BOX_HOLD_SEC * detector.fps)), 1)
    panel_width = get_panel_width(UI_MODE)
    available_frames = max(frame_count - args.start_frame, 0) if frame_count else 0
    target_frames = available_frames
    if args.max_frames > 0:
        target_frames = min(target_frames, args.max_frames) if target_frames else args.max_frames
    started_at = time.monotonic()

    print("[INFO] Display: disabled (headless)")
    if target_frames:
        print(
            f"[INFO] Workload: {target_frames} frames "
            f"({_format_duration(target_frames / detector.fps)} of video)"
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            vis = detector.process_frame(frame, frame_idx)

            for event in detector.consume_pending_confirmations():
                x, y, w, h = event.bbox
                cx = x + w / 2.0
                cy = y + h / 2.0
                easting, northing = coordinate(
                    geo.lon,
                    geo.lat,
                    geo.alt,
                    geo.yaw,
                    geo.pitch,
                    geo.roll,
                    cx,
                    cy,
                )
                lon, lat = twd97_to_wgs84(easting, northing)
                logged = LoggedEvent(
                    event_id=event.event_id,
                    frame_idx=event.frame_idx,
                    timestamp_sec=event.timestamp_sec,
                    bbox=event.bbox,
                    easting=easting,
                    northing=northing,
                    lon=lon,
                    lat=lat,
                )
                all_events.append(logged)
                recent_events.append(logged)
                recent_red_events.append(logged)

            recent_events = [
                event for event in recent_events if frame_idx - event.frame_idx <= keep_frames]
            recent_red_events = [
                event for event in recent_red_events if frame_idx - event.frame_idx <= red_hold_frames]

            vis = draw_confirmed_event_overlays(vis, recent_red_events)

            info_panel = build_side_panel(
                vis.shape[0],
                panel_width,
                recent_events,
                UI_MODE,
                video_path.name,
            )
            combined = np.hstack((vis, info_panel))

            if DISPLAY_SCALE != 1.0:
                combined = cv2.resize(
                    combined,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA,
                )

            if writer is None:
                height, width = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_video_path), fourcc, detector.fps, (width, height))

            writer.write(combined)

            frame_idx += 1
            processed += 1
            report_every = max(args.report_every, 1)
            if processed % report_every == 0 or (target_frames and processed >= target_frames):
                elapsed = max(time.monotonic() - started_at, 1e-6)
                processing_fps = processed / elapsed
                progress = processed / target_frames if target_frames else 0.0
                remaining = (
                    (target_frames - processed) / processing_fps
                    if target_frames and processing_fps > 0 else 0.0
                )
                print(
                    f"[PROGRESS] {processed}/{target_frames or '?'} frames "
                    f"({progress * 100:6.2f}%) "
                    f"| video={_format_duration(frame_idx / detector.fps)} "
                    f"| speed={processing_fps:.2f} fps "
                    f"| elapsed={_format_duration(elapsed)} "
                    f"| ETA={_format_duration(remaining)}",
                    flush=True,
                )
            if args.max_frames > 0 and processed >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    log_path = write_coordinate_log(output_dir, base_name, all_events)
    print(f"[INFO] Processed frames: {processed}")
    print(f"[INFO] Confirmed events: {len(all_events)}")
    guard_summary = detector.camera_guard_summary()
    if guard_summary:
        summary_text = ", ".join(
            f"{reason}={count}" for reason, count in sorted(guard_summary.items()))
        print(f"[INFO] Camera guard skipped frames: {summary_text}")
    print(f"[INFO] Image metadata source: {image_path or 'configured defaults'}")
    print(
        f"[INFO] Geo reference: lon={geo.lon:.7f}, lat={geo.lat:.7f}, "
        f"alt={geo.alt:.3f}, yaw={geo.yaw:.3f}, pitch={geo.pitch:.3f}, roll={geo.roll:.3f}"
    )
    print(f"[INFO] Input video: {video_path}")
    print(f"[INFO] Output video: {output_video_path}")
    print(f"[INFO] Coordinate log: {log_path}")


if __name__ == "__main__":
    main()
