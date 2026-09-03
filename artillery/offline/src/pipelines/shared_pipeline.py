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

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artillery.offline.src.detectors import (  # noqa: E402
    PIDNetSmokeImpactDetector,
    PIDNetSmokeTrackerConfig,
)
from artillery.offline.src.detectors.optical_flow_smoke_detector import (  # noqa: E402
    DetectorConfig,
    InstantSmokeDustDetector,
)

IMAGE_PATH: Optional[str] = None
VIDEO_PATH: Optional[str] = None
OUTPUT_ROOT = PROJECT_ROOT / "output" / "offline"

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
DIFF_INTERVAL_SEC = 0.12
DIFF_THRESHOLD = 18
MIN_BLOB_AREA = 15.0
MIN_CONFIRM_AREA = 35.0
MIN_GROWTH_RATIO = 1.3
OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 7
CLOSE_ITERATIONS = 1
MIN_RESIDUAL_POLARITY_RATIO = 0.60
MIN_RESIDUAL_SIGNED_COHERENCE = 0.24
MIN_CANDIDATE_FILL_RATIO = 0.16
MAX_INITIAL_MOTION_HISTORY_OVERLAP = 0.55
SMALL_MIN_BLOB_AREA = 4.0
SMALL_DIFF_FLOOR = 8.0
SMALL_NOISE_WINDOW_SIZE = 31
SMALL_NOISE_SIGMA = 2.2
SMALL_MIN_SIGNAL_TO_NOISE = 2.2
SMALL_MIN_GROWTH_RATIO = 1.35
SMALL_MAX_CENTER_SHIFT_RATIO = 1.0
SMALL_CONFIRMATION_HITS_REQUIRED = 3
SMALL_MIN_CUMULATIVE_SIGNAL = 700.0
MAX_DISTRIBUTED_CANDIDATES = 8
MIN_CANDIDATE_FIELD_SPAN_RATIO = 0.30
CONFIRMATION_HITS_REQUIRED = 2
STOP_GROWTH_RATIO = 1.08
MAX_EVENT_AGE_SEC = 3.0
CONFIRMED_BOX_HOLD_SEC = 1.5
TRACK_MATCH_DISTANCE = 200.0
TRAJECTORY_MATCH_DISTANCE = 270.0
RECENT_CONFIRMED_MATCH_SEC = 10.0
POST_EVENT_MEMORY_MAX_SEC = 600.0
POST_EVENT_MEMORY_IDLE_SEC = 600.0
POST_EVENT_SPATIAL_HISTORY_SEC = 600.0
POST_EVENT_MATCH_DISTANCE = 110.0
POST_EVENT_TRAJECTORY_DISTANCE = 80.0
POST_EVENT_RETRIGGER_MIN_AREA = 70.0
POST_EVENT_RETRIGGER_MIN_GROWTH_RATIO = 4.5
MAX_MISSED_FRAMES = 6
MIN_DIMENSION_GROWTH_RATIO = 1.08
MAX_CENTER_SHIFT_RATIO = 0.65
SPLIT_BLOB_AREA = 2500.0
ENABLE_RADIAL_FLOW_CHECK = True
DRAW_DEBUG_TILES = False
SHOW_CANDIDATE_BOXES = True

# PIDNet-S semantic smoke segmentation + new-smoke tracking.  The first second
# is the only interval in which a newly created track may become an impact event.
DETECTOR_BACKEND = "pidnet"
PIDNET_MODEL_PATH = PROJECT_ROOT / "artillery" / "offline" / "model" / "sam_sup_pidnet_s.pt"
PIDNET_DEVICE = "auto"
PIDNET_INPUT_WIDTH = 960
PIDNET_INPUT_HEIGHT = 544
PIDNET_THRESHOLD = 0.80
PIDNET_MIN_COMPONENT_AREA = 30
PIDNET_WARMUP_SEC = 0.50
NEW_SMOKE_WINDOW_SEC = 1.0
PIDNET_CONFIRMATION_HITS = 2
PIDNET_MIN_CONFIRMATION_CONFIDENCE = 0.72


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
    confidence: float
    impact_point: Tuple[float, float]
    confirmed_frame_idx: int
    confirmation_delay_sec: float


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
            f"IMPACT {event.event_id}  NEW  {event.confidence:.2f}",
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        impact_x, impact_y = (int(round(v)) for v in event.impact_point)
        cv2.drawMarker(
            canvas,
            (impact_x, impact_y),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            22,
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


def build_arg_parser(
    default_detector: str = DETECTOR_BACKEND,
    *,
    allow_detector_selection: bool = True,
) -> argparse.ArgumentParser:
    algorithm_name = "PIDNet-S" if default_detector == "pidnet" else "optical flow"
    parser = argparse.ArgumentParser(
        description=f"Offline artillery-impact detection using {algorithm_name}."
    )
    parser.add_argument("--video", type=Path,
                        help="Input video; defaults to the first video under data/.")
    parser.add_argument("--reference-image", type=Path,
                        help="Optional DJI image containing GPS/XMP metadata.")
    parser.add_argument("--start-frame", type=int, default=START_FRAME)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--report-every", type=int, default=60,
                        help="Print progress every N frames (default: 60).")
    if allow_detector_selection:
        parser.add_argument(
            "--detector",
            choices=("pidnet", "motion"),
            default=default_detector,
            help="Select PIDNet-S or traditional optical-flow detection.",
        )
    else:
        parser.set_defaults(detector=default_detector)
    if allow_detector_selection or default_detector == "pidnet":
        parser.add_argument("--model-path", type=Path, default=PIDNET_MODEL_PATH)
        parser.add_argument(
            "--device", choices=("auto", "cpu", "cuda"), default=PIDNET_DEVICE
        )
        parser.add_argument("--seg-threshold", type=float, default=PIDNET_THRESHOLD)
        parser.add_argument("--model-width", type=int, default=PIDNET_INPUT_WIDTH)
        parser.add_argument("--model-height", type=int, default=PIDNET_INPUT_HEIGHT)
        parser.add_argument("--inference-stride", type=int, default=1)
        parser.add_argument(
            "--new-smoke-window-sec", type=float, default=NEW_SMOKE_WINDOW_SEC
        )
        parser.add_argument("--warmup-sec", type=float, default=PIDNET_WARMUP_SEC)
    parser.add_argument("--box-hold-sec", type=float, default=CONFIRMED_BOX_HOLD_SEC)
    parser.add_argument("--panel-hold-sec", type=float, default=DISPLAY_SECONDS)
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
    debug_mode = UI_MODE.lower() == "debug"
    return PIDNetSmokeTrackerConfig(
        model_path=args.model_path,
        device=args.device,
        input_width=args.model_width,
        input_height=args.model_height,
        inference_stride=max(args.inference_stride, 1),
        segmentation_threshold=args.seg_threshold,
        min_component_area=PIDNET_MIN_COMPONENT_AREA,
        warmup_sec=args.warmup_sec,
        new_smoke_window_sec=args.new_smoke_window_sec,
        confirmation_hits=PIDNET_CONFIRMATION_HITS,
        min_confirmation_confidence=PIDNET_MIN_CONFIRMATION_CONFIDENCE,
        track_match_distance=TRACK_MATCH_DISTANCE,
        post_event_memory_sec=POST_EVENT_MEMORY_MAX_SEC,
        show_status_overlay=debug_mode,
        show_candidate_boxes=SHOW_CANDIDATE_BOXES if debug_mode else False,
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
    _draw_text_inplace(
        panel,
        f"NEW  {event.confidence:.2f}",
        (right - 142, top + 22),
        19 if compact else 23,
        (22, 92, 185),
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


def resolve_output_paths(
    video_stem: str,
    detector_name: str,
    output_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Return paths in a method-labelled folder without overwriting prior runs."""
    root = OUTPUT_ROOT if output_root is None else output_root
    method_name = "pidnet" if detector_name == "pidnet" else "optical_flow"
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
            fh.write(
                f"ID {event.event_id}: "
                f"t={event.timestamp_sec:.2f}s, "
                f"confidence={event.confidence:.3f}, "
                f"confirmation_delay={event.confirmation_delay_sec:.3f}s, "
                f"impact_px=({event.impact_point[0]:.1f}, {event.impact_point[1]:.1f}), "
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
    image_width: float = 2840.0,
    image_height: float = 2840.0,
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
    # The original calibration is expressed on a 2840 x 2840 raster. Map the
    # event pixel into that raster so video resolution does not change the FOV.
    calibration_width = 2840.0
    calibration_height = 2840.0
    calibrated_x = orix * calibration_width / max(image_width, 1.0)
    calibrated_y = oriy * calibration_height / max(image_height, 1.0)
    dx_pixel = calibrated_x - calibration_width / 2
    dy_pixel = -calibrated_y + calibration_height / 2
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


def main(
    default_detector: str = DETECTOR_BACKEND,
    *,
    allow_detector_selection: bool = True,
) -> None:
    args = build_arg_parser(
        default_detector=default_detector,
        allow_detector_selection=allow_detector_selection,
    ).parse_args()
    if args.box_hold_sec < 0 or args.panel_hold_sec < 0:
        raise ValueError("hold durations cannot be negative")
    if args.detector == "pidnet":
        if not 0.0 < args.seg_threshold < 1.0:
            raise ValueError("--seg-threshold must be between 0 and 1")
        if args.new_smoke_window_sec <= 0:
            raise ValueError("--new-smoke-window-sec must be positive")
        if args.warmup_sec < 0:
            raise ValueError("--warmup-sec cannot be negative")
        if args.model_width <= 0 or args.model_height <= 0 or args.inference_stride <= 0:
            raise ValueError(
                "model dimensions and --inference-stride must be positive")
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
    if args.detector == "pidnet":
        detector = PIDNetSmokeImpactDetector(build_pidnet_detector_config(args), fps)
    else:
        detector = InstantSmokeDustDetector(build_detector_config(), fps)

    base_name = video_path.stem
    output_video_path, coordinate_log_path = resolve_output_paths(
        base_name, args.detector
    )

    writer = None
    frame_idx = args.start_frame
    processed = 0
    recent_events: List[LoggedEvent] = []
    recent_red_events: List[LoggedEvent] = []
    all_events: List[LoggedEvent] = []
    keep_frames = max(int(round(args.panel_hold_sec * detector.fps)), 1)
    red_hold_frames = max(int(round(args.box_hold_sec * detector.fps)), 1)
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
                impact_point = getattr(event, "impact_point", (x + w / 2.0, y + h / 2.0))
                confidence = float(getattr(event, "confidence", 1.0))
                confirmed_frame_idx = int(
                    getattr(event, "confirm_frame_idx", event.frame_idx)
                )
                easting, northing = coordinate(
                    geo.lon,
                    geo.lat,
                    geo.alt,
                    geo.yaw,
                    geo.pitch,
                    geo.roll,
                    impact_point[0],
                    impact_point[1],
                    image_width=frame.shape[1],
                    image_height=frame.shape[0],
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
                    confidence=confidence,
                    impact_point=impact_point,
                    confirmed_frame_idx=confirmed_frame_idx,
                    confirmation_delay_sec=(
                        confirmed_frame_idx - event.frame_idx
                    ) / detector.fps,
                )
                all_events.append(logged)
                recent_events.append(logged)
                recent_red_events.append(logged)

            recent_events = [
                event
                for event in recent_events
                if frame_idx - event.confirmed_frame_idx <= keep_frames
            ]
            recent_red_events = [
                event
                for event in recent_red_events
                if frame_idx - event.confirmed_frame_idx <= red_hold_frames
            ]

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

    log_path = write_coordinate_log(coordinate_log_path, all_events)
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
    print(f"[INFO] Detector: {args.detector}")
    print(f"[INFO] Output video: {output_video_path}")
    print(f"[INFO] Coordinate log: {log_path}")


if __name__ == "__main__":
    main()
