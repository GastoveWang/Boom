"""
==============================================================================
Boom 離線執行期 - 舊版靜態參考照片地理資訊讀取 (reference.py)
==============================================================================

【檔案定位】
本檔案提供舊版靜態無人機參考照片的 Exif / XMP 讀取策略（用於無地圖模式下的
簡易相機姿態備援）。航拍圖資與視覺里程計 (VO) 所需的完整姿態中繼資料則由
`localization/photo.py` 處理。

【核心功能】
- `read_geo_reference_from_image(image_path) -> GeoReference`:
  從參考照片讀取無人機經緯度、高度與雲台姿態角，若缺失則回退至預設值。
==============================================================================
"""
from __future__ import annotations


from PIL import Image
from PIL.ExifTags import GPSTAGS
from boom.config.defaults import DRONE_ALT
from boom.config.defaults import DRONE_LAT
from boom.config.defaults import DRONE_LON
from boom.config.defaults import DRONE_PITCH
from boom.config.defaults import DRONE_ROLL
from boom.config.defaults import DRONE_YAW
from boom.core.events import GeoReference
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple
import re


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
