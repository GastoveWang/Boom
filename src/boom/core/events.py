"""
==============================================================================
Boom 核心領域實體 - 煙霧事件與地理參考資料結構 (events.py)
==============================================================================

【檔案定位】
本檔案定義 Boom 系統的通用資料模型，包含偵測器輸出的確認起煙事件、
執行期解算後的地理座標記錄、以及無人機相機幾何參考。
==============================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GeoReference:
    """無人機相機先驗/備援姿態與地理位置"""
    lon: float = 121.54060381534254
    lat: float = 25.013457612687223
    alt: float = 100.0
    yaw: float = 0.0
    pitch: float = -45.0
    roll: float = 0.0


@dataclass
class ConfirmedEvent:
    """
    偵測器確認起煙之原始事件記錄。
    跨 PIDNet, Optical Flow, Fusion 與未來所有 Detector 之統一介面載體。
    """
    event_id: int
    frame_idx: int
    timestamp_sec: float
    bbox: Tuple[int, int, int, int]
    growth_ratio: float = 1.0
    radial_ratio: Optional[float] = None
    area: float = 0.0
    confirmed_by: str = "unknown"
    impact_point: Optional[Tuple[float, float]] = None
    confirm_frame_idx: Optional[int] = None
    first_seen_frame_idx: Optional[int] = None
    status: str = "new-impact-smoke"
    confidence: float = 1.0


@dataclass
class LoggedEvent:
    """經由 Localizer 解算為真實大地座標 (TWD97 / WGS84) 之完整事件記錄"""
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
