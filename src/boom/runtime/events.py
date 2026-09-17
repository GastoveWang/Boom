"""
==============================================================================
Boom 離線執行期 - 事件地理定位與生命週期管理 (events.py)
==============================================================================

【檔案定位】
本檔案負責將偵測器產出的原始煙霧事件（像素框、起煙點像素）轉換為具備地理座標
（經緯度、TWD97）與時間戳記的標準執行期記錄，並獨立管理 UI 畫面上的外框顯示期限。

【核心功能】
1. 事件地理座標解算 (`localize_event`)：
   - 支援解析度縮放對齊（光流可視化尺寸 vs 原圖尺寸）。
   - 若啟用 `MapLocalizer`，利用無人機在起煙瞬間（onset frame）的相機姿態將像素投影至地面；
     若未啟用，則使用靜態相機模型備援計算。
2. UI 顯示生命週期控制 (`EventHistory`)：
   - 分離「畫面疊圖外框保留時長 (`box_hold_sec`)」與「資訊面板記錄保留時長 (`panel_hold_sec`)」，
     確保不干擾偵測器內部的追蹤邏輯，且不遺失任何歷史證據記錄。

【核心類別與函式清單】
- `localize_event(...) -> LoggedEvent`: 將事件像素轉換為地理座標並封裝為 LoggedEvent。
- `EventHistory`: 管理所有事件歷史、當前疊圖事件與面板顯示事件。
==============================================================================
"""
import math
from boom.core.events import LoggedEvent
from boom.core.coordinates import coordinate, twd97_to_wgs84


def localize_event(event, frame_shape, visualization_shape, fps, geo, localizer=None):
    x, y, w, h = event.bbox
    impact_point = getattr(event, "impact_point", (x + w / 2.0, y + h / 2.0))
    confidence = float(getattr(event, "confidence", 1.0))
    confirmed_frame_idx = int(
        getattr(event, "confirm_frame_idx", event.frame_idx)
    )
    if localizer:
        # Optical-flow events use the resized visualization raster;
        # localization history uses the original camera raster.
        source_point = (impact_point[0]*frame_shape[1]/visualization_shape[1],
                        impact_point[1]*frame_shape[0]/visualization_shape[0])
        position = localizer.event_position(event.event_id, event.frame_idx, source_point)
        if position is None:
            lon = lat = easting = northing = float("nan")
        else:
            lon, lat = position
            # Zero-height center ray uses only the existing WGS84 -> TWD97 projection.
            easting, northing = coordinate(lon, lat, 0., 0., -90., 0., 1420., 1420.)
    else:
        easting, northing = coordinate(
            geo.lon, geo.lat, geo.alt, geo.yaw, geo.pitch, geo.roll,
            impact_point[0], impact_point[1],
            image_width=frame_shape[1], image_height=frame_shape[0],
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
        ) / fps,
        coordinate_status=("unavailable" if not math.isfinite(lon) else
                           ("map_registered_ground_estimate" if localizer.reference_valid else "gps_seeded_ground_estimate"))
        if localizer else "static_pose_estimate",
    )
    return logged


class EventHistory:
    def __init__(self, fps, panel_hold_sec, box_hold_sec):
        self.all_events = []
        self.panel_events = []
        self.overlay_events = []
        self.panel_frames = max(int(round(panel_hold_sec * fps)), 1)
        self.box_frames = max(int(round(box_hold_sec * fps)), 1)

    def append(self, event):
        self.all_events.append(event)
        self.panel_events.append(event)
        self.overlay_events.append(event)

    def expire(self, frame_idx):
        self.panel_events = [e for e in self.panel_events if frame_idx-e.confirmed_frame_idx <= self.panel_frames]
        self.overlay_events = [e for e in self.overlay_events if frame_idx-e.confirmed_frame_idx <= self.box_frames]
