"""
==============================================================================
Boom 離線使用者介面 - 座標數值側邊欄與事件卡片渲染 (panels.py)
==============================================================================

【檔案定位】
本檔案負責視覺化介面中側邊欄資訊面板的繪製，包含無人機姿態數據面板、
煙霧事件詳細資訊卡片（起煙影格、經緯度、TWD97 座標、置信度等）。

【核心功能】
1. 經典數值側邊欄 (`build_side_panel`)：
   - 在純偵測模式下，繪製垂直資訊面板，顯示航高、姿態角、影片名稱與最新事件。
2. 事件資訊卡片 (`build_map_event_panel`)：
   - 在地圖模式下，於右下側渲染現代化深色事件卡片，顯示各煙霧源的定位座標與延遲。
==============================================================================
"""
from __future__ import annotations

import math
from typing import List
import cv2
import numpy as np

from boom.config.defaults import PANEL_WIDTH
from boom.core.events import LoggedEvent
from boom.ui.drawing import _draw_text_inplace
from .theme import rounded_surface, CARD


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
        f"事件 {event.event_id:02d}  Event {event.event_id:02d}",
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

    _draw_text_inplace(panel, "TWD97 E", (label_x, row_y), label_size, (40, 52, 72))
    _draw_text_inplace(panel, f"{event.easting:.2f}", (value_x, row_y), value_size, (40, 52, 72))

    _draw_text_inplace(panel, "TWD97 N", (label_x, row_y + row_gap), label_size, (40, 52, 72))
    _draw_text_inplace(panel, f"{event.northing:.2f}", (value_x, row_y + row_gap), value_size, (40, 52, 72))

    _draw_text_inplace(panel, "WGS84 Lat", (label_x, row_y + row_gap * 2), label_size, (40, 52, 72))
    _draw_text_inplace(panel, f"{event.lat:.6f}", (value_x, row_y + row_gap * 2), value_size, (40, 52, 72))

    _draw_text_inplace(panel, "WGS84 Lon", (label_x, row_y + row_gap * 3), label_size, (40, 52, 72))
    _draw_text_inplace(panel, f"{event.lon:.6f}", (value_x, row_y + row_gap * 3), value_size, (40, 52, 72))
    return bottom + 14


def build_debug_side_panel(height: int, width: int, active_events: List[LoggedEvent]) -> np.ndarray:
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width, 92), (234, 240, 248), -1)
    _draw_text_inplace(panel, "煙霧事件與座標列表", (18, 10), 35, (28, 41, 58))
    _draw_text_inplace(panel, "Smoke Event & Coordinates", (18, 50), 22, (78, 92, 112))

    top = 112
    for event in active_events[-4:]:
        top = _draw_event_card(panel, top, width, event, compact=False)
        if top >= height - 150:
            break
    return panel


def _draw_client_header(panel: np.ndarray, video_name: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 150), (18, 34, 62), -1)
    cv2.rectangle(panel, (0, 150), (panel.shape[1], 158), (108, 176, 255), -1)
    _draw_text_inplace(panel, "煙霧事件偵測", (18, 12), 36, (247, 250, 255))
    _draw_text_inplace(panel, "Wilderness Smoke Detection", (20, 58), 23, (214, 228, 250))
    _draw_text_inplace(panel, video_name, (20, 102), 18, (214, 228, 250))


def _draw_empty_client_card(panel: np.ndarray, top: int, width: int) -> None:
    left = 20
    right = width - 20
    bottom = min(top + 108, panel.shape[0] - 20)
    cv2.rectangle(panel, (left, top), (right, bottom), (244, 247, 252), -1)
    cv2.rectangle(panel, (left, top), (left + 8, bottom), (83, 148, 255), -1)
    cv2.rectangle(panel, (left, top), (right, bottom), (208, 217, 230), 1)
    _draw_text_inplace(panel, "尚無事件", (left + 18, top + 14), 28, (34, 46, 64))
    _draw_text_inplace(panel, "No Event", (left + 20, top + 58), 20, (88, 100, 119))


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


def build_map_event_panel(height: int, width: int, active_events, video_name: str) -> np.ndarray:
    """Render sleek commercial event side panel at final screen resolution."""
    from .event_style import event_color, event_label

    panel = np.full((height, width, 3), (36, 31, 26), np.uint8)

    # Panel header
    cv2.rectangle(panel, (0, 0), (width, 50), (48, 42, 35), -1)
    cv2.line(panel, (0, 50), (width, 50), (52, 64, 84), 1)

    _draw_text_inplace(panel, "即時煙霧事件紀錄 | SMOKE EVENTS", (18, 12), 22, (245, 248, 252))
    cv2.rectangle(panel, (width - 150, 12), (width - 16, 38), (65, 56, 46), -1)
    cv2.rectangle(panel, (width - 150, 12), (width - 16, 38), (70, 85, 110), 1)
    _draw_text_inplace(panel, f"總計: {len(active_events)} 筆", (width - 138, 16), 16, (220, 235, 255))

    card_height = 150
    card_gap = 14
    top_offset = 64
    capacity = max(0, (height - top_offset - 30) // (card_height + card_gap))

    events = list(reversed(active_events))[:capacity]

    if not events:
        empty_box = panel[top_offset : top_offset + 120, 16 : width - 16]
        rounded_surface(panel, (16, top_offset, width-32, 120), CARD)
        _draw_text_inplace(panel, "● 目前沒有顯示中的煙霧事件", (36, top_offset + 32), 22, (220, 212, 197))
        _draw_text_inplace(panel, "事件出現後，將顯示對應編號與座標", (36, top_offset + 68), 17, (175, 161, 144))
    else:
        for index, event in enumerate(events):
            top = top_offset + index * (card_height + card_gap)
            card = panel[top : top + card_height, 16 : width - 16]
            rounded_surface(panel, (16, top, width-32, card_height), CARD)

            color = event_color(event.event_id)

            # Left color indicator bar
            card[12:-12, :5] = color

            # Card border

            # Header: Event ID & Confidence & Timestamp
            _draw_text_inplace(card, f"{event_label(event.event_id)}  疑似煙霧源", (22, 10), 24, color)

            _draw_text_inplace(
                card,
                f"對應時間: {event.timestamp_sec:.1f}s",
                (width - 340, 14),
                18,
                (225, 235, 248),
            )

            # Coordinates
            if math.isfinite(event.lat) and math.isfinite(event.lon):
                _draw_text_inplace(card, f"WGS84   {event.lat:.6f}°, {event.lon:.6f}°", (22, 48), 21, (242, 246, 252))
                _draw_text_inplace(card, f"TWD97   E {event.easting:.2f} m,  N {event.northing:.2f} m", (22, 80), 19, (200, 218, 238))
            else:
                _draw_text_inplace(card, "WGS84   —\nTWD97   —", (22, 48), 21, (200, 218, 238))


    remaining = len(active_events) - len(events)
    if remaining > 0:
        _draw_text_inplace(panel, f"另有 {remaining} 筆歷史紀錄已儲存", (20, height - 22), 16, (160, 180, 205))

    return panel
