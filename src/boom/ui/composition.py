"""
==============================================================================
Boom 離線使用者介面 - 多面板畫面合成 (composition.py)
==============================================================================

【檔案定位】
本檔案為離線視覺化呈現的最後合成階段。負責將「即時影像畫面（含煙霧標註框）」、
「航拍地圖視窗（含無人機軌跡與視野錐體）」與「飛控/事件座標側邊面板」拼接為
高解析度（1920x1080）的最終輸出視訊畫布。

【核心功能】
- `compose_frame(vis, history, video_name, localizer)`:
  1. 若啟用 `localizer`（地圖模式）：
     畫布左側呈現高動態地圖面板 (`localizer.render_panel`)，右側垂直堆疊
     無人機航拍視訊預覽視窗與最新事件座標資訊卡片。
  2. 若未啟用 `localizer`（經典純偵測模式）：
     主畫面保留原始比例並覆蓋事件標籤，右側附加經典數值側邊欄 (`build_side_panel`)。
==============================================================================
"""
import cv2
import numpy as np
from boom.config.defaults import UI_MODE, DISPLAY_SCALE
from .drawing import draw_confirmed_event_overlays
from .panels import build_side_panel, get_panel_width
from .panels import build_map_event_panel


def compose_frame(vis, history, video_name, localizer=None):
    if localizer:
        height, width = 1080, 1920
        header_h = 56
        map_width = 1180
        sidebar_width = width - map_width

        canvas = np.full((height, width, 3), (36, 31, 26), np.uint8)

        # 1. Top System Header Bar
        cv2.rectangle(canvas, (0, 0), (width, header_h), (48, 42, 35), -1)
        cv2.line(canvas, (0, header_h - 1), (width, header_h - 1), (50, 64, 86), 1)

        from .drawing import _draw_text_inplace

        _draw_text_inplace(canvas, "BOOM 荒野煙霧即時監控系統  |  WILDERNESS SMOKE MONITOR", (20, 14), 22, (245, 248, 255))

        # Right side info
        _draw_text_inplace(canvas, f"來源: {video_name[:36]}", (width - 480, 17), 16, (180, 200, 225))

        # 2. Main content area below header
        content_h = height - header_h

        # Left: GeoMap panel
        canvas[header_h:, :map_width] = localizer.panel(content_h, map_width)

        # Right: Video Feed + Event Panel
        video_area_h = 420
        scale = min(sidebar_width / vis.shape[1], video_area_h / vis.shape[0])
        vw, vh = round(vis.shape[1] * scale), round(vis.shape[0] * scale)
        video = cv2.resize(vis, (vw, vh), interpolation=cv2.INTER_AREA)
        video = draw_confirmed_event_overlays(video, history.overlay_events, vis.shape)

        # Video container background
        v_top = header_h
        cv2.rectangle(canvas, (map_width, v_top), (width, v_top + video_area_h), (36, 31, 26), -1)
        left = map_width + (sidebar_width - vw) // 2
        top_offset = v_top + (video_area_h - vh) // 2
        canvas[top_offset : top_offset + vh, left : left + vw] = video

        # Label badge on video feed
        cv2.rectangle(canvas, (map_width + 12, v_top + 10), (map_width + 200, v_top + 36), (48, 42, 35), -1)
        cv2.rectangle(canvas, (map_width + 12, v_top + 10), (map_width + 200, v_top + 36), (60, 75, 98), 1)
        _draw_text_inplace(canvas, "無人機即時鏡頭畫面", (map_width + 22, v_top + 14), 16, (235, 242, 252))

        # Event panel below video feed
        event_panel_top = v_top + video_area_h
        canvas[event_panel_top:, map_width:] = build_map_event_panel(
            height - event_panel_top, sidebar_width, history.panel_events, video_name
        )

        return canvas

    panel_width = get_panel_width(UI_MODE)
    vis = draw_confirmed_event_overlays(vis, history.overlay_events)

    info_panel = build_side_panel(vis.shape[0], panel_width, history.panel_events, UI_MODE, video_name)
    combined = np.hstack((vis, info_panel))

    if DISPLAY_SCALE != 1.0:
        combined = cv2.resize(
            combined,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_AREA,
        )

    return combined
