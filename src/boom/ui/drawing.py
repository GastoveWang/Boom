"""
==============================================================================
Boom 離線使用者介面 - 中文文字渲染與煙霧標註框繪製 (drawing.py)
==============================================================================

【檔案定位】
本檔案提供基於 OpenCV 與 PIL 的文字繪製輔助工具，以及煙霧事件標註框（Bounding Box）、
置信度文字標籤與起煙點十字線的畫面渲染繪製。

【核心功能】
1. 中文文字繪製 (`_draw_text_inplace`)：
   - 支援自動搜尋系統中文字型（微軟正黑體、思源黑體等），在影像陣列上以 PIL
     繪製平滑反鋸齒中文文字後轉回 numpy 陣列。
2. 事件標註疊圖 (`draw_confirmed_event_overlays`)：
   - 繪製煙霧邊界框、半透明文字標籤底色、事件編號、置信度及預估座標提示。
==============================================================================
"""
from __future__ import annotations

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from boom.config.defaults import UI_FONT_PATHS
from boom.core.events import LoggedEvent
from pathlib import Path
from typing import List
from typing import Tuple
import cv2
import numpy as np
from .event_style import event_color, event_label


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


def draw_confirmed_event_overlays(frame: np.ndarray, active_events: List[LoggedEvent], source_shape=None) -> np.ndarray:
    canvas = frame.copy()
    sx = frame.shape[1] / source_shape[1] if source_shape is not None else 1.
    sy = frame.shape[0] / source_shape[0] if source_shape is not None else 1.
    for event in active_events:
        x, y, w, h = (int(round(v*s)) for v, s in zip(event.bbox, (sx, sy, sx, sy)))
        color = event_color(event.event_id)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
        text = event_label(event.event_id)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .9, 2)
        tx = max(0, min(x, canvas.shape[1]-tw-12))
        ty = max(th+10, min(y-10, canvas.shape[0]-baseline-5))
        cv2.rectangle(canvas, (tx, ty-th-7), (tx+tw+10, ty+baseline+4), (24, 28, 36), -1)
        cv2.putText(canvas, text, (tx+5, ty), cv2.FONT_HERSHEY_SIMPLEX, .9, color, 2, cv2.LINE_AA)
        impact_x, impact_y = (int(round(v*s)) for v, s in zip(event.impact_point, (sx, sy)))
        cv2.drawMarker(
            canvas,
            (impact_x, impact_y),
            color,
            cv2.MARKER_CROSS,
            22,
            2,
            cv2.LINE_AA,
        )
    return canvas
