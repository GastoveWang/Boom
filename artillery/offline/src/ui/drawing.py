"""Shared text and offline event overlays."""
from __future__ import annotations

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from artillery.common.defaults import UI_FONT_PATHS
from artillery.common.events import LoggedEvent
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
