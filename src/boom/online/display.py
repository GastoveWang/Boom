"""Live event overlays."""
from __future__ import annotations
from typing import Any
import cv2
from boom.core.events import ConfirmedEvent

def _draw_confirmed_box(canvas: Any, event: ConfirmedEvent) -> None:
    x, y, w, h = event.bbox
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.putText(
        canvas,
        f"IMPACT CANDIDATE {event.event_id}",
        (x, max(y - 10, 26)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
