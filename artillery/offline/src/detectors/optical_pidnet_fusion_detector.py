from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from artillery.offline.src.detectors.optical_flow_smoke_detector import (
    DetectorConfig,
    InstantSmokeDustDetector,
    OpticalCandidateEvent,
)
from artillery.offline.src.detectors.pidnet_smoke_detector import (
    ConfirmedEvent as PIDNetConfirmedEvent,
    PIDNetSmokeImpactDetector,
    PIDNetSmokeTrackerConfig,
)


BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]


@dataclass
class OpticalPIDNetFusionConfig:
    confirmation_window_sec: float = 1.0
    pretrigger_tolerance_sec: float = 0.15
    match_distance_px: float = 220.0
    min_bbox_iou: float = 0.01
    draw_pending_candidates: bool = True


@dataclass
class FusionConfirmedEvent:
    event_id: int
    frame_idx: int
    timestamp_sec: float
    bbox: BBox
    area: float
    growth_ratio: float
    radial_ratio: Optional[float]
    confidence: float
    impact_point: Point
    confirm_frame_idx: int
    first_seen_frame_idx: int
    status: str = "optical-pidnet-confirmed-impact"


@dataclass
class _PendingOpticalCandidate:
    event: OpticalCandidateEvent
    bbox: BBox
    impact_point: Point
    area: float


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(right - left, 0) * max(bottom - top, 0)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


class OpticalPIDNetFusionDetector:
    """Optical onset proposal followed by independent PIDNet confirmation."""

    def __init__(
        self,
        optical_config: DetectorConfig,
        pidnet_config: PIDNetSmokeTrackerConfig,
        fps: float,
        fusion_config: Optional[OpticalPIDNetFusionConfig] = None,
        *,
        optical_detector: Optional[InstantSmokeDustDetector] = None,
        pidnet_detector: Optional[PIDNetSmokeImpactDetector] = None,
    ) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.config = fusion_config or OpticalPIDNetFusionConfig()
        self.optical_detector = (
            optical_detector
            if optical_detector is not None
            else InstantSmokeDustDetector(optical_config, self.fps)
        )
        self.pidnet_detector = (
            pidnet_detector
            if pidnet_detector is not None
            else PIDNetSmokeImpactDetector(pidnet_config, self.fps)
        )
        self.window_frames = max(
            int(round(self.config.confirmation_window_sec * self.fps)), 1
        )
        self.pretrigger_tolerance_frames = max(
            int(round(self.config.pretrigger_tolerance_sec * self.fps)), 0
        )
        self.pending_optical: List[_PendingOpticalCandidate] = []
        self.pending_pidnet: List[PIDNetConfirmedEvent] = []
        self.pending_confirmations: List[FusionConfirmedEvent] = []
        self.next_event_id = 1

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        self.optical_detector.process_frame(frame_bgr, frame_idx)
        visualization = self.pidnet_detector.process_frame(frame_bgr, frame_idx)

        for candidate in self.optical_detector.consume_pending_candidates():
            bbox = self._optical_bbox_to_source(candidate.bbox, frame_bgr.shape)
            x, y, width, height = bbox
            self.pending_optical.append(
                _PendingOpticalCandidate(
                    event=candidate,
                    bbox=bbox,
                    impact_point=(x + width / 2.0, float(y + height - 1)),
                    area=candidate.area
                    / max(float(self.optical_detector.config.downscale) ** 2, 1e-6),
                )
            )
        self.pending_pidnet.extend(
            self.pidnet_detector.consume_pending_confirmations()
        )

        self._match_confirmations()
        self._expire_pending(frame_idx)
        if self.config.draw_pending_candidates:
            self._draw_pending(visualization)
        return visualization

    def consume_pending_confirmations(self) -> List[FusionConfirmedEvent]:
        events = self.pending_confirmations
        self.pending_confirmations = []
        return events

    def camera_guard_summary(self) -> Dict[str, int]:
        return {
            f"optical_{reason}": count
            for reason, count in self.optical_detector.camera_guard_summary().items()
        }

    def _match_confirmations(self) -> None:
        pairings: List[Tuple[float, int, int]] = []
        for optical_idx, optical in enumerate(self.pending_optical):
            for pidnet_idx, pidnet in enumerate(self.pending_pidnet):
                confirm_frame = int(
                    getattr(pidnet, "confirm_frame_idx", pidnet.frame_idx)
                )
                delay = confirm_frame - optical.event.frame_idx
                if not -self.pretrigger_tolerance_frames <= delay <= self.window_frames:
                    continue

                pidnet_point = getattr(pidnet, "impact_point", None)
                if pidnet_point is None:
                    x, y, width, height = pidnet.bbox
                    pidnet_point = (x + width / 2.0, float(y + height - 1))
                distance = math.hypot(
                    optical.impact_point[0] - pidnet_point[0],
                    optical.impact_point[1] - pidnet_point[1],
                )
                iou = _bbox_iou(optical.bbox, pidnet.bbox)
                adaptive_distance = max(
                    float(self.config.match_distance_px),
                    0.5 * math.hypot(pidnet.bbox[2], pidnet.bbox[3]),
                )
                if iou < self.config.min_bbox_iou and distance > adaptive_distance:
                    continue
                temporal_score = abs(delay) / max(self.window_frames, 1)
                spatial_score = distance / max(adaptive_distance, 1.0)
                score = temporal_score + spatial_score + (1.0 - iou)
                pairings.append((score, optical_idx, pidnet_idx))

        used_optical = set()
        used_pidnet = set()
        for _, optical_idx, pidnet_idx in sorted(pairings):
            if optical_idx in used_optical or pidnet_idx in used_pidnet:
                continue
            self._confirm_pair(
                self.pending_optical[optical_idx], self.pending_pidnet[pidnet_idx]
            )
            used_optical.add(optical_idx)
            used_pidnet.add(pidnet_idx)

        self.pending_optical = [
            item
            for idx, item in enumerate(self.pending_optical)
            if idx not in used_optical
        ]
        self.pending_pidnet = [
            item
            for idx, item in enumerate(self.pending_pidnet)
            if idx not in used_pidnet
        ]

    def _confirm_pair(
        self,
        optical: _PendingOpticalCandidate,
        pidnet: PIDNetConfirmedEvent,
    ) -> None:
        pidnet_confidence = float(getattr(pidnet, "confidence", 0.75))
        snr_score = min(max(optical.event.signal_to_noise / 5.0, 0.0), 1.0)
        polarity_score = min(
            max(optical.event.residual_polarity_ratio, 0.0), 1.0
        )
        optical_confidence = 0.6 * snr_score + 0.4 * polarity_score
        confidence = 0.75 * pidnet_confidence + 0.25 * optical_confidence
        confirm_frame = int(getattr(pidnet, "confirm_frame_idx", pidnet.frame_idx))

        event = FusionConfirmedEvent(
            event_id=self.next_event_id,
            frame_idx=optical.event.frame_idx,
            timestamp_sec=optical.event.frame_idx / self.fps,
            bbox=optical.bbox,
            area=optical.area,
            growth_ratio=float(getattr(pidnet, "growth_ratio", 1.0)),
            radial_ratio=None,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            impact_point=optical.impact_point,
            confirm_frame_idx=confirm_frame,
            first_seen_frame_idx=optical.event.frame_idx,
        )
        self.next_event_id += 1
        self.pending_confirmations.append(event)
        print(
            f"[INFO] Fused impact {event.event_id}: "
            f"optical_frame={event.frame_idx}, pidnet_frame={confirm_frame}, "
            f"confidence={event.confidence:.3f}, "
            f"impact_px=({event.impact_point[0]:.1f},{event.impact_point[1]:.1f})"
        )

    def _expire_pending(self, frame_idx: int) -> None:
        self.pending_optical = [
            item
            for item in self.pending_optical
            if frame_idx - item.event.frame_idx <= self.window_frames
        ]
        self.pending_pidnet = [
            item
            for item in self.pending_pidnet
            if frame_idx
            - int(getattr(item, "confirm_frame_idx", item.frame_idx))
            <= self.window_frames
        ]

    def _optical_bbox_to_source(
        self, bbox: BBox, frame_shape: Tuple[int, ...]
    ) -> BBox:
        scale = max(float(self.optical_detector.config.downscale), 1e-6)
        height, width = frame_shape[:2]
        x, y, box_width, box_height = bbox
        left = int(np.clip(round(x / scale), 0, width - 1))
        top = int(np.clip(round(y / scale), 0, height - 1))
        right = int(np.clip(round((x + box_width) / scale), left + 1, width))
        bottom = int(np.clip(round((y + box_height) / scale), top + 1, height))
        return left, top, right - left, bottom - top

    def _draw_pending(self, visualization: np.ndarray) -> None:
        for candidate in self.pending_optical:
            x, y, width, height = candidate.bbox
            cv2.rectangle(
                visualization,
                (x, y),
                (x + width, y + height),
                (0, 220, 255),
                2,
            )
            cv2.putText(
                visualization,
                "OF candidate - waiting PIDNet",
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
