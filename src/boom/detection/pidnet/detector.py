"""
==============================================================================
Boom 煙霧偵測演算法 - PIDNet-S 輕量化語意分割偵測器 (pidnet_smoke_detector.py)
==============================================================================

【檔案定位】
本檔案為 Boom 系統的核心深度學習煙霧偵測演算法實作。基於 PIDNet-S 輕量化
雙邊架構，專門針對無人機空拍環境下的荒野煙霧進行逐像素語意分割、連通域分析、
新煙霧軌跡追蹤與起煙點鎖定。

【核心功能】
1. 輕量化模型推論 (`PIDNetSmokeImpactDetector`)：
   - 支援 CUDA GPU / CPU 自動切換，利用 PIDNet-S 架構對輸入航拍影像進行即時推論。
   - 雙重機率門檻：弱煙霧早期觸發門檻 (`early_candidate_threshold`) 保留初期起煙時間，
     正式確認門檻 (`seg_threshold`) 確保低誤報率。
2. 煙霧生命週期追蹤 (`SmokeCandidateTracker`)：
   - 時間滑動視窗分析面積膨脹比、徑向擴散特徵，過濾靜態雲霧、高光反光與地形干擾。
   - 估算疑似煙霧源的起煙點（impact_point），並發布確認事件 (`ConfirmedEvent`)。

【核心類別清單】
- `PIDNetSmokeTrackerConfig`: 偵測器與追蹤門檻超參數設定 dataclass。
- `ConfirmedEvent`: 統一的確認事件資料載體。
- `PIDNetSmokeImpactDetector`: 實作 `process_frame` 與 `consume_pending_confirmations` 統一介面。
==============================================================================
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]
ProbabilityProvider = Callable[[np.ndarray], np.ndarray]


@dataclass
class PIDNetSmokeTrackerConfig:
    """Field-tuning parameters for semantic smoke detection and lifecycle tracking."""

    model_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "model"
        / "sam_sup_pidnet_s.pt"
    )
    device: str = "auto"
    input_width: int = 960
    input_height: int = 544
    inference_stride: int = 1
    segmentation_threshold: float = 0.80
    min_component_area: int = 30
    morphology_kernel: int = 5
    overlay_alpha: float = 0.24

    # A lower PIDNet threshold keeps the first faint smoke pixels as an onset
    # candidate. It never confirms an event by itself.
    early_candidate_threshold: float = 0.45
    early_candidate_min_area: int = 10

    # Anything visible while the detector warms up is treated as pre-existing.
    warmup_sec: float = 0.50
    # A candidate must be confirmed inside this interval. Older smoke is silent.
    new_smoke_window_sec: float = 1.00
    confirmation_hits: int = 2
    min_confirmation_confidence: float = 0.72
    min_growth_ratio: float = 1.03
    min_new_pixel_ratio: float = 0.55

    track_match_distance: float = 160.0
    track_min_iou: float = 0.03
    max_missed_sec: float = 0.75
    post_event_memory_sec: float = 600.0
    known_smoke_dilate_px: int = 9

    # Lightweight global motion compensation keeps old-smoke memory aligned
    # while the drone translates, rotates or changes scale slightly.
    motion_compensation: bool = True
    motion_estimation_width: int = 640
    motion_max_corners: int = 500
    motion_min_inliers: int = 15
    motion_min_inlier_ratio: float = 0.40
    motion_max_rotation_deg: float = 10.0
    motion_min_scale: float = 0.85
    motion_max_scale: float = 1.15

    draw_mask_overlay: bool = True
    show_status_overlay: bool = False
    show_candidate_boxes: bool = False


@dataclass
class ConfirmedEvent:
    """One newly appeared smoke event, timestamped at its first observation."""

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
    status: str = "new-impact-smoke"


@dataclass
class _Component:
    bbox: BBox
    centroid: Point
    area: int
    mean_probability: float
    new_pixel_ratio: float
    ys: np.ndarray
    xs: np.ndarray

    @property
    def impact_point(self) -> Point:
        # The lower centre of the first smoke mask is the best image-only proxy
        # for the ground contact point. It is intentionally frozen at first sight.
        x, y, w, h = self.bbox
        return x + w / 2.0, float(y + h - 1)


@dataclass
class _SmokeTrack:
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    initial_bbox: BBox
    bbox: BBox
    initial_centroid: Point
    centroid: Point
    initial_impact_point: Point
    initial_area: float
    area: float
    max_area: float
    peak_probability: float
    peak_new_pixel_ratio: float
    preexisting: bool
    observations: int = 1
    missed_frames: int = 0
    event_id: Optional[int] = None
    expired_new_window: bool = False

    @property
    def confirmed(self) -> bool:
        return self.event_id is not None


@dataclass
class _EarlyTrack:
    """Low-confidence PIDNet observations retained until semantic confirmation."""

    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    initial_bbox: BBox
    bbox: BBox
    initial_centroid: Point
    centroid: Point
    initial_impact_point: Point
    initial_area: float
    max_area: float
    peak_probability: float


@dataclass
class _TrackMemory:
    bbox: BBox
    centroid: Point
    expires_frame: int


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


from boom.interfaces.detector import BaseSmokeDetector

class PIDNetSmokeImpactDetector(BaseSmokeDetector):
    """PIDNet smoke segmentation plus tracking focused on impact onset.

    The class deliberately keeps the same two-method runtime API as the legacy
    detector: ``process_frame`` and ``consume_pending_confirmations``. Torch and
    the research model are loaded lazily so tracking can be unit-tested without
    a GPU or even a torch installation.
    """

    def __init__(
        self,
        config: PIDNetSmokeTrackerConfig,
        fps: float,
        *,
        probability_provider: Optional[ProbabilityProvider] = None,
    ) -> None:
        self.config = config
        self.fps = fps if fps > 0 else 30.0
        self.warmup_frames = max(int(round(config.warmup_sec * self.fps)), 0)
        self.new_window_frames = max(
            int(round(config.new_smoke_window_sec * self.fps)), 1
        )
        self.max_missed_frames = max(
            int(round(config.max_missed_sec * self.fps)), 1)
        self.memory_frames = max(
            int(round(config.post_event_memory_sec * self.fps)), 1)
        self.probability_provider = probability_provider
        self.pending_confirmations: List[ConfirmedEvent] = []
        self.tracks: Dict[int, _SmokeTrack] = {}
        self.early_tracks: Dict[int, _EarlyTrack] = {}
        self.memories: List[_TrackMemory] = []
        self.next_track_id = 1
        self.next_early_track_id = 1
        self.next_event_id = 1
        self.known_smoke_mask: Optional[np.ndarray] = None
        self.last_probability: Optional[np.ndarray] = None
        self.last_binary_mask: Optional[np.ndarray] = None
        self.last_inference_ms = 0.0
        self._model = None
        self._torch = None
        self._device = None
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._start_frame_idx: Optional[int] = None
        self._previous_motion_gray: Optional[np.ndarray] = None

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr must be a non-empty BGR image")
        height, width = frame_bgr.shape[:2]
        if self._frame_shape != (height, width):
            self._reset_for_shape(height, width)
            self._start_frame_idx = frame_idx
        elif self._start_frame_idx is None:
            self._start_frame_idx = frame_idx

        transform = self._estimate_camera_motion(frame_bgr)
        if transform is not None:
            self._warp_tracking_state(transform, width, height)

        stride = max(int(self.config.inference_stride), 1)
        should_infer = self.last_probability is None or frame_idx % stride == 0
        if should_infer:
            started = time.perf_counter()
            probability = self._predict_probability(frame_bgr)
            self.last_inference_ms = (time.perf_counter() - started) * 1000.0
            probability = np.asarray(probability, dtype=np.float32)
            if probability.shape != (height, width):
                probability = cv2.resize(
                    probability, (width,
                                  height), interpolation=cv2.INTER_LINEAR
                )
            probability = np.clip(probability, 0.0, 1.0)
            binary = probability >= float(self.config.segmentation_threshold)
            binary = self._clean_mask(binary)
            early_threshold = min(
                float(self.config.early_candidate_threshold),
                float(self.config.segmentation_threshold),
            )
            early_binary = probability >= early_threshold
            early_binary = self._clean_mask(early_binary)
            self.last_probability = probability
            self.last_binary_mask = binary
            early_components = self._components(
                probability,
                early_binary,
                min_area=self.config.early_candidate_min_area,
            )
            self._update_early_tracks(early_components, frame_idx)
            self._update_tracks(probability, binary, frame_idx)
        else:
            self._age_tracks(frame_idx)

        return self._draw_visualization(frame_bgr, frame_idx)

    def consume_pending_confirmations(self) -> List[ConfirmedEvent]:
        events = self.pending_confirmations
        self.pending_confirmations = []
        return events

    def camera_guard_summary(self) -> Dict[str, int]:
        return {}

    def _reset_for_shape(self, height: int, width: int) -> None:
        self._frame_shape = (height, width)
        self.known_smoke_mask = np.zeros((height, width), dtype=np.uint8)
        self.last_probability = None
        self.last_binary_mask = None
        self.tracks.clear()
        self.early_tracks.clear()
        self.memories.clear()
        self._previous_motion_gray = None

    def _predict_probability(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.probability_provider is not None:
            return self.probability_provider(frame_bgr)
        if self._model is None:
            self._load_model()

        torch = self._torch
        assert torch is not None and self._model is not None and self._device is not None
        resized = cv2.resize(
            frame_bgr,
            (int(self.config.input_width), int(self.config.input_height)),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        tensor = tensor.unsqueeze(0).to(
            self._device, non_blocking=True).float().div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=self._device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=self._device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        with torch.inference_mode():
            output = self._model(tensor)
            if isinstance(output, (list, tuple)):
                output = output[1] if len(output) > 1 else output[0]
            probability = torch.sigmoid(output)
            probability = torch.nn.functional.interpolate(
                probability,
                size=frame_bgr.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        return probability.detach().float().cpu().numpy()

    def _load_model(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PIDNet needs PyTorch. Run this pipeline with the wildfire_seg "
                "environment or install artillery/offline/requirements.txt."
            ) from exc

        model_path = Path(self.config.model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"PIDNet checkpoint does not exist: {model_path}")

        from boom.detection.pidnet.model import get_pred_model

        model = get_pred_model(name="pidnet_s", num_classes=1)

        try:
            checkpoint = torch.load(
                model_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f"Unsupported PIDNet checkpoint format: {type(checkpoint)!r}")

        expected = model.state_dict()
        weights = {}
        for key, value in checkpoint.items():
            normalized = key
            for prefix in ("module.model.", "model.", "module."):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    break
            if normalized in expected and expected[normalized].shape == value.shape:
                weights[normalized] = value
        if len(weights) < int(len(expected) * 0.90):
            raise RuntimeError(
                f"Checkpoint/model mismatch: loaded {len(weights)} of {len(expected)} tensors"
            )
        model.load_state_dict(weights, strict=False)

        requested = self.config.device.lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but CUDA is unavailable")
        device = torch.device(requested)
        model.eval().to(device)
        self._torch = torch
        self._device = device
        self._model = model
        print(
            f"[INFO] PIDNet-S loaded: checkpoint={model_path}, device={device}, "
            f"input={self.config.input_width}x{self.config.input_height}"
        )

    def _clean_mask(self, binary: np.ndarray) -> np.ndarray:
        mask = binary.astype(np.uint8) * 255
        size = max(int(self.config.morphology_kernel), 1)
        if size % 2 == 0:
            size += 1
        if size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask > 0

    def _estimate_camera_motion(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self.config.motion_compensation:
            return None
        height, width = frame_bgr.shape[:2]
        target_width = min(
            max(int(self.config.motion_estimation_width), 64), width)
        target_height = max(int(round(height * target_width / width)), 48)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if (target_width, target_height) != (width, height):
            gray = cv2.resize(gray, (target_width, target_height),
                              interpolation=cv2.INTER_AREA)
        previous = self._previous_motion_gray
        self._previous_motion_gray = gray
        if previous is None or previous.shape != gray.shape:
            return None

        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=max(int(self.config.motion_max_corners), 20),
            qualityLevel=0.01,
            minDistance=8.0,
            blockSize=7,
        )
        if points is None or len(points) < self.config.motion_min_inliers:
            return None
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if current_points is None or status is None:
            return None
        valid = status.reshape(-1) == 1
        source = points.reshape(-1, 2)[valid]
        target = current_points.reshape(-1, 2)[valid]
        if len(source) < self.config.motion_min_inliers:
            return None
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
            confidence=0.99,
        )
        if affine is None or inlier_mask is None:
            return None
        inliers = int(np.count_nonzero(inlier_mask))
        if inliers < self.config.motion_min_inliers:
            return None
        if inliers / max(len(source), 1) < self.config.motion_min_inlier_ratio:
            return None

        scale = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
        rotation = math.degrees(math.atan2(
            float(affine[1, 0]), float(affine[0, 0])))
        if not self.config.motion_min_scale <= scale <= self.config.motion_max_scale:
            return None
        if abs(rotation) > self.config.motion_max_rotation_deg:
            return None

        scale_x = target_width / width
        scale_y = target_height / height
        small_to_full = np.array(
            [[1.0 / scale_x, 0.0, 0.0], [0.0, 1.0 / scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        full_to_small = np.array(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        affine_3x3 = np.vstack((affine, [0.0, 0.0, 1.0]))
        return (small_to_full @ affine_3x3 @ full_to_small)[:2].astype(np.float32)

    def _warp_tracking_state(
        self, transform: np.ndarray, width: int, height: int
    ) -> None:
        if self.known_smoke_mask is not None:
            self.known_smoke_mask = cv2.warpAffine(
                self.known_smoke_mask,
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        if self.last_binary_mask is not None:
            warped = cv2.warpAffine(
                self.last_binary_mask.astype(np.uint8),
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            self.last_binary_mask = warped > 0
        if self.last_probability is not None:
            self.last_probability = cv2.warpAffine(
                self.last_probability,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        for track in self.tracks.values():
            track.centroid = self._transform_point(track.centroid, transform)
            track.bbox = self._transform_bbox(
                track.bbox, transform, width, height)
        for track in self.early_tracks.values():
            track.centroid = self._transform_point(track.centroid, transform)
            track.bbox = self._transform_bbox(
                track.bbox, transform, width, height)
        for memory in self.memories:
            memory.centroid = self._transform_point(memory.centroid, transform)
            memory.bbox = self._transform_bbox(
                memory.bbox, transform, width, height)

    @staticmethod
    def _transform_point(point: Point, transform: np.ndarray) -> Point:
        x, y = point
        return (
            float(transform[0, 0] * x + transform[0, 1] * y + transform[0, 2]),
            float(transform[1, 0] * x + transform[1, 1] * y + transform[1, 2]),
        )

    @classmethod
    def _transform_bbox(
        cls, bbox: BBox, transform: np.ndarray, width: int, height: int
    ) -> BBox:
        x, y, w, h = bbox
        corners = [
            cls._transform_point((x, y), transform),
            cls._transform_point((x + w, y), transform),
            cls._transform_point((x, y + h), transform),
            cls._transform_point((x + w, y + h), transform),
        ]
        left = int(np.clip(math.floor(min(p[0]
                   for p in corners)), 0, width - 1))
        top = int(np.clip(math.floor(min(p[1]
                  for p in corners)), 0, height - 1))
        right = int(np.clip(math.ceil(max(p[0]
                    for p in corners)), left + 1, width))
        bottom = int(
            np.clip(math.ceil(max(p[1] for p in corners)), top + 1, height))
        return left, top, right - left, bottom - top

    def _components(
        self,
        probability: np.ndarray,
        binary: np.ndarray,
        *,
        min_area: Optional[int] = None,
    ) -> List[_Component]:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        components: List[_Component] = []
        known = self.known_smoke_mask
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            required_area = (
                self.config.min_component_area if min_area is None else min_area
            )
            if area < max(int(required_area), 1):
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            local_labels = labels[y: y + h, x: x + w]
            local_ys, local_xs = np.where(local_labels == label)
            ys = local_ys + y
            xs = local_xs + x
            mean_probability = float(probability[ys, xs].mean())
            if known is None:
                new_ratio = 1.0
            else:
                new_ratio = float(np.count_nonzero(
                    known[ys, xs] == 0) / max(area, 1))
            components.append(
                _Component(
                    bbox=(x, y, w, h),
                    centroid=(float(centroids[label, 0]),
                              float(centroids[label, 1])),
                    area=area,
                    mean_probability=mean_probability,
                    new_pixel_ratio=new_ratio,
                    ys=ys,
                    xs=xs,
                )
            )
        return components

    def _update_early_tracks(
        self, components: List[_Component], frame_idx: int
    ) -> None:
        """Track faint PIDNet regions without allowing them to confirm events."""
        pairings: List[Tuple[float, int, int]] = []
        for track_id, track in self.early_tracks.items():
            for component_idx, component in enumerate(components):
                iou = _bbox_iou(track.bbox, component.bbox)
                dist = _distance(track.centroid, component.centroid)
                adaptive_distance = max(
                    self.config.track_match_distance,
                    0.6
                    * max(
                        math.hypot(track.bbox[2], track.bbox[3]),
                        math.hypot(component.bbox[2], component.bbox[3]),
                    ),
                )
                if iou >= self.config.track_min_iou or dist <= adaptive_distance:
                    score = (1.0 - iou) + dist / max(adaptive_distance, 1.0)
                    pairings.append((score, track_id, component_idx))

        pairings.sort()
        used_tracks = set()
        used_components = set()
        for _, track_id, component_idx in pairings:
            if track_id in used_tracks or component_idx in used_components:
                continue
            track = self.early_tracks[track_id]
            component = components[component_idx]
            track.last_seen_frame = frame_idx
            track.bbox = component.bbox
            track.centroid = component.centroid
            track.max_area = max(track.max_area, float(component.area))
            track.peak_probability = max(
                track.peak_probability, component.mean_probability
            )
            used_tracks.add(track_id)
            used_components.add(component_idx)

        for component_idx, component in enumerate(components):
            if component_idx in used_components:
                continue
            track_id = self.next_early_track_id
            self.next_early_track_id += 1
            self.early_tracks[track_id] = _EarlyTrack(
                track_id=track_id,
                first_seen_frame=frame_idx,
                last_seen_frame=frame_idx,
                initial_bbox=component.bbox,
                bbox=component.bbox,
                initial_centroid=component.centroid,
                centroid=component.centroid,
                initial_impact_point=component.impact_point,
                initial_area=float(component.area),
                max_area=float(component.area),
                peak_probability=component.mean_probability,
            )

        stale_ids = [
            track_id
            for track_id, track in self.early_tracks.items()
            if frame_idx - track.last_seen_frame > self.new_window_frames
        ]
        for track_id in stale_ids:
            del self.early_tracks[track_id]

    def _matching_early_track(
        self, component: _Component
    ) -> Optional[_EarlyTrack]:
        matches: List[Tuple[float, int, _EarlyTrack]] = []
        for track in self.early_tracks.values():
            iou = _bbox_iou(track.bbox, component.bbox)
            dist = _distance(track.centroid, component.centroid)
            adaptive_distance = max(
                self.config.track_match_distance,
                0.6
                * max(
                    math.hypot(track.bbox[2], track.bbox[3]),
                    math.hypot(component.bbox[2], component.bbox[3]),
                ),
            )
            if iou >= self.config.track_min_iou or dist <= adaptive_distance:
                score = (1.0 - iou) + dist / max(adaptive_distance, 1.0)
                matches.append((score, track.track_id, track))
        if not matches:
            return None
        return min(matches, key=lambda item: (item[0], item[1]))[2]

    def _update_tracks(
        self, probability: np.ndarray, binary: np.ndarray, frame_idx: int
    ) -> None:
        self._prune_memories(frame_idx)
        components = self._components(probability, binary)
        matches, unmatched_tracks, unmatched_components = self._associate(
            components)

        for track_id, component_idx in matches:
            self._update_track(
                self.tracks[track_id], components[component_idx], frame_idx)
        for track_id in unmatched_tracks:
            self.tracks[track_id].missed_frames += 1
        for component_idx in unmatched_components:
            component = components[component_idx]
            start_frame = self._start_frame_idx if self._start_frame_idx is not None else frame_idx
            in_warmup = frame_idx - start_frame < self.warmup_frames
            preexisting = in_warmup or self._matches_memory(component)
            self._create_track(component, frame_idx, preexisting=preexisting)

        self._age_tracks(frame_idx)

    def _associate(
        self, components: List[_Component]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        pairings: List[Tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for component_idx, component in enumerate(components):
                iou = _bbox_iou(track.bbox, component.bbox)
                dist = _distance(track.centroid, component.centroid)
                tx, ty, tw, th = track.bbox
                cx, cy, cw, ch = component.bbox
                adaptive_distance = max(
                    self.config.track_match_distance,
                    0.6 * max(math.hypot(tw, th), math.hypot(cw, ch)),
                )
                if iou >= self.config.track_min_iou or dist <= adaptive_distance:
                    score = (1.0 - iou) + dist / max(adaptive_distance, 1.0)
                    pairings.append((score, track_id, component_idx))
        pairings.sort()
        used_tracks = set()
        used_components = set()
        matches: List[Tuple[int, int]] = []
        for _, track_id, component_idx in pairings:
            if track_id in used_tracks or component_idx in used_components:
                continue
            used_tracks.add(track_id)
            used_components.add(component_idx)
            matches.append((track_id, component_idx))
        return (
            matches,
            [track_id for track_id in self.tracks if track_id not in used_tracks],
            [idx for idx in range(len(components))
             if idx not in used_components],
        )

    def _create_track(
        self, component: _Component, frame_idx: int, *, preexisting: bool
    ) -> None:
        early_track = self._matching_early_track(component)
        first_seen_frame = frame_idx
        initial_bbox = component.bbox
        initial_centroid = component.centroid
        initial_impact_point = component.impact_point
        initial_area = float(component.area)
        expired_new_window = False
        if early_track is not None:
            first_seen_frame = early_track.first_seen_frame
            initial_bbox = early_track.initial_bbox
            initial_centroid = early_track.initial_centroid
            initial_impact_point = early_track.initial_impact_point
            initial_area = early_track.initial_area
            candidate_age = frame_idx - first_seen_frame
            expired_new_window = candidate_age > self.new_window_frames
            start_frame = (
                self._start_frame_idx
                if self._start_frame_idx is not None
                else first_seen_frame
            )
            preexisting = preexisting or (
                first_seen_frame - start_frame < self.warmup_frames
            ) or expired_new_window

        track = _SmokeTrack(
            track_id=self.next_track_id,
            first_seen_frame=first_seen_frame,
            last_seen_frame=frame_idx,
            initial_bbox=initial_bbox,
            bbox=component.bbox,
            initial_centroid=initial_centroid,
            centroid=component.centroid,
            initial_impact_point=initial_impact_point,
            initial_area=initial_area,
            area=float(component.area),
            max_area=float(component.area),
            peak_probability=component.mean_probability,
            peak_new_pixel_ratio=component.new_pixel_ratio,
            preexisting=preexisting,
            expired_new_window=expired_new_window,
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1
        if preexisting:
            self._remember_pixels(component)

    def _update_track(
        self, track: _SmokeTrack, component: _Component, frame_idx: int
    ) -> None:
        track.last_seen_frame = frame_idx
        track.bbox = component.bbox
        track.centroid = component.centroid
        track.area = float(component.area)
        track.max_area = max(track.max_area, track.area)
        track.peak_probability = max(
            track.peak_probability, component.mean_probability)
        track.peak_new_pixel_ratio = max(
            track.peak_new_pixel_ratio, component.new_pixel_ratio
        )
        track.observations += 1
        track.missed_frames = 0

        age_frames = frame_idx - track.first_seen_frame
        if age_frames > self.new_window_frames:
            track.expired_new_window = True
        if track.preexisting or track.confirmed or track.expired_new_window:
            self._remember_pixels(component)
            return

        growth = track.max_area / max(track.initial_area, 1.0)
        persistence_score = min(track.observations /
                                max(self.config.confirmation_hits + 1, 1), 1.0)
        growth_score = min(max(growth - 1.0, 0.0) / 0.30, 1.0)
        confidence = float(
            0.60 * track.peak_probability
            + 0.25 * persistence_score
            + 0.15 * growth_score
        )
        is_new = track.peak_new_pixel_ratio >= self.config.min_new_pixel_ratio
        enough_hits = track.observations >= max(
            self.config.confirmation_hits, 1)
        has_growth = growth >= self.config.min_growth_ratio
        high_semantic_confidence = track.peak_probability >= min(
            self.config.segmentation_threshold + 0.08, 0.98
        )
        if (
            age_frames <= self.new_window_frames
            and is_new
            and enough_hits
            and (has_growth or high_semantic_confidence)
            and confidence >= self.config.min_confirmation_confidence
        ):
            event_id = self.next_event_id
            self.next_event_id += 1
            track.event_id = event_id
            event = ConfirmedEvent(
                event_id=event_id,
                frame_idx=track.first_seen_frame,
                timestamp_sec=track.first_seen_frame / self.fps,
                bbox=track.initial_bbox,
                area=track.initial_area,
                growth_ratio=float(growth),
                radial_ratio=None,
                confidence=confidence,
                impact_point=track.initial_impact_point,
                confirm_frame_idx=frame_idx,
                first_seen_frame_idx=track.first_seen_frame,
            )
            self.pending_confirmations.append(event)
            self._remember_pixels(component)
            print(
                f"[INFO] New impact smoke {event_id}: first_frame={event.frame_idx}, "
                f"confirm_frame={frame_idx}, confidence={confidence:.3f}, "
                f"impact_px=({event.impact_point[0]:.1f},{event.impact_point[1]:.1f})"
            )

    def _age_tracks(self, frame_idx: int) -> None:
        expired_ids = []
        for track_id, track in self.tracks.items():
            if frame_idx - track.first_seen_frame > self.new_window_frames:
                track.expired_new_window = True
            if track.missed_frames > self.max_missed_frames:
                expired_ids.append(track_id)
        for track_id in expired_ids:
            track = self.tracks.pop(track_id)
            self.memories.append(
                _TrackMemory(
                    bbox=track.bbox,
                    centroid=track.centroid,
                    expires_frame=frame_idx + self.memory_frames,
                )
            )

    def _matches_memory(self, component: _Component) -> bool:
        for memory in self.memories:
            if _bbox_iou(memory.bbox, component.bbox) >= self.config.track_min_iou:
                return True
            if _distance(memory.centroid, component.centroid) <= self.config.track_match_distance:
                return True
        return component.new_pixel_ratio < self.config.min_new_pixel_ratio

    def _prune_memories(self, frame_idx: int) -> None:
        self.memories = [
            m for m in self.memories if frame_idx <= m.expires_frame]

    def _remember_pixels(self, component: _Component) -> None:
        if self.known_smoke_mask is None:
            return
        observation = np.zeros_like(self.known_smoke_mask)
        observation[component.ys, component.xs] = 255
        radius = max(int(self.config.known_smoke_dilate_px), 0)
        if radius > 0:
            size = radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            observation = cv2.dilate(observation, kernel)
        self.known_smoke_mask = cv2.bitwise_or(
            self.known_smoke_mask, observation)

    def _draw_visualization(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        canvas = frame_bgr.copy()
        if (
            self.config.draw_mask_overlay
            and self.last_binary_mask is not None
            and np.any(self.last_binary_mask)
        ):
            color = np.zeros_like(canvas)
            color[self.last_binary_mask] = (0, 170, 255)
            canvas = cv2.addWeighted(
                canvas, 1.0, color, float(self.config.overlay_alpha), 0.0
            )
        if self.config.show_candidate_boxes:
            for track in self.tracks.values():
                x, y, w, h = track.bbox
                box_color = (128, 128, 128) if track.preexisting else (
                    0, 180, 255)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), box_color, 1)
        if self.config.show_status_overlay:
            active_new = sum(
                1
                for track in self.tracks.values()
                if not track.preexisting and not track.expired_new_window
            )
            start_frame = self._start_frame_idx if self._start_frame_idx is not None else frame_idx
            warmup = max(self.warmup_frames -
                         (frame_idx - start_frame), 0) / self.fps
            lines = [
                f"PIDNet-S  threshold={self.config.segmentation_threshold:.2f}",
                f"new_candidates={active_new} tracks={len(self.tracks)}",
                f"inference={self.last_inference_ms:.1f} ms",
            ]
            if warmup > 0:
                lines.append(f"learning existing smoke: {warmup:.1f}s")
            for idx, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (18, 28 + idx * 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        return canvas
