"""
==============================================================================
Boom 煙霧偵測演算法 - YOLO 實例分割 / 物件偵測器 (yolo/detector.py)
==============================================================================

【檔案定位】
本檔案為 Boom 系統基於 YOLO (ONNX 權重) 的目標辨識演算法實作。
載入訓練完成的 YOLOv8/v11/v26 segmentation 或 detection 模型
(如 boom_v1_yolo26m-seg.onnx)，專門針對空拍視角下的荒野起爆/煙霧 (boom)
進行偵測、目標追蹤與座標估算。

【核心特性】
1. 依據需求專注辨識 "boom" 類別，過濾非 boom 目標。
2. 座標推估點：取 Bounding Box 正中心點 (x + w/2, y + h/2) 作為疑似源點 (impact_point)。
3. 支援 Ultralytics YOLO 與純 ONNXRuntime 雙推論引擎相容切換。
4. 實作 BaseSmokeDetector 抽象介面契約，支援統一 ConfirmedEvent 事件發布。
==============================================================================
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from boom.core.events import ConfirmedEvent
from boom.interfaces.detector import BaseSmokeDetector

BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]


@dataclass
class YOLOSmokeDetectorConfig:
    """YOLO 煙霧/起爆偵測器設定參數"""

    model_path: Union[str, Path] = field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "models"
        / "checkpoints"
        / "boom_v1_yolo26m-seg.onnx"
    )
    device: str = "auto"
    precision: str = "fp32"
    input_size: int = 640
    conf_threshold: float = 0.50  # 調高偵測辨識閥值（預設 0.50），過濾雜訊
    iou_threshold: float = 0.45
    inference_stride: int = 1

    # 目標類別清單：依使用者需求，預設只判斷 "boom"
    target_classes: Tuple[str, ...] = ("boom",)

    # 追蹤與時序確認參數
    confirmation_hits: int = 2
    warmup_sec: float = 0.0
    new_smoke_window_sec: float = 2.0
    track_match_distance: float = 140.0
    track_min_iou: float = 0.10
    max_missed_sec: float = 2.0  # 提高容忍未檢出時間至 2.0 秒，避免掉幀斷追蹤
    post_event_memory_sec: float = 300.0  # 歷史已確認事件記憶時間 (300 秒)，防止下一秒重複報警

    # 相機全域運動補償 (保持無人機飛行旋轉時軌跡與歷史記憶精確對齊)
    motion_compensation: bool = True
    motion_estimation_width: int = 640
    motion_max_corners: int = 400
    motion_min_inliers: int = 15
    motion_min_inlier_ratio: float = 0.40
    motion_max_rotation_deg: float = 10.0
    motion_min_scale: float = 0.85
    motion_max_scale: float = 1.15

    # 視覺化設定
    draw_mask_overlay: bool = True
    overlay_alpha: float = 0.35
    show_status_overlay: bool = True
    show_candidate_boxes: bool = True


@dataclass
class _TrackMemory:
    """已確認起爆/煙霧事件之歷史空間記憶，防止同地點短時間內重複辨識觸發"""

    event_id: int
    bbox: BBox
    center: Point
    class_name: str
    expires_frame: int


@dataclass
class _YOLOTrack:
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    initial_bbox: BBox
    bbox: BBox
    initial_center: Point
    center: Point
    initial_area: float
    area: float
    peak_confidence: float
    class_name: str
    class_id: int
    mask: Optional[np.ndarray] = None
    observations: int = 1
    missed_frames: int = 0
    event_id: Optional[int] = None
    preexisting: bool = False  # 標記是否為既有事件 / 歷史事件延續
    expired_new_window: bool = False

    @property
    def confirmed(self) -> bool:
        return self.event_id is not None


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


class YOLOSmokeImpactDetector(BaseSmokeDetector):
    """
    基於 YOLO 的煙霧/起爆偵測器，實作 BaseSmokeDetector 介面。
    預設鎖定 'boom' 類別，並採用 Bounding Box 中心點推估地表接觸座標。
    """

    def __init__(
        self,
        config: Optional[YOLOSmokeDetectorConfig] = None,
        fps: float = 30.0,
    ) -> None:
        self.config = config or YOLOSmokeDetectorConfig()
        self.fps = float(fps) if fps > 0 else 30.0
        self.warmup_frames = max(int(round(self.config.warmup_sec * self.fps)), 0)
        self.new_window_frames = max(
            int(round(self.config.new_smoke_window_sec * self.fps)), 1
        )
        self.max_missed_frames = max(
            int(round(self.config.max_missed_sec * self.fps)), 1
        )

        self.pending_confirmations: List[ConfirmedEvent] = []
        self.tracks: Dict[int, _YOLOTrack] = {}
        self.memories: List[_TrackMemory] = []
        self.memory_frames = max(
            int(round(self.config.post_event_memory_sec * self.fps)), 1
        )
        self.next_track_id = 1
        self.next_event_id = 1

        self.last_inference_ms = 0.0
        self._model = None
        self._backend = "none"  # "ultralytics" or "onnxruntime"
        self._class_names: Dict[int, str] = {0: "smoke", 1: "boom"}
        self._validated_target_classes = False
        self._start_frame_idx: Optional[int] = None
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._previous_motion_gray: Optional[np.ndarray] = None
        self._last_detections: List[Dict] = []

        self._init_model()

    def _init_model(self) -> None:
        """初始化 YOLO 模型推論引擎 (優先使用 ultralytics，無則使用 onnxruntime)"""
        model_path = Path(self.config.model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"找不到 YOLO 模型權重檔案：{model_path}")

        # 優先嘗試 ultralytics
        try:
            from ultralytics import YOLO
            from ultralytics.cfg import DEFAULT_CFG_DICT

            device = self.config.device
            if device == "auto":
                try:
                    import torch

                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    if device.startswith("cuda") and model_path.suffix.lower() == ".onnx":
                        import onnxruntime as ort
                        if "CUDAExecutionProvider" not in ort.get_available_providers():
                            device = "cpu"
                except Exception:
                    device = "cpu"
            self._inference_device = device
            self._use_half = self.config.precision == "fp16" and str(device).startswith("cuda") and model_path.suffix.lower() == ".pt"
            self._precision_options = (
                {"quantize": 16 if self._use_half else None}
                if "quantize" in DEFAULT_CFG_DICT else {"half": self._use_half}
            )
            self._checked_runtime_device = False
            self._model = YOLO(str(model_path))
            self._backend = "ultralytics"
            # Exported models may initialize a CPU backend just to read names.
            # Let the first predict(device=...) initialize it, then use Results.names.
            if model_path.suffix.lower() == ".pt" and self._model.names:
                self._class_names = dict(self._model.names)
            return
        except ImportError:
            pass

        # 後備嘗試 onnxruntime
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self.config.device in ("auto", "cuda"):
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")
            if self.config.device == "cuda" and "CUDAExecutionProvider" not in providers:
                raise RuntimeError("CUDA requested but ONNX Runtime CUDA provider is unavailable")
            self._model = ort.InferenceSession(str(model_path), providers=providers)
            if self.config.device == "cuda" and "CUDAExecutionProvider" not in self._model.get_providers():
                raise RuntimeError("ONNX Runtime could not initialize CUDAExecutionProvider")
            print(f"[INFO] YOLO ONNX providers: {self._model.get_providers()}")
            self._backend = "onnxruntime"
            return
        except ImportError:
            pass

        raise RuntimeError(
            "無法載入 YOLO 推論引擎：請安裝 ultralytics (`pip install ultralytics`) 或 onnxruntime (`pip install onnxruntime-gpu` / `onnxruntime`)"
        )

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        處理單一影格：相機運動補償、YOLO 推論、目標追蹤與去重、回傳視覺化影格。
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr 必須為非空影像陣列")

        height, width = frame_bgr.shape[:2]
        if self._frame_shape != (height, width):
            self._frame_shape = (height, width)
            self._previous_motion_gray = None
            self._start_frame_idx = frame_idx
        elif self._start_frame_idx is None:
            self._start_frame_idx = frame_idx

        # 1. 估計相機全域運動並對追蹤中軌跡與歷史記憶進行座標補償
        if self.config.motion_compensation:
            transform = self._estimate_camera_motion(frame_bgr)
            if transform is not None:
                self._warp_tracking_state(transform, width, height)

        stride = max(int(self.config.inference_stride), 1)
        should_infer = (frame_idx - self._start_frame_idx) % stride == 0

        if should_infer:
            started = time.perf_counter()
            detections = self._infer(frame_bgr)
            self.last_inference_ms = (time.perf_counter() - started) * 1000.0
            self._last_detections = detections
            self._update_tracks(detections, frame_idx)
        else:
            self._age_tracks(frame_idx)

        return self._draw_visualization(frame_bgr, frame_idx)

    def consume_pending_confirmations(self) -> List[ConfirmedEvent]:
        """取出正式確認的起煙/起爆事件清單並清空緩衝區"""
        events = self.pending_confirmations
        self.pending_confirmations = []
        return events

    def camera_guard_summary(self) -> Dict[str, int]:
        """相容 Runtime 摘要報告之相機動態防護狀態統計"""
        return {}

    def _infer(self, frame_bgr: np.ndarray) -> List[Dict]:
        """執行推論並篩選出符合 target_classes (預設 boom) 的目標"""
        height, width = frame_bgr.shape[:2]
        target_classes_lower = {str(c).lower().strip() for c in self.config.target_classes}
        results: List[Dict] = []

        if self._backend == "ultralytics":
            preds = self._model.predict(
                frame_bgr,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.input_size,
                device=self._inference_device,
                **self._precision_options,
                verbose=False,
            )
            if not self._checked_runtime_device:
                backend = self._model.predictor.model
                session = getattr(backend, "session", None)
                if session is not None and str(self._inference_device).startswith("cuda"):
                    if "CUDAExecutionProvider" not in session.get_providers():
                        if self.config.device == "cuda":
                            raise RuntimeError("YOLO ONNX fell back to CPU despite --device cuda")
                        self._inference_device = "cpu"
                        print("[WARN] YOLO ONNX CUDA provider unavailable at runtime; using CPU")
                self._checked_runtime_device = True
            if not preds:
                return results

            pred = preds[0]
            self._class_names = dict(pred.names)
            if not self._validated_target_classes:
                available = {str(name).lower().strip() for name in self._class_names.values()}
                missing = target_classes_lower - available
                if missing:
                    raise ValueError(
                        f"YOLO model has no target class(es): {', '.join(sorted(missing))}. "
                        f"Available classes: {', '.join(sorted(available))}"
                    )
                self._validated_target_classes = True
            if pred.boxes is None or len(pred.boxes) == 0:
                return results

            boxes = pred.boxes.xyxy.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            clss = pred.boxes.cls.cpu().numpy().astype(int)
            masks = None
            if pred.masks is not None and pred.masks.data is not None:
                # 取得原圖尺寸的遮罩
                masks = pred.masks.data.cpu().numpy()

            for i in range(len(boxes)):
                cls_id = int(clss[i])
                cls_name = self._class_names.get(cls_id, str(cls_id)).lower().strip()

                # 依需求只保留目標類別（如 boom）
                if target_classes_lower and cls_name not in target_classes_lower:
                    continue

                x1, y1, x2, y2 = boxes[i]
                bx, by = int(round(x1)), int(round(y1))
                bw, bh = max(int(round(x2 - x1)), 1), max(int(round(y2 - y1)), 1)
                bbox: BBox = (bx, by, bw, bh)

                # 關鍵需求：中心點推估座標
                center: Point = (bx + bw / 2.0, by + bh / 2.0)

                mask_resized = None
                if masks is not None and i < len(masks):
                    m = masks[i]
                    if m.shape[:2] != (height, width):
                        mask_resized = cv2.resize(
                            (m > 0.5).astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    else:
                        mask_resized = (m > 0.5).astype(np.uint8)

                results.append(
                    {
                        "bbox": bbox,
                        "center": center,
                        "conf": float(confs[i]),
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "area": float(bw * bh),
                        "mask": mask_resized,
                    }
                )

        elif self._backend == "onnxruntime":
            results = self._infer_onnxruntime(frame_bgr)

        return results

    def _infer_onnxruntime(self, frame_bgr: np.ndarray) -> List[Dict]:
        """純 ONNXRuntime 後備推論實作"""
        sess = self._model
        input_meta = sess.get_inputs()[0]
        h_orig, w_orig = frame_bgr.shape[:2]
        size = self.config.input_size

        # Letterbox 縮放
        scale = min(size / h_orig, size / w_orig)
        nw, nh = int(round(w_orig * scale)), int(round(h_orig * scale))
        dx = (size - nw) // 2
        dy = (size - nh) // 2

        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[dy : dy + nh, dx : dx + nw] = resized

        # NCHW Float32 RGB
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)

        outputs = sess.run(None, {input_meta.name: tensor})
        out0 = outputs[0]
        if out0.ndim == 3 and out0.shape[1] < out0.shape[2] and out0.shape[1] in (38, 6, 7):
            out0 = out0.transpose(0, 2, 1)

        detections = out0[0]
        results: List[Dict] = []
        target_classes_lower = {str(c).lower().strip() for c in self.config.target_classes}

        for row in detections:
            if len(row) >= 6:
                class_scores = row[4:6]
                cls_id = int(np.argmax(class_scores))
                conf = float(class_scores[cls_id])
                if conf < self.config.conf_threshold:
                    continue

                cls_name = self._class_names.get(cls_id, str(cls_id)).lower().strip()
                if target_classes_lower and cls_name not in target_classes_lower:
                    continue

                cx, cy, w, h = row[0], row[1], row[2], row[3]
                rx1 = (cx - w / 2.0 - dx) / scale
                ry1 = (cy - h / 2.0 - dy) / scale
                rw = w / scale
                rh = h / scale

                bx = int(max(0, min(w_orig - 1, round(rx1))))
                by = int(max(0, min(h_orig - 1, round(ry1))))
                bw = int(max(1, min(w_orig - bx, round(rw))))
                bh = int(max(1, min(h_orig - by, round(rh))))

                bbox: BBox = (bx, by, bw, bh)
                center: Point = (bx + bw / 2.0, by + bh / 2.0)

                results.append(
                    {
                        "bbox": bbox,
                        "center": center,
                        "conf": conf,
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "area": float(bw * bh),
                        "mask": None,
                    }
                )

        return results

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
        left = int(np.clip(math.floor(min(p[0] for p in corners)), 0, width - 1))
        top = int(np.clip(math.floor(min(p[1] for p in corners)), 0, height - 1))
        right = int(np.clip(math.ceil(max(p[0] for p in corners)), left + 1, width))
        bottom = int(np.clip(math.ceil(max(p[1] for p in corners)), top + 1, height))
        return left, top, max(right - left, 1), max(bottom - top, 1)

    def _estimate_camera_motion(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """使用 LK 光流估計前後影格之相機全域平移與旋轉變換 (Affine)"""
        height, width = frame_bgr.shape[:2]
        target_width = min(max(int(self.config.motion_estimation_width), 64), width)
        target_height = max(int(round(height * target_width / width)), 48)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if (target_width, target_height) != (width, height):
            gray = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)

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
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
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
        rotation = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
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

    def _warp_tracking_state(self, transform: np.ndarray, width: int, height: int) -> None:
        """依照相機運動補償矩陣轉換所有 active tracks 與歷史記憶的位置"""
        for track in self.tracks.values():
            track.center = self._transform_point(track.center, transform)
            track.bbox = self._transform_bbox(track.bbox, transform, width, height)
        for memory in self.memories:
            memory.center = self._transform_point(memory.center, transform)
            memory.bbox = self._transform_bbox(memory.bbox, transform, width, height)

    def _find_matching_memory_or_event(self, det: Dict) -> Optional[int]:
        """
        檢查新偵測目標是否落在已確認 active track 或歷史事件記憶 (self.memories) 附近。
        若有符合，回傳該事件的 event_id，用以抑制重複發布事件並延續追蹤。
        """
        det_bbox = det["bbox"]
        det_center = det["center"]
        dw, dh = det_bbox[2], det_bbox[3]

        # 1. 先比對目前已確認的 active tracks
        for track in self.tracks.values():
            if not track.confirmed:
                continue
            iou = _bbox_iou(track.bbox, det_bbox)
            dist = _distance(track.center, det_center)
            tw, th = track.bbox[2], track.bbox[3]
            adaptive_distance = max(
                self.config.track_match_distance,
                0.6 * max(math.hypot(tw, th), math.hypot(dw, dh)),
            )
            if iou >= self.config.track_min_iou or dist <= adaptive_distance:
                return track.event_id

        # 2. 比對歷史記憶 (已消失但仍在記憶有效期內的事件)
        for memory in self.memories:
            iou = _bbox_iou(memory.bbox, det_bbox)
            dist = _distance(memory.center, det_center)
            mw, mh = memory.bbox[2], memory.bbox[3]
            adaptive_distance = max(
                self.config.track_match_distance,
                0.6 * max(math.hypot(mw, mh), math.hypot(dw, dh)),
            )
            if iou >= self.config.track_min_iou or dist <= adaptive_distance:
                return memory.event_id

        return None

    def _update_tracks(self, detections: List[Dict], frame_idx: int) -> None:
        """更新目標軌跡、歷史記憶比對、去重並判定起爆確認事件"""
        self._prune_memories(frame_idx)
        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        for track_id, det_idx in matches:
            det = detections[det_idx]
            track = self.tracks[track_id]
            track.last_seen_frame = frame_idx
            track.bbox = det["bbox"]
            track.center = det["center"]
            track.area = det["area"]
            track.peak_confidence = max(track.peak_confidence, det["conf"])
            track.mask = det["mask"]
            track.observations += 1
            track.missed_frames = 0

            # 若尚未確認且非既有歷史事件，且累積幀數達標，正式確認新事件
            if (
                not track.confirmed
                and not track.preexisting
                and track.observations >= self.config.confirmation_hits
            ):
                self._confirm_event(track, frame_idx)

        for track_id in unmatched_tracks:
            self.tracks[track_id].missed_frames += 1

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            # 關鍵防重複：檢查是否為已確認事件或歷史記憶中的同一個起爆/煙霧點
            matched_event_id = self._find_matching_memory_or_event(det)
            is_preexisting = matched_event_id is not None

            track = _YOLOTrack(
                track_id=self.next_track_id,
                first_seen_frame=frame_idx,
                last_seen_frame=frame_idx,
                initial_bbox=det["bbox"],
                bbox=det["bbox"],
                initial_center=det["center"],
                center=det["center"],
                initial_area=det["area"],
                area=det["area"],
                peak_confidence=det["conf"],
                class_name=det["cls_name"],
                class_id=det["cls_id"],
                mask=det["mask"],
                event_id=matched_event_id,
                preexisting=is_preexisting,
            )
            self.tracks[track.track_id] = track
            self.next_track_id += 1

            # 只有全新的獨立事件且 confirmation_hits <= 1 才會立即確認
            if not is_preexisting and self.config.confirmation_hits <= 1:
                self._confirm_event(track, frame_idx)

        self._age_tracks(frame_idx)

    def _confirm_event(self, track: _YOLOTrack, frame_idx: int) -> None:
        """將軌跡轉為已確認起爆事件 (使用 Bounding Box 中心為座標推估點)"""
        event_id = self.next_event_id
        self.next_event_id += 1
        track.event_id = event_id

        # 關鍵需求：取 Bounding Box 中心點作為座標推估 (impact_point)
        impact_point: Point = track.initial_center

        event = ConfirmedEvent(
            event_id=event_id,
            frame_idx=track.first_seen_frame,
            timestamp_sec=track.first_seen_frame / self.fps,
            bbox=track.initial_bbox,
            growth_ratio=float(track.area / max(track.initial_area, 1.0)),
            radial_ratio=None,
            area=float(track.initial_area),
            confirmed_by="yolo",
            impact_point=impact_point,
            confirm_frame_idx=frame_idx,
            first_seen_frame_idx=track.first_seen_frame,
            status=("new-impact-boom" if track.class_name == "boom"
                    else "new-impact-smoke" if track.class_name == "smoke"
                    else f"detected-{track.class_name}"),
            confidence=float(track.peak_confidence),
        )
        self.pending_confirmations.append(event)
        print(
            f"[INFO] [YOLO] Confirmed '{track.class_name}' event #{event_id}: "
            f"first_frame={event.frame_idx}, confirm_frame={frame_idx}, "
            f"conf={track.peak_confidence:.2f}, center_point=({impact_point[0]:.1f}, {impact_point[1]:.1f})"
        )

    def _associate(
        self, detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """透過 IoU 與自適應中心點距離將偵測結果與軌跡進行關聯"""
        pairings: List[Tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            tx, ty, tw, th = track.bbox
            for det_idx, det in enumerate(detections):
                dw, dh = det["bbox"][2], det["bbox"][3]
                iou = _bbox_iou(track.bbox, det["bbox"])
                dist = _distance(track.center, det["center"])
                adaptive_distance = max(
                    self.config.track_match_distance,
                    0.6 * max(math.hypot(tw, th), math.hypot(dw, dh)),
                )
                if iou >= self.config.track_min_iou or dist <= adaptive_distance:
                    score = (1.0 - iou) + (dist / max(adaptive_distance, 1.0))
                    pairings.append((score, track_id, det_idx))

        pairings.sort(key=lambda item: item[0])
        used_tracks = set()
        used_dets = set()
        matches: List[Tuple[int, int]] = []

        for _, track_id, det_idx in pairings:
            if track_id in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(det_idx)
            matches.append((track_id, det_idx))

        unmatched_tracks = [t for t in self.tracks if t not in used_tracks]
        unmatched_dets = [d for d in range(len(detections)) if d not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _age_tracks(self, frame_idx: int) -> None:
        """老化並清理過期軌跡，若為已確認事件則轉入歷史記憶防止未來重複觸發"""
        expired = []
        for track_id, track in self.tracks.items():
            if track.missed_frames > self.max_missed_frames:
                expired.append(track_id)
        for track_id in expired:
            track = self.tracks.pop(track_id, None)
            if track and track.confirmed and track.event_id is not None:
                self.memories.append(
                    _TrackMemory(
                        event_id=track.event_id,
                        bbox=track.bbox,
                        center=track.center,
                        class_name=track.class_name,
                        expires_frame=frame_idx + self.memory_frames,
                    )
                )

    def _prune_memories(self, frame_idx: int) -> None:
        """清理超過保留時間的歷史事件記憶"""
        self.memories = [m for m in self.memories if frame_idx <= m.expires_frame]

    def _draw_visualization(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        """在影格上繪製煙霧/起爆視覺化標註、中心十字點與 HUD 狀態"""
        canvas = frame_bgr.copy()

        # 1. 繪製 Segmentation 遮罩 (若有)
        if self.config.draw_mask_overlay:
            mask_canvas = np.zeros_like(canvas)
            has_mask = False
            for track in self.tracks.values():
                if track.mask is not None and np.any(track.mask):
                    mask_canvas[track.mask > 0] = (0, 165, 255)
                    has_mask = True
            if has_mask:
                canvas = cv2.addWeighted(
                    canvas, 1.0, mask_canvas, float(self.config.overlay_alpha), 0.0
                )

        # 2. 繪製 Bounding Box 與中心點 (Impact Point)
        for track in self.tracks.values():
            x, y, w, h = track.bbox
            is_confirmed = track.confirmed
            box_color = (0, 69, 255) if is_confirmed else (0, 215, 255)
            thickness = 2 if is_confirmed else 1

            cv2.rectangle(canvas, (x, y), (x + w, y + h), box_color, thickness)

            # 中心點座標推估十字準心
            cx, cy = int(round(track.center[0])), int(round(track.center[1]))
            cross_size = 6
            cv2.drawMarker(
                canvas,
                (cx, cy),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=cross_size * 2,
                thickness=2,
            )

            # 標籤文字 (已確認事件顯示編號，未確認顯示候選觀察進度)
            if track.event_id is not None:
                label = f"#{track.event_id} {track.class_name.upper()} {track.peak_confidence:.2f}"
            else:
                label = f"[?] {track.class_name.upper()} {track.peak_confidence:.2f} ({track.observations}/{self.config.confirmation_hits})"
            cv2.putText(
                canvas,
                label,
                (x, max(y - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA,
            )

        # 3. 狀態資訊 HUD
        if self.config.show_status_overlay:
            active_count = len(self.tracks)
            confirmed_count = sum(1 for t in self.tracks.values() if t.confirmed)
            lines = [
                f"YOLO Detector [{Path(self.config.model_path).name}]",
                f"conf: {self.config.conf_threshold:.2f}  tracks: {active_count}  events: {confirmed_count}",
                f"inference: {self.last_inference_ms:.1f} ms ({self._backend})",
            ]
            for idx, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (18, 28 + idx * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return canvas
