# 檔案功能：固定尺寸 EDM ONNX 影像對匹配後端。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""EDM's fixed-size ONNX pair matcher adapted to Boom's feature backend API.

The model and postprocessing follow chicleee/EDM/deploy/run_onnx.py. EDM
processes both images together, so ``extract`` retains the source image and
``match`` performs the single pair inference.
"""

from pathlib import Path
from importlib import metadata
import warnings

import cv2
import numpy as np


class EDMBackend:
    def __init__(self, config, session=None):
        self.config = config
        self.extractor = None
        self.matcher = self
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("EDM requires onnxruntime or onnxruntime-gpu") from exc
            path = Path(config.edm_model_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f"EDM ONNX model not found: {path}. Download the fixed model "
                    "from https://github.com/chicleee/EDM and set --map-edm-model-path."
                )
            available = ort.get_available_providers()
            if config.device == "cuda" and "CUDAExecutionProvider" not in available:
                raise RuntimeError("--map-device cuda requested but ONNX Runtime CUDA is unavailable")
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if config.device != "cpu" and "CUDAExecutionProvider" in available
                         else ["CPUExecutionProvider"])
            if providers[0] == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
                try:
                    metadata.version("nvidia-cudnn-cu12")
                except metadata.PackageNotFoundError:
                    pass
                else:
                    # Load CUDA 12 cuDNN explicitly. A PyTorch installation in
                    # the same environment may instead bundle CUDA 13 DLLs.
                    ort.preload_dlls(cuda=False, cudnn=True, msvc=False, directory="")
            session = ort.InferenceSession(str(path), providers=providers)
            if providers[0] == "CUDAExecutionProvider" and "CUDAExecutionProvider" not in session.get_providers():
                message = "EDM CUDA provider failed to load; check CUDA 12 and cuDNN 9 runtime dependencies"
                if config.device == "cuda":
                    raise RuntimeError(message)
                warnings.warn(message + "; using CPU", RuntimeWarning, stacklevel=2)
        self.session = session
        inputs = session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].shape[1] != 2:
            raise ValueError("EDM ONNX model must have one [1, 2, H, W] input")
        _, _, height, width = inputs[0].shape
        if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
            raise ValueError("EDM ONNX model must have fixed positive height and width")
        self.height, self.width = height, width
        self.input_name = inputs[0].name

    def extract(self, gray, mask=None):
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("EDM expects a uint8 grayscale image")
        if mask is not None and mask.shape != gray.shape:
            raise ValueError("EDM mask must have the same shape as the image")
        return gray, mask

    def match(self, a, b):
        image0, mask0 = a
        image1, mask1 = b
        inputs = np.stack((cv2.resize(image0, (self.width, self.height)),
                           cv2.resize(image1, (self.width, self.height))), axis=0)
        inputs = np.ascontiguousarray(inputs[None], dtype=np.float32) / 255.0
        raw = np.asarray(self.session.run(None, {self.input_name: inputs})[0])
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim != 2 or raw.shape[1] != 11:
            raise ValueError(f"Unsupported EDM output shape: {raw.shape}; expected [K, 11]")
        if len(raw) == 0:
            return self._empty()
        # Official bidirectional refinement: retain the direction with the
        # stronger fine score and apply the same confidence/sigma gates.
        coarse0, coarse1 = raw[:, :2], raw[:, 2:4]
        forward = raw[:, 8] > raw[:, 9]
        p0 = np.where(forward[:, None], coarse0, coarse0 + raw[:, 6:8] * 8)
        p1 = np.where(forward[:, None], coarse1 + raw[:, 4:6] * 8, coarse1)
        confidence = raw[:, 10]
        fine_score = np.where(forward, raw[:, 8], raw[:, 9])
        border = 16  # EDM demo removes two 8-pixel cells at each edge.
        valid = (np.isfinite(raw).all(axis=1)
                 & (confidence > self.config.edm_conf_threshold)
                 & (fine_score > self.config.edm_sigma_threshold))
        for points in (p0, p1):
            valid &= ((points[:, 0] >= border) & (points[:, 0] <= self.width-border)
                      & (points[:, 1] >= border) & (points[:, 1] <= self.height-border))
        p0 = self._to_source(p0[valid], image0.shape)
        p1 = self._to_source(p1[valid], image1.shape)
        scores = confidence[valid].astype(np.float64)
        keep = np.ones(len(scores), dtype=bool)
        for points, mask, image in ((p0, mask0, image0), (p1, mask1, image1)):
            keep &= ((points[:, 0] >= 0) & (points[:, 0] < image.shape[1])
                     & (points[:, 1] >= 0) & (points[:, 1] < image.shape[0]))
            if mask is not None:
                xy = np.rint(points).astype(np.int64)
                x = np.clip(xy[:, 0], 0, mask.shape[1]-1)
                y = np.clip(xy[:, 1], 0, mask.shape[0]-1)
                keep &= mask[y, x] > 0
        return p0[keep], p1[keep], scores[keep]

    def _to_source(self, points, shape):
        height, width = shape
        # Invert OpenCV's pixel-center resize transform.
        scale = np.array([width/self.width, height/self.height])
        return ((points + .5) * scale - .5).astype(np.float64)

    @staticmethod
    def _empty():
        return np.empty((0, 2), np.float64), np.empty((0, 2), np.float64), np.empty(0)
