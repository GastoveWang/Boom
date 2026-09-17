"""
==============================================================================
Boom 航拍定位 - 特徵點提取與比對後端 (feature_matching.py)
==============================================================================

【檔案定位】
本檔案提供統一抽象的特徵匹配後端介面。支援深度學習 SuperPoint + LightGlue
高精度比對後端，以及經典 SIFT 特徵比對後端。

【核心後端】
- `SuperPointLightGlueBackend`: 採用深度神經網路進行特徵點提取與圖神經網路匹配，具備極佳的視角與光照變化穩健性。
- `SIFTBackend`: 基於 OpenCV SIFT 的傳統特徵比對後端。
- `create_feature_backend(config)`: 工廠函式，依據設定建立匹配後端實例。
==============================================================================
"""
import cv2
import numpy as np


class SIFTBackend:
    """Explicit legacy option, never an automatic fallback for LightGlue."""

    def __init__(self, config, extractor=None, matcher=None):
        self.config = config
        self.extractor = extractor if extractor is not None else cv2.SIFT_create(
            nfeatures=config.max_features, contrastThreshold=.02)
        self.matcher = matcher if matcher is not None else cv2.BFMatcher()

    def extract(self, gray, mask=None):
        return self.extractor.detectAndCompute(gray, mask)

    def match(self, a, b):
        from .registration import match_features
        pa, pb = match_features(self.matcher, a, b, self.config.ratio_threshold)
        return pa, pb, None


class SuperPointLightGlueBackend:
    def __init__(self, config):
        try:
            import torch
            from lightglue import SuperPoint, LightGlue
        except ImportError as exc:
            raise RuntimeError(
                "SuperPoint + LightGlue requires the offline requirements. "
                "Install with: python -m pip install -r artillery/offline/requirements.txt"
            ) from exc
        self.torch = torch
        self.max_image_size = config.max_image_size
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--map-device cuda requested but CUDA is unavailable")
        self.extractor = SuperPoint(max_num_keypoints=config.max_features).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint",
                                filter_threshold=config.lightglue_filter_threshold).eval().to(self.device)

    def extract(self, gray, mask=None):
        torch = self.torch
        image = torch.from_numpy(np.ascontiguousarray(gray)).float()[None].to(self.device) / 255.
        with torch.inference_mode():
            features = self.extractor.extract(
                image, resize=self.max_image_size if max(gray.shape) > self.max_image_size else None)
            if mask is not None:
                xy = features["keypoints"][0].round().long()
                height, width = mask.shape
                valid = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
                valid &= torch.as_tensor(mask, device=self.device)[
                    xy[:, 1].clamp(0, height-1), xy[:, 0].clamp(0, width-1)] > 0
                # Keep descriptor/keypoint alignment and batched image_size.
                features = {key: value[:, valid] if key in (
                    "keypoints", "keypoint_scores", "descriptors") else value
                            for key, value in features.items()}
        return features

    def match(self, a, b):
        if a["keypoints"].shape[1] == 0 or b["keypoints"].shape[1] == 0:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        with self.torch.inference_mode():
            output = self.matcher({"image0": a, "image1": b})
            pairs = output["matches"][0]
            pa = a["keypoints"][0, pairs[:, 0]].cpu().numpy().astype(np.float64)
            pb = b["keypoints"][0, pairs[:, 1]].cpu().numpy().astype(np.float64)
            scores = output["scores"][0].cpu().numpy()
        return pa, pb, scores


def create_feature_backend(config, extractor=None, matcher=None):
    if extractor is not None or matcher is not None or config.matcher_backend == "sift":
        return SIFTBackend(config, extractor, matcher)
    return SuperPointLightGlueBackend(config)
