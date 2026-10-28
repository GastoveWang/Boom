"""Frozen outdoor LoFTR + RANSAC background similarity registration.

Matrices map destination/current pixel centres to source/previous pixel centres.
Failed pairs have identity matrices, zero residuals AND false validity masks;
identity is a storage placeholder, never evidence of successful registration.
"""
from copy import deepcopy
from pathlib import Path
import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = ROOT / 'models/pretrained/registration/loftr_outdoor.ckpt'
QUALITY_FIELDS = ('accepted', 'inlier_ratio', 'normalized_reprojection_error',
                  'previous_coverage', 'current_coverage', 'overlap',
                  'mean_inlier_confidence', 'inlier_support')


def fit_background(previous, current, confidence, height, width, config):
    """Fit current -> previous in matching-resolution pixels; no ground truth ROI."""
    keep = (np.isfinite(previous).all(1) & np.isfinite(current).all(1)
            & np.isfinite(confidence) & (confidence >= config.registration_confidence))
    previous, current, confidence = previous[keep], current[keep], confidence[keep]
    quality = np.zeros(8, np.float32)
    if len(previous) < config.registration_min_inliers:
        return None, quality
    try:
        matrix, inliers = cv2.estimateAffinePartial2D(
            current, previous, method=cv2.RANSAC,
            ransacReprojThreshold=config.registration_ransac_threshold,
            maxIters=2000, confidence=.995, refineIters=10)
    except cv2.error:
        return None, quality
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return None, quality
    inside = inliers.ravel().astype(bool)
    count = int(inside.sum())
    ratio = count / len(inside)
    if count < 3:
        return None, quality
    p, q = previous[inside], current[inside]
    error = float(np.median(np.linalg.norm(q @ matrix[:, :2].T + matrix[:, 2] - p, axis=1)))
    coverage0 = cv2.contourArea(cv2.convexHull(p.astype(np.float32))) / (height * width)
    coverage1 = cv2.contourArea(cv2.convexHull(q.astype(np.float32))) / (height * width)
    scale = math.sqrt(abs(float(np.linalg.det(matrix[:, :2]))))
    quality[1:5] = ratio, error / config.registration_ransac_threshold, coverage0, coverage1
    quality[6:] = float(confidence[inside].mean()), min(count / 100., 1.)
    accepted = (count >= config.registration_min_inliers and ratio >= config.registration_min_ratio
                and min(coverage0, coverage1) >= config.registration_min_coverage
                and error <= config.registration_ransac_threshold and .5 <= scale <= 2.)
    if not accepted:
        return None, quality
    return np.vstack((matrix, (0., 0., 1.))), quality


class LoFTRRegistration(nn.Module):
    """No learned motion head or photometric loss; matcher is always frozen/eval.

    Adjacent pairs are processed serially to bound matching memory. There is no
    implicit stream cache, so shuffled training and independent video streams
    cannot contaminate each other. Full checkpoints embed the frozen matcher.
    """
    def __init__(self, config):
        super().__init__()
        from kornia.feature import LoFTR
        from kornia.feature.loftr.loftr import default_cfg
        self.config = config
        self.matcher = LoFTR(pretrained=None, config=deepcopy(default_cfg)).eval()
        self.matcher.requires_grad_(False)
        self.register_buffer('weights_loaded', torch.tensor(False))

    def load_weights(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'LoFTR weights missing: {path}. Run: python src/st2 download-weights')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.matcher.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
        self.weights_loaded.fill_(True)
        self.matcher.eval()

    def train(self, mode=True):
        super().train(mode)
        self.matcher.eval()
        return self

    @torch.no_grad()
    def forward(self, sequence):
        if not bool(self.weights_loaded):
            raise RuntimeError('Load outdoor LoFTR weights or a complete ST2 v2 checkpoint before forward')
        b, length, channels, h, w = sequence.shape
        if channels != 3 or length < 2:
            raise ValueError('Registration expects at least two RGB frames')
        steps = length - 1
        factor = min(1., self.config.registration_max_side / max(h, w))
        mh, mw = max(32, int(h * factor) // 8 * 8), max(32, int(w * factor) // 8 * 8)
        rgb = sequence.detach().reshape(-1, 3, h, w).float()
        gray = (rgb * rgb.new_tensor((.299, .587, .114))[None, :, None, None]).sum(1, keepdim=True)
        small = F.interpolate(gray, (mh, mw), mode='bilinear', align_corners=False).reshape(b, length, 1, mh, mw)
        # Inverse of align_corners=False resize: matching pixel -> input pixel.
        resize = np.array(((w / mw, 0, (w / mw - 1) / 2),
                           (0, h / mh, (h / mh - 1) / 2), (0, 0, 1)), np.float64)
        resize_inverse = np.linalg.inv(resize)
        matrices, qualities = [], []
        for sample in range(b):
            for step in range(steps):
                if min(float(small[sample, step].std()), float(small[sample, step+1].std())) < self.config.registration_min_std:
                    matrices.append(np.eye(3))
                    qualities.append(np.zeros(8, np.float32))
                    continue
                matches = self.matcher({'image0': small[sample:sample+1, step],
                                        'image1': small[sample:sample+1, step+1]})
                p = matches['keypoints0'].float().cpu().numpy()
                q = matches['keypoints1'].float().cpu().numpy()
                confidence = matches['confidence'].float().cpu().numpy()
                matrix, quality = fit_background(p, q, confidence, mh, mw, self.config)
                if matrix is not None:
                    quality[0] = 1.
                    matrix = resize @ matrix @ resize_inverse
                else:
                    matrix = np.eye(3)
                matrices.append(matrix)
                qualities.append(quality)
        matrix = torch.as_tensor(np.stack(matrices), device=sequence.device, dtype=torch.float32)
        quality = torch.as_tensor(np.stack(qualities), device=sequence.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(torch.arange(h, device=sequence.device, dtype=torch.float32),
                                torch.arange(w, device=sequence.device, dtype=torch.float32), indexing='ij')
        coordinates = torch.stack((xx, yy, torch.ones_like(xx)), -1)
        source = torch.einsum('nij,hwj->nhwi', matrix, coordinates)[..., :2]
        grid = (source + .5) * source.new_tensor((2 / w, 2 / h)) - 1
        valid = ((source[..., 0] >= 0) & (source[..., 0] <= w-1)
                 & (source[..., 1] >= 0) & (source[..., 1] <= h-1))[:, None]
        overlap = valid.float().mean((1, 2, 3))
        accepted = quality[:, 0].bool() & (overlap >= self.config.registration_min_overlap)
        quality[:, 0] = accepted.float()
        quality[:, 5] = torch.where(accepted, overlap, torch.zeros_like(overlap))
        valid &= accepted[:, None, None, None]
        previous = sequence[:, :-1].reshape(-1, 3, h, w).float()
        current = sequence[:, 1:].reshape(-1, 3, h, w).float()
        warped = F.grid_sample(previous, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        residual = (warped-current).square().mean(1, keepdim=True) * valid
        # Failed matrices are explicitly identity placeholders, even on low overlap.
        matrix = torch.where(accepted[:, None, None], matrix, torch.eye(3, device=matrix.device)[None])
        return {'registration_mse': residual.reshape(b, steps, h, w),
                'registration_valid': valid.reshape(b, steps, h, w),
                'registration_matrix': matrix.reshape(b, steps, 3, 3),
                'registration_quality': quality.reshape(b, steps, 8),
                'registration_success': accepted.reshape(b, steps)}
