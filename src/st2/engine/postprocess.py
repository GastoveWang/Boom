"""Fixed score scaling and causal, model-independent camera-motion checks."""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ScoreNormalizer:
    maximum: float = .02
    knee: float = .0001

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (self.maximum, self.knee)):
            raise ValueError('Score maximum and knee must be positive finite values')

    def __call__(self, score):
        if not math.isfinite(score) or score < 0:
            raise ValueError('MSE scores must be nonnegative and finite')
        # Same mapping for raw/weighted/final scores and every video. Not a probability.
        return float(np.clip(math.log1p(score / self.knee) / math.log1p(self.maximum / self.knee), 0, 1))


class CameraMotionGuard:
    """Robust previous->current similarity transform; never alters model inputs.

    LK forward/backward tracks plus RANSAC estimate global background motion.
    The residual support is an extra change check, not smoke classification.
    Unreliable/rapid motion produces an invalid score, not a zero/negative result.
    """
    def __init__(self, history_frames=8, max_side=640, rotation_limit=60., translation_limit=.75,
                 residual_floor=.015):
        if type(history_frames) is not int or history_frames < 1 or max_side < 32:
            raise ValueError('Invalid camera-guard history or image size')
        if any(not math.isfinite(v) or v <= 0 for v in (rotation_limit, translation_limit, residual_floor)):
            raise ValueError('Camera-guard thresholds must be positive finite values')
        self.history_frames, self.max_side = history_frames, max_side
        self.rotation_limit, self.translation_limit = rotation_limit, translation_limit
        self.residual_floor = residual_floor
        self.previous = None
        self.recovery = 0

    def update(self, frame, fps, output_shape):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Camera guard requires positive finite FPS')
        h, w = frame.shape[:2]
        ratio = min(1., self.max_side / max(h, w))
        gray = cv2.cvtColor(cv2.resize(frame, (max(1, round(w * ratio)), max(1, round(h * ratio)))), cv2.COLOR_BGR2GRAY)
        previous, self.previous = self.previous, gray
        meta = {'motion_status': 'initial', 'motion_score_valid': False, 'rotation_deg_per_second': None,
                'translation_diagonal_per_second': None, 'motion_inlier_ratio': None,
                'motion_overlap': None, 'motion_residual_threshold': None, 'motion_support_fraction': None}
        support = np.ones(output_shape, dtype=np.float32)
        if previous is None or previous.shape != gray.shape:
            return support, meta
        try:
            support, details = self._estimate(previous, gray, fps)
            meta.update(details)
        except cv2.error:
            meta.update(motion_status='tracking_failed', motion_score_valid=False)
        if not meta['motion_score_valid']:
            self.recovery = self.history_frames
        elif self.recovery:
            self.recovery -= 1
            meta.update(motion_status='recovering_history', motion_score_valid=False)
        support = cv2.resize(support, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_LINEAR)
        meta['motion_support_fraction'] = float(support.mean())
        return support, meta

    def _estimate(self, previous, current, fps):
        h, w = current.shape
        transform = None
        ratio = None
        points = cv2.goodFeaturesToTrack(previous, maxCorners=400, qualityLevel=.01, minDistance=8, blockSize=7)
        if points is not None and len(points) >= 20:
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                                        winSize=(21, 21), maxLevel=3)
            if tracked is not None and status is not None:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, tracked, None,
                                                               winSize=(21, 21), maxLevel=3)
                if back is not None and back_status is not None:
                    valid = ((status[:, 0] > 0) & (back_status[:, 0] > 0)
                             & (np.linalg.norm(points[:, 0] - back[:, 0], axis=1) < 1.5))
                    a, b = points[valid, 0], tracked[valid, 0]
                    if len(a) >= 20:
                        transform, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                                         ransacReprojThreshold=2., maxIters=2000)
                        if transform is not None and inliers is not None:
                            ratio = float(inliers.mean())
        if transform is None:
            # Identity is allowed only for low global change, not an untracked pan.
            changed = float((cv2.absdiff(previous, current) > 5).mean())
            if changed > .15:
                return np.ones_like(current, dtype=np.float32), {'motion_status': 'tracking_failed',
                                                                 'motion_score_valid': False}
            transform = np.array([[1., 0., 0.], [0., 1., 0.]])
            status_name = 'low_global_change'
        else:
            status_name = 'compensated'
        if not np.isfinite(transform).all():
            return np.ones_like(current, dtype=np.float32), {'motion_status': 'tracking_failed', 'motion_score_valid': False}
        rotation = abs(math.degrees(math.atan2(transform[1, 0], transform[0, 0]))) * fps
        scale = math.hypot(transform[0, 0], transform[1, 0])
        center = np.array([w / 2, h / 2, 1.])
        displacement = np.linalg.norm(transform @ center - center[:2]) / math.hypot(h, w) * fps
        warped = cv2.warpAffine(previous, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.warpAffine(np.ones_like(previous), transform, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0
        overlap = float(valid.mean())
        quality = overlap >= .7 and (ratio is None or ratio >= .55) and .9 <= scale <= 1.1
        rapid = rotation > self.rotation_limit or displacement > self.translation_limit or abs(scale - 1) * fps > 1.
        if not quality or rapid:
            return np.ones_like(current, dtype=np.float32), {
                'motion_status': 'rapid_camera_motion' if rapid else 'unreliable_alignment',
                'motion_score_valid': False, 'rotation_deg_per_second': rotation,
                'translation_diagonal_per_second': float(displacement), 'motion_inlier_ratio': ratio,
                'motion_overlap': overlap}
        residual = cv2.absdiff(warped, current).astype(np.float32) / 255.
        values = residual[valid]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) * 1.4826
        threshold = max(self.residual_floor, median + 3 * mad)
        noise = max(1 / 255, median)
        support = np.clip((residual - noise) / max(threshold - noise, 1e-6), 0, 1)
        support = cv2.dilate(support, np.ones((3, 3), np.uint8)) * valid
        return support, {'motion_status': status_name, 'motion_score_valid': True,
                         'rotation_deg_per_second': rotation, 'translation_diagonal_per_second': float(displacement),
                         'motion_inlier_ratio': ratio, 'motion_overlap': overlap,
                         'motion_residual_threshold': threshold}
