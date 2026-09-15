"""Ground-plane projection and validated PnP pose estimation."""
from __future__ import annotations

import cv2
import numpy as np


def plane_homography(k, rotation, center):
    return k @ np.column_stack((rotation[:, 0], rotation[:, 1], -rotation @ center))


def transform(points, matrix):
    if len(points) == 0:
        return np.empty((0, 2), np.float64)
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), matrix).reshape(-1, 2)


def estimate_pose(world, pixels, k, max_shift, previous_center, reference_height):
    if len(world) < 12:
        return None
    xyz = np.column_stack((world, np.zeros(len(world)))).astype(np.float64)
    ok, rv, tv, indices = cv2.solvePnPRansac(xyz, pixels.astype(np.float64), k, None,
                                           iterationsCount=200, reprojectionError=3., confidence=.999,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or indices is None or len(indices) < 12 or len(indices)/len(world) < .45:
        return None
    idx = indices.ravel()
    rv, tv = cv2.solvePnPRefineLM(xyz[idx], pixels[idx], k, None, rv, tv)
    rotation = cv2.Rodrigues(rv)[0]
    center = (-rotation.T @ tv).ravel()
    projected = cv2.projectPoints(xyz[idx], rv, tv, k, None)[0].reshape(-1, 2)
    err = float(np.median(np.linalg.norm(projected-pixels[idx], axis=1)))
    span = np.ptp(pixels[idx], axis=0)
    if (not np.all(np.isfinite(center)) or not 10 < center[2] < reference_height*3
            or err > 2.5 or min(span) < 60
            or np.linalg.norm(center-previous_center) > max_shift
            or np.mean((rotation @ (xyz[idx]-center).T)[2] > 0) < .95):
        return None
    quality = {"inliers": len(idx), "total": len(world), "median_reprojection_px": err}
    return rotation, center, quality
