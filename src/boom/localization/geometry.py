"""
==============================================================================
Boom 航拍定位 - 相機幾何模型與地平投影 (geometry.py)
==============================================================================

【檔案定位】
本檔案包含相機三維外參/內參幾何計算、地平面單應性變換（Plane Homography）、
PnP 相機姿態估計、以及相機水平航向角（Heading/Bearing）的三角推導。

【核心功能】
- `camera_heading(rotation)`: 從相機旋轉矩陣解算無人機光軸相對於真北的水平方位角。
- `plane_homography(k, rotation, center)`: 構建相機成像平面到局部地平面的單應性變換矩陣 H。
- `estimate_pose(...)`: 利用 SolvePnPRansac 從 2D-3D 特徵對中估算相機外參矩陣。
- `transform(points, matrix)`: 齊次座標二維透視投影變換。
==============================================================================
"""
from __future__ import annotations

import cv2
import numpy as np


def camera_heading(rotation):
    """Horizontal optical-axis bearing: north=0, east=90; nadir is undefined.

    Equals drone heading only under the configured physical assumption that
    the camera and aircraft face the same horizontal direction.
    """
    forward = np.asarray(rotation).T[:, 2][:2]
    if not np.isfinite(forward).all() or np.linalg.norm(forward) < 1e-5:
        return None
    return float(np.degrees(np.arctan2(forward[0], forward[1])) % 360)


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
    if ok and indices is not None and len(indices) >= 12 and len(indices)/len(world) >= .45:
        result = _refine_pose(xyz, pixels, k, rv, tv, indices.ravel(), max_shift,
                              previous_center, reference_height)
        if result is not None:
            return result
    # Minimal PnP RANSAC samples can be unstable on the all-planar map points.
    # Verify correspondences with a homography, then evaluate both IPPE poses.
    h, inliers = cv2.findHomography(world, pixels, cv2.RANSAC, 3.)
    if h is None or inliers is None:
        return None
    idx = np.flatnonzero(inliers.ravel())
    if len(idx) < 12 or len(idx)/len(world) < .45:
        return None
    ok, rotations, translations, _ = cv2.solvePnPGeneric(
        xyz[idx], pixels[idx].astype(np.float64), k, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    candidates = [_refine_pose(xyz, pixels, k, rv, tv, idx, max_shift,
                               previous_center, reference_height)
                  for rv, tv in zip(rotations, translations)]
    valid = [candidate for candidate in candidates if candidate is not None]
    return min(valid, key=lambda result: result[2]["median_reprojection_px"]) if valid else None


def _refine_pose(xyz, pixels, k, rv, tv, idx, max_shift, previous_center, reference_height):
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
    quality = {"inliers": len(idx), "total": len(xyz), "median_reprojection_px": err,
               "heading_deg": camera_heading(rotation)}
    return rotation, center, quality
