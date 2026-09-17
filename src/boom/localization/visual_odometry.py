"""
==============================================================================
Boom 航拍定位 - 視覺里程計與逐格相機姿態追蹤 (visual_odometry.py)
==============================================================================

【檔案定位】
本檔案實作輕量化單目視覺里程計 (Visual Odometry, VO)。在飛行過程中透過連續影格
之間的地面特徵點光流追蹤，估算無人機相機相對於上一刻的姿態旋轉矩陣與三維平移向量。

【核心功能】
- `replenish(...)`: 當追蹤特徵點數量不足時，在地面區域重新補充 Harris / Shi-Tomasi 角點。
- `track_frame(...)`: 利用 Lucas-Kanade 光流金字塔追蹤特徵點，求解本影格的相機姿態與無人機軌跡。
==============================================================================
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import plane_homography, transform
from .photo import resize_work


def replenish(localizer, gray, k):
    mask = np.zeros_like(gray)
    mask[int(gray.shape[0]*.35):] = 255
    points = cv2.goodFeaturesToTrack(gray, 900, .015, 14, mask=mask)
    if points is None:
        localizer.points = localizer.world_points = None
        return
    points = points.reshape(-1, 2)
    inverse = np.linalg.inv(plane_homography(k, localizer.rotation, localizer.center))
    world = transform(points, inverse)
    positive_depth = np.column_stack((points, np.ones(len(points)))) @ inverse[2] > 0
    valid = positive_depth & np.array([localizer.map.contains(p) and np.linalg.norm(p-localizer.center[:2]) < 1800 for p in world])
    localizer.points = points[valid].astype(np.float32)
    localizer.world_points = world[valid]


def process(localizer, frame, frame_idx, fps):
    small, scale = resize_work(frame)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    k = localizer.photo.camera_matrix(gray.shape[1], gray.shape[0])
    if localizer.video_focal_scale is not None:
        k[0, 0] *= localizer.video_focal_scale
        k[1, 1] *= localizer.video_focal_scale
    valid = False
    corrected = False
    if localizer.reference_valid or localizer.allow_gps_seed:
        if localizer.prev_gray is None or localizer.status == "LOST":
            if frame_idx-localizer.last_relocalization >= max(int(fps), 1):
                localizer.last_relocalization = frame_idx
                valid = localizer.reference_link(gray, k)
                if valid:
                    localizer.last_map_fix = frame_idx
                    localizer.points = None
        elif localizer.points is not None and len(localizer.points) >= 16:
            q, ok, _ = cv2.calcOpticalFlowPyrLK(localizer.prev_gray, gray, localizer.points, None,
                                                winSize=(31, 31), maxLevel=3)
            back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, localizer.prev_gray, q, None,
                                                      winSize=(31, 31), maxLevel=3)
            good = (ok.ravel()!=0) & (back_ok.ravel()!=0) & (np.linalg.norm(back-localizer.points, axis=1)<1.)
            valid = localizer.fit_pose(localizer.world_points[good], q[good], k, max_shift=max(2., 40./fps))
            if valid:
                localizer.points, localizer.world_points = q[good], localizer.world_points[good]
        if valid and frame_idx-localizer.last_map_check >= int(fps*5):
            localizer.last_map_check = frame_idx
            corrected = localizer.map_correction(gray, k)
            if corrected:
                localizer.last_map_fix = frame_idx
                localizer.points = None
        if valid and localizer.reference_valid and localizer.last_map_fix is not None and frame_idx-localizer.last_map_fix > int(fps*15):
            valid = False
        localizer.status = ("VO_TRACKING" if localizer.reference_valid else "VO_GPS_ESTIMATE") if valid else "LOST"
    if valid:
        localizer.trajectory.append(localizer.center[:2].copy())
        h = plane_homography(k, localizer.rotation, localizer.center)
        localizer.history[frame_idx] = (np.linalg.inv(h) @ np.diag([scale, scale, 1.]), localizer.center.copy())
        if localizer.points is None or len(localizer.points) < 250 or frame_idx % 30 == 0:
            localizer.replenish(gray, k)
    else:
        localizer.history[frame_idx] = None
    # Bound frame history while retaining more than detector confirmation windows.
    cutoff = frame_idx-max(int(fps*30), 300)
    for old in list(localizer.history):
        if old < cutoff:
            del localizer.history[old]
    localizer.prev_gray = gray
    localizer.record({"stage": "video", "frame": frame_idx, "time_sec": frame_idx/fps,
                 "status": localizer.status, "position_gps": localizer.map.to_gps(localizer.center) if valid else None,
                 "camera_center_enh_m": localizer.center.tolist() if valid else None,
                 "heading_deg": localizer.heading_deg if valid else None,
                 "heading_source": "visual_pose_camera_aligned_with_body" if valid and localizer.reference_valid else None,
                 "map_corrected": corrected,
                 "map_alignment": "visual" if localizer.reference_valid else "gps_metadata_only",
                 "seconds_since_map_fix": (frame_idx-localizer.last_map_fix)/fps if localizer.last_map_fix is not None else None,
                 "quality": localizer.quality if valid else None})
