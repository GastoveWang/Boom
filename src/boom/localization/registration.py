# 檔案功能：參考照片／地圖配準與矩陣校正。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""
==============================================================================
Boom 航拍定位 - 照片/地圖配準與漂移校正 (registration.py)
==============================================================================

【檔案定位】
本檔案負責無人機空拍照片與 GeoTIFF 航拍地圖的全局配準、參考照片與即時影片影格的
特徵錨定（Photo/Video Linking），以及長時間飛行下視覺里程計累積漂移的週期性地圖校正。

【核心功能】
- `register_photo_map(...)`: 照片與地圖特徵比對，求得單應性矩陣。
- `link_photo_video(...)`: 將影片第一格與參考照片對齊。
- `correct_map_drift(...)`: 週期性比對當前影格與地圖，修正累積誤差。
==============================================================================
"""
from __future__ import annotations

import time
import cv2
import numpy as np

from .geometry import plane_homography, transform
from .match_output import geometric_quality, match_directory, write_match


def _report(localizer, stage, started, image, reference, a, b, h, selected,
            accepted, reason, quality=None):
    metrics = geometric_quality(a, b, h, selected, image.shape) if quality is None else dict(quality)
    metrics.update(accepted=bool(accepted), reason=reason,
                   elapsed_ms=(time.monotonic() - started) * 1000,
                   matcher_backend=localizer.nearby.config.matcher_backend)
    directory = match_directory(localizer.output, stage)
    data = write_match(directory, stage, image, reference, a, b, selected, metrics,
                       frame=getattr(localizer, "current_frame_idx", None),
                       time_sec=getattr(localizer, "current_time_sec", None))
    localizer.record(data)


def matches(localizer, a, b, mask_a=None, mask_b=None):
    if hasattr(localizer, "nearby"):
        backend = localizer.nearby.backend
        pa, pb, _ = backend.match(backend.extract(a, mask_a), backend.extract(b, mask_b))
        return pa, pb
    ka, da = localizer.sift.detectAndCompute(a, mask_a)
    kb, db = localizer.sift.detectAndCompute(b, mask_b)
    config = getattr(getattr(localizer, "nearby", None), "config", None)
    return match_features(localizer.matcher, (ka, da), (kb, db),
                          getattr(config, "ratio_threshold", .70))


def match_features(matcher, features_a, features_b, ratio_threshold=.70):
    """Shared SIFT matching, also accepting cached ROI descriptors."""
    ka, da = features_a
    kb, db = features_b
    if da is None or db is None:
        return np.empty((0, 2)), np.empty((0, 2))
    pairs = matcher.knnMatch(da, db, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < ratio_threshold*n.distance]
    # A destination descriptor may support only one correspondence.
    good = sorted(good, key=lambda m: m.distance)
    used, unique = set(), []
    for m in good:
        if m.trainIdx not in used:
            unique.append(m)
            used.add(m.trainIdx)
    return (np.array([ka[m.queryIdx].pt for m in unique], np.float64).reshape(-1, 2),
            np.array([kb[m.trainIdx].pt for m in unique], np.float64).reshape(-1, 2))


def register_reference(localizer):
    started = time.monotonic()
    # Rectify the oblique photo with the metadata prior before feature matching.
    view = localizer.map.matching_view(localizer.nearby, localizer.photo.lon, localizer.photo.lat)
    if view is None:
        localizer.reference_valid = False
        localizer.status = "REFERENCE_MATCH_FAILED"
        localizer.ref_center, localizer.ref_rotation = localizer.center.copy(), localizer.rotation.copy()
        localizer.record({"stage": "photo_initialization", "status": localizer.status,
                          "reason": "no_candidate_maps"})
        return
    to_map = np.linalg.inv(view.world_from_pixel) @ np.linalg.inv(localizer.ref_h)
    size = (view.gray.shape[1], view.gray.shape[0])
    warped = cv2.warpPerspective(localizer.ref_gray, to_map, size)
    mask = cv2.warpPerspective(np.full_like(localizer.ref_gray, 255), to_map, size)
    mask = cv2.erode(mask, np.ones((11, 11), np.uint8))
    mask = cv2.bitwise_and(mask, view.coverage)
    a, b = localizer.matches(warped, view.gray, mask, view.coverage)
    accepted = False
    inliers = 0
    correction, select, match_quality = localizer.nearby._verify(
        a, b, warped.shape, view.coverage)
    inliers = int(select.sum())
    if match_quality["accepted"]:
        photo_pts = transform(a[select], np.linalg.inv(to_map))
        world = transform(b[select], view.world_from_pixel)
        accepted = localizer.fit_pose(world, photo_pts, localizer.ref_k, max_shift=100.)
        if accepted:
            localizer.ref_h = plane_homography(localizer.ref_k, localizer.rotation, localizer.center)
    _report(localizer, "map_match_rectified", started, warped, view.gray, a, b,
            correction, select, accepted,
            "accepted" if accepted else ("pose_rejected" if match_quality["accepted"] else match_quality["reason"]),
            quality=match_quality)
    localizer.reference_valid = accepted
    localizer.status = "REFERENCE_MATCHED" if accepted else "REFERENCE_MATCH_FAILED"
    if localizer.nearby.config.debug:
        cv2.imwrite(str(localizer.output / "reference_ground_projection.jpg"), warped)
    localizer.record({"stage": "photo_initialization", "status": localizer.status,
                 "matches": len(a), "inliers": inliers, "gps": [localizer.photo.lon, localizer.photo.lat],
                 "assumptions": "flat ground; RelativeAltitude above ground; EXIF equivalent focal; centered video crop",
                 "quality": localizer.quality, "match_quality": match_quality,
                 "heading_deg": localizer.heading_deg,
                 "heading_source": "visual_pose_camera_aligned_with_body" if accepted else None,
                 "submap": localizer.nearby.submap})
    print(f"[LOCALIZATION] {localizer.status}: matches={len(a)}, inliers={inliers}", flush=True)
    localizer.ref_center = localizer.center.copy()
    localizer.ref_rotation = localizer.rotation.copy()


def reference_link(localizer, gray, k):
    started = time.monotonic()
    a, b = localizer.matches(localizer.ref_gray, gray)
    h = None
    selected = np.zeros(len(a), bool)
    def finish(accepted, reason):
        _report(localizer, "photo_video_match", started, localizer.ref_gray, gray,
                a, b, h, selected, accepted, reason)
        return accepted
    if len(a) < 16:
        return finish(False, "insufficient_matches")
    h, good = cv2.findHomography(a, b, cv2.RANSAC, 3.)
    if good is not None:
        selected = good.ravel() != 0
    if h is None or good is None or good.sum() < 16:
        return finish(False, "insufficient_inliers")
    world = transform(a[selected], np.linalg.inv(localizer.ref_h))
    valid = np.array([localizer.map.contains(p) for p in world])
    if localizer.video_focal_scale is not None:
        accepted = localizer.fit_pose(world[valid], b[selected][valid], k, max_shift=150., previous=localizer.ref_center)
        return finish(accepted, "accepted" if accepted else "pose_rejected")
    # A video may additionally crop horizontally. Fit a bounded focal-length
    # search against the photo-ground correspondences, then keep it fixed.
    best = None
    for factor in np.linspace(.9, 1.2, 16):
        candidate = k.copy()
        candidate[0, 0] *= factor
        candidate[1, 1] *= factor
        if localizer.fit_pose(world[valid], b[selected][valid], candidate, max_shift=150., previous=localizer.ref_center):
            score = localizer.quality["inliers"]
            if best is None or score > best[0]:
                best = (score, factor, localizer.center.copy(), localizer.rotation.copy(), localizer.quality.copy())
    if best is None:
        localizer.center, localizer.rotation = localizer.ref_center.copy(), localizer.ref_rotation.copy()
        return finish(False, "pose_rejected")
    _, localizer.video_focal_scale, localizer.center, localizer.rotation, localizer.quality = best
    k[0, 0] *= localizer.video_focal_scale
    k[1, 1] *= localizer.video_focal_scale
    localizer.record({"stage": "photo_video_link", "focal_scale": localizer.video_focal_scale,
                 "quality": localizer.quality, "map_alignment": "visual" if localizer.reference_valid else "gps_metadata_only"})
    return finish(True, "accepted")


def map_correction(localizer, gray, k):
    started = time.monotonic()
    lon, lat = localizer.map.to_gps(localizer.center)
    view = localizer.map.matching_view(localizer.nearby, lon, lat)
    if view is None:
        return False
    to_map = np.linalg.inv(view.world_from_pixel) @ np.linalg.inv(plane_homography(k, localizer.rotation, localizer.center))
    size = (view.gray.shape[1], view.gray.shape[0])
    warped = cv2.warpPerspective(gray, to_map, size)
    valid_image = np.zeros_like(gray)
    valid_image[int(gray.shape[0]*.35):] = 255
    mask = cv2.warpPerspective(valid_image, to_map, size)
    mask = cv2.bitwise_and(cv2.erode(mask, np.ones((11, 11), np.uint8)), view.coverage)
    a, b = localizer.matches(warped, view.gray, mask, view.coverage)
    correction = None
    select = np.zeros(len(a), bool)
    def finish(accepted, reason):
        _report(localizer, "map_correction", started, warped, view.gray, a, b,
                correction, select, accepted, reason)
        return accepted
    if len(a) < 16:
        return finish(False, "insufficient_matches")
    correction, good = cv2.findHomography(a, b, cv2.RANSAC, 3.)
    if good is not None:
        select = good.ravel() != 0
    if correction is None or good is None or good.sum() < 16 or good.mean() < .25:
        return finish(False, "insufficient_inliers")
    accepted = localizer.fit_pose(transform(b[select], view.world_from_pixel),
                         transform(a[select], np.linalg.inv(to_map)), k, max_shift=15.)
    return finish(accepted, "accepted" if accepted else "pose_rejected")
