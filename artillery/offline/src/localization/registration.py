"""Photo/map registration, photo/video linking and map drift correction."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import plane_homography, transform


def matches(localizer, a, b, mask_a=None, mask_b=None):
    ka, da = localizer.sift.detectAndCompute(a, mask_a)
    kb, db = localizer.sift.detectAndCompute(b, mask_b)
    if da is None or db is None:
        return np.empty((0, 2)), np.empty((0, 2))
    pairs = localizer.matcher.knnMatch(da, db, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < .70*n.distance]
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
    # Rectify the oblique photo with the metadata prior before feature matching.
    to_map = np.linalg.inv(localizer.map.world_from_pixel) @ np.linalg.inv(localizer.ref_h)
    size = (localizer.map.canvas.shape[1], localizer.map.canvas.shape[0])
    warped = cv2.warpPerspective(localizer.ref_gray, to_map, size)
    mask = cv2.warpPerspective(np.full_like(localizer.ref_gray, 255), to_map, size)
    mask = cv2.erode(mask, np.ones((11, 11), np.uint8))
    mask = cv2.bitwise_and(mask, localizer.map.coverage)
    a, b = localizer.matches(warped, localizer.map.gray, mask, localizer.map.coverage)
    accepted = False
    inliers = 0
    if len(a) >= 16:
        correction, good = cv2.findHomography(a, b, cv2.RANSAC, 3.)
        if correction is not None and good is not None:
            select = good.ravel().astype(bool)
            inliers = int(select.sum())
            if inliers >= 16 and inliers/len(a) >= .25:
                photo_pts = transform(a[select], np.linalg.inv(to_map))
                world = transform(b[select], localizer.map.world_from_pixel)
                accepted = localizer.fit_pose(world, photo_pts, localizer.ref_k, max_shift=100.)
                if accepted:
                    localizer.ref_h = plane_homography(localizer.ref_k, localizer.rotation, localizer.center)
    localizer.reference_valid = accepted
    localizer.status = "REFERENCE_MATCHED" if accepted else "REFERENCE_MATCH_FAILED"
    cv2.imwrite(str(localizer.output / "reference_ground_projection.jpg"), warped)
    localizer.record({"stage": "photo_initialization", "status": localizer.status,
                 "matches": len(a), "inliers": inliers, "gps": [localizer.photo.lon, localizer.photo.lat],
                 "assumptions": "flat ground; RelativeAltitude above ground; EXIF equivalent focal; centered video crop",
                 "quality": localizer.quality})
    print(f"[LOCALIZATION] {localizer.status}: matches={len(a)}, inliers={inliers}", flush=True)
    localizer.ref_center = localizer.center.copy()
    localizer.ref_rotation = localizer.rotation.copy()


def reference_link(localizer, gray, k):
    a, b = localizer.matches(localizer.ref_gray, gray)
    if len(a) < 16:
        return False
    h, good = cv2.findHomography(a, b, cv2.RANSAC, 3.)
    if h is None or good is None or good.sum() < 16:
        return False
    a, b = a[good.ravel() != 0], b[good.ravel() != 0]
    world = transform(a, np.linalg.inv(localizer.ref_h))
    valid = np.array([localizer.map.contains(p) for p in world])
    if localizer.video_focal_scale is not None:
        return localizer.fit_pose(world[valid], b[valid], k, max_shift=150., previous=localizer.ref_center)
    # A video may additionally crop horizontally. Fit a bounded focal-length
    # search against the photo-ground correspondences, then keep it fixed.
    best = None
    for factor in np.linspace(.9, 1.2, 16):
        candidate = k.copy()
        candidate[0, 0] *= factor
        candidate[1, 1] *= factor
        if localizer.fit_pose(world[valid], b[valid], candidate, max_shift=150., previous=localizer.ref_center):
            score = localizer.quality["inliers"]
            if best is None or score > best[0]:
                best = (score, factor, localizer.center.copy(), localizer.rotation.copy(), localizer.quality.copy())
    if best is None:
        localizer.center, localizer.rotation = localizer.ref_center.copy(), localizer.ref_rotation.copy()
        return False
    _, localizer.video_focal_scale, localizer.center, localizer.rotation, localizer.quality = best
    k[0, 0] *= localizer.video_focal_scale
    k[1, 1] *= localizer.video_focal_scale
    localizer.record({"stage": "photo_video_link", "focal_scale": localizer.video_focal_scale,
                 "quality": localizer.quality, "map_alignment": "visual" if localizer.reference_valid else "gps_metadata_only"})
    return True


def map_correction(localizer, gray, k):
    to_map = np.linalg.inv(localizer.map.world_from_pixel) @ np.linalg.inv(plane_homography(k, localizer.rotation, localizer.center))
    size = (localizer.map.canvas.shape[1], localizer.map.canvas.shape[0])
    warped = cv2.warpPerspective(gray, to_map, size)
    valid_image = np.zeros_like(gray)
    valid_image[int(gray.shape[0]*.35):] = 255
    mask = cv2.warpPerspective(valid_image, to_map, size)
    mask = cv2.bitwise_and(cv2.erode(mask, np.ones((11, 11), np.uint8)), localizer.map.coverage)
    a, b = localizer.matches(warped, localizer.map.gray, mask, localizer.map.coverage)
    if len(a) < 16:
        return False
    correction, good = cv2.findHomography(a, b, cv2.RANSAC, 3.)
    if correction is None or good is None or good.sum() < 16 or good.mean() < .25:
        return False
    select = good.ravel() != 0
    return localizer.fit_pose(transform(b[select], localizer.map.world_from_pixel),
                         transform(a[select], np.linalg.inv(to_map)), k, max_shift=15.)
