"""
==============================================================================
Boom 航拍定位 - 局部 ROI 地圖特徵比對與品質驗證 (nearby.py)
==============================================================================

【檔案定位】
本檔案實作基於先驗位置（如 GPS 或前一格估計姿態）的局部航拍地圖特徵比對。
獨立於具體相機與視訊 I/O，專注於演算法驗證與 ROI 特徵匹配。

【核心功能】
1. 局部航拍 ROI 提取：
   - 根據無人機位置與搜尋半徑，從 GeoTIFF 圖資中僅擷取感興趣區域（ROI），避免載入整張巨大地圖。
2. 特徵匹配與幾何品質檢核 (`NearbyMapLocalizer`)：
   - 調用 SuperPoint + LightGlue 或 SIFT 進行特徵比對。
   - 計算 RANSAC 單應性變換矩陣，驗證內點數量 (`min_inliers`)、內點比例 (`min_inlier_ratio`)、
     重投影誤差 (`max_reprojection_error_px`) 與視角覆蓋率 (`min_coverage`)。
3. 像素到 GPS 映射 (`MapMatchResult`)：
   - 提供成功配準後的影像像素至真實經緯度轉換函式。
==============================================================================
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from boom.config.defaults import MapMatchingConfig
from .geo_map import read_roi, roi_window
from .geometry import transform
from .regions import GeoTIFFIndex, gps_bounds, select_nearest_submap
from .feature_matching import create_feature_backend


@dataclass
class MapMatchResult:
    success: bool
    reason: str
    candidates: list = field(default_factory=list)
    roi: object = None
    homography: object = None
    image_shape: tuple = ()
    image_points: object = None
    map_points: object = None
    inliers: object = None
    quality: dict = field(default_factory=dict)
    match_scores: object = None
    submap: str = ""

    def pixel_to_geo(self, u, v):
        """Return (lon, lat), or None for failed/invalid/uncovered pixels.

        u/v refer to the ORIGINAL input frame. This is a ground-map point,
        not the camera's 3D location.
        """
        if not self.success or not np.isfinite([u, v]).all():
            return None
        height, width = self.image_shape[:2]
        if not (0 <= u < width and 0 <= v < height):
            return None
        projected = self.homography @ np.array([u, v, 1.])
        if abs(projected[2]) < 1e-10:
            return None
        x, y = projected[:2] / projected[2]
        h, w = self.roi.mask.shape
        if not np.isfinite([x, y]).all() or not (0 <= x < w and 0 <= y < h):
            return None
        if not self.roi.mask[min(round(y), h-1), min(round(x), w-1)]:
            return None
        geo = self.roi.pixel_to_geo([(x, y)])[0]
        return tuple(map(float, geo)) if np.isfinite(geo).all() else None


class NearbyMapLocalizer:
    """Initialize once; reuse index, extractor, matcher and bounded ROI cache.

    SuperPoint + LightGlue is the default. OpenCV injections retain the legacy
    test/extension contract. All registration paths share this backend.
    """

    def __init__(self, root, config=None, *, index=None, extractor=None, matcher=None):
        self.config = config or MapMatchingConfig()
        self.index = index if index is not None else GeoTIFFIndex(root)
        self.backend = create_feature_backend(self.config, extractor, matcher)
        self.sift = self.backend.extractor  # compatibility attributes
        self.matcher = self.backend.matcher
        self.submap = None
        self.cache = OrderedDict()

    def candidate_rois(self, longitude, latitude, search_radius_m=None):
        radius = self.config.search_radius_m if search_radius_m is None else search_radius_m
        submap = self.submap
        if submap is None:
            _, selection = select_nearest_submap(self.index.tiles, longitude, latitude)
            submap = selection["selected_submap"]
        tiles = self.index.nearby(longitude, latitude, radius, self.config.max_candidates, submap=submap)
        for tile in tiles:
            window = roi_window(tile, longitude, latitude, self.config.roi_size_m, radius)
            if window is None:
                continue
            key = (tile.path, window.col_off, window.row_off, window.width,
                   window.height, self.config.max_image_size)
            if key not in self.cache:
                self.cache[key] = [read_roi(tile, window, self.config.max_image_size), None]
            self.cache.move_to_end(key)
            while len(self.cache) > self.config.cache_entries:
                self.cache.popitem(last=False)
            yield self.cache[key]

    def localize(self, frame, longitude, latitude, *, search_radius_m=None, debug_dir=None):
        radius = self.config.search_radius_m if search_radius_m is None else search_radius_m
        gps_bounds(longitude, latitude, radius)  # validate even for featureless frames
        if self.config.debug and debug_dir is None:
            raise ValueError("debug_dir is required when debug=True")
        if frame is None or frame.dtype != np.uint8 or frame.ndim not in (2, 3):
            raise ValueError("frame must be a uint8 grayscale or BGR image")
        if frame.ndim == 3 and frame.shape[2] != 3:
            raise ValueError("frame must have three BGR channels")
        height, width = frame.shape[:2]
        if min(height, width) < 2:
            raise ValueError("frame is too small")
        scale = min(1., self.config.max_image_size / max(height, width))
        work = cv2.resize(frame, (max(2, round(width*scale)), max(2, round(height*scale))))
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
        features = self.backend.extract(gray)
        # OpenCV resize maps pixel centers, not raster outer corners.
        sx, sy = work.shape[1]/width, work.shape[0]/height
        to_work = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])
        reports, best = [], None
        debug_path = None
        if self.config.debug:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            debug_path = Path(tempfile.mkdtemp(prefix="map_match_", dir=debug_dir))
            cv2.imwrite(str(debug_path / "uav.jpg"), frame)
        for number, entry in enumerate(self.candidate_rois(longitude, latitude, radius)):
            roi, map_features = entry
            if map_features is None:
                map_gray = cv2.cvtColor(roi.image, cv2.COLOR_BGR2GRAY)
                mask = cv2.erode(roi.mask, np.ones((5, 5), np.uint8))
                map_features = self.backend.extract(map_gray, mask)
                entry[1] = map_features
            a, b, scores = self.backend.match(features, map_features)
            h, selected, quality = self._verify(a, b, gray.shape, roi.mask)
            quality["matcher_backend"] = self.config.matcher_backend
            quality["mean_match_confidence"] = float(np.mean(scores)) if scores is not None and len(scores) else None
            report = {"filepath": str(roi.tile.path), **quality}
            reports.append(report)
            if debug_path:
                self._debug(debug_path, number, work, roi, a, b, h, selected)
            if not quality["accepted"]:
                continue
            original_points = transform(a, np.linalg.inv(to_work))
            result = MapMatchResult(True, "matched", roi=roi, homography=h @ to_work,
                                    image_shape=frame.shape, image_points=original_points,
                                    map_points=b, inliers=selected, quality=quality,
                                    match_scores=scores, submap=roi.tile.region)
            # A plausible match must map the frame center into valid map data.
            center_geo = result.pixel_to_geo((width-1)/2, (height-1)/2)
            if center_geo is None:
                report.update(accepted=False, reason="center_outside_coverage")
                continue
            from pyproj import Geod
            distance = Geod(ellps="WGS84").inv(longitude, latitude, *center_geo)[2]
            if distance > radius:
                report.update(accepted=False, reason="outside_search_radius")
                continue
            if best is None or quality["score"] > best.quality["score"]:
                best = result
        result = best or MapMatchResult(False, "quality_rejected" if reports else "no_candidate_maps")
        result.candidates = reports
        if debug_path:
            (debug_path / "quality.json").write_text(json.dumps(
                {"success": result.success, "reason": result.reason, "candidates": reports},
                indent=2, allow_nan=False), encoding="utf-8")
        return result

    def _verify(self, a, b, shape, mask):
        cfg = self.config
        quality = {"matches": len(a), "inliers": 0, "inlier_ratio": 0.,
                   "reprojection_error_px": None, "confidence": 0., "score": 0.,
                   "accepted": False, "reason": "insufficient_matches"}
        selected = np.zeros(len(a), bool)
        if len(a) < cfg.min_inliers:
            return None, selected, quality
        h, inliers = cv2.findHomography(a, b, cv2.RANSAC, cfg.ransac_threshold_px)
        if h is None or inliers is None or not np.isfinite(h).all():
            quality["reason"] = "homography_failed"
            return None, selected, quality
        selected = inliers.ravel() != 0
        count = int(selected.sum())
        ratio = count / len(a)
        quality.update(inliers=count, inlier_ratio=ratio, reason="insufficient_inliers")
        if count < cfg.min_inliers or ratio < cfg.min_inlier_ratio or abs(np.linalg.det(h)) < 1e-12:
            return h, selected, quality
        error = float(np.sqrt(np.mean(np.sum((transform(a[selected], h)-b[selected])**2, axis=1))))
        quality["reprojection_error_px"] = error if np.isfinite(error) else None
        height, width = shape
        coverage = cv2.contourArea(cv2.convexHull(a[selected].astype(np.float32))) / (width*height)
        corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float64)
        denominators = np.column_stack((corners, np.ones(4))) @ h[2]
        footprint = transform(corners, h).astype(np.float32)
        valid_geometry = (np.isfinite(footprint).all() and np.all(np.abs(denominators) > 1e-8)
                          and (np.all(denominators > 0) or np.all(denominators < 0))
                          and cv2.isContourConvex(footprint)
                          and 16 < abs(cv2.contourArea(footprint)) <= mask.size*4)
        accepted = bool(valid_geometry and coverage >= cfg.min_coverage
                        and np.isfinite(error) and error <= cfg.max_reprojection_error_px)
        confidence = ratio * min(1., count/(2*cfg.min_inliers)) / (1+error) if np.isfinite(error) else 0.
        quality.update(coverage=float(coverage), confidence=float(confidence),
                       score=float(count*confidence), accepted=accepted,
                       reason="accepted" if accepted else "geometry_rejected")
        return h, selected, quality

    @staticmethod
    def pixel_to_geo(result, u, v):
        return result.pixel_to_geo(u, v)

    @staticmethod
    def _debug(directory, number, frame, roi, a, b, h, selected):
        cv2.imwrite(str(directory / f"{number:02d}_roi.jpg"), roi.image)
        ka = [cv2.KeyPoint(float(x), float(y), 1) for x, y in a]
        kb = [cv2.KeyPoint(float(x), float(y), 1) for x, y in b]
        pairs = [cv2.DMatch(i, i, 0) for i in range(len(a))]
        for name, selection in (("matches", None), ("inliers", selected.astype(int).tolist())):
            canvas = cv2.drawMatches(frame, ka, roi.image, kb, pairs, None,
                                     matchesMask=selection, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(str(directory / f"{number:02d}_{name}.jpg"), canvas)
        footprint = roi.image.copy()
        if h is not None:
            height, width = frame.shape[:2]
            points = transform([(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)], h)
            if np.isfinite(points).all() and np.abs(points).max() < 1e7:
                cv2.polylines(footprint, [np.round(points).astype(np.int32)], True, (0, 255, 255), 2)
        cv2.imwrite(str(directory / f"{number:02d}_footprint.jpg"), footprint)
