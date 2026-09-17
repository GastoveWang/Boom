"""
==============================================================================
Boom 航拍定位 - GeoTIFF 航拍圖資讀取與座標轉換 (geo_map.py)
==============================================================================

【檔案定位】
本檔案負責地理圖資管理，包含 GeoTIFF 大圖的局部視窗讀取、最近子圖（submap）載入、
快取機制與大地座標系（TWD97 / WGS84）轉換。

【核心功能】
1. 局部航拍視窗讀取 (`read_roi`, `roi_window`)：
   - 根據給定經緯度範圍僅讀取所需的圖資區塊，大幅降低記憶體與 I/O 開銷。
2. 地圖拼貼與展示 (`GeoMap`)：
   - 管理離線航拍地圖面板底圖、地理邊界、每像素解析度 (mpp)，並提供像素至經緯度的雙向映射。
==============================================================================
"""
from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
from .regions import gps_bounds, transformer, GeoTIFFIndex, select_nearest_submap


@dataclass
class MapROI:
    tile: object
    image: np.ndarray
    mask: np.ndarray
    transform: object
    window: object

    def pixel_to_geo(self, points):
        """OpenCV pixel centers -> source CRS -> (longitude, latitude)."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        x, y = self.transform * (points[:, 0] + .5, points[:, 1] + .5)
        lon, lat = transformer(self.tile.crs, "EPSG:4326").transform(x, y)
        return np.column_stack((lon, lat))


def roi_window(tile, lon, lat, size_m, search_radius_m):
    """Clip a local ROI to both the tile and the prior's search envelope."""
    from rasterio.windows import Window
    # Center on the closest location in each candidate, never its filename.
    x, y = transformer("EPSG:4326", tile.crs).transform(lon, lat)
    left, bottom, right, top = tile.bounds
    x, y = min(max(x, left), right), min(max(y, bottom), top)
    anchor = transformer(tile.crs, "EPSG:4326").transform(x, y)
    a, b, c, d = gps_bounds(*anchor, size_m / 2)
    e, f, g, h = gps_bounds(lon, lat, search_radius_m)
    bounds = max(a, e), max(b, f), min(c, g), min(d, h)
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        return None
    a, b, c, d = transformer("EPSG:4326", tile.crs).transform_bounds(
        *bounds, densify_pts=21)
    pixels = [~tile.transform * p for p in ((a, b), (a, d), (c, b), (c, d))]
    x0 = max(0, math.floor(min(p[0] for p in pixels)))
    y0 = max(0, math.floor(min(p[1] for p in pixels)))
    x1 = min(tile.width, math.ceil(max(p[0] for p in pixels)))
    y1 = min(tile.height, math.ceil(max(p[1] for p in pixels)))
    return Window(x0, y0, x1-x0, y1-y0) if x1 > x0 and y1 > y0 else None


def read_roi(tile, window, max_image_size):
    """Window read with bounded output, nodata/alpha mask and exact scale."""
    import rasterio
    from rasterio.enums import Resampling, ColorInterp
    from affine import Affine
    scale = min(1., max_image_size / max(window.width, window.height))
    width, height = max(1, round(window.width*scale)), max(1, round(window.height*scale))
    with rasterio.open(tile.path) as src:
        colors = src.colorinterp
        if all(c in colors for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
            bands = [colors.index(c)+1 for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)]
        elif src.count == 1 or colors[0] == ColorInterp.gray:
            bands = [1, 1, 1]
        else:
            raise ValueError(f"Expected RGB or grayscale GeoTIFF: {tile.path}")
        data = src.read(bands, window=window, out_shape=(3, height, width),
                        resampling=Resampling.bilinear)
        valid = src.read_masks(bands, window=window, out_shape=(3, height, width),
                               resampling=Resampling.nearest).min(axis=0) > 0
        if ColorInterp.alpha in colors:
            alpha = src.read(colors.index(ColorInterp.alpha)+1, window=window,
                             out_shape=(height, width), resampling=Resampling.nearest)
            valid &= alpha > 0
        valid &= np.isfinite(data).all(axis=0)
        if data.dtype != np.uint8:
            converted = np.zeros(data.shape, np.uint8)
            for i, band in enumerate(data):
                if valid.any():
                    lo, hi = np.percentile(band[valid], (1, 99))
                    if hi > lo:
                        converted[i] = np.clip((np.nan_to_num(band)-lo)*255/(hi-lo), 0, 255).astype(np.uint8)
            data = converted
        image = np.moveaxis(data[::-1], 0, -1).copy()
        image[~valid] = 0
        affine = src.window_transform(window) * Affine.scale(window.width/width, window.height/height)
    return MapROI(tile, image, valid.astype(np.uint8)*255, affine, window)


class GeoMap:
    def __init__(self, root, photo, output, metres_per_pixel=0.75, nearby=None):
        from affine import Affine
        from rasterio.warp import reproject, Resampling
        from rasterio.windows import Window
        if not math.isfinite(metres_per_pixel) or metres_per_pixel <= 0:
            raise ValueError("metres_per_pixel must be finite and positive")
        index = nearby.index if nearby is not None else GeoTIFFIndex(root)
        self.lon, self.lat = photo.lon, photo.lat
        self.mx = 6378137 * math.cos(math.radians(self.lat)) * math.pi / 180
        self.my = 6378137 * math.pi / 180
        self.mpp = metres_per_pixel
        tiles, self.region_selection = select_nearest_submap(index.tiles, photo.lon, photo.lat)
        (output / "region_selection.json").write_text(
            json.dumps(self.region_selection, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[MAP] Region selection: " + json.dumps({
            "status": self.region_selection["status"],
            "selected": self.region_selection["selected_regions"],
        }, ensure_ascii=True), flush=True)
        extents = []
        for tile in tiles:
            extents.extend([self.to_en(tile.west, tile.south), self.to_en(tile.east, tile.north)])
        self.xmin = min(p[0] for p in extents)
        self.ymax = max(p[1] for p in extents)
        width = max(1, math.ceil((max(p[0] for p in extents)-self.xmin)/self.mpp))
        height = max(1, math.ceil((self.ymax-min(p[1] for p in extents))/self.mpp))
        if width*height > 30_000_000:
            raise ValueError("Map exceeds preview budget; increase --map-mpp")
        self.world_from_pixel = np.array([[self.mpp, 0, self.xmin], [0, -self.mpp, self.ymax], [0, 0, 1.]])
        fingerprint = repr(("submap-v1", self.lon, self.lat, self.mpp,
                            [(str(t.path.resolve()), t.path.stat().st_size,
                              t.path.stat().st_mtime_ns) for t in tiles]))
        cache_dir = output.parent / "_map_cache"
        cache_dir.mkdir(exist_ok=True)
        cache = cache_dir / (hashlib.sha256(fingerprint.encode()).hexdigest()[:24]+".npz")
        if cache.exists():
            with np.load(cache) as data:
                self.canvas, self.coverage = data["canvas"], data["coverage"]
            self.gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(output / "map_preview.jpg"), self.canvas)
            return
        self.canvas = np.full((height, width, 3), 32, np.uint8)
        self.coverage = np.zeros((height, width), np.uint8)
        target = Affine(self.mpp/self.mx, 0, self.lon+(self.xmin-self.mpp/2)/self.mx,
                        0, -self.mpp/self.my, self.lat+(self.ymax+self.mpp/2)/self.my)
        for tile in tiles:
            # Include every tile of the selected submap at display resolution.
            # Matching radius/candidate/ROI settings never limit map loading.
            a, b = np.floor(self.pixel(self.to_en(tile.west, tile.north))).astype(int)
            c, d = np.ceil(self.pixel(self.to_en(tile.east, tile.south))).astype(int) + 1
            a, b, c, d = max(a, 0), max(b, 0), min(c, width), min(d, height)
            roi = read_roi(tile, Window(0, 0, tile.width, tile.height), max(c-a, d-b))
            options = dict(src_transform=roi.transform, src_crs=roi.tile.crs,
                           dst_transform=target * Affine.translation(a, b), dst_crs="EPSG:4326")
            coverage = np.zeros((d-b, c-a), np.uint8)
            reproject(roi.mask, coverage, resampling=Resampling.nearest, **options)
            valid = coverage > 0
            for band in range(3):
                pixels = np.zeros_like(coverage)
                reproject(roi.image[:, :, band], pixels, resampling=Resampling.bilinear, **options)
                self.canvas[b:d, a:c, band][valid] = pixels[valid]
            self.coverage[b:d, a:c][valid] = 255
        self.gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        np.savez_compressed(cache, canvas=self.canvas, coverage=self.coverage)
        cv2.imwrite(str(output / "map_preview.jpg"), self.canvas)

    def matching_view(self, nearby, lon, lat):
        """Crop the loaded map for matching without shrinking its display extent."""
        cfg = nearby.config
        polygons = []
        for tile in nearby.index.nearby(lon, lat, cfg.search_radius_m, cfg.max_candidates,
                                       submap=self.region_selection["selected_submap"]):
            window = roi_window(tile, lon, lat, cfg.roi_size_m, cfg.search_radius_m)
            if window is None:
                continue
            x0, y0 = window.col_off, window.row_off
            x1, y1 = x0+window.width, y0+window.height
            points = []
            for pixel in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                gps = transformer(tile.crs, "EPSG:4326").transform(*(tile.transform * pixel))
                points.append(self.pixel(self.to_en(*gps)))
            polygons.append(np.array(points))
        if not polygons:
            return None
        points = np.concatenate(polygons)
        x0, y0 = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(points.max(axis=0)).astype(int)+1,
                            [self.gray.shape[1], self.gray.shape[0]])
        if x1 <= x0 or y1 <= y0:
            return None
        mask = np.zeros((y1-y0, x1-x0), np.uint8)
        for polygon in polygons:
            cv2.fillConvexPoly(mask, np.round(polygon-[x0, y0]).astype(np.int32), 255)
        mask = cv2.bitwise_and(mask, self.coverage[y0:y1, x0:x1])
        return SimpleNamespace(gray=self.gray[y0:y1, x0:x1], coverage=mask,
                               world_from_pixel=self.world_from_pixel @ np.array(
                                   [[1., 0, x0], [0, 1., y0], [0, 0, 1.]]))

    def to_en(self, lon, lat):
        return (lon-self.lon)*self.mx, (lat-self.lat)*self.my

    def to_gps(self, xy):
        return self.lon+float(xy[0])/self.mx, self.lat+float(xy[1])/self.my

    def pixel(self, xy):
        return np.array([(xy[0]-self.xmin)/self.mpp, (self.ymax-xy[1])/self.mpp])

    def contains(self, xy):
        x, y = self.pixel(xy)
        return (0 <= y < self.coverage.shape[0] and 0 <= x < self.coverage.shape[1]
                and self.coverage[int(y), int(x)] != 0)
