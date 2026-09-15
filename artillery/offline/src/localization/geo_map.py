"""WGS84 GeoTIFF loading, alpha-aware mosaics, cache and local coordinates."""
from __future__ import annotations

import hashlib
import json
import math

import cv2
import numpy as np
from PIL import Image
from .regions import select_map_tiles


class GeoMap:
    def __init__(self, root, photo, output, metres_per_pixel=0.75):
        self.lon, self.lat = photo.lon, photo.lat
        self.mx = 6378137 * math.cos(math.radians(self.lat)) * math.pi / 180
        self.my = 6378137 * math.pi / 180
        self.mpp = metres_per_pixel
        selected_tiles, self.region_selection = select_map_tiles(root, photo)
        (output / "region_selection.json").write_text(
            json.dumps(self.region_selection, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[MAP] Region selection: " + json.dumps({
            "status": self.region_selection["status"],
            "selected": self.region_selection["selected_regions"],
        }, ensure_ascii=True), flush=True)
        tiles = []
        for tile in selected_tiles:
            x0, y1 = self.to_en(tile.west, tile.north)
            x1, y0 = self.to_en(tile.east, tile.south)
            tiles.append((tile.path, x0, y0, x1, y1))
        self.xmin = min(t[1] for t in tiles)
        self.ymax = max(t[4] for t in tiles)
        width = math.ceil((max(t[3] for t in tiles)-self.xmin)/self.mpp)
        height = math.ceil((self.ymax-min(t[2] for t in tiles))/self.mpp)
        if width*height > 30_000_000:
            raise ValueError("Map exceeds preview budget; increase --map-mpp")
        fingerprint = repr(("rgba-v2", self.lon, self.lat, self.mpp,
                            [(str(t[0].resolve()), t[0].stat().st_size, t[0].stat().st_mtime_ns) for t in tiles]))
        cache_dir = output.parent / "_map_cache"
        cache_dir.mkdir(exist_ok=True)
        cache = cache_dir / (hashlib.sha256(fingerprint.encode()).hexdigest()[:24]+".npz")
        self.world_from_pixel = np.array([[self.mpp, 0, self.xmin], [0, -self.mpp, self.ymax], [0, 0, 1.]])
        if cache.exists():
            with np.load(cache) as data:
                self.canvas, self.coverage = data["canvas"], data["coverage"]
            self.gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(output / "map_preview.jpg"), self.canvas)
            print("[MAP] Using georeferenced preview cache", flush=True)
            return
        self.canvas = np.full((height, width, 3), 32, np.uint8)
        self.coverage = np.zeros((height, width), np.uint8)
        for i, (path, x0, y0, x1, y1) in enumerate(tiles):
            a, b = self.pixel((x0, y1)).astype(int)
            c, d = np.ceil(self.pixel((x1, y0))).astype(int)
            c, d = min(c, width), min(d, height)
            with Image.open(path) as im:
                rgba = np.array(im.convert("RGBA").resize((c-a, d-b), Image.Resampling.BILINEAR))
                small = rgba[:, :, :3][:, :, ::-1]
            valid = rgba[:, :, 3] > 240
            region = self.canvas[b:d, a:c]
            region[valid] = small[valid]
            self.coverage[b:d, a:c][valid] = 255
            if i % 100 == 0:
                print(f"[MAP] Loading tile {i+1}/{len(tiles)}", flush=True)
        self.gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        np.savez_compressed(cache, canvas=self.canvas, coverage=self.coverage)
        cv2.imwrite(str(output / "map_preview.jpg"), self.canvas)

    def to_en(self, lon, lat):
        return (lon-self.lon)*self.mx, (lat-self.lat)*self.my

    def to_gps(self, xy):
        return self.lon+float(xy[0])/self.mx, self.lat+float(xy[1])/self.my

    def pixel(self, xy):
        return np.array([(xy[0]-self.xmin)/self.mpp, (self.ymax-xy[1])/self.mpp])

    def contains(self, xy):
        x, y = self.pixel(xy).astype(int)
        return 0 <= y < self.coverage.shape[0] and 0 <= x < self.coverage.shape[1] and self.coverage[y, x] != 0
