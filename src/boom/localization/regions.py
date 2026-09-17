"""
==============================================================================
Boom 航拍定位 - GeoTIFF 航拍圖資空間索引與區域篩選 (regions.py)
==============================================================================

【檔案定位】
本檔案負責對磁碟上的 GeoTIFF 圖資進行快速空間中繼資料索引 (`GeoTIFFIndex`)。
依據無人機當前 GPS 座標或預估視角，迅速篩選出候選瓦片圖資。

【核心功能】
- `GeoTIFFIndex`: 建立多圖資邊界索引，支援點包含性查詢與半徑鄰域檢索。
- `select_nearest_submap`: 依據 GPS 尋找歐幾里得距離最近之局部子圖資。
==============================================================================
"""
from dataclasses import dataclass
import math
from pathlib import Path
from functools import lru_cache


@dataclass(frozen=True)
class MapTile:
    path: Path
    region: str
    west: float
    south: float
    east: float
    north: float
    crs: str = "EPSG:4326"
    bounds: tuple = ()
    transform: object = None
    width: int = 0
    height: int = 0

    def contains(self, lon, lat):
        return self.west <= lon <= self.east and self.south <= lat <= self.north


def read_tile(path, region):
    import rasterio
    with rasterio.open(path) as src:
        if src.crs is None or abs(src.transform.determinant) < 1e-20:
            raise ValueError(f"Missing or invalid GeoTIFF georeferencing: {path}")
        if not (src.crs.is_geographic or src.crs.is_projected):
            raise ValueError(f"GeoTIFF needs a geographic or projected CRS: {path}")
        crs = src.crs.to_wkt()
        bounds = transformer(crs, "EPSG:4326").transform_bounds(
            *src.bounds, densify_pts=21)
        if not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[2]:
            raise ValueError(f"Unsupported geographic extent: {path}")
        return MapTile(Path(path), region, *bounds, crs, tuple(src.bounds),
                       src.transform, src.width, src.height)


@lru_cache(maxsize=64)
def transformer(source, target):
    from pyproj import Transformer
    return Transformer.from_crs(source, target, always_xy=True)


def gps_bounds(lon, lat, radius_m):
    """Conservative metric search envelope; reject polar/dateline ambiguity."""
    from pyproj import Geod
    if not (-180 <= lon <= 180 and -85 < lat < 85
            and math.isfinite(radius_m) and radius_m > 0):
        raise ValueError("Expected finite GPS (-85 < latitude < 85) and positive radius")
    geod = Geod(ellps="WGS84")
    # A circumscribed square supports a conservative radius search.
    corners = [geod.fwd(lon, lat, az, radius_m * math.sqrt(2))[:2]
               for az in (45, 135, 225, 315)]
    west, east = min(p[0] for p in corners), max(p[0] for p in corners)
    south, north = min(p[1] for p in corners), max(p[1] for p in corners)
    if east - west > 180 or south <= -85 or north >= 85:
        raise ValueError("Search crossing the antimeridian or polar region is unsupported")
    return west, south, east, north


class GeoTIFFIndex:
    """One metadata scan per session, followed by an in-memory geographic grid."""

    cell_size = 0.05

    def __init__(self, root):
        self.tiles = discover_tiles(Path(root))
        self.cells, self.large_tiles = {}, set()
        for i, tile in enumerate(self.tiles):
            x0, y0, x1, y1 = self._cells((tile.west, tile.south, tile.east, tile.north))
            if (x1-x0+1)*(y1-y0+1) > 10000:
                self.large_tiles.add(i)
                continue
            for x in range(x0, x1+1):
                for y in range(y0, y1+1):
                    self.cells.setdefault((x, y), set()).add(i)

    def _cells(self, bounds):
        return tuple(math.floor(v / self.cell_size) for v in bounds)

    def nearby(self, lon, lat, radius_m, max_candidates, submap=None):
        if not isinstance(max_candidates, int) or max_candidates <= 0:
            raise ValueError("max_candidates must be a positive integer")
        bounds = gps_bounds(lon, lat, radius_m)
        x0, y0, x1, y1 = self._cells(bounds)
        ids = set(self.large_tiles)
        if (x1-x0+1)*(y1-y0+1) > 100000:
            raise ValueError("Search radius exceeds spatial-index query budget")
        for x in range(x0, x1+1):
            for y in range(y0, y1+1):
                ids.update(self.cells.get((x, y), ()))
        candidates = []
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        for i in ids:
            tile = self.tiles[i]
            if submap is not None and tile.region != submap:
                continue
            if (tile.east < bounds[0] or tile.west > bounds[2]
                    or tile.north < bounds[1] or tile.south > bounds[3]):
                continue
            near_lon = min(max(lon, tile.west), tile.east)
            near_lat = min(max(lat, tile.south), tile.north)
            distance = geod.inv(lon, lat, near_lon, near_lat)[2]
            if distance <= radius_m:
                candidates.append((distance, str(tile.path), tile))
        return [t for _, _, t in sorted(candidates)[:max_candidates]]


def discover_tiles(root):
    tiles = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
            relative = path.relative_to(root)
            region = relative.parts[0] if len(relative.parts) > 1 else root.name
            tiles.append(read_tile(path, region))
    if not tiles:
        raise ValueError(f"No GeoTIFF tiles under {root}")
    return tiles


def region_bounds(tiles):
    result = []
    for name in sorted({t.region for t in tiles}):
        group = [t for t in tiles if t.region == name]
        result.append({"name": name, "tiles": len(group),
                       "west": min(t.west for t in group), "south": min(t.south for t in group),
                       "east": max(t.east for t in group), "north": max(t.north for t in group)})
    return result


def select_nearest_submap(tiles, lon, lat):
    """Choose one folder by its nearest tile bounds, then keep all its tiles.

    A folder's enclosing rectangle may contain large gaps, so distances use
    individual tile rectangles. Overlap ties use tile-center distance, then
    folder name for deterministic selection. This is display selection only.
    """
    try:
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        def _dist(p1, p2):
            return abs(geod.inv(p1[0], p1[1], p2[0], p2[1])[2])
    except ImportError:
        def _dist(p1, p2):
            deg_lat = 111132.954
            deg_lon = 111412.84 * math.cos(math.radians((p1[1] + p2[1]) / 2))
            dx = (p1[0] - p2[0]) * deg_lon
            dy = (p1[1] - p2[1]) * deg_lat
            return math.hypot(dx, dy)

    if not tiles:
        raise ValueError("Cannot select a submap without GeoTIFF tiles")
    if not (math.isfinite(lon) and math.isfinite(lat)
            and -180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("Expected finite WGS84 longitude/latitude")
    distances = {}
    for tile in tiles:
        nearest = (min(max(lon, tile.west), tile.east),
                   min(max(lat, tile.south), tile.north))
        distance = _dist((lon, lat), nearest)
        center_distance = _dist((lon, lat), ((tile.west+tile.east)/2, (tile.south+tile.north)/2))
        score = (distance, center_distance)
        distances[tile.region] = min(distances.get(tile.region, score), score)
    name = min(distances, key=lambda key: (*distances[key], key))
    selected = [tile for tile in tiles if tile.region == name]
    return selected, {
        "status": "nearest_submap",
        "selected_regions": [name],
        "selected_submap": name,
        "photo_gps": [lon, lat],
        "distance_m": distances[name][0],
        "regions": region_bounds(tiles),
        "loaded_files": [str(tile.path) for tile in selected],
        "note": "Nearest folder by tile bounds; display selection, not a visual position fix.",
    }


def select_regions(tiles, photo):
    """Select candidates intersecting the photo GPS or estimated center view.

    Multiple hits stay candidates; no hits keeps all maps available. We do not
    assume the nearest map is correct. Center view uses the existing flat-ground
    and relative-height approximation and is bounded to the VO ground range.
    """
    gps_regions = sorted({t.region for t in tiles if t.contains(photo.lon, photo.lat)})
    view_gps = None
    forward = photo.rotation().T[:, 2]
    if forward[2] < -1e-6 and photo.height > 0:
        east, north = forward[:2]*(-photo.height/forward[2])
        if math.hypot(east, north) <= 1800:
            metres_lat = 6378137*math.pi/180
            metres_lon = metres_lat*math.cos(math.radians(photo.lat))
            if abs(metres_lon) > 1e-6:
                view_gps = [photo.lon+float(east)/metres_lon, photo.lat+float(north)/metres_lat]
    view_regions = sorted({t.region for t in tiles if view_gps and t.contains(*view_gps)})
    candidates = sorted(set(gps_regions) | set(view_regions))
    all_names = sorted({t.region for t in tiles})
    selected = candidates or all_names
    if len(candidates) == 1:
        status = "gps_and_view_candidate" if gps_regions and view_regions else (
            "gps_bounds_candidate" if gps_regions else "estimated_view_candidate")
    else:
        status = "multiple_candidates" if candidates else "unresolved_keep_all"
    return {"status": status, "selected_regions": selected,
            "gps_regions": gps_regions, "estimated_view_regions": view_regions,
            "photo_gps": [photo.lon, photo.lat], "estimated_view_gps": view_gps,
            "regions": region_bounds(tiles),
            "note": "Raster-bound candidates only; view uses approximate ground height. Not visual matching or administrative identification."}


def select_map_tiles(root, photo):
    tiles = discover_tiles(root)
    selection = select_regions(tiles, photo)
    return [t for t in tiles if t.region in selection["selected_regions"]], selection
