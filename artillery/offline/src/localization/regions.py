"""Metadata-only region discovery and GPS/view-based map selection.

Bounds describe raster rectangles, not administrative boundaries or guaranteed
valid imagery. A selected region is a search candidate, not visual localization.
"""
from dataclasses import dataclass
import math
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class MapTile:
    path: Path
    region: str
    west: float
    south: float
    east: float
    north: float

    def contains(self, lon, lat):
        return self.west <= lon <= self.east and self.south <= lat <= self.north


def read_tile(path, region):
    with Image.open(path) as im:
        tags = im.tag_v2
        tie, scale, keys = tags.get(33922), tags.get(33550), tags.get(34735)
        if not tie or not scale or not keys or tags.get(34264):
            raise ValueError(f"Unsupported GeoTIFF georeferencing: {path}")
        entries = {keys[i]: tuple(keys[i+1:i+4]) for i in range(4, len(keys), 4)}
        if entries.get(2048) != (0, 1, 4326) or entries.get(1025) != (0, 1, 1):
            raise ValueError(f"Expected WGS84 PixelIsArea TIFF: {path}")
        left = tie[3]-tie[0]*scale[0]
        top = tie[4]+tie[1]*scale[1]
        return MapTile(path, region, left, top-im.height*scale[1], left+im.width*scale[0], top)


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
