"""Offline geographic overview, north-up compass and camera bearing widgets."""
import json
import math
from functools import lru_cache

import cv2
import numpy as np
from artillery.common.defaults import PROJECT_ROOT
from ..ui.drawing import _draw_text_inplace
from ..ui.theme import SURFACE, BORDER, ACCENT, TEXT, rounded_surface


def bearing_vector(degrees):
    radians = math.radians(degrees)
    return np.array([math.sin(radians), -math.cos(radians)])


def camera_bearing(rotation):
    """Use gimbal bearing as drone heading under the aligned-heading assumption."""
    forward = rotation.T[:, 2][:2]
    if np.linalg.norm(forward) < 1e-5:
        return None
    return math.degrees(math.atan2(forward[0], forward[1])) % 360


def taiwan_pixel(lon, lat):
    # Equirectangular overview with longitude corrected at Taiwan's latitude.
    return np.array([20+(lon-118.0)*49*math.cos(math.radians(24)),
                     270-(lat-21.7)*59])


@lru_cache(maxsize=1)
def _taiwan_base():
    card = np.full((304, 232, 3), SURFACE, np.uint8)
    path = PROJECT_ROOT / "asset/ui/taiwan_natural_earth.geojson"
    geometry = json.loads(path.read_text(encoding="utf-8-sig"))["geometry"]
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    for polygon in polygons:
        rings = [np.array([taiwan_pixel(*xy[:2]) for xy in ring], np.int32) for ring in polygon]
        cv2.fillPoly(card, rings, (116, 109, 80), cv2.LINE_AA)
        cv2.polylines(card, rings, True, (183, 185, 147), 1, cv2.LINE_AA)
    _draw_text_inplace(card, "台灣全圖", (14, 10), 23, (245, 242, 230))
    _draw_text_inplace(card, "框線：目前地圖範圍", (14, 277), 17, (170, 225, 255))
    cv2.rectangle(card, (0, 0), (231, 303), (110, 96, 73), 1)
    return card


def draw_taiwan_overview(panel, position, bounds):
    card = _taiwan_base().copy()
    west, south, east, north = bounds
    a, b = taiwan_pixel(west, north), taiwan_pixel(east, south)
    center = (a+b)/2
    # At national scale a ~1 km viewport is subpixel: retain its center and
    # use a minimum visible locator box, rather than implying pixel accuracy.
    half = np.maximum(np.abs(b-a)/2, 6)
    lo, hi = np.round(center-half).astype(int), np.round(center+half).astype(int)
    cv2.rectangle(card, tuple(lo), tuple(hi), (65, 200, 255), 2, cv2.LINE_AA)
    cv2.circle(card, tuple(np.round(center).astype(int)), 2, (255, 255, 255), -1)
    x, y = position
    panel[y:y+304, x:x+232] = card


@lru_cache(maxsize=1)
def _compass():
    card = np.full((184, 184, 3), SURFACE, np.uint8)
    center = np.array([92, 92])
    cv2.circle(card, tuple(center), 86, (105, 91, 69), 1, cv2.LINE_AA)
    for degree in range(0, 360, 15):
        v = bearing_vector(degree)
        a, b = center+v*64, center+v*(56 if degree % 45 == 0 else 60)
        cv2.line(card, tuple(a.astype(int)), tuple(b.astype(int)), (155, 156, 145), 1, cv2.LINE_AA)
    for label, degree in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        pos = center+bearing_vector(degree)*75
        cv2.putText(card, label, (int(pos[0])-7, int(pos[1])+6), cv2.FONT_HERSHEY_SIMPLEX, .6,
                    ACCENT if degree == 0 else TEXT, 2, cv2.LINE_AA)
    for degree in (0, 90, 180, 270):
        v, right = bearing_vector(degree), bearing_vector(degree+90)
        tip = center+v*(49 if degree % 180 == 0 else 39)
        color = ACCENT if degree == 0 else (180, 169, 150)
        cv2.fillConvexPoly(card, np.array([tip, center+right*10, center], np.int32), color, cv2.LINE_AA)
        cv2.fillConvexPoly(card, np.array([tip, center-right*10, center], np.int32), tuple(int(c*.55) for c in color), cv2.LINE_AA)
    cv2.circle(card, tuple(center), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return card


def draw_compass(panel, position):
    x, y = position
    panel[y:y+184, x:x+184] = _compass()


def draw_navigation_card(panel, position, bounds, bearing):
    """One shared surface, without separate rectangular widget backgrounds."""
    x, y = position
    rounded_surface(panel, (x+3, y+5, 366, 218), (23, 20, 17), 14)
    rounded_surface(panel, (x, y, 366, 218), SURFACE, 14)
    card = panel[y:y+218, x:x+366]
    _draw_text_inplace(card, "位置與方向  |  NAVIGATION", (16, 10), 19, TEXT)
    overview = np.zeros((304, 232, 3), np.uint8)
    draw_taiwan_overview(overview, (0, 0), bounds)
    card[42:194, 12:165] = cv2.resize(overview[44:274, 1:231], (153, 152), interpolation=cv2.INTER_AREA)
    cv2.line(card, (179, 48), (179, 185), BORDER, 1)
    card[40:184, 195:339] = cv2.resize(_compass(), (144, 144), interpolation=cv2.INTER_AREA)
    _draw_text_inplace(card, "台灣 · 目前範圍", (17, 194), 14, (185, 204, 220))
    caption = f"無人機朝向 {bearing:03.0f}°" if bearing is not None else "無人機朝向 —"
    _draw_text_inplace(card, caption, (196, 187), 16, ACCENT)


def draw_camera_direction(panel, point, bearing):
    if bearing is None:
        return
    p = np.array(point)
    angles = np.linspace(bearing-24, bearing+24, 24)
    polygon = np.array([p, *[p+bearing_vector(a)*128 for a in angles]], np.int32)
    overlay = panel.copy()
    cv2.fillConvexPoly(overlay, polygon, ACCENT, cv2.LINE_AA)
    cv2.addWeighted(overlay, .30, panel, .70, 0, dst=panel)
    cv2.polylines(panel, [polygon], True, ACCENT, 2, cv2.LINE_AA)
