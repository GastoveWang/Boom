"""Map panel rendering: map-centered satellite view, auto-expanding bounds for drone, rotated drone icon with high-contrast visibility."""
from __future__ import annotations

import math
import cv2
import numpy as np
from ..ui.drawing import _draw_text_inplace
from ..ui.event_style import event_color, event_label
from .map_widgets import camera_bearing, draw_camera_direction, draw_navigation_card
from ..ui.theme import BACKGROUND, SURFACE, BORDER, ACCENT, rounded_surface


def _draw_rotated_drone_icon(panel: np.ndarray, p: tuple[int, int], heading_deg: float, icon: np.ndarray | None) -> None:
    px, py = p
    icon_size = 56
    h, w = panel.shape[:2]

    # High-contrast glowing halo rings & radar crosshairs around drone position
    cv2.circle(panel, p, 30, (225, 172, 105), 2, cv2.LINE_AA)     # Bright Neon Gold/Yellow outer ring
    cv2.circle(panel, p, 22, (255, 255, 255), 2, cv2.LINE_AA)   # Pure White inner ring

    # 4 radar crosshair ticks
    cv2.line(panel, (px - 36, py), (px - 26, py), (225, 172, 105), 2, cv2.LINE_AA)
    cv2.line(panel, (px + 26, py), (px + 36, py), (225, 172, 105), 2, cv2.LINE_AA)
    cv2.line(panel, (px, py - 36), (px, py - 26), (225, 172, 105), 2, cv2.LINE_AA)
    cv2.line(panel, (px, py + 26), (px, py + 36), (225, 172, 105), 2, cv2.LINE_AA)

    if icon is not None and icon.ndim == 3 and icon.shape[2] == 4:
        # Resize RGBA icon
        base_icon = cv2.resize(icon, (icon_size, icon_size), interpolation=cv2.INTER_AREA)

        # Rotate icon (clockwise rotation)
        M = cv2.getRotationMatrix2D((icon_size / 2.0, icon_size / 2.0), -heading_deg, 1.0)
        rotated = cv2.warpAffine(
            base_icon,
            M,
            (icon_size, icon_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        x0, y0 = px - icon_size // 2, py - icon_size // 2

        rx0, ry0 = max(0, x0), max(0, y0)
        rx1, ry1 = min(w, x0 + icon_size), min(h, y0 + icon_size)

        if rx1 > rx0 and ry1 > ry0:
            ix0, iy0 = rx0 - x0, ry0 - y0
            ix1, iy1 = ix0 + (rx1 - rx0), iy0 + (ry1 - ry0)

            crop_icon = rotated[iy0:iy1, ix0:ix1]
            alpha = crop_icon[:, :, 3:4] / 255.0
            rgb = crop_icon[:, :, :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            roi = panel[ry0:ry1, rx0:rx1]

            # Translucent bright backing disc to ensure dark drone icon pops out on dark terrain
            backing = roi.copy()
            cv2.circle(backing, (px - rx0, py - ry0), 22, (255, 255, 255), -1)
            roi_with_backing = cv2.addWeighted(backing, 0.45, roi, 0.55, 0)

            panel[ry0:ry1, rx0:rx1] = np.uint8(bgr * alpha + roi_with_backing * (1.0 - alpha))

        # Bright center point
        cv2.circle(panel, p, 4, (225, 172, 105), -1, cv2.LINE_AA)
        return

    # Fallback: High-contrast rotated quadcopter vector graphic
    rad = math.radians(heading_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    arm_len = 24
    rotor_rad = 8
    angles = [45, 135, 225, 315]

    for idx, ang in enumerate(angles):
        r_rad = math.radians(heading_deg + ang)
        rx = int(px + arm_len * math.sin(r_rad))
        ry = int(py - arm_len * math.cos(r_rad))
        # High contrast white arms
        cv2.line(panel, p, (rx, ry), (255, 255, 255), 3, cv2.LINE_AA)

        # Front rotors in bright neon yellow/gold, rear in bright neon cyan
        rotor_color = (225, 172, 105) if idx in (0, 3) else (255, 240, 0)
        cv2.circle(panel, (rx, ry), rotor_rad, rotor_color, -1, cv2.LINE_AA)
        cv2.circle(panel, (rx, ry), rotor_rad, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(panel, (rx, ry), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # Front orientation nose dot
    nose_x = int(px + (arm_len + 6) * sin_a)
    nose_y = int(py - (arm_len + 6) * cos_a)
    cv2.circle(panel, (nose_x, nose_y), 5, (50, 255, 100), -1, cv2.LINE_AA)
    cv2.circle(panel, (nose_x, nose_y), 6, (255, 255, 255), 1, cv2.LINE_AA)


def render_panel(localizer, height: int, width: int) -> np.ndarray:
    """Render full-bleed interactive GeoMap panel centered at the map, auto-zoom out for drone."""
    panel = np.full((height, width, 3), BACKGROUND, np.uint8)

    # Full-bleed layout bounds
    ox, oy = 10, 56
    map_width = max(100, width - 20)
    map_height = max(100, height - 110)

    # 1. Calculate GeoMap center in world coordinates (MAP-CENTERED, NOT DRONE-CENTERED)
    map_w_m = localizer.map.canvas.shape[1] * localizer.map.mpp
    map_h_m = localizer.map.canvas.shape[0] * localizer.map.mpp
    user_span_m = float(getattr(localizer, "map_span_m", 1100.0))
    drone_pos = localizer.center[:2]
    # Reserve navigation space above the fitted geography. Include the drone
    # in the fit so neither map imagery nor aircraft starts behind the card.
    safe_top = 242
    safe_width, safe_height = map_width-64, max(100, map_height-safe_top-64)
    xmin = min(localizer.map.xmin, drone_pos[0])
    xmax = max(localizer.map.xmin+map_w_m, drone_pos[0])
    ymin = min(localizer.map.ymax-map_h_m, drone_pos[1])
    ymax = max(localizer.map.ymax, drone_pos[1])
    pixels_per_metre = min(safe_width/max(xmax-xmin, 1), safe_height/max(ymax-ymin, 1),
                           safe_width/max(user_span_m, 50))*.90
    scale = pixels_per_metre*localizer.map.mpp
    view_center = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
    map_center_px = localizer.map.pixel(view_center)
    origin = map_center_px - np.array([map_width/2, safe_top+32+safe_height/2])/scale
    span_m = map_width/pixels_per_metre

    # Full-bleed affine warp
    affine = np.array([[scale, 0, -origin[0] * scale], [0, scale, -origin[1] * scale]])
    thumb = cv2.warpAffine(
        localizer.map.canvas,
        affine,
        (map_width, map_height),
        borderValue=BACKGROUND,
    )
    coverage = getattr(localizer.map, "coverage", None)
    if coverage is not None:
        valid = cv2.warpAffine(coverage, affine, (map_width, map_height), flags=cv2.INTER_NEAREST)
        thumb[valid == 0] = BACKGROUND
    panel[oy : oy + map_height, ox : ox + map_width] = thumb

    # Thin border around map
    cv2.rectangle(panel, (ox, oy), (ox + map_width - 1, oy + map_height - 1), (85, 74, 61), 1)

    def point(xy) -> tuple[int, int]:
        pt = (localizer.map.pixel(xy) - origin) * scale + [ox, oy]
        return (int(round(pt[0])), int(round(pt[1])))

    # 1. Trajectory line
    if len(localizer.trajectory) > 1:
        pts = np.array([point(p) for p in localizer.trajectory], np.int32)
        cv2.polylines(panel, [pts], False, (225, 172, 105), 3, cv2.LINE_AA)

    # 2. Event markers
    for event_id, xy in localizer.events:
        p = point(xy)
        if not (ox <= p[0] < ox + map_width and oy <= p[1] < oy + map_height):
            continue
        color = event_color(event_id)
        cv2.circle(panel, p, 10, color, -1, cv2.LINE_AA)
        cv2.circle(panel, p, 12, (245, 248, 255), 2, cv2.LINE_AA)

        text = event_label(event_id)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        tx = min(p[0] + 14, ox + map_width - tw - 10)
        ty = max(oy + th + 8, p[1] - 12)
        cv2.rectangle(panel, (tx - 5, ty - th - 5), (tx + tw + 5, ty + baseline + 3), (20, 26, 36), -1)
        cv2.rectangle(panel, (tx - 5, ty - th - 5), (tx + tw + 5, ty + baseline + 3), color, 1)
        cv2.putText(panel, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    # 3. High-contrast Rotated Drone Icon
    drone_pt = point(drone_pos)
    heading_deg = camera_bearing(localizer.rotation)

    draw_camera_direction(panel, drone_pt, heading_deg)
    _draw_rotated_drone_icon(panel, drone_pt, heading_deg or 0., localizer.icon)

    # The inset tracks the actual rendered viewport, including auto-zoom.
    west_en = localizer.map.xmin + origin[0]*localizer.map.mpp
    north_en = localizer.map.ymax - origin[1]*localizer.map.mpp
    east_en = west_en + map_width/scale*localizer.map.mpp
    south_en = north_en - map_height/scale*localizer.map.mpp
    west, north = localizer.map.to_gps((west_en, north_en))
    east, south = localizer.map.to_gps((east_en, south_en))
    draw_navigation_card(panel, (ox+map_width-382, oy+14), (west, south, east, north), heading_deg)

    # Top overlay bar inside map panel
    bar_h = 42
    bar = panel[10 : 10 + bar_h, ox : ox + map_width]
    overlay = bar.copy()
    cv2.rectangle(overlay, (0, 0), (map_width, bar_h), (48, 42, 35), -1)
    panel[10 : 10 + bar_h, ox : ox + map_width] = cv2.addWeighted(overlay, 0.85, bar, 0.15, 0)
    cv2.rectangle(panel, (ox, 10), (ox + map_width - 1, 10 + bar_h), (85, 74, 61), 1)

    _draw_text_inplace(panel, "全域衛星地圖 | GEOMAP", (ox + 14, 18), 17, (240, 245, 252))

    # Zoom & Auto-bounds hint badge
    zoom_hint = f"滾輪縮放 | 視場: {span_m / 1000.0:.1f} km" if span_m >= 1000 else f"滾輪縮放 | 視場: {int(span_m)} m"
    if span_m > user_span_m:
        zoom_hint += " [動態視野]"
    _draw_text_inplace(panel, zoom_hint, (ox + map_width - 335, 18), 16, (255, 215, 90))

    # Region selection / map candidates
    selection = getattr(localizer.map, "region_selection", None)
    if selection and selection.get("selected_regions"):
        names = " / ".join(selection["selected_regions"])
        _draw_text_inplace(panel, "地圖：" + names, (ox + 300, 18), 17, (140, 200, 255))

    # Bottom status telemetry bar
    lon, lat = localizer.map.to_gps(localizer.center)
    pos_title = "無人機位置"

    info_y = height - 44
    _draw_text_inplace(panel, f"{pos_title}: {lat:.6f}°, {lon:.6f}°", (ox + 10, info_y), 17, (235, 242, 250))
    _draw_text_inplace(
        panel,
        f"地圖事件: {len(localizer.events)} 筆  |  方向扇形：無人機朝向",
        (ox + 400, info_y),
        17,
        (180, 210, 235),
    )

    return panel
