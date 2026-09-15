"""Existing geographic projection and static camera model; no runner dependencies."""
from __future__ import annotations

from typing import Tuple
import math


def coordinate(
    lon: float,
    lat: float,
    alt: float,
    yaw: float,
    pitch: float,
    roll: float,
    orix: float,
    oriy: float,
    image_width: float = 2840.0,
    image_height: float = 2840.0,
) -> Tuple[float, float]:
    a = 6378137.0
    b = 6356752.3142451
    lon0 = 121 * math.pi / 180
    k0 = 0.9999
    dx = 250000
    dy = 0
    e = 1 - b**2 / a**2
    e2 = e / (1 - e)

    lon = (lon - math.floor((lon + 180) / 360) * 360) * math.pi / 180
    lat = lat * math.pi / 180
    v = a / math.sqrt(1 - e * math.sin(lat) ** 2)
    t = math.tan(lat) ** 2
    c = e2 * math.cos(lat) ** 2
    a_term = math.cos(lat) * (lon - lon0)
    m = a * (
        (1 - e / 4 - 3 * e**2 / 64 - 5 * e**3 / 256) * lat
        - (3 * e / 8 + 3 * e**2 / 32 + 45 * e**3 / 1024) * math.sin(2 * lat)
        + (15 * e**2 / 256 + 45 * e**3 / 1024) * math.sin(4 * lat)
        - (35 * e**3 / 3072) * math.sin(6 * lat)
    )

    twd_e = dx + k0 * v * (
        a_term
        + (1 - t + c) * a_term**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * e2) * a_term**5 / 120
    )
    twd_n = dy + k0 * (
        m
        + v
        * math.tan(lat)
        * (
            a_term**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a_term**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * e2) * a_term**6 / 720
        )
    )

    yaw = yaw * math.pi / 180
    pitch = pitch * math.pi / 180
    roll = roll * math.pi / 180

    m11 = math.sin(yaw) * math.sin(pitch) * math.sin(roll) + \
        math.cos(yaw) * math.cos(roll)
    m12 = math.cos(yaw) * math.sin(pitch) * math.sin(roll) - \
        math.sin(yaw) * math.cos(roll)
    m13 = -math.cos(pitch) * math.sin(roll)
    m21 = math.sin(yaw) * math.cos(pitch)
    m22 = math.cos(yaw) * math.cos(pitch)
    m23 = math.sin(pitch)
    m31 = -(math.sin(yaw) * math.sin(pitch) *
            math.cos(roll) - math.cos(yaw) * math.sin(roll))
    m32 = -(math.cos(yaw) * math.sin(pitch) *
            math.cos(roll) + math.sin(yaw) * math.sin(roll))
    m33 = math.cos(pitch) * math.cos(roll)

    f = 4.5 / 0.00274
    f2 = 4.5
    pixel_size = 0.00274
    # The original calibration is expressed on a 2840 x 2840 raster. Map the
    # event pixel into that raster so video resolution does not change the FOV.
    calibration_width = 2840.0
    calibration_height = 2840.0
    calibrated_x = orix * calibration_width / max(image_width, 1.0)
    calibrated_y = oriy * calibration_height / max(image_height, 1.0)
    dx_pixel = calibrated_x - calibration_width / 2
    dy_pixel = -calibrated_y + calibration_height / 2
    dist = math.sqrt(dx_pixel**2 + dy_pixel**2)
    if dist == 0:
        dist = 1e-6

    theta = dist / f
    radius = f * math.tan(theta)
    x = dx_pixel * radius / dist * pixel_size
    y = dy_pixel * radius / dist * pixel_size

    easting = -alt * (m11 * x + m21 * y - m31 * f2) / \
        (m13 * x + m23 * y - m33 * f2) + twd_e
    northing = -alt * (m12 * x + m22 * y - m32 * f2) / \
        (m13 * x + m23 * y - m33 * f2) + twd_n
    return easting, northing


def twd97_to_wgs84(easting: float, northing: float) -> Tuple[float, float]:
    a = 6378137.0
    b = 6356752.3142451
    lon0 = 121 * math.pi / 180
    k0 = 0.9999
    dx = 250000
    e = math.sqrt(1 - (b**2 / a**2))
    x = easting - dx
    y = northing

    m = y / k0
    mu = m / (a * (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256))
    e1 = (1 - math.sqrt(1 - e**2)) / (1 + math.sqrt(1 - e**2))

    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512

    fp = (
        mu
        + j1 * math.sin(2 * mu)
        + j2 * math.sin(4 * mu)
        + j3 * math.sin(6 * mu)
        + j4 * math.sin(8 * mu)
    )

    e2 = e**2 / (1 - e**2)
    c1 = e2 * math.cos(fp) ** 2
    t1 = math.tan(fp) ** 2
    r1 = a * (1 - e**2) / ((1 - e**2 * math.sin(fp) ** 2) ** 1.5)
    n1 = a / math.sqrt(1 - e**2 * math.sin(fp) ** 2)
    d = x / (n1 * k0)

    q1 = n1 * math.tan(fp) / r1
    q2 = d**2 / 2
    q3 = (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e2) * d**4 / 24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * t1 **
          2 - 252 * e2 - 3 * c1**2) * d**6 / 720
    lat = fp - q1 * (q2 - q3 + q4)

    q5 = d
    q6 = (1 + 2 * t1 + c1) * d**3 / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e2 + 24 * t1**2) * d**5 / 120
    lon = lon0 + (q5 - q6 + q7) / math.cos(fp)

    return math.degrees(lon), math.degrees(lat)
