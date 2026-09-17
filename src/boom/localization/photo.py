"""
==============================================================================
Boom 航拍定位 - DJI 照片 Exif/XMP 解析與相機內參估算 (photo.py)
==============================================================================

【檔案定位】
本檔案負責讀取 DJI 無人機參考照片（.JPG）中嵌入的 Exif 與 XMP 飛控中繼資料，
提取相機經緯度、相對/絕對高度、雲台姿態角（Pitch, Roll, Yaw），並建構近似相機內參矩陣 K。

【核心功能】
- `PhotoPose.read(path)`: 解析照片並返回包含相機姿態與 GPS 的 `PhotoPose` 結構。
- `read_bgr(path)`: 安全讀取影像並轉換為 OpenCV BGR 格式。
- `resize_work(image, width)`: 等比例縮放至工作解析度，加速後續特徵運算。
==============================================================================
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


def read_bgr(path):
    with Image.open(path) as im:
        return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def resize_work(image, width=1280):
    scale = min(1.0, width / image.shape[1])
    return cv2.resize(image, None, fx=scale, fy=scale), scale


def gps_number(values, direction):
    value = float(values[0]) + float(values[1]) / 60 + float(values[2]) / 3600
    return -value if direction in ("S", "W") else value


@dataclass
class PhotoPose:
    lon: float
    lat: float
    height: float
    yaw: float
    pitch: float
    roll: float
    focal: float
    width: int
    height_px: int

    @classmethod
    def read(cls, path, ground_height=None):
        with Image.open(path) as im:
            width, height = im.size
            exif = im.getexif()
            gps = exif.get_ifd(34853)
            detail = exif.get_ifd(34665)
        if not all(k in gps for k in (1, 2, 3, 4)):
            raise ValueError("Initialization photo requires GPS EXIF.")
        raw = path.read_bytes().decode("utf-8", errors="ignore")
        def field(name):
            match = re.search(rf'{name}="([^"]+)"', raw)
            if match is None:
                raise ValueError(f"Initialization photo missing {name}")
            return float(match.group(1))
        h = ground_height if ground_height is not None else field("RelativeAltitude")
        if h <= 0:
            raise ValueError("Height above the assumed ground plane must be positive")
        focal35 = float(detail.get(41989, 0))
        if focal35 <= 0:
            raise ValueError("Photo needs FocalLengthIn35mmFilm for approximate intrinsics")
        return cls(gps_number(gps[4], gps[3]), gps_number(gps[2], gps[1]),
                   h, field("GimbalYawDegree"), field("GimbalPitchDegree"),
                   field("GimbalRollDegree"), focal35 / math.hypot(36, 24) * math.hypot(width, height),
                   width, height)

    def camera_matrix(self, width, height):
        # Same horizontal FOV; videos may vertically crop the photo sensor.
        f = self.focal * width / self.width
        return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.]], dtype=float)

    def rotation(self):
        yaw, pitch, roll = np.radians([self.yaw, self.pitch, self.roll])
        right = np.array([math.cos(yaw), -math.sin(yaw), 0.])
        forward = np.array([math.sin(yaw)*math.cos(pitch), math.cos(yaw)*math.cos(pitch), math.sin(pitch)])
        down = np.cross(forward, right)
        return np.column_stack((right*math.cos(roll)+down*math.sin(roll),
                                down*math.cos(roll)-right*math.sin(roll), forward)).T
