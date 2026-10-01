# 檔案功能：localization 套件初始化與公開匯出；不直接執行。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""
==============================================================================
Boom 航拍定位與圖資模組 (boom.localization)
==============================================================================

提供無人機姿態追蹤、視覺里程計、GeoTIFF 航拍圖資對齊與煙霧地理座標反解算：
- MapLocalizer: 核心定位協調器，實作 BaseLocalizer 介面
- GeoMap: GeoTIFF 圖資幾何索引與投影管理
- NearbyMapLocalizer: 局部航拍影像特徵搜尋與匹配定位器
- PhotoPose: 相機姿態與 GPS EXIF 幾何解算
- plane_homography, transform: 幾何投影變換核心函式
==============================================================================
"""

from .geo_map import GeoMap
from .geometry import plane_homography, transform, camera_heading
from .localizer import MapLocalizer
from .photo import PhotoPose, gps_number, read_bgr, resize_work
from .nearby import NearbyMapLocalizer, MapMatchResult
from .regions import GeoTIFFIndex

__all__ = [
    "GeoMap",
    "MapLocalizer",
    "PhotoPose",
    "gps_number",
    "plane_homography",
    "read_bgr",
    "resize_work",
    "transform",
    "camera_heading",
    "NearbyMapLocalizer",
    "MapMatchResult",
    "GeoTIFFIndex",
]
