"""
==============================================================================
Boom 離線管線 - 地圖定位符號相容性轉接層 (map_localization.py)
==============================================================================

【檔案定位】
本檔案為歷史版本相容轉接模組。實際的相機姿態追蹤、地圖匹配與地面投影實作
已全面模組化搬遷至 `artillery.offline.src.localization`。

【相容性保證】
保留原本從 `artillery.offline.src.pipelines.map_localization` 匯入符號的程式碼相容性，
避免破壞現有腳本或舊測試案例。

【轉接符號】
- `GeoMap`, `MapLocalizer`, `PhotoPose`, `gps_number`, `plane_homography`
- `read_bgr`, `resize_work`, `transform`
==============================================================================
"""
from boom.localization import (
    GeoMap, MapLocalizer, PhotoPose, gps_number, plane_homography,
    read_bgr, resize_work, transform,
)

__all__ = [
    "GeoMap", "MapLocalizer", "PhotoPose", "gps_number", "plane_homography",
    "read_bgr", "resize_work", "transform",
]
