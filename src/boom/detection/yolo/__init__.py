# 檔案功能：yolo 套件初始化與公開匯出；不直接執行。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""
==============================================================================
Boom 煙霧/起爆辨識 - YOLO 偵測模組 (boom.detection.yolo)
==============================================================================

提供基於 YOLO 實例分割 / 物件辨識的無人機煙霧與起爆偵測器：
- YOLOSmokeImpactDetector: 核心偵測器，實作 BaseSmokeDetector 介面
- YOLOSmokeDetectorConfig: 參數設定資料類別
==============================================================================
"""

from .detector import (
    YOLOSmokeDetectorConfig,
    YOLOSmokeImpactDetector,
)

__all__ = [
    "YOLOSmokeDetectorConfig",
    "YOLOSmokeImpactDetector",
]
