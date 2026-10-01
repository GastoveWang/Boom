# 檔案功能：detection 套件初始化與公開匯出；不直接執行。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""
==============================================================================
Boom 煙霧辨識模組層 (detection)
==============================================================================

包含系統所有煙霧識別演算法實作：
- optical_flow: 基於相機動態補償與光流膨脹比的即時煙霧偵測
- yolo: 物件偵測與追蹤
- factory: 統一偵測器建立入口
==============================================================================
"""

from .factory import create_detector
from .optical_flow import DetectorConfig, InstantSmokeDustDetector
from .yolo import YOLOSmokeDetectorConfig, YOLOSmokeImpactDetector

__all__ = [
    "create_detector",
    "DetectorConfig",
    "InstantSmokeDustDetector",
    "YOLOSmokeDetectorConfig",
    "YOLOSmokeImpactDetector",
]
