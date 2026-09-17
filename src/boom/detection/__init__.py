"""
==============================================================================
Boom 煙霧辨識模組層 (detection)
==============================================================================

包含系統所有煙霧識別演算法實作：
- pidnet: 基於雙邊分支語意分割的煙霧偵測與生命週期追蹤
- optical_flow: 基於相機動態補償與光流膨脹比的即時煙霧偵測
- fusion: 結合光流高速觸發與 PIDNet 特徵驗證的兩階段融合偵測
- factory: 統一偵測器建立入口
==============================================================================
"""

from .factory import create_detector
from .pidnet import PIDNetSmokeTrackerConfig, PIDNetSmokeImpactDetector
from .optical_flow import DetectorConfig, InstantSmokeDustDetector
from .fusion import OpticalPIDNetFusionConfig, OpticalPIDNetFusionDetector

__all__ = [
    "create_detector",
    "PIDNetSmokeTrackerConfig",
    "PIDNetSmokeImpactDetector",
    "DetectorConfig",
    "InstantSmokeDustDetector",
    "OpticalPIDNetFusionConfig",
    "OpticalPIDNetFusionDetector",
]
