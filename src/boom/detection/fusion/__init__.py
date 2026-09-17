"""
==============================================================================
Boom 煙霧辨識 - 光流與語意分割兩階段融合模組 (boom.detection.fusion)
==============================================================================

結合光流動態高靈敏觸發與 PIDNet 深度學習特徵確認的兩階段融合偵測器：
- OpticalPIDNetFusionDetector: 核心融合偵測器，實作 BaseSmokeDetector 介面
- OpticalPIDNetFusionConfig: 融合時間窗與距離閾值參數設定
- FusionConfirmedEvent: 融合確認事件資料結構
==============================================================================
"""

from .detector import (
    OpticalPIDNetFusionConfig,
    FusionConfirmedEvent,
    OpticalPIDNetFusionDetector,
)

__all__ = [
    "OpticalPIDNetFusionConfig",
    "FusionConfirmedEvent",
    "OpticalPIDNetFusionDetector",
]
