"""
==============================================================================
Boom 煙霧辨識 - PIDNet-S 輕量化語意分割模組 (boom.detection.pidnet)
==============================================================================

提供基於深度學習的無人機荒野煙霧分割與新煙霧追蹤器：
- PIDNetSmokeImpactDetector: 核心偵測器，實作 BaseSmokeDetector 介面
- PIDNetSmokeTrackerConfig: 參數設定資料類別
==============================================================================
"""

from .detector import (
    PIDNetSmokeTrackerConfig,
    PIDNetSmokeImpactDetector,
    ConfirmedEvent,
)

__all__ = [
    "PIDNetSmokeTrackerConfig",
    "PIDNetSmokeImpactDetector",
    "ConfirmedEvent",
]
