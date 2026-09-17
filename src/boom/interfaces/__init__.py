"""
==============================================================================
Boom 抽象介面層 (interfaces)
==============================================================================

定義系統核心抽象介面，實踐 High Cohesion / Low Coupling 與依賴反轉原則：
- BaseSmokeDetector: 煙霧辨識器標準介面
- BaseFeatureMatcher: 圖像特徵點匹配標準介面
- BaseLocalizer: 地面座標投影與航拍定位標準介面
- BaseFrameSource: 影格擷取輸入來源標準介面
==============================================================================
"""

from .detector import BaseSmokeDetector
from .matcher import BaseFeatureMatcher
from .localizer import BaseLocalizer
from .sensor import BaseFrameSource

__all__ = [
    "BaseSmokeDetector",
    "BaseFeatureMatcher",
    "BaseLocalizer",
    "BaseFrameSource",
]
