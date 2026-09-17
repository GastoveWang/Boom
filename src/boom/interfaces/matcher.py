"""
==============================================================================
Boom 抽象介面 - 特徵匹配器標準契約 (matcher.py)
==============================================================================

【介面定義】
定義影像特徵點比對與幾何匹配（如 SIFT, LightGlue, SuperPoint）之通用介面。
可用於照片與地圖匹配、影格間特徵跟蹤。
==============================================================================
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np


class BaseFeatureMatcher(ABC):
    """影像特徵匹配演算法抽象基底類別"""

    @abstractmethod
    def match(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        mask_a: Optional[np.ndarray] = None,
        mask_b: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        比對兩張輸入影像並輸出匹配的座標點對。

        :param image_a: 第一張輸入影像 (BGR 或灰階)
        :param image_b: 第二張輸入影像 (BGR 或灰階)
        :param mask_a: 影像 A 的有效區域遮罩 (可選)
        :param mask_b: 影像 B 的有效區域遮罩 (可選)
        :return: (pts_a, pts_b, metrics)
                 - pts_a: 匹配點在影像 A 上的座標陣列 (N, 2)
                 - pts_b: 匹配點在影像 B 上的座標陣列 (N, 2)
                 - metrics: 匹配品質字典 (如 confidence, inlier_count, reproj_error)
        """
        pass
