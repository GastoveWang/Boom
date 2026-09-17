"""
==============================================================================
Boom 抽象介面 - 地面定位器標準契約 (localizer.py)
==============================================================================

【介面定義】
定義無人機飛行姿態估算與地面煙霧事件座標解算的標準介面。
==============================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BaseLocalizer(ABC):
    """地面空間定位與視覺里程計抽象基底類別"""

    @abstractmethod
    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """
        接收逐格影像，更新視覺里程計與無人機目前估算姿態。

        :param frame_bgr: 當前影格
        :param frame_idx: 影格編號
        :return: 包含 pose, status, tracking_quality 等資訊的狀態字典
        """
        pass

    @abstractmethod
    def estimate_ground_coordinates(
        self,
        pixel_xy: Tuple[float, float],
        frame_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        根據相機幾何與圖資，將像素座標反投影至地面真實座標。

        :param pixel_xy: 影像上的像素座標 (x, y)
        :param frame_idx: 所在影格編號
        :return: 包含 wgs84 (lon, lat), twd97 (e, n), confidence 等資訊，若無法估算則回傳 None
        """
        pass

    @abstractmethod
    def render_map_panel(self, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        繪製當前航跡、無人機視野與地面事件標註的地圖面板影像。

        :param target_size: 輸出的 (寬, 高)；若為 None 則維持預設尺寸
        :return: 繪製好的地圖面板 BGR 影像
        """
        pass
