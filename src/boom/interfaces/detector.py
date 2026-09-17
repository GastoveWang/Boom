"""
==============================================================================
Boom 抽象介面 - 煙霧偵測器標準契約 (detector.py)
==============================================================================

【介面定義】
所有煙霧偵測演算法（PIDNet 語意分割、光流法、兩階段融合、以及未來的 YOLO 模型）
皆必須實作此抽象介面。這保證了上層 Pipeline 在抽換演算法時不需要修改任何調度邏輯。
==============================================================================
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from boom.core.events import ConfirmedEvent


class BaseSmokeDetector(ABC):
    """煙霧偵測演算法抽象基底類別"""

    @abstractmethod
    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        接收當前輸入影像，執行推論與特徵分析。

        :param frame_bgr: 原始 BGR 格式圖像
        :param frame_idx: 當前影格編號 (0-indexed)
        :return: 繪製有特徵、遮罩或標註的視覺化影像
        """
        pass

    @abstractmethod
    def consume_pending_confirmations(self) -> List[ConfirmedEvent]:
        """
        取出當前影格判定為「正式確認起煙」的事件清單。
        調用後應清空內部的待回傳緩衝區。

        :return: ConfirmedEvent 清單
        """
        pass

    @property
    def fps(self) -> float:
        """偵測器設定之運作 FPS"""
        return getattr(self, "_fps", 30.0)

    @fps.setter
    def fps(self, value: float) -> None:
        self._fps = float(value)
