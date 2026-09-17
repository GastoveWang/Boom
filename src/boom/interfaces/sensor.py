"""
==============================================================================
Boom 抽象介面 - 影像感測來源標準契約 (sensor.py)
==============================================================================

【介面定義】
定義影片檔、相機串流 (RTSP/USB/V4L2) 或模擬產生器的共通影格擷取介面。
解耦離線與線上推論流程。
==============================================================================
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class BaseFrameSource(ABC):
    """影格輸入來源抽象基底類別"""

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """
        讀取下一影格。

        :return: (success, frame_bgr, frame_index)
                 若結束或讀取失敗，success 為 False，frame_bgr 為 None
        """
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """影像來源的影格率"""
        pass

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """影像來源解析度 (寬, 高)"""
        pass

    @abstractmethod
    def close(self) -> None:
        """釋放影片檔或串流相機資源"""
        pass
