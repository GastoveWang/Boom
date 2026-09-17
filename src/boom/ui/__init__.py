"""
==============================================================================
Boom 離線使用者介面與畫面呈現模組 (artillery.offline.src.ui)
==============================================================================

【模組定位】
本模組負責離線視訊處理過程中的所有視覺呈現與畫面合成，包括煙霧標註框疊圖、
文字資訊面板、色彩主題規範與多面板畫布合成。

【核心元件與職責】
1. `composition.py`: 最終畫面合成 (`compose_frame`)，將偵測畫面、地圖視窗與資訊面板拼接。
2. `drawing.py`: 煙霧標記繪製、中文文字渲染與幾何繪圖輔助。
3. `panels.py`: 座標與狀態資訊面板繪製 (`build_side_panel`, `build_map_event_panel`)。
4. `theme.py`: 現代化暗色系調色盤與圓角卡片工具 (`SURFACE`, `BORDER`, `ACCENT`, `TEXT`)。
5. `event_style.py`: 煙霧事件外框顏色與樣式定義。

【導出類別與函式】
- `compose_frame`: 畫面合成主函式
- `build_side_panel`, `build_map_event_panel`, `get_panel_width`: 側邊面板渲染
- `SURFACE`, `BORDER`, `ACCENT`, `TEXT`: UI 調色盤常數
==============================================================================
"""

from .composition import compose_frame
from .panels import build_map_event_panel, build_side_panel, get_panel_width
from .theme import ACCENT, BORDER, CARD, SURFACE, TEXT, rounded_surface

__all__ = [
    "compose_frame",
    "build_side_panel",
    "build_map_event_panel",
    "get_panel_width",
    "SURFACE",
    "BORDER",
    "CARD",
    "ACCENT",
    "TEXT",
    "rounded_surface",
]
