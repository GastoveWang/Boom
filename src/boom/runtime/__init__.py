"""
==============================================================================
Boom 離線執行期核心模組 (artillery.offline.src.runtime)
==============================================================================

【模組定位】
本模組負責離線煙霧偵測與定位管線的流程排程與執行。協調「輸入讀取 → 地圖定位
→ 演算法偵測 → 事件追蹤 → 畫面合成 → 結果輸出」的單向資料流。

【核心元件與職責】
1. `runner.py`: 主執行迴圈 (`run`)，管理資源 Context、影格處理迴圈與鍵盤互動。
2. `inputs.py`: 視訊與照片輸入解析 (`VideoSource`, `resolve_inputs`)。
3. `events.py`: 事件歷史追蹤、像素至地理座標綁定與 UI 顯示期限 (`EventHistory`, `localize_event`)。
4. `output.py`: 編碼視訊輸出、事件座標日誌寫入 (`VideoOutput`, `write_coordinate_log`)。
5. `progress.py`: 終端執行進度條與統計報告 (`report_progress`, `report_result`)。
6. `reference.py`: 舊版靜態照片 GPS/XMP 參考資訊讀取。

【導出類別與函式】
- `run`: 離線管線執行主函式
- `VideoSource`, `resolve_inputs`: 視訊擷取與輸入解析
- `EventHistory`, `localize_event`: 事件生命週期管理與定位
- `VideoOutput`, `resolve_output_paths`, `write_coordinate_log`: 結果保存
- `report_progress`, `report_result`: 進度與報表
==============================================================================
"""

from .events import EventHistory, localize_event
from .inputs import VideoSource, resolve_inputs
from .output import VideoOutput, resolve_output_paths, write_coordinate_log
from .progress import report_progress, report_result
from .runner import run

__all__ = [
    "run",
    "VideoSource",
    "resolve_inputs",
    "EventHistory",
    "localize_event",
    "VideoOutput",
    "resolve_output_paths",
    "write_coordinate_log",
    "report_progress",
    "report_result",
]
