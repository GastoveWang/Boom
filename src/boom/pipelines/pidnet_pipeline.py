"""
==============================================================================
Boom 離線管線 - PIDNet-S 深度學習煙霧辨識入口 (pidnet_pipeline.py)
==============================================================================

【檔案定位】
PIDNet-S 語意分割演算法的專屬命令列執行入口。將預設偵測器固定為 `pidnet`，
適合以深度學習分析煙霧紋理與外觀特徵。

【核心功能】
- `main(argv=None)`: 鎖定 detector="pidnet" 並調用 `pipelines.main._run_pipeline`。

【使用範例】
python artillery/offline/src/pipelines/pidnet_pipeline.py --video data/0603/DJI_001_V.MP4 --display
==============================================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boom.pipelines.main import main as _run_pipeline


def main(argv=None):
    """Run PIDNet detection with this method's own CLI options."""
    return _run_pipeline(default_detector="pidnet", allow_detector_selection=False, argv=argv)


if __name__ == "__main__":
    main()
