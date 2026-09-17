"""
==============================================================================
Boom 離線管線 - 光流動態煙霧偵測入口 (optical_flow_pipeline.py)
==============================================================================

【檔案定位】
傳統影格差分與光流動態煙塵偵測演算法的專屬命令列執行入口。
將預設偵測器固定為 `motion`，適合分析爆炸衝擊與起煙瞬間的運動擴散。

【核心功能】
- `main(argv=None)`: 鎖定 detector="motion" 並調用 `pipelines.main._run_pipeline`。

【使用範例】
python artillery/offline/src/pipelines/optical_flow_pipeline.py --video data/0603/DJI_001_V.MP4 --display
==============================================================================
"""

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
for p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from boom.pipelines.main import main as _run_pipeline


def main(argv=None):
    """Run optical-flow detection with this method's own CLI options."""
    return _run_pipeline(default_detector="motion", allow_detector_selection=False, argv=argv)


if __name__ == "__main__":
    main()
