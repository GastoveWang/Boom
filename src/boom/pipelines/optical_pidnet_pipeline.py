"""
==============================================================================
Boom 離線管線 - 光流觸發與 PIDNet 確認之兩階段融合入口 (optical_pidnet_pipeline.py)
==============================================================================

【檔案定位】
光流 + PIDNet 兩階段融合偵測演算法的專屬命令列執行入口。
將預設偵測器固定為 `fusion`。由光流進行低延遲初步起煙觸發，再由 PIDNet 進行
語意分割特徵確認，兼具低延遲與高可靠度。

【核心功能】
- `main(argv=None)`: 鎖定 detector="fusion" 並調用 `pipelines.main._run_pipeline`。

【使用範例】
python artillery/offline/src/pipelines/optical_pidnet_pipeline.py --video data/0603/DJI_001_V.MP4 --display
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
    """Run optical-flow/PIDNet fusion with this method's own CLI options."""
    return _run_pipeline(default_detector="fusion", allow_detector_selection=False, argv=argv)


if __name__ == "__main__":
    main()
