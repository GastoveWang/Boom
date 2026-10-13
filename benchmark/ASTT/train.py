# 檔案功能：正式訓練入口：正常片段訓練、獨立 validation 與保存最佳權重。
# 所屬分工：根目錄入口：相當於 Boom 的 run_offline.py。
# 輸入／輸出：CLI 參數與資料／checkpoint → output/astt/<experiment-name>/。
# 執行方式：從 Boom 根目錄執行 python benchmark/ASTT/train.py --help。
# 設定與說明：正式參數見 src/config.py；說明見 doc/RUN_SETTINGS.md。

import sys
from pathlib import Path

_source = str(Path(__file__).resolve().parents[2])
if _source not in sys.path:
    sys.path.insert(0, _source)

from benchmark.ASTT.src.config import parser
from benchmark.ASTT.src.runner import train

if __name__ == "__main__":
    train(parser(training=True).parse_args())
