# 檔案功能：正式驗證／測試入口：載入本機 checkpoint，產生異常候選結果。
# 所屬分工：根目錄入口：相當於 Boom 的 run_offline.py。
# 輸入／輸出：CLI 參數與資料／checkpoint → output/astt/<experiment-name>/。
# 執行方式：從 Boom 根目錄執行 python benchmark/ASTT/test.py --help。
# 設定與說明：正式參數見 src/config/cli.py；說明見 doc/RUN_SETTINGS.md。

import sys
from pathlib import Path

_source = str(Path(__file__).resolve().parents[2])
if _source not in sys.path:
    sys.path.insert(0, _source)

from benchmark.ASTT.src.config.cli import parser
from benchmark.ASTT.src.runtime.runner import infer

if __name__ == "__main__":
    infer(parser(training=False).parse_args())
