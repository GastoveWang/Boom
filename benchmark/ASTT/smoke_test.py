# 檔案功能：自我檢查相容入口，轉交 tests/smoke_test.py。
# 所屬分工：根目錄入口：保留原有自我檢查指令。
# 輸入／輸出：暫存 fixtures → PASS／例外。
# 執行方式：從 Boom 根目錄執行 python benchmark/ASTT/smoke_test.py。
# 設定與說明：正式參數見 src/config/cli.py；說明見 doc/RUN_SETTINGS.md。

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "tests" / "smoke_test.py"), run_name="__main__")
