# 檔案功能：舊匯入路徑相容入口；實作已移至 src/candidates.py。
# 所屬分工：相容層：保留 from model／dataset／localization／runner 的舊用法。
# 輸入／輸出：舊模組 import → src 的正式模組。
# 執行方式：不直接執行；修改功能請開啟對應 src 檔案。
# 設定與說明：正式參數見 src/config.py；說明見 doc/RUN_SETTINGS.md。

import importlib as _importlib
import sys as _sys
from pathlib import Path as _Path

_source = str(_Path(__file__).resolve().parents[2])
if _source not in _sys.path:
    _sys.path.insert(0, _source)
_sys.modules[__name__] = _importlib.import_module("benchmark.ASTT.src.candidates")
