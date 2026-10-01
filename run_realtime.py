#!/usr/bin/env python3
# 檔案功能：Spinnaker 即時相機入口：取像、偵測、畫面顯示與事件保存。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""
==============================================================================
Boom 即時無人機煙霧偵測執行器 (run_realtime.py)
==============================================================================

【模組定位】
本腳本為無人機機載即時煙霧偵測與定位的主執行入口。
透過 Spinnaker SDK 連接 FLIR 機載相機，即時擷取影格，並依指定路線執行光流或 YOLO 偵測。

【使用範例】
  python run_realtime.py --help
  python run_realtime.py motion --serial 12345678 --exposure-us 5000
  python run_realtime.py yolo --display
==============================================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from boom.online.cli import main

if __name__ == "__main__":
    sys.exit(main() or 0)
