# 檔案功能：舊抽幀指令相容入口，轉交 tools/extract_frames.py。
# 所屬分工：相容層：原有 scripts 路徑可繼續使用。
# 輸入／輸出：與 tools/extract_frames.py 相同。
# 執行方式：原指令可用；新的操作請使用 tools/extract_frames.py。
# 設定與說明：正式參數見 src/config.py；說明見 doc/RUN_SETTINGS.md。

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.extract_frames import main

if __name__ == "__main__":
    main()
