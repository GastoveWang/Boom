#!/usr/bin/env python3
"""
==============================================================================
Boom 離線無人機荒野煙霧偵測與定位系統 - 根目錄快速啟動器 (run_offline.py)
==============================================================================

【檔案定位】
本腳本為 Boom 離線處理流程的頂層直覺啟動入口。讓研發人員與使用者不必輸入深層的
模組路徑，即可一鍵執行煙霧偵測、地圖定位或不同演算法比對。

【核心功能】
1. 支援子命令快捷語法：可直接指定 `pidnet`、`motion` 或 `fusion` 作為首個參數。
2. 智慧輸入容錯：自動配對無人機照片（如 DJI_001_P.JPG 與 DJI_001_V.MP4），無照片時自動降級無地圖模式。
3. 包含完整執行參數提示與常用範例指令。

【常用執行指令範例】
1. 使用預設 PIDNet-S 進行煙霧辨識：
   python run_offline.py --video data/0603/DJI_001_V.MP4

2. 指定使用光流法 (Motion) 快速偵測：
   python run_offline.py motion --video data/0603/DJI_001_V.MP4

3. 指定使用兩階段融合演算法 (Fusion = Optical onset + PIDNet confirmation)：
   python run_offline.py fusion --video data/0603/DJI_001_V.MP4

4. 啟用即時預覽視窗（顯示標註框與疊圖）：
   python run_offline.py pidnet --video data/0603/DJI_001_V.MP4 --display

5. 啟用全功能地圖匹配與無人機視覺里程計 (VO) 定位：
   python run_offline.py pidnet --video data/0603/DJI_001_V.MP4 --reference-image data/0603/DJI_001_P.JPG --map-dir asset/maps --allow-gps-seed --display

==============================================================================
"""

import sys
from pathlib import Path

# 確保專案根目錄加入 Python 搜尋路徑
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boom.config import get_config
from boom.pipelines.main import main as run_pipeline


def print_quick_help():
    print("""
Boom 離線無人機荒野煙霧偵測與定位 - 快速執行指南
====================================================================
使用方式:
  python run_offline.py [演算法] [選項...]

可用演算法 (預設為 pidnet):
  pidnet   : PIDNet-S 語意分割偵測器（適合煙霧外觀特徵辨識）
  motion   : 光流與影格差分偵測器（適合起煙瞬間與動態煙塵）
  fusion   : 兩階段融合偵測器（光流即時觸發 + PIDNet 特徵確認）

常用選項:
  --video <路徑>           輸入影片路徑 (預設自動搜尋 data/ 下的影片)
  --reference-image <路徑> DJI GPS/XMP 照片 (若有同名 _P.JPG 會自動配對)
  --no-map                 停用地圖與 VO，僅進行純影片煙霧偵測
  --map-dir <目錄>         GeoTIFF 航拍地圖目錄 (預設為 asset/maps)
  --allow-gps-seed         允許在特徵匹配不足時採用照片 GPS 暫估初始位置
  --display                在桌面彈出即時處理視窗與地圖追蹤視窗
  --max-frames <數量>      限制處理影格數 (用於快速驗證測試)
  --device <cpu|cuda|auto> 指定模型推論硬體設備

範例:
  python run_offline.py --video data/0603/DJI_001_V.MP4
  python run_offline.py motion --video data/0603/DJI_001_V.MP4 --display
  python run_offline.py fusion --video data/0603/DJI_001_V.MP4 --max-frames 100
====================================================================
""")


def main():
    args = list(sys.argv[1:])

    # 顯示客製快速指南
    if "-h" in args or "--help" in args:
        print_quick_help()
        # 呼叫原生 argparse --help 輸出完整選項列表
        try:
            from boom.pipelines.cli import build_arg_parser
            build_arg_parser().print_help()
        except Exception:
            pass
        return 0

    # 檢查第一個參數是否為演算法快捷名稱
    detector = get_config()["detector"]["backend"]
    if args and args[0].lower() in ("pidnet", "motion", "fusion"):
        detector = args[0].lower()
        args = args[1:]
    elif args and args[0].lower() in ("optical_flow", "flow"):
        detector = "motion"
        args = args[1:]

    # 如果 args 中未指定 --detector，主動帶入快捷演算法
    if not any(arg == "--detector" or arg.startswith("--detector=") for arg in args):
        args = ["--detector", detector] + args

    return run_pipeline(default_detector=detector, allow_detector_selection=True, argv=args)


if __name__ == "__main__":
    sys.exit(main() or 0)
