# Boom 文件導覽

更新日期：2026-10-27。本文是目前操作的入口；歷史方法與文獻快照另行分類。

## 先看哪一份

| 想做的事 | 文件 |
|---|---|
| 了解目錄與 Boom／ST² 的分工 | [專案結構](PROJECT_STRUCTURE.md) |
| 訓練、續訓或推論 ST² | [ST² 模型與操作](ST2.md) |
| 標註影片與產生 ST² 資料集 | [ST² 資料集](ST2_DATASET.md) |
| 執行離線影片與即時相機辨識 | [執行設定](RUN_SETTINGS.md)、[即時相機](ONLINE.md) |
| 找檔案負責什麼 | [逐檔功能索引](FILE_INDEX.md) |

目前 Boom 提供 motion／YOLO 偵測器；ST² 提供獨立模型與逐幀熱圖介面，尚未接入即時煙霧事件確認。舊 Smoke Heatmap 入口與 baseline／camera-aware 研究程式已移除，既有資料、權重與輸出保留。

## Boom 執行與工具

- [執行入口、YAML 與 CLI 參數](RUN_SETTINGS.md)
- [即時相機與顯示流程](ONLINE.md)
- [YOLO 與 Motion 加速](ACCELERATION.md)
- [地圖與座標定位](MAP_LOCALIZATION.md)
- [資料準備、相機檢查與訓練工具](TOOLS.md)
- [地圖 UI 資料來源](ASSET_UI.md)

## ST² 模型與紀錄

- [ST² 架構、訓練、續訓、推論與限制](ST2.md)
- [ST² 逐幀框標註、資料格式與匯出](ST2_DATASET.md)
- [ST² torchview 模型視覺化](ST2_VISUALIZATION.md)
- [目前架構與 ASTT 的差異及歷史演進](MODEL_EVOLUTION.md)
- [完整模型與推論變更紀錄](MODEL_CHANGELOG.md)

`MODEL_CHANGELOG.md` 依時間保留紀錄；其中舊路徑與指令是當時的狀態，不是目前入口。

## 比較模型、研究與歷史

- [ASTT 快速使用](../benchmark/ASTT/doc/START_HERE.md)
- [HSTforU 套件與驗證](../benchmark/HSTforU/README.md)
- [HSTforU 與目前 ST² 的差異](HSTFORU_COMPARISON.md)
- [ASTT 資料準備](../benchmark/ASTT/doc/DATASET_PREPARATION.md)
- [論文研究索引](research/README.md)
- [歷史方法、搬移與盤點紀錄](history/README.md)

`benchmark/ASTT/` 保留比較模型，`origin/` 是官方快照；其維護與操作規則見該專案文件。歷史輸出不因文件分類而搬移或改寫。
