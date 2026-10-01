# Boom 文件

- [專案結構與執行](PROJECT_STRUCTURE.md)
- [逐檔功能索引](FILE_INDEX.md)
- [執行入口與完整設定](RUN_SETTINGS.md)
- [ASTT 快速使用與檔案分類](../benchmark/ASTT/START_HERE.md)
- [即時相機流程](ONLINE.md)
- [YOLO 與 Motion 加速](ACCELERATION.md)
- [地圖與定位](MAP_LOCALIZATION.md)
- [資料準備與 YOLO 訓練工具](TOOLS.md)
- [地圖 UI 資料來源](ASSET_UI.md)

## 碩論比較專案

新的獨立專案可放在根目錄的 `benchmark/<專案名稱>/`，保留各專案原本的目錄、授權和文件。除 `benchmark/ASTT/` 已納入 Boom Git 外，其他比較專案預設不追蹤。ASTT 使用一般原始碼目錄，不是巢狀 Git 或 submodule；上游資訊保留在 README。Boom 的偵測程式和輸出仍分別放在 `src/boom/` 與 `output/`。

現有的 `X-AnyLabeling/` 保留在根目錄，不移動。
