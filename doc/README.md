# Boom 文件

- [專案結構與執行](PROJECT_STRUCTURE.md)
- [逐檔功能索引](FILE_INDEX.md)
- [執行入口與完整設定](RUN_SETTINGS.md)
- [即時相機流程](ONLINE.md)
- [YOLO 與 Motion 加速](ACCELERATION.md)
- [地圖與定位](MAP_LOCALIZATION.md)
- [資料準備與模型訓練工具](TOOLS.md)
- [地圖 UI 資料來源](ASSET_UI.md)

## Git 分支

日常開發與推送使用 `dev`，可包含 benchmark 原始碼。`master` 只保留 Boom 專案本身，整個 `benchmark/` 已排除。更新 master 時只同步專案變更，不直接合併含 benchmark 的 dev。資料、模型與輸出仍依忽略規則保留在本機。
