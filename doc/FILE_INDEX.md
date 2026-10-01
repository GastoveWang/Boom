# Boom 逐檔功能索引

Boom 是一套用於飛行中無人機的荒野煙霧偵測與定位系統，目標是在影像中即時標示煙霧區域，並估算疑似煙霧源的地面座標。

此索引涵蓋專案原始碼、工具、設定與文件；Python 檔案頂端也有功能註記。資料、權重、輸出、第三方標註軟體與 Git/快取不逐檔註記，以免修改使用者資料。

直接執行的入口和參數見 [RUN_SETTINGS.md](RUN_SETTINGS.md)，ASTT 分組見 [START_HERE.md](../benchmark/ASTT/START_HERE.md)。

| 檔案 | 功能 |
|---|---|
| [.gitignore](../.gitignore) | 版本控制／目錄保留設定。 |
| [AGENTS.md](../AGENTS.md) | 專案目標、功能優先順序與修改／驗證原則。 |
| [benchmark/ASTT/.gitignore](../benchmark/ASTT/.gitignore) | 版本控制／目錄保留設定。 |
| [benchmark/ASTT/__init__.py](../benchmark/ASTT/__init__.py) | ASTT 套件初始化與公開匯出；不直接執行。 |
| [benchmark/ASTT/check_jax.py](../benchmark/ASTT/check_jax.py) | 歷史 JAX 權重檢查入口。 |
| [benchmark/ASTT/checkpoint.py](../benchmark/ASTT/checkpoint.py) | 歷史 JAX/ViT 權重載入與轉換。 |
| [benchmark/ASTT/config.py](../benchmark/ASTT/config.py) | 歷史 ViT argparse 設定；正式入口不使用。 |
| [benchmark/ASTT/data_loaders.py](../benchmark/ASTT/data_loaders.py) | 歷史 CIFAR/ImageNet loader；正式入口不使用。 |
| [benchmark/ASTT/data_utils.py](../benchmark/ASTT/data_utils.py) | 歷史影格 loader；正式入口改用 dataset.py。 |
| [benchmark/ASTT/dataset.py](../benchmark/ASTT/dataset.py) | 每影片連續影格窗口、缺幀／標籤校驗與可選 ECC。 |
| [benchmark/ASTT/eval.py](../benchmark/ASTT/eval.py) | 歷史影像分類評估入口，非本機異常測試。 |
| [benchmark/ASTT/localization.py](../benchmark/ASTT/localization.py) | 誤差候選 bbox 與事件去抖合併，非 GPS 定位。 |
| [benchmark/ASTT/model.py](../benchmark/ASTT/model.py) | 本機 STE/TTE/CSA 模型與下一幀預測 decoder。 |
| [benchmark/ASTT/model_baseline_ca.py](../benchmark/ASTT/model_baseline_ca.py) | 歷史 baseline + cross attention 模型，研究參考。 |
| [benchmark/ASTT/model_original.py](../benchmark/ASTT/model_original.py) | 歷史原始模型，研究參考。 |
| [benchmark/ASTT/model_ste_tte.py](../benchmark/ASTT/model_ste_tte.py) | 歷史 STE/TTE 模型，研究參考。 |
| [benchmark/ASTT/README.md](../benchmark/ASTT/README.md) | ASTT 上游論文、作者與引用資訊。 |
| [benchmark/ASTT/README_LOCAL.md](../benchmark/ASTT/README_LOCAL.md) | ASTT 本機適配、模型解讀、資料格式、設定與限制。 |
| [benchmark/ASTT/requirements.txt](../benchmark/ASTT/requirements.txt) | 正式 ASTT 流程的 torch、NumPy、OpenCV 依賴。 |
| [benchmark/ASTT/runner.py](../benchmark/ASTT/runner.py) | 訓練、驗證、推論、CLI、checkpoint 與輸出核心。 |
| [benchmark/ASTT/scripts/extract_frames.py](../benchmark/ASTT/scripts/extract_frames.py) | 正式 MP4 抽幀入口，保存影格與來源幀號／時間 manifest。 |
| [benchmark/ASTT/smoke_test.py](../benchmark/ASTT/smoke_test.py) | 暫存資料上的模型、資料、事件与指標自我檢查。 |
| [benchmark/ASTT/START_HERE.md](../benchmark/ASTT/START_HERE.md) | ASTT 正式／歷史檔案分组與訓練、測試快速操作。 |
| [benchmark/ASTT/test.py](../benchmark/ASTT/test.py) | 正式驗證／測試入口，必須提供本機 checkpoint。 |
| [benchmark/ASTT/train.py](../benchmark/ASTT/train.py) | 正式訓練入口，設定由 runner.parser 定義。 |
| [benchmark/ASTT/Train_and_eval.py](../benchmark/ASTT/Train_and_eval.py) | 歷史資料巡覽與訓練評估腳本，含上游路徑。 |
| [benchmark/ASTT/train_val_UIT_ADrone.py](../benchmark/ASTT/train_val_UIT_ADrone.py) | 歷史 UIT_ADrone 資料集／模型實驗；含上游路徑，非正式入口。 |
| [benchmark/ASTT/train_val_UIT_ADrone_baseline_ca.py](../benchmark/ASTT/train_val_UIT_ADrone_baseline_ca.py) | 歷史 UIT_ADrone_baseline_ca 資料集／模型實驗；含上游路徑，非正式入口。 |
| [benchmark/ASTT/train_val_UIT_ADrone_ste_tte.py](../benchmark/ASTT/train_val_UIT_ADrone_ste_tte.py) | 歷史 UIT_ADrone_ste_tte 資料集／模型實驗；含上游路徑，非正式入口。 |
| [benchmark/ASTT/utils.py](../benchmark/ASTT/utils.py) | 歷史 metrics、logging 與設定工具。 |
| [configs/default.yaml](../configs/default.yaml) | 執行設定；用途與合併順序見 configs/README.md。 |
| [configs/detector-motion.yaml](../configs/detector-motion.yaml) | 執行設定；用途與合併順序見 configs/README.md。 |
| [configs/detector-yolo.yaml](../configs/detector-yolo.yaml) | 執行設定；用途與合併順序見 configs/README.md。 |
| [configs/detector.yaml](../configs/detector.yaml) | 執行設定；用途與合併順序見 configs/README.md。 |
| [configs/localization.yaml](../configs/localization.yaml) | 執行設定；用途與合併順序見 configs/README.md。 |
| [configs/matching.yaml](../configs/matching.yaml) | 執行設定；用途與合併順序見 configs/README.md。 |
| [configs/pipeline.yaml](../configs/pipeline.yaml) | 執行設定；用途與合併順序見 configs/README.md。 |
| [configs/README.md](../configs/README.md) | 分類 YAML 用途、覆寫順序與地圖匹配設定。 |
| [configs/system.yaml](../configs/system.yaml) | 執行設定；用途與合併順序見 configs/README.md。 |
| [configs/ui.yaml](../configs/ui.yaml) | 執行設定；用途與合併順序見 configs/README.md。 |
| [doc/ACCELERATION.md](../doc/ACCELERATION.md) | YOLO／Motion 執行效能、GPU 與顯示加速。 |
| [doc/ASSET_UI.md](../doc/ASSET_UI.md) | 地圖與 UI 圖示資源的來源與用途。 |
| [doc/FILE_INDEX.md](../doc/FILE_INDEX.md) | 專案逐檔用途與資源目錄索引。 |
| [doc/MAP_LOCALIZATION.md](../doc/MAP_LOCALIZATION.md) | GeoTIFF 地圖、特徵匹配、參考照片與地面定位。 |
| [doc/ONLINE.md](../doc/ONLINE.md) | 即時 Spinnaker 相機、偵測、輸出與設備設定。 |
| [doc/PROJECT_STRUCTURE.md](../doc/PROJECT_STRUCTURE.md) | 專案架構、偵測流程與離線／即時執行範例。 |
| [doc/README.md](../doc/README.md) | 文件入口與比較專案的 Git 追蹤規則。 |
| [doc/RUN_SETTINGS.md](../doc/RUN_SETTINGS.md) | 執行入口、目前 YAML 值與 CLI 完整參數表。 |
| [doc/TOOLS.md](../doc/TOOLS.md) | 抽幀、YOLO 資料切分、訓練與 ONNX 匯出指令。 |
| [pyproject.toml](../pyproject.toml) | Boom 套件、Python 版本、基礎／ML／ONNX／地圖依賴與安裝設定。 |
| [run_offline.py](../run_offline.py) | 離線影片入口：選擇 motion 或 YOLO，執行偵測、顯示、保存與可選定位。 |
| [run_realtime.py](../run_realtime.py) | Spinnaker 即時相機入口：取像、偵測、畫面顯示與事件保存。 |
| [src/boom/__init__.py](../src/boom/__init__.py) | boom 套件初始化與公開匯出；不直接執行。 |
| [src/boom/config/__init__.py](../src/boom/config/__init__.py) | config 套件初始化與公開匯出；不直接執行。 |
| [src/boom/config/defaults.py](../src/boom/config/defaults.py) | 共用預設常數與地圖匹配設定結構。 |
| [src/boom/config/loader.py](../src/boom/config/loader.py) | 分類 YAML 讀取、深度合併與設定存取。 |
| [src/boom/core/coordinates.py](../src/boom/core/coordinates.py) | WGS84/TWD97 座標轉換與地面投影。 |
| [src/boom/core/events.py](../src/boom/core/events.py) | 地理參考、確認事件與紀錄資料結構。 |
| [src/boom/detection/__init__.py](../src/boom/detection/__init__.py) | detection 套件初始化與公開匯出；不直接執行。 |
| [src/boom/detection/factory.py](../src/boom/detection/factory.py) | 建立 motion/YOLO 偵測器及對應設定。 |
| [src/boom/detection/optical_flow/__init__.py](../src/boom/detection/optical_flow/__init__.py) | optical_flow 套件初始化與公開匯出；不直接執行。 |
| [src/boom/detection/optical_flow/detector.py](../src/boom/detection/optical_flow/detector.py) | 影像變化、相機運動補償、光流與煙塵候選事件追蹤。 |
| [src/boom/detection/optical_flow/runner.py](../src/boom/detection/optical_flow/runner.py) | motion 單影片／批次 runner 與獨立 CLI。 |
| [src/boom/detection/optical_flow/types.py](../src/boom/detection/optical_flow/types.py) | motion 設定、影格、候選區域、追蹤與輸出資料結構。 |
| [src/boom/detection/yolo/__init__.py](../src/boom/detection/yolo/__init__.py) | yolo 套件初始化與公開匯出；不直接執行。 |
| [src/boom/detection/yolo/detector.py](../src/boom/detection/yolo/detector.py) | YOLO 推論、目標類別篩選與事件追蹤。 |
| [src/boom/interfaces/__init__.py](../src/boom/interfaces/__init__.py) | interfaces 套件初始化與公開匯出；不直接執行。 |
| [src/boom/interfaces/detector.py](../src/boom/interfaces/detector.py) | 偵測器共用抽象介面。 |
| [src/boom/interfaces/localizer.py](../src/boom/interfaces/localizer.py) | 定位器共用抽象介面。 |
| [src/boom/interfaces/matcher.py](../src/boom/interfaces/matcher.py) | 特徵匹配器共用抽象介面。 |
| [src/boom/interfaces/sensor.py](../src/boom/interfaces/sensor.py) | 影格來源共用抽象介面。 |
| [src/boom/localization/__init__.py](../src/boom/localization/__init__.py) | localization 套件初始化與公開匯出；不直接執行。 |
| [src/boom/localization/edm_matching.py](../src/boom/localization/edm_matching.py) | 固定尺寸 EDM ONNX 影像對匹配後端。 |
| [src/boom/localization/feature_matching.py](../src/boom/localization/feature_matching.py) | SIFT、SuperPoint/LightGlue 後端與建立工廠。 |
| [src/boom/localization/geo_map.py](../src/boom/localization/geo_map.py) | GeoTIFF 讀取、ROI 與地圖座標轉換。 |
| [src/boom/localization/geometry.py](../src/boom/localization/geometry.py) | 相機姿態、平面 homography 與姿態估計。 |
| [src/boom/localization/localizer.py](../src/boom/localization/localizer.py) | 地圖定位器外觀與 session 管理。 |
| [src/boom/localization/map_panel.py](../src/boom/localization/map_panel.py) | 地圖畫面與無人機圖示合成。 |
| [src/boom/localization/map_widgets.py](../src/boom/localization/map_widgets.py) | 方位、羅盤、台灣概覽與導航卡。 |
| [src/boom/localization/nearby.py](../src/boom/localization/nearby.py) | 鄰近地圖 ROI 特徵匹配與定位結果。 |
| [src/boom/localization/photo.py](../src/boom/localization/photo.py) | DJI 照片 Exif/XMP 與相機姿態讀取。 |
| [src/boom/localization/regions.py](../src/boom/localization/regions.py) | GeoTIFF 索引、邊界與鄰近圖幅選擇。 |
| [src/boom/localization/registration.py](../src/boom/localization/registration.py) | 參考照片／地圖配準與矩陣校正。 |
| [src/boom/localization/visual_odometry.py](../src/boom/localization/visual_odometry.py) | 影片視覺里程與相機姿態追蹤。 |
| [src/boom/online/__init__.py](../src/boom/online/__init__.py) | online 套件初始化與公開匯出；不直接執行。 |
| [src/boom/online/cli.py](../src/boom/online/cli.py) | 即時應用程式啟動入口。 |
| [src/boom/online/configuration.py](../src/boom/online/configuration.py) | 即時 CLI、相機及偵測設定校驗。 |
| [src/boom/online/display.py](../src/boom/online/display.py) | 即時事件疊圖。 |
| [src/boom/online/logging_setup.py](../src/boom/online/logging_setup.py) | 主控台與 session 檔案 logging。 |
| [src/boom/online/pipeline.py](../src/boom/online/pipeline.py) | 取像、即時偵測與證據輸出協調。 |
| [src/boom/online/positioning.py](../src/boom/online/positioning.py) | 可選固定姿態的地面位置估計。 |
| [src/boom/online/runner.py](../src/boom/online/runner.py) | 既有即時入口的相容匯出。 |
| [src/boom/online/spin_camera.py](../src/boom/online/spin_camera.py) | PySpin 相機設定、取像與最新影格執行緒。 |
| [src/boom/online/spinnaker_camera.py](../src/boom/online/spinnaker_camera.py) | 公開 Spinnaker 相機相容模組。 |
| [src/boom/online/storage.py](../src/boom/online/storage.py) | 事件 JSONL/CSV、座標與截圖保存。 |
| [src/boom/pipelines/__init__.py](../src/boom/pipelines/__init__.py) | pipelines 套件初始化與公開匯出；不直接執行。 |
| [src/boom/pipelines/cli.py](../src/boom/pipelines/cli.py) | 離線 CLI 定義、參數校驗與地圖匹配設定。 |
| [src/boom/pipelines/main.py](../src/boom/pipelines/main.py) | 離線流程選擇與啟動。 |
| [src/boom/pipelines/map_localization.py](../src/boom/pipelines/map_localization.py) | 地圖定位舊匯入路徑相容層。 |
| [src/boom/pipelines/optical_flow_pipeline.py](../src/boom/pipelines/optical_flow_pipeline.py) | motion 離線流程相容入口。 |
| [src/boom/runtime/__init__.py](../src/boom/runtime/__init__.py) | runtime 套件初始化與公開匯出；不直接執行。 |
| [src/boom/runtime/async_output.py](../src/boom/runtime/async_output.py) | 非同步影片輸出與佇列管理。 |
| [src/boom/runtime/events.py](../src/boom/runtime/events.py) | 事件生命週期、地理定位與歷史。 |
| [src/boom/runtime/inputs.py](../src/boom/runtime/inputs.py) | 影片及參考照片路徑解析與影片來源。 |
| [src/boom/runtime/live_preview.py](../src/boom/runtime/live_preview.py) | 主執行緒 HighGUI 與最新預覽畫面傳遞。 |
| [src/boom/runtime/output.py](../src/boom/runtime/output.py) | 輸出路徑、座標日誌與影片寫入。 |
| [src/boom/runtime/preview.py](../src/boom/runtime/preview.py) | 預覽縮放，保持偵測座標不變。 |
| [src/boom/runtime/progress.py](../src/boom/runtime/progress.py) | 終端進度、耗時與結果統計。 |
| [src/boom/runtime/reference.py](../src/boom/runtime/reference.py) | 參考照片 GPS/XMP 地理資訊解析。 |
| [src/boom/runtime/runner.py](../src/boom/runtime/runner.py) | 離線影格處理、偵測、定位與輸出主迴圈。 |
| [src/boom/ui/__init__.py](../src/boom/ui/__init__.py) | ui 套件初始化與公開匯出；不直接執行。 |
| [src/boom/ui/composition.py](../src/boom/ui/composition.py) | 影片、側欄與地圖多面板合成。 |
| [src/boom/ui/drawing.py](../src/boom/ui/drawing.py) | 文字與確認事件標記繪製。 |
| [src/boom/ui/event_style.py](../src/boom/ui/event_style.py) | 事件標籤與顏色。 |
| [src/boom/ui/panels.py](../src/boom/ui/panels.py) | 事件卡、debug/client 側欄與地圖事件面板。 |
| [src/boom/ui/theme.py](../src/boom/ui/theme.py) | 配色與圓角卡片繪圖。 |
| [tools/benchmark_matchers.py](../tools/benchmark_matchers.py) | 同影像對比較 EDM 與 SuperPoint/LightGlue 的匹配表現。 |
| [tools/check_camera.py](../tools/check_camera.py) | 列出 PySpin SDK 與相機，可指定幀數驗證取像。 |
| [tools/split_dataset.py](../tools/split_dataset.py) | YOLO 偵測／分割資料版本管理、平衡切分與 GUI。 |
| [tools/train_yolo.py](../tools/train_yolo.py) | 互動式 YOLO 訓練、選擇本機模型與 ONNX 匯出。 |
| [tools/update_file_index.py](../tools/update_file_index.py) | 更新逐檔用途索引、Python 功能註記與 CLI 設定表。 |
| [tools/video_to_frames.py](../tools/video_to_frames.py) | 將影片依時間間隔抽成圖片，保留原始解析度。 |

## 資源與未納入 Git 的目錄

| 目錄 | 功能 |
|---|---|
| data/picture | 使用者影格與標註；不自動改寫或視為正常資料。 |
| data/astt | ASTT 的 train、val、test 影格與逐幀標籤。 |
| data/yolo_datasets | YOLO 原始／版本化訓練資料。 |
| data/video、data/raw_video | 影片輸入。 |
| models | 預訓練權重、checkpoint 與 ONNX，不提交大型模型。 |
| asset | UI、字型、圖示與地圖資源；大型地圖不提交。 |
| output | 離線、即時、訓練與比較結果，不提交。 |
| tests | 本機驗證程式；目前根 .gitignore 排除，未更改此規則。 |
| X-AnyLabeling、cvat | 外部標註軟體；不修改、不加入 Boom Git。 |

更新索引：`python tools/update_file_index.py`（使用 boom 環境）。
