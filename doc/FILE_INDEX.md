# Boom 逐檔功能索引

Boom 是一套用於飛行中無人機的荒野煙霧偵測與定位系統，目標是在影像中即時標示煙霧區域，並估算疑似煙霧源的地面座標。

此索引涵蓋專案原始碼、工具、設定與文件；Python 檔案頂端也有功能註記。資料、權重、輸出、第三方標註軟體與 Git/快取不逐檔註記，以免修改使用者資料。

直接執行的入口和參數見 [RUN_SETTINGS.md](RUN_SETTINGS.md)，ASTT 分組見 [START_HERE.md](../benchmark/ASTT/doc/START_HERE.md)。

| 檔案 | 功能 |
|---|---|
| [data/st2/st2_v1/](../data/st2/st2_v1/) | 本機ST²資料集；Git僅保留零位元.gitkeep的資料夾結構。 |
| [src/st2/engine/visualize.py](../src/st2/engine/visualize.py) | torchview計算圖追蹤、分支匯出與Graphviz尋找。 |
| [src/st2/cli/visualize.py](../src/st2/cli/visualize.py) | ST² visualize命令與參數。 |
| [doc/ST2_VISUALIZATION.md](ST2_VISUALIZATION.md) | torchview操作、實際SVG快照與限制。 |
| [train_st2.py](../train_st2.py) | 根目錄ST²訓練＋best.pt自動影片推論入口，支援續訓與來源覆寫。 |
| [src/st2/__main__.py](../src/st2/__main__.py) | ST² st2：__main__。 |
| [src/st2/cli/__init__.py](../src/st2/cli/__init__.py) | ST² cli：__init__。 |
| [src/st2/common/__init__.py](../src/st2/common/__init__.py) | ST² common：__init__。 |
| [src/st2/data/__init__.py](../src/st2/data/__init__.py) | ST² data：__init__。 |
| [src/st2/engine/__init__.py](../src/st2/engine/__init__.py) | ST² engine：__init__。 |
| [src/st2/models/__init__.py](../src/st2/models/__init__.py) | ST² models：__init__。 |
| [src/st2/__init__.py](../src/st2/__init__.py) | ST² 獨立套件入口，不預先載入模型。 |
| [src/st2/models/network.py](../src/st2/models/network.py) | ST2Config、ST2Model、ST2Criterion 與主要 loss。 |
| [src/st2/models/motion.py](../src/st2/models/motion.py) | 背景配準 loss、幾何運算與連續相機增強。 |
| [src/st2/data/sequence.py](../src/st2/data/sequence.py) | 來源分組、連續窗口與資料 fingerprint。 |
| [src/st2/data/dataset.py](../src/st2/data/dataset.py) | ST2Dataset：8幀歷史、當前影格與9幀ROI遮罩。 |
| [src/st2/data/annotations.py](../src/st2/data/annotations.py) | 逐幀標註、來源分組及安全的版本匯出。 |
| [src/st2/data/editor.py](../src/st2/data/editor.py) | ST² 影片區段與逐幀框標註介面。 |
| [src/st2/cli/prepare.py](../src/st2/cli/prepare.py) | 標註、分組預覽與匯出命令列實作。 |
| [src/st2/engine/postprocess.py](../src/st2/engine/postprocess.py) | 固定分數換算與可選相機運動檢查。 |
| [src/st2/engine/view.py](../src/st2/engine/view.py) | ST² 熱圖影片與分數曲線顯示。 |
| [src/st2/common/utils.py](../src/st2/common/utils.py) | 前處理、checkpoint、JSON與抽樣工具。 |
| [src/st2/engine/tensorboard.py](../src/st2/engine/tensorboard.py) | 既有訓練數值轉TensorBoard，不改來源。 |
| [src/st2/README.md](../src/st2/README.md) | 獨立套件分工與快速入口。 |
| [README.md](../README.md) | 專案根目錄的閱讀入口。 |
| [doc/history/README.md](history/README.md) | 歷史方法及整理快照索引。 |
| [doc/research/README.md](research/README.md) | 論文研究快照索引。 |
| [doc/MODEL_CHANGELOG.md](MODEL_CHANGELOG.md) | MODEL_CHANGELOG |
| [doc/MODEL_EVOLUTION.md](MODEL_EVOLUTION.md) | MODEL_EVOLUTION |
| [doc/history/ASTT_BBOX_INFERENCE.md](history/ASTT_BBOX_INFERENCE.md) | 歷史文件：ASTT_BBOX_INFERENCE |
| [doc/history/ASTT_BBOX_TRAINING.md](history/ASTT_BBOX_TRAINING.md) | 歷史文件：ASTT_BBOX_TRAINING |
| [doc/history/BBOX_TEMPORAL_MODEL.md](history/BBOX_TEMPORAL_MODEL.md) | 歷史文件：BBOX_TEMPORAL_MODEL |
| [doc/history/FILE_USAGE_AUDIT.md](history/FILE_USAGE_AUDIT.md) | 歷史文件：FILE_USAGE_AUDIT |
| [doc/history/SMOKE_HEATMAP_TRANSFORMER.md](history/SMOKE_HEATMAP_TRANSFORMER.md) | 歷史文件：SMOKE_HEATMAP_TRANSFORMER |
| [doc/research/PAPER_REVIEW_20261026.md](research/PAPER_REVIEW_20261026.md) | 研究快照：PAPER_REVIEW_20261026 |
| [doc/research/TRANSFORMER_PAPERS_20261026.md](research/TRANSFORMER_PAPERS_20261026.md) | 研究快照：TRANSFORMER_PAPERS_20261026 |
| [src/st2/engine/train.py](../src/st2/engine/train.py) | 獨立ST²訓練與續訓。 |
| [src/st2/engine/infer.py](../src/st2/engine/infer.py) | ST²影片推論與結果。 |
| [src/st2/engine/runtime.py](../src/st2/engine/runtime.py) | 供執行環境替換模型的純逐幀推論介面。 |
| [doc/ST2_DATASET.md](ST2_DATASET.md) | ST²標註與匯出格式。 |
| [doc/history/ST2_MIGRATION.md](history/ST2_MIGRATION.md) | ST²與Boom分離及搬移紀錄。 |
| [src/st2/cli/train.py](../src/st2/cli/train.py) | ST²訓練入口。 |
| [src/st2/cli/infer.py](../src/st2/cli/infer.py) | ST²影片推論入口。 |
| [src/st2/cli/prepare.py](../src/st2/cli/prepare.py) | ST²資料標註與匯出入口。 |
| [src/st2/cli/export_tensorboard.py](../src/st2/cli/export_tensorboard.py) | 訓練紀錄轉TensorBoard。 |
| [doc/ST2.md](ST2.md) | ST² v1.0 正式名稱、三分支架構、訓練／推論與驗證紀錄。 |
| [src/st2/models/transformer.py](../src/st2/models/transformer.py) | ST² 純 Transformer 主實作：正常預測、背景運動及目標關注。 |
| [.gitignore](../.gitignore) | 版本控制／目錄保留設定。 |
| [AGENTS.md](../AGENTS.md) | 專案目標、功能優先順序與修改／驗證原則。 |
| [benchmark/ASTT/origin/](../benchmark/ASTT/origin) | 官方原始碼完整快照，禁止修改；來源與校驗见 ASTT/doc/ORIGIN.md。 |
| [benchmark/ASTT/tools/verify_origin.py](../benchmark/ASTT/tools/verify_origin.py) | 檢查官方快照檔案清單與 SHA-256。 |
| [benchmark/ASTT/.gitignore](../benchmark/ASTT/.gitignore) | ASTT 快取、本地環境、標記紀錄與產生資料集的忽略規則。 |
| [benchmark/ASTT/README.md](../benchmark/ASTT/README.md) | ASTT 本機專案總覽與文件入口。 |
| [benchmark/ASTT/doc/START_HERE.md](../benchmark/ASTT/doc/START_HERE.md) | 整理後的正式入口、閱讀順序與訓練／測試快速操作。 |
| [benchmark/ASTT/__init__.py](../benchmark/ASTT/__init__.py) | ASTT 外層套件標記。 |
| [benchmark/ASTT/doc/FILE_INDEX.md](../benchmark/ASTT/doc/FILE_INDEX.md) | ASTT 每個檔案的用途與點擊入口。 |
| [benchmark/ASTT/doc/DATASET_PREPARATION.md](../benchmark/ASTT/doc/DATASET_PREPARATION.md) | ASTT 異常區段標記、train／val 7:3 分組與 .npy 操作。 |
| [benchmark/ASTT/doc/PROJECT_STRUCTURE.md](../benchmark/ASTT/doc/PROJECT_STRUCTURE.md) | ASTT 目錄結構、流程及 Boom 功能對照。 |
| [benchmark/ASTT/doc/README_LOCAL.md](../benchmark/ASTT/doc/README_LOCAL.md) | 模型解讀、資料格式、訓練／推論協定與限制。 |
| [benchmark/ASTT/doc/RUN_SETTINGS.md](../benchmark/ASTT/doc/RUN_SETTINGS.md) | ASTT 正式設定位置、模型預設值與執行指令。 |
| [benchmark/ASTT/requirements.txt](../benchmark/ASTT/requirements.txt) | 正式 ASTT 的 torch、NumPy、OpenCV 依賴。 |
| [benchmark/ASTT/src/__init__.py](../benchmark/ASTT/src/__init__.py) | 正式 ASTT 原始碼套件。 |
| [benchmark/ASTT/src/config.py](../benchmark/ASTT/src/config.py) | 正式 CLI、模型參數與 data/output 預設路徑。 |
| [benchmark/ASTT/src/model.py](../benchmark/ASTT/src/model.py) | 完整 STE／TTE／CSA 與下一幀預測模型。 |
| [benchmark/ASTT/src/candidates.py](../benchmark/ASTT/src/candidates.py) | 誤差候選框與事件合併，非地理定位。 |
| [benchmark/ASTT/src/annotations.py](../benchmark/ASTT/src/annotations.py) | 標記專案、來源群組切分與安全資料版本匯出。 |
| [benchmark/ASTT/src/dataset.py](../benchmark/ASTT/src/dataset.py) | 連續影格窗口、標籤與缺幀檢查、可選 ECC 配準。 |
| [benchmark/ASTT/src/runner.py](../benchmark/ASTT/src/runner.py) | 正式訓練、驗證、推論、checkpoint 與結果保存。 |
| [benchmark/ASTT/src/ui/__init__.py](../benchmark/ASTT/src/ui/__init__.py) | ASTT 資料標記介面套件。 |
| [benchmark/ASTT/src/ui/annotation_app.py](../benchmark/ASTT/src/ui/annotation_app.py) | 影片播放、異常區段標記與資料準備 GUI。 |
| [benchmark/ASTT/scripts/test.py](../benchmark/ASTT/scripts/test.py) | 正式驗證／測試入口：載入本機 checkpoint，產生異常候選結果。 |
| [benchmark/ASTT/tools/__init__.py](../benchmark/ASTT/tools/__init__.py) | ASTT 資料準備工具套件。 |
| [benchmark/ASTT/tools/extract_frames.py](../benchmark/ASTT/tools/extract_frames.py) | 正式 MP4 抽幀工具及來源幀號／時間 manifest。 |
| [benchmark/ASTT/tools/prepare_dataset.py](../benchmark/ASTT/tools/prepare_dataset.py) | ASTT 標記 GUI、分組預覽與 .npy 匯出入口。 |
| [benchmark/ASTT/scripts/train.py](../benchmark/ASTT/scripts/train.py) | 正式訓練入口：正常片段訓練、獨立 validation 與保存最佳權重。 |
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
| [doc/ACCELERATION.md](ACCELERATION.md) | YOLO／Motion 執行效能、GPU 與顯示加速。 |
| [doc/ASSET_UI.md](ASSET_UI.md) | 地圖與 UI 圖示資源的來源與用途。 |
| [doc/FILE_INDEX.md](FILE_INDEX.md) | 專案逐檔用途與資源目錄索引。 |
| [doc/MAP_LOCALIZATION.md](MAP_LOCALIZATION.md) | GeoTIFF 地圖、特徵匹配、參考照片與地面定位。 |
| [doc/ONLINE.md](ONLINE.md) | 即時 Spinnaker 相機、偵測、輸出與設備設定。 |
| [doc/PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 專案架構、偵測流程與離線／即時執行範例。 |
| [doc/README.md](README.md) | 文件入口與比較專案的 Git 追蹤規則。 |
| [doc/RUN_SETTINGS.md](RUN_SETTINGS.md) | 執行入口、目前 YAML 值與 CLI 完整參數表。 |
| [doc/TOOLS.md](TOOLS.md) | 抽幀、ASTT 標記、YOLO 資料切分、訓練與 ONNX 匯出指令。 |
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
| benchmark/ASTT/data/astt | ASTT 的 train、val、test 影格與逐幀標籤。 |
| data/yolo_datasets | YOLO 原始／版本化訓練資料。 |
| data/video、data/raw_video | 影片輸入。 |
| models | 預訓練權重、checkpoint 與 ONNX，不提交大型模型。 |
| asset | UI、字型、圖示與地圖資源；大型地圖不提交。 |
| output | 離線、即時、訓練與比較結果，不提交。 |
| tmp | 臨時測試程式、資料與產物，執行後立即刪除，不納入 Git。 |
| X-AnyLabeling、cvat | 外部標註軟體；不修改、不加入 Boom Git。 |

更新索引：`python tools/update_file_index.py`（使用 boom 環境）。


## Smoke Heatmap 模型與工具

| 檔案 | 功能 |
|---|---|
| [doc/history/SMOKE_HEATMAP.md](history/SMOKE_HEATMAP.md) | 歷史：已移除的 Smoke Heatmap 方法與指令。 |
| [doc/history/SMOKE_HEATMAP_DATASET.md](history/SMOKE_HEATMAP_DATASET.md) | 時間區段與手動逐幀 BBox 標註 UI、資料格式及匯出操作。 |
