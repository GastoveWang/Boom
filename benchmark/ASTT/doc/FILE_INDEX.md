# ASTT 逐檔功能索引

正式程式的用途註記在檔案開頭；核心模組採平坦 src 結構。origin 的用途說明只寫在外部文件，不修改原檔。

| 檔案 | 功能 |
|---|---|
| [origin/](../origin) | 官方固定版本的全部 18 個原始檔案，禁止修改；校驗見 ORIGIN.md。 |
| [.gitattributes](../.gitattributes) | 停用 origin 快照的換行轉換，保留原始 Git blob 位元內容。 |
| [.gitignore](../.gitignore) | ASTT 快取、本地環境、標記紀錄與產生資料集的忽略規則。 |
| [__init__.py](../__init__.py) | ASTT 外層套件標記。 |
| [AGENTS.md](../AGENTS.md) | 本機版維護、原始碼保護、相容性與驗證規則。 |
| [doc/DATASET_PREPARATION.md](DATASET_PREPARATION.md) | 異常區段標記、train／val 7:3 分組、逐幀標籤與資料版本操作。 |
| [doc/FILE_INDEX.md](FILE_INDEX.md) | ASTT 每個檔案的用途與點擊入口。 |
| [doc/ORIGIN.md](ORIGIN.md) | 官方來源、固定提交版本與原檔保護方式。 |
| [doc/ORIGIN_MANIFEST.json](ORIGIN_MANIFEST.json) | 官方快照每個檔案的 SHA-256 校驗清單。 |
| [doc/PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | ASTT 目錄結構、流程及 Boom 功能對照。 |
| [doc/README_LOCAL.md](README_LOCAL.md) | 模型解讀、資料格式、訓練／推論協定與限制。 |
| [doc/RUN_SETTINGS.md](RUN_SETTINGS.md) | ASTT 正式設定位置、模型預設值與執行指令。 |
| [doc/VIDEO_INFERENCE.md](VIDEO_INFERENCE.md) | doc/VIDEO_INFERENCE.md |
| [scripts/inference.py](../scripts/inference.py) | 載入 ASTT 權重，直接對一支新影片推論並輸出標記 MP4。 |
| [README.md](../README.md) | ASTT 本機專案總覽與文件入口。 |
| [requirements.txt](../requirements.txt) | 正式 ASTT 的 torch、NumPy、OpenCV 依賴。 |
| [src/__init__.py](../src/__init__.py) | 本機適配版核心套件；核心模組平坦排列，UI 使用子套件。 |
| [src/annotations.py](../src/annotations.py) | 人工區段標記、30 FPS 時間軸、來源分組與安全版本匯出。 |
| [src/candidates.py](../src/candidates.py) | 高誤差候選框與跨影格事件合併；不提供煙霧分類或 GPS。 |
| [src/config.py](../src/config.py) | 訓練／測試 CLI、模型預設參數與預設資料／輸出路徑。 |
| [src/dataset.py](../src/dataset.py) | 連續影格窗口、標籤／缺幀檢查與可選 ECC 配準。 |
| [src/model.py](../src/model.py) | STE、TTE、CSA 與 Decoder，從歷史影格預測下一幀。 |
| [src/runner.py](../src/runner.py) | 訓練、驗證、測試、checkpoint 與分數／事件輸出。 |
| [src/ui/__init__.py](../src/ui/__init__.py) | 本機桌面 UI 套件；資料邏輯由 annotations.py 負責。 |
| [src/ui/annotation_app.py](../src/ui/annotation_app.py) | Tkinter 影片播放、異常區段標記與匯出介面。 |
| [src/ui/inference_view.py](../src/ui/inference_view.py) | 原圖、熱圖、疊圖與分數曲線的畫面合成。 |
| [src/video_inference.py](../src/video_inference.py) | 新影片串流、批次推論、30 FPS 標記 MP4 與 CSV。 |
| [doc/START_HERE.md](START_HERE.md) | 整理後的正式入口、閱讀順序與訓練／測試快速操作。 |
| [scripts/test.py](../scripts/test.py) | 正式驗證／測試入口：載入本機 checkpoint，產生異常候選結果。 |
| [tools/__init__.py](../tools/__init__.py) | ASTT 資料準備工具套件。 |
| [tools/extract_frames.py](../tools/extract_frames.py) | 正式 MP4 抽幀工具及來源幀號／時間 manifest。 |
| [tools/prepare_dataset.py](../tools/prepare_dataset.py) | 時間區段標記 GUI 入口，以及專案預覽／匯出 CLI。 |
| [tools/verify_origin.py](../tools/verify_origin.py) | 核對官方快照的完整檔案清單與 SHA-256。 |
| [scripts/train.py](../scripts/train.py) | 正式訓練入口：正常片段訓練、獨立 validation 與保存最佳權重。 |
