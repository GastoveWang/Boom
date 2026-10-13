<!-- 檔案功能：ASTT 每個檔案的用途與點擊入口。 -->

# ASTT 逐檔功能索引

每個檔案開頭也有中文用途註記。只列原始碼、工具與文件；資料、模型權重和快取不修改。

| 檔案 | 功能 |
|---|---|
| [.gitignore](../.gitignore) | ASTT 快取、本地環境、標記紀錄與產生資料集的忽略規則。 |
| [README.md](../README.md) | ASTT 本機專案總覽與文件入口。 |
| [README_LOCAL.md](../README_LOCAL.md) | 舊本機說明連結，導向 doc/README_LOCAL.md。 |
| [START_HERE.md](../START_HERE.md) | 整理後的正式入口、閱讀順序與訓練／測試快速操作。 |
| [__init__.py](../__init__.py) | ASTT 外層套件標記。 |
| [dataset.py](../dataset.py) | 舊匯入路徑相容入口；實作已移至 src/runtime/dataset.py。 |
| [doc/FILE_INDEX.md](../doc/FILE_INDEX.md) | ASTT 每個檔案的用途與點擊入口。 |
| [doc/DATASET_PREPARATION.md](../doc/DATASET_PREPARATION.md) | 異常區段標記、train／val 7:3 分組、逐幀標籤與資料版本操作。 |
| [doc/PROJECT_STRUCTURE.md](../doc/PROJECT_STRUCTURE.md) | ASTT 目錄結構、流程及 Boom 功能對照。 |
| [doc/README_LOCAL.md](../doc/README_LOCAL.md) | 模型解讀、資料格式、訓練／推論協定與限制。 |
| [doc/RUN_SETTINGS.md](../doc/RUN_SETTINGS.md) | ASTT 正式設定位置、模型預設值與執行指令。 |
| [legacy/README.md](../legacy/README.md) | 上游歷史檔案分類、用途與執行限制。 |
| [legacy/README_UPSTREAM.md](../legacy/README_UPSTREAM.md) | 保留作者的上游 README、論文背景與引用資訊。 |
| [legacy/Train_and_eval.py](../legacy/Train_and_eval.py) | 作者歷史資料巡覽與訓練評估腳本。 |
| [legacy/__init__.py](../legacy/__init__.py) | 上游歷史研究檔案的目錄標記。 |
| [legacy/check_jax.py](../legacy/check_jax.py) | 作者歷史 JAX 權重檢查腳本。 |
| [legacy/checkpoint.py](../legacy/checkpoint.py) | 作者歷史 JAX／ViT 權重載入與轉換工具。 |
| [legacy/config.py](../legacy/config.py) | 作者歷史 ViT 參數，不控制正式 train.py／test.py。 |
| [legacy/data_loaders.py](../legacy/data_loaders.py) | 作者歷史 CIFAR／ImageNet loader，正式流程不使用。 |
| [legacy/data_utils.py](../legacy/data_utils.py) | 作者歷史連續影格讀取工具，正式流程使用 runtime/dataset.py。 |
| [legacy/eval.py](../legacy/eval.py) | 作者歷史影像分類評估入口，非本機異常測試。 |
| [legacy/model_baseline_ca.py](../legacy/model_baseline_ca.py) | 作者 baseline 加 cross-attention 模型，供消融研究。 |
| [legacy/model_original.py](../legacy/model_original.py) | 作者原始時空 Transformer 模型，供架構比較。 |
| [legacy/model_ste_tte.py](../legacy/model_ste_tte.py) | 作者 STE＋TTE 模型，供正式模型的來源與架構比較。 |
| [legacy/train_val_UIT_ADrone.py](../legacy/train_val_UIT_ADrone.py) | 作者 UIT-ADrone 主模型歷史實驗入口。 |
| [legacy/train_val_UIT_ADrone_baseline_ca.py](../legacy/train_val_UIT_ADrone_baseline_ca.py) | 作者 UIT-ADrone baseline＋CSA 歷史實驗入口。 |
| [legacy/train_val_UIT_ADrone_ste_tte.py](../legacy/train_val_UIT_ADrone_ste_tte.py) | 作者 UIT-ADrone STE＋TTE 歷史實驗入口。 |
| [legacy/utils.py](../legacy/utils.py) | 作者歷史指標、logging、裝置與設定工具。 |
| [localization.py](../localization.py) | 舊匯入路徑相容入口；實作已移至 src/localization/candidates.py。 |
| [model.py](../model.py) | 舊匯入路徑相容入口；實作已移至 src/detection/model.py。 |
| [requirements.txt](../requirements.txt) | 正式 ASTT 的 torch、NumPy、OpenCV 依賴。 |
| [runner.py](../runner.py) | 舊匯入路徑相容入口；實作已移至 src/runtime/runner.py。 |
| [scripts/extract_frames.py](../scripts/extract_frames.py) | 舊 scripts 抽幀指令相容入口。 |
| [smoke_test.py](../smoke_test.py) | 根目錄自我檢查相容入口。 |
| [src/__init__.py](../src/__init__.py) | 正式 ASTT 原始碼套件。 |
| [src/config/__init__.py](../src/config/__init__.py) | 正式 CLI 設定套件。 |
| [src/config/cli.py](../src/config/cli.py) | 正式 CLI、模型參數與 data/output 預設路徑。 |
| [src/detection/__init__.py](../src/detection/__init__.py) | ASTT 模型演算法套件。 |
| [src/detection/model.py](../src/detection/model.py) | 完整 STE／TTE／CSA 與下一幀預測模型。 |
| [src/localization/__init__.py](../src/localization/__init__.py) | ASTT 影像候選區域套件。 |
| [src/localization/candidates.py](../src/localization/candidates.py) | 誤差候選框與事件合併，非地理定位。 |
| [src/runtime/__init__.py](../src/runtime/__init__.py) | ASTT 資料讀取與執行流程套件。 |
| [src/runtime/annotations.py](../src/runtime/annotations.py) | 來源與異常區段、標記專案、來源群組切分與安全版本匯出。 |
| [src/runtime/dataset.py](../src/runtime/dataset.py) | 連續影格窗口、標籤與缺幀檢查、可選 ECC 配準。 |
| [src/runtime/runner.py](../src/runtime/runner.py) | 正式訓練、驗證、推論、checkpoint 與結果保存。 |
| [src/ui/__init__.py](../src/ui/__init__.py) | ASTT 資料標記介面套件。 |
| [src/ui/annotation_app.py](../src/ui/annotation_app.py) | 影片／影格播放、區段標記、專案續編與資料匯出 GUI。 |
| [test.py](../test.py) | 正式驗證／測試入口：載入本機 checkpoint，產生異常候選結果。 |
| [tests/__init__.py](../tests/__init__.py) | ASTT 自我檢查套件標記，區別根目錄相容入口。 |
| [tests/smoke_test.py](../tests/smoke_test.py) | 暫存資料上的模型、事件、CLI、續訓與抽幀整體自我檢查。 |
| [tests/test_annotations.py](../tests/test_annotations.py) | 暫存來源上的標記、標籤對齊、分組、匯出與 GUI 測試。 |
| [tools/__init__.py](../tools/__init__.py) | ASTT 資料準備工具套件。 |
| [tools/extract_frames.py](../tools/extract_frames.py) | 正式 MP4 抽幀工具及來源幀號／時間 manifest。 |
| [tools/prepare_dataset.py](../tools/prepare_dataset.py) | 時間區段標記 GUI 入口，以及專案預覽／匯出 CLI。 |
| [train.py](../train.py) | 正式訓練入口：正常片段訓練、獨立 validation 與保存最佳權重。 |
