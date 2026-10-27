# ASTT 專案結構

本機版採用一般 Python 專案的「執行入口、核心程式、工具、測試、文件」結構。核心以單一模組負責一類工作，只有桌面介面使用 `ui/` 子套件。

```text
ASTT/
├── origin/                 # 官方原始碼完整快照，禁止修改
├── src/                    # 本機適配版核心程式
│   ├── config.py           # 訓練／測試 CLI、模型預設參數、預設路徑
│   ├── model.py            # STE、TTE、CSA、Decoder 與下一幀預測模型
│   ├── dataset.py          # 連續影格窗口、標籤與缺幀檢查、ECC 配準
│   ├── annotations.py      # 人工標記、30 FPS 時間軸、來源分組、資料匯出
│   ├── candidates.py       # 高誤差候選框、事件編號與跨影格合併
│   ├── runner.py           # 訓練、驗證、測試、checkpoint 與結果保存
│   ├── video_inference.py  # 新影片串流、批次推論、MP4／CSV 輸出
│   └── ui/
│       ├── annotation_app.py # Tkinter 影片標記介面
│       └── inference_view.py # 原圖、熱圖、疊圖與分數曲線版面
├── scripts/                # 正式執行入口
│   ├── train.py            # 訓練與獨立 validation
│   ├── test.py             # 已準備資料集的驗證／測試
│   └── inference.py        # 用本機權重直接跑新影片
├── tools/
│   ├── prepare_dataset.py  # 影片標記 GUI、分組預覽與資料匯出入口
│   ├── extract_frames.py   # 影片抽幀與來源幀號／時間 manifest
│   └── verify_origin.py    # 驗證官方快照的檔案清單與 SHA-256
├── data/                   # ASTT 資料集，忽略於 Git
├── output/                 # 分層的訓練及結果，忽略於 Git
├── doc/                    # 方法、資料格式、操作說明、來源紀錄
├── requirements.txt
├── README.md               # 專案簡介與文件入口
├── AGENTS.md               # 維護與原檔保護規則
├── __init__.py             # ASTT 套件標記
└── .gitignore、.gitattributes # 資料忽略與原檔換行保護
```

正式操作使用 `scripts/`；抽幀與資料準備使用 `tools/`；臨時驗證一律使用 Boom 根目錄的 `tmp/`，完成後清理。舊版參考與根目錄轉接檔已移除；功能實作集中在 `src/`。

## 資料與輸出位置

ASTT 資料與輸出已回到 `benchmark/ASTT/`，保留訓練、最後結果及人工標記的分層：

```text
benchmark/ASTT/
├── data/
│   ├── prepared/<名稱>_vN/     # 匯出版本，其下 astt/train、astt/val
│   └── astt/train、val、test/ # 手動資料的預設位置
└── output/astt/
    ├── training/<實驗名稱>/  # 權重、history、progress 與訓練後 validation
    ├── results/<實驗名稱>/   # 資料集評估或影片推論的最後結果
    ├── annotations/         # 人工標記專案 JSON
    └── cache/               # 保留的既有快取
```

所有下列指令從 Boom 根目錄執行。訓練版本使用 `--data-dir benchmark/ASTT/data/prepared/<名稱>_vN`，其下仍包含 `astt/`。手動資料不指定參數時讀取 `benchmark/ASTT/data/astt/<split>/`。

`--output-dir` 預設為 `benchmark/ASTT/output`；程式繼續附加 `astt/training` 或 `astt/results`。自訂 output-dir 的行為不變，同名實驗已存在時拒絕覆寫。

搬回前後已比對完整的檔名、檔案數、大小及修改時間，未重寫既有資料、checkpoint 或歷史結果。先前的 `path_migration_20261013.json` 保留歷史紀錄；這次對照另存於 `output/astt/path_return_*.json`。歷史 README/config/split_report 仍可能記錄以前位置，最新使用路徑見本頁及資料集內新增的 `README_RETURNED_PATHS_*.md`。

## 原始碼與本機版

`origin/` 保存 [Tungufm/ASTT](https://github.com/Tungufm/ASTT) 的固定提交 `1d05305a94fb47b33897c7aa085c5351bcbbc8f9`，含全部 18 個版本控制檔案，內容與原始 Git blob 相同。來源與校驗值在 [ORIGIN_MANIFEST.json](ORIGIN_MANIFEST.json)；說明在 [ORIGIN.md](ORIGIN.md)。不要在快照中加註解、格式化或執行會生成快取／結果的指令。

`src/` 是本機 STE＋TTE＋CSA 實作與 Boom 的資料準備、候選事件及輸出擴充。正式流程不匯入 `origin/`。官方模型與實驗程式完整保留在 origin，先前重複的 legacy 參考已移除。

## 程式資料流

1. `tools/prepare_dataset.py` 呼叫 `ui/annotation_app.py` 標記影片；`annotations.py` 處理時間軸、來源分組及資料匯出。
2. `scripts/train.py`／`scripts/test.py` 讀取 `src/config.py` 的參數，再呼叫 `src/runner.py`。
3. `dataset.py` 提供歷史影格和下一幀 target；`model.py` 預測下一幀。
4. `runner.py` 計算 MSE、更新權重、評估獨立 validation，保存 checkpoint、分數及結果。
5. `scripts/inference.py` 呼叫 `src/video_inference.py` 直接讀新影片；`src/candidates.py` 將高誤差區域合併成候選事件，`src/ui/inference_view.py` 組合輸出畫面。

候選框代表高預測誤差，不能直接當成確認煙霧；本機 ASTT 尚未接入 Boom 即時 runner 或 GPS 地理定位。演算法、CLI、checkpoint 權重欄位、資料／輸出格式維持既有行為。

## 執行與驗證

從 Boom 根目錄、已安裝 requirements 的環境執行：

```powershell
python benchmark/ASTT/scripts/train.py --help
python benchmark/ASTT/scripts/test.py --help
python benchmark/ASTT/scripts/inference.py --help
python benchmark/ASTT/tools/prepare_dataset.py --help
python benchmark/ASTT/tools/verify_origin.py
```

實際訓練與資料準備見 [START_HERE.md](START_HERE.md)、[RUN_SETTINGS.md](RUN_SETTINGS.md) 及 [DATASET_PREPARATION.md](DATASET_PREPARATION.md)。Boom 的 YAML 不控制 ASTT。
