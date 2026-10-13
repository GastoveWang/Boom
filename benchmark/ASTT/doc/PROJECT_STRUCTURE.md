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
├── train.py                # 正式訓練入口
├── test.py                 # 已準備資料集的驗證／測試入口
├── inference.py            # 用本機權重直接跑新影片
├── tools/
│   ├── prepare_dataset.py  # 影片標記 GUI、分組預覽與資料匯出入口
│   ├── extract_frames.py   # 影片抽幀與來源幀號／時間 manifest
│   └── verify_origin.py    # 驗證官方快照的檔案清單與 SHA-256
├── tests/                  # 模型、標記、推論及原檔保護測試
├── doc/                    # 方法、資料格式、操作說明、來源紀錄
├── legacy/                 # 先前已修改過的歷史參考，不是官方原檔
├── data/prepared/          # 匯出的版本資料集，不納入 Git
├── output/                 # 標記與影片結果，不納入 Git
├── requirements.txt
├── README.md、START_HERE.md
└── model.py、dataset.py、runner.py、localization.py、smoke_test.py
                            # 舊匯入／指令相容層
```

`scripts/extract_frames.py` 也是舊指令相容層。新功能請修改 `src/` 或 `tools/` 的實作；相容層不放第二份演算法。

## 原始碼與本機版

`origin/` 保存 [Tungufm/ASTT](https://github.com/Tungufm/ASTT) 的固定提交 `1d05305a94fb47b33897c7aa085c5351bcbbc8f9`，含全部 18 個版本控制檔案，內容與原始 Git blob 相同。來源與校驗值在 [ORIGIN_MANIFEST.json](ORIGIN_MANIFEST.json)；說明在 [ORIGIN.md](ORIGIN.md)。不要在快照中加註解、格式化或執行會生成快取／結果的指令。

`src/` 是本機 STE＋TTE＋CSA 實作與 Boom 的資料準備、候選事件及輸出擴充。正式流程不匯入 `origin/` 或 `legacy/`。`legacy/` 保留先前加過註解與匯入調整的研究參考，以免丟失既有內容。

## 程式資料流

1. `tools/prepare_dataset.py` 呼叫 `ui/annotation_app.py` 標記影片；`annotations.py` 處理時間軸、來源分組及資料匯出。
2. `train.py`／`test.py` 讀取 `config.py` 的參數，再呼叫 `runner.py`。
3. `dataset.py` 提供歷史影格和下一幀 target；`model.py` 預測下一幀。
4. `runner.py` 計算 MSE、更新權重、評估獨立 validation，保存 checkpoint、分數及結果。
5. `inference.py` 呼叫 `video_inference.py` 直接讀新影片；`candidates.py` 將高誤差區域合併成候選事件，`ui/inference_view.py` 組合輸出畫面。

候選框代表高預測誤差，不能直接當成確認煙霧；本機 ASTT 尚未接入 Boom 即時 runner 或 GPS 地理定位。演算法、CLI、checkpoint 權重欄位、資料／輸出格式維持既有行為。

## 執行與驗證

從 Boom 根目錄、已安裝 requirements 的環境執行：

```powershell
python benchmark/ASTT/train.py --help
python benchmark/ASTT/test.py --help
python benchmark/ASTT/inference.py --help
python benchmark/ASTT/tools/prepare_dataset.py --help
python benchmark/ASTT/tools/verify_origin.py
python -m unittest discover -s benchmark/ASTT/tests -p "test_*.py"
python benchmark/ASTT/smoke_test.py
```

實際訓練與資料準備見 [START_HERE.md](../START_HERE.md)、[RUN_SETTINGS.md](RUN_SETTINGS.md) 及 [DATASET_PREPARATION.md](DATASET_PREPARATION.md)。Boom 的 YAML 不控制 ASTT。
