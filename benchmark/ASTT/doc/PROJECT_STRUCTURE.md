<!-- 檔案功能：ASTT 目錄結構、流程及 Boom 功能對照。 -->

# ASTT 專案結構

與 Boom 相同：**根目錄執行入口 → src 演算法與流程 → tools／tests／doc 輔助目錄**。這裡只整理功能分工，完整模型與資料／輸出格式延續原本本機版。

```text
ASTT/
├── train.py                       # 正式訓練入口
├── test.py                        # 正式驗證／測試入口
├── inference.py                   # 用權重直接跑新影片並保存標記 MP4
├── smoke_test.py                  # 自我檢查相容入口
├── README.md、START_HERE.md        # 總覽、快速操作
├── requirements.txt              # 正式流程依賴
├── src/
│   ├── config/cli.py              # 命令列參數、完整模型預設值與路徑
│   ├── detection/model.py         # STE＋TTE＋CSA → 預測下一幀
│   ├── runtime/dataset.py         # 連續窗口、標籤／缺幀檢查、ECC
│   ├── runtime/annotations.py     # 人工區段、來源分組與安全匯出
│   ├── runtime/runner.py          # 訓練、驗證、推論、checkpoint／輸出
│   ├── runtime/video_inference.py # 30 FPS 影片串流、候選框、MP4／CSV
│   ├── ui/annotation_app.py       # 影片標記桌面介面
│   ├── ui/inference_view.py       # 原畫面／熱圖／疊圖與分數曲線的成果版面
│   └── localization/candidates.py # 誤差候選框、事件編號與去抖合併
├── tools/prepare_dataset.py       # 標記 GUI／分組／.npy 匯出入口
├── tools/extract_frames.py        # 影片抽幀與 frames.csv
├── tests/smoke_test.py            # 暫存資料整體自我檢查
├── tests/test_annotations.py      # 標記、分組與匯出相容性測試
├── tests/test_video_inference.py  # 新影片推論、時間軸與輸出測試
├── data/prepared/                 # 標記工具匯出的版本資料集（執行後建立）
├── output/annotations/            # 人工時間區段標記 JSON（執行後建立）
├── doc/
│   ├── PROJECT_STRUCTURE.md       # 你正在讀的結構說明
│   ├── FILE_INDEX.md              # 每個檔案的用途
│   ├── DATASET_PREPARATION.md      # 標記操作與資料版本使用方式
│   ├── RUN_SETTINGS.md            # 正式參數與操作
│   ├── VIDEO_INFERENCE.md         # 用權重跑新影片
│   └── README_LOCAL.md            # 方法、格式、輸出與限制
├── legacy/                       # 作者舊模型、消融腳本、工具與 README
└── model.py、dataset.py、runner.py、localization.py
                                  # 舊 import 相容層；實作在 src
```

`scripts/extract_frames.py` 與根目錄 `README_LOCAL.md` 也保留舊指令／連結。功能修改請進入 `src/`，不要修改相容層。

## 與 Boom 對照

| 分工 | Boom | ASTT | 你什麼時候看它 |
|---|---|---|---|
| 執行入口 | run_offline.py／run_realtime.py | train.py／test.py | 開始實驗 |
| 執行設定 | src/boom/config、configs | src/config/cli.py | 改模型大小、epochs、路徑、閾值 |
| 偵測演算法 | src/boom/detection | src/detection/model.py | 理解 STE／TTE／CSA |
| 資料讀取 | src/boom/runtime/inputs.py | src/runtime/dataset.py | 檢查影格窗口與標籤 |
| 執行流程 | src/boom/runtime/runner.py | src/runtime/runner.py | 看訓練、推論與結果保存 |
| 定位 | src/boom/localization | src/localization/candidates.py | ASTT 僅有影像候選框，沒有 GPS |
| 資料工具 | tools | tools | 影片標記、自動分組、抽幀與標籤 |
| 操作介面 | src/boom/ui | src/ui/annotation_app.py | ASTT 的離線資料標記介面 |
| 驗證 | tests | tests | 確認程式能正常跑通 |
| 說明 | doc | doc | 看設定、方法與限制 |
| 模型權重 | models、output | Boom/output/astt/<experiment>/checkpoints | 訓練產生的權重 |

ASTT 正式設定使用 CLI；Boom 的 `configs/*.yaml` 不會控制 ASTT。標記 GUI 用於離線資料準備，ASTT 尚無即時偵測 runner 或 GPS 介面，目錄名稱相似不表示它已具有 Boom 的全部能力。

## 正式資料流

資料準備：`tools/prepare_dataset.py` → `ui/annotation_app.py` 播放與人工標記 → `runtime/annotations.py` 依來源群組匯出影格、`.npy`、標記快照與分組報告。

1. `train.py`／`test.py` 使用 `config/cli.py` 讀取參數。
2. `runtime/dataset.py` 從每支影片取得連續歷史幀與下一幀 target。
3. `detection/model.py` 用歷史幀預測下一幀。
4. `runtime/runner.py` 計算預測與真實 target 的 MSE；訓練時更新權重。
5. 輸出階段由 `localization/candidates.py` 擷取高誤差候選框、合併事件，再保存分數、熱圖、標註圖片與 CSV。

第一輪實驗使用完整主模型。歷史 `legacy/` 供查閱來源與消融研究，正式 train/test 不匯入它。
