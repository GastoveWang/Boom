<!-- 檔案功能：ASTT 正式設定位置、模型預設值與執行指令。 -->

# ASTT 執行設定

所有指令從 **Boom 根目錄** 執行。正式參數集中在 [../src/config/cli.py](../src/config/cli.py)，抽幀參數在 [../tools/extract_frames.py](../tools/extract_frames.py)。`legacy/config.py` 與 Boom 的 YAML 不影響正式 ASTT。

完整 CLI 表沿用 [Boom 執行設定索引](../../../doc/RUN_SETTINGS.md#astt-訓練-cli)，也可以查看：

```powershell
python benchmark/ASTT/train.py --help
python benchmark/ASTT/test.py --help
python benchmark/ASTT/inference.py --help
python benchmark/ASTT/tools/extract_frames.py --help
python benchmark/ASTT/tools/prepare_dataset.py --help
```

| 設定 | 預設 | 用途 |
|---|---|---|
| image-size／patch-size | 256／32 | 模型影像與 patch 大小 |
| emb-dim／mlp-dim | 768／3072 | Transformer 維度 |
| spatial-layers／temporal-layers | 12／12 | 完整 STE／TTE 深度 |
| num-heads／num-frames | 8／4 | 注意力頭數與歷史幀數 |
| batch-size／epochs | 8／30 | 本機預設批次與 epoch 數；30 是本機排程預設 |
| lr／weight-decay | 1e-4／1e-4 | AdamW；OneCycleLR 排程 |
| selection | mse | 以獨立 validation 的 MSE 選最佳；全標註且含兩類時可選 auc |
| registration | 關閉 | ECC 將歷史幀對齊到最後 input |
| threshold／min-area | 0.1／25 | 誤差候選區域閾值與模型像素面積 |
| min-event-frames／event-max-gap | 3／2 | 連續命中確認與事件漏幀容忍 |

第一輪直接使用完整 STE＋TTE＋CSA；完整模型 GPU 預檢在目前 RTX 5080、batch 8、FP32 上通過一次更新，這不等於正式效果或長時間效能驗證。資料準備完後：

```powershell
conda activate boom
python benchmark/ASTT/train.py --experiment-name astt_full_001 --device cuda --batch-size 8 --epochs 30
python benchmark/ASTT/test.py --split val --checkpoint output/astt/astt_full_001/checkpoints/best.pth --experiment-name astt_full_val_001 --device cuda --batch-size 8
```

每次實驗使用新名稱。模型設定從 checkpoint 載入時以權重內容為準；registration 要與訓練一致。資料與輸出根預設分別為 Boom 的 `data/`、`output/`，再附加 `astt/`。

## 先確認能否訓練

短測試使用實際資料與目前完整模型，只取最多四個 train batch、兩個 val batch，跑一輪並保存／重新載入最佳權重。這是執行診斷，短測試的 AUC 不代表正式效果；原資料集不會被修改。

```powershell
python benchmark/ASTT/train.py --data-dir benchmark/ASTT/data/prepared/artillery-shell_v1 --output-dir benchmark/ASTT/output --experiment-name astt_preflight_001 --device cuda --batch-size 8 --epochs 1 --selection auc --preflight
```

`--preflight` 必須搭配 `--epochs 1`，不能搭配 `--resume`。val 子集會分別從正常、異常窗口取樣。若 val 只有一類，改用 `--selection mse`。每次診斷也需要新的實驗名稱，避免覆寫已有輸出。正式訓練移除 `--preflight` 並設定 `--epochs 30`，使用另一個新實驗名稱。

訓練與驗證現在會顯示 `完成 batch／總 batch`、比例、已花時間與該階段剩餘時間估計；第一個 batch 完成後即印出，之後約每五秒更新一次（在 batch 完成時）。首次讀入及權重保存／重新載入也有階段訊息。執行中的狀態同步保存到實驗資料夾 `progress.json`；`history.json` 與權重仍是每輪訓練及驗證完成後保存。ETA 包含圖片讀取與運算時間，初期可能因初始化而不準確。若第一個 batch 尚未完成，會顯示目前正在讀取或處理。

## 時間區段標記與資料版本

```powershell
python benchmark/ASTT/tools/prepare_dataset.py
```

操作見 [DATASET_PREPARATION.md](DATASET_PREPARATION.md)。預設比例 7:3、固定 seed 2026、標記與匯出固定 30 FPS、名稱 `artillery-shell`、輸出父目錄 `benchmark/ASTT/data/prepared/`。每次匯出建立新的 `<名稱>_vN/`，至少兩個獨立來源群組。

不開 GUI 時，用 `--project <標記JSON> --preview` 預覽或 `--export` 匯出。可搭配 `--output-parent`、`--name`、`--ratios TRAIN VAL`、`--seed`。影片高幀率取樣、低幀率重複影格，統一 30 FPS；圖片資料夾必須已是 30 FPS 序列。此標記工具沒有 `--fps` 參數。比例按來源群組近似分配。舊版標記載入後需重新確認並按完成。

此工具只產生 train／val。訓練結束會輸出最佳模型的 val 結果；另行查看時執行 `test.py --split val`。舊手動資料集仍可使用獨立 test。

訓練／測試的 `--data-dir` 指向版本目錄，例如 `benchmark/ASTT/data/prepared/artillery-shell_v1`。該版本 README 會依 val 可評分類別生成 `--selection auc` 或 `mse` 指令。
