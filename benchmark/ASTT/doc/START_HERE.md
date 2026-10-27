<!-- 檔案功能：整理後的正式入口、閱讀順序與訓練／測試快速操作。 -->

# ASTT 從這裡開始

先看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 的目錄與資料流，了解各模組用途；再看 [RUN_SETTINGS.md](RUN_SETTINGS.md) 調整參數。完整資料格式與模型解讀在 [README_LOCAL.md](README_LOCAL.md)。

## 你會直接執行的檔案

| 工作 | 入口 |
|---|---|
| 影片時間區段標記、自動分組與 .npy | tools/prepare_dataset.py |
| 正常片段訓練 | scripts/train.py |
| 載入權重測試／輸出異常候選 | scripts/test.py |
| 用權重直接跑新影片、輸出標記 MP4 | scripts/inference.py |
| 從影片抽幀 | tools/extract_frames.py |

其餘 `src/` 由入口呼叫。`origin/` 是禁止修改的官方快照，不參與正式流程。舊版參考與轉接入口已移除。原檔校驗：`python benchmark/ASTT/tools/verify_origin.py`。

用權重跑一支新影片，不需標記或建立資料集：

```bat
python benchmark/ASTT/scripts/inference.py --video "C:\videos\new_video.mp4" --checkpoint benchmark/ASTT/output/astt/training/astt_artillery_shell_v1_001/checkpoints/best.pth --device cuda
```

影片直接轉為 30 FPS 推論，輸出新的標記 MP4 與候選事件紀錄；操作見 [VIDEO_INFERENCE.md](VIDEO_INFERENCE.md)。

## 準備資料

現在可直接啟動標記工具：

```powershell
conda activate boom
python benchmark/ASTT/tools/prepare_dataset.py
```

加入影片 → S 標記異常開始、E 標記異常結束 → 每支按「完成此來源標記」→ 預覽分組 → 產生資料集與 `.npy`。標記與匯出固定 **30 FPS**，高幀率取樣、低幀率重複影格；圖片資料夾須已是 30 FPS 序列。未標記影格依本工具規則視為正常；至少兩個獨立來源群組，依 7:3 分 train／val。預設建立 `benchmark/ASTT/data/prepared/artillery-shell_v1/`，再次匯出建立新版本。操作與完整模型指令見 [DATASET_PREPARATION.md](DATASET_PREPARATION.md)，匯出版本內也有 README。訓練時 `--data-dir` 指向該版本目錄；查看效果使用 `scripts/test.py --split val`。

以下是自行準備資料時的格式：

1. 人工確認無煙霧的連續片段放入 `benchmark/ASTT/data/astt/train/<片段名稱>/`；缺少 JSON 不代表正常。
2. 以不同原影片留出 val，分別放在 `benchmark/ASTT/data/astt/val/frames/<影片名稱>/` 與 `benchmark/ASTT/data/astt/test/frames/<影片名稱>/`。不要隨機拆相鄰幀。
3. 預設每片段至少五張連續 JPG／PNG。剪接或不連續片段要拆開。
4. LabelMe JSON 不能直接輸入 ASTT。計算 AUROC 需與抽幀排序一一對應的 binary `.npy`，放入 split 的 `test_frame_masks/<影片名稱>.npy`；0 正常、1 異常。缺標籤仍可輸出候選，但不能計算該影片 AUROC。

## 執行

從 **Boom 根目錄**：

```powershell
conda activate boom
python -m pip install -r benchmark/ASTT/requirements.txt
python benchmark/ASTT/scripts/train.py --experiment-name astt_full_001 --epochs 30 --batch-size 8 --device cuda
python benchmark/ASTT/scripts/test.py --split val --checkpoint benchmark/ASTT/output/astt/training/astt_full_001/checkpoints/best.pth --experiment-name astt_full_val_001 --batch-size 8 --device cuda
```

資料準備完才執行正式訓練；每次使用新 experiment name。完整模型預設為 STE＋TTE＋CSA，降低層數／維度會成為另一個小模型實驗。CPU 可用 `--device cpu`，顯存不足可調整 batch size 並記錄。

```powershell
python benchmark/ASTT/tools/extract_frames.py data/video/normal.mp4 --split train --name normal_001 --fps 30
python benchmark/ASTT/scripts/train.py --resume benchmark/ASTT/output/astt/training/astt_full_001/checkpoints/last.pth --experiment-name astt_resume_001 --epochs 30 --batch-size 8
```

訓練輸出放在 `benchmark/ASTT/output/astt/training/<experiment-name>/`，包含 checkpoint、訓練紀錄與最佳模型的 validation 結果。`scripts/test.py` 和 `scripts/inference.py` 的最後結果放在 `benchmark/ASTT/output/astt/results/<experiment-name>/`。候選框位於模型解析度；高誤差不保證是煙霧，ASTT 尚未接入即時相機或 GPS 定位。
