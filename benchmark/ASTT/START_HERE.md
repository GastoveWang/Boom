# ASTT 從這裡開始

ASTT 是連續影格預測的異常偵測方法。本機版可訓練、驗證、測試並產生高預測誤差的候選區域；尚未接入 Boom 即時 runner，也不能把候選區域當成已確認煙霧或 GPS 煙源。

## 檔案分組

| 用途 | 檔案 | 你是否需要直接執行 |
|---|---|---|
| 正式訓練 | train.py | 是 |
| 正式測試／推論 | test.py | 是，必須指定本機版 checkpoint |
| 影片抽幀 | scripts/extract_frames.py | 只有資料還是影片時 |
| 自我檢查 | smoke_test.py | 安裝後可執行，不是正式訓練 |
| 正式流程核心 | runner.py | 否，train/test 呼叫它；CLI 設定定義在 parser() |
| 連續影格資料集 | dataset.py | 否；按影片建窗口、檢查缺幀與標籤、可選 ECC |
| 正式模型 | model.py | 否；STE、TTE、CSA 與下一幀 decoder |
| 候選區域與事件 | localization.py | 否；誤差區域、bbox、IoU 合併與去抖，非地理定位 |
| 環境 | requirements.txt | pip 安裝使用 |
| 詳細適配說明 | README_LOCAL.md | 資料格式、模型解讀、恢復訓練與限制 |
| 上游說明 | README.md | 論文、作者、原專案背景 |
| 歷史模型 | model_original.py、model_baseline_ca.py、model_ste_tte.py | 研究參考，不是正式入口 |
| 歷史實驗入口 | Train_and_eval.py、train_val_*.py、eval.py | 研究參考，仍有作者路徑及額外依賴 |
| 歷史共用模組 | config.py、utils.py、data_utils.py、data_loaders.py | 正式流程不使用；不要在 config.py 改本機訓練設定 |
| 歷史權重工具 | checkpoint.py、check_jax.py | JAX/ViT 權重轉換與檢查；正式流程不使用 |
| 套件標記 | __init__.py | 否 |

保留歷史檔案的位置，避免破壞它們原來的 import。檔案頂端新增功能／入口註記；正式設定完整列在 [執行設定索引](../../doc/RUN_SETTINGS.md)。

## 用現有 data/picture 訓練

1. 先人工找出無煙霧的連續片段，複製 JPG/PNG 到 `data/astt/train/<片段名稱>/`。
2. 不同來源影片留作 val/test，分別放入 `data/astt/val/frames/<影片名稱>/` 和 `data/astt/test/frames/<影片名稱>/`。不要把同影片相鄰幀隨機切分。
3. 每個資料夾至少 5 幀，支援 `frame_000001.jpg`。不要跨剪接拼接；缺少 JSON 不代表正常。
4. LabelMe JSON 不直接輸入 ASTT。評估 AUC 需每影片一個長度與排序影格一致的 binary `.npy`（0 正常、1 異常），保存到各 split 的 `test_frame_masks/<影片名稱>.npy`。無標籤仍可算 MSE 和輸出候選區域。

以下從 **Boom 根目錄** 執行；先準備上述資料，路徑為範例，不會自動搬移你的 picture：

```powershell
conda activate boom
python -m pip install -r benchmark/ASTT/requirements.txt
python benchmark/ASTT/smoke_test.py
python benchmark/ASTT/train.py --experiment-name astt_first --epochs 30 --batch-size 1 --device auto
python benchmark/ASTT/test.py --checkpoint output/astt/astt_first/checkpoints/best.pth --experiment-name astt_test --batch-size 1 --device auto
```

每次使用新 experiment name，避免覆蓋。默认 train 使用正常資料；以 val MSE 選 best，test 不參與訓練。完整預設模型很大，先從 batch 1 開始；降低 layers/embedding 屬於小模型實驗，必須記錄，不能直接當論文基準。

```powershell
# 抽幀（如果已經有連續圖片，省略此步）
python benchmark/ASTT/scripts/extract_frames.py data/video/normal.mp4 --split train --name normal_001 --fps 10
# 恢復原本 30 epoch 排程，使用新輸出目錄
python benchmark/ASTT/train.py --resume output/astt/astt_first/checkpoints/last.pth --experiment-name astt_resume --epochs 30 --batch-size 1
# 使用 checkpoint 初始化新排程
python benchmark/ASTT/train.py --checkpoint output/astt/astt_first/checkpoints/best.pth --experiment-name astt_finetune --epochs 10 --batch-size 1
```

輸出位於 `output/astt/<experiment-name>/`：checkpoints、逐幀 scores、heatmaps、visualizations、events.csv、config.json、metrics.json。BBox 位於模型解析度；不是原圖像素或地面座標。詳見 README_LOCAL.md。
