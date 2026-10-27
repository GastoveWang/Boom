# 新版 ASTT_BBOX 訓練

> 歷史文件：保留當時的方法、指令與驗證紀錄；其中舊程式可能已移除，請勿當成目前操作指引。目前操作請見 [ST²](../ST2.md) 與 [文件首頁](../README.md)。

> 歷史文件：2026-10-22 已移除直接 BBox 回歸模型與其入口。下列程式路徑／指令僅供舊實驗追溯，不能再執行；目前流程見 [Smoke Heatmap](SMOKE_HEATMAP.md)。資料、權重與輸出仍保留。

新版 ASTT_BBOX 是 Boom 自有的單類 ROI 框監督時序研究模型，使用 `BBoxTemporalModel` 與 `BBoxSetCriterion`。不是 `benchmark/ASTT` 的下一幀預測模型，不相容其 checkpoint，也不會更動其程式。輸出表示使用者標註的關注區域；沒有煙霧類別真值時，不能把它宣稱為煙霧偵測器。

## 開始訓練

從 Boom 根目錄執行；`boom` 環境若已有 PyTorch、SciPy、Pillow、NumPy，不需重裝。缺少依賴時使用 `python -m pip install -e ".[bbox]"`。

```powershell
conda activate boom
python tools/train_astt_bbox.py --inspect
python tools/train_astt_bbox.py --data "C:\Users\roger\IDEA\Boom\data\astt_bbox\prepared\astt_bbox_v1" --epochs 30 --history-frames 8 --batch-size 4 --device auto
```

未指定 `--data` 時預設上述 v1 路徑。`auto` 有 CUDA 時用 GPU，否則 CPU；指定 `--device cuda` 而無可用 CUDA 時直接報錯。

現有 v1 的原始匯出有 32,399 個 train 窗口（765 個有框）與 12,591 個 val 窗口（560 個有框），歷史長度 4，單幀最多 3 個框。訓練預設抽樣有框與無框各約一半，採有放回抽樣，每個 epoch 抽取與 train 原大小相同數量的窗口；不是增加獨立正樣本。可用 `--positive-fraction 0.25` 調整，或 `--positive-fraction 0` 使用原始分布並逐窗口訓練。val 不重抽樣、不打亂，預設使用完整集合。

新訓練預設輸入 H=576、W=768（寬×高為 768×576，4:3），八幀皆轉成 RGB [0,1]，直接縮放、不裁切、不 padding；cxcywh 的相對座標不變。相較原本 256×192，長寬各提高 3 倍、像素數提高 9 倍，原本中位數約 9.4×7.6 像素的訓練 ROI 框會變成約 28.3×22.9 像素。資料集匯出已保留原圖尺寸，因此不需重新匯出或改標註。

記憶體與運算時間會增加；可用 `--height 768 --width 1024` 再提高解析度，或明確指定其他尺寸；長寬須至少 16 且為 8 的倍數。非原圖比例會變形，訓練及推論必須採相同映射。GPU 記憶體不足時先將 `--batch-size` 降至 2 或 1。舊 checkpoint 的驗證與續訓會恢復其原本輸入尺寸，不受新預設或 CLI 尺寸覆蓋；要使用較大尺寸請開新訓練。

新訓練預設 30 epochs、`--history-frames 8`。每個窗口包含目前幀及之前 7 幀，預測窗口最後一幀的 BBox，不是未來第 9 幀。`--batch-size 4` 表示一次處理 4 個窗口，每個窗口各有 8 張圖；兩個參數不同。

目前 v1 在 8 幀設定下實際可用 32,363 個 train 窗口（765 個有框）與 12,575 個 val 窗口（560 個有框）。已在本機 RTX 5080 以 batch 4、8 幀、768×576、FP32 完成一次前向、反向與 optimizer 更新，PyTorch 峰值配置顯存約 5.31 GiB；此為單批載入檢查，不代表完整訓練的峰值或辨識效能。

現有 v1 匯出的是 4 幀窗口，loader 會在同一來源內將連續、互相重疊的有效窗口組成 8 幀，不改寫資料集，不跨略過／缺失窗口或影片邊界。不補幀，每個目標幀最多產生一個窗口，標註沿用最後一幀；各連續片段開頭不足 8 幀的目標會略去。可用 `--inspect` 查看實際窗口數，或用 `--history-frames 4` 作舊版對照。指定長度須至少為該資料集匯出的窗口長度。舊 checkpoint 驗證和續訓會恢復 `model_config.num_frames`（包含沒有新 CLI 欄位的 4 幀 checkpoint）；要改成 8 幀請開新訓練。

模型預設 embedding=128、queries=16。正負窗口同時參與分類，正樣本透過 Hungarian 配對參與 L1/GIoU/NWD 框回歸。目前沒有直接的注意力 mask loss。Windows 預設 `--workers 0`；可自行評估增加 workers 是否改善資料讀取速度。

## 小目標 BBox loss

新訓練的實驗起始設定為：

```text
L = 1 × L_class + 5 × L_L1 + 2 × L_GIoU + 2 × L_NWD
```

- `L_class`：物體／背景的加權 cross entropy；未配對 query 為背景，背景權重 0.1。空負樣本只計算分類，框 loss 為可反向傳播的零。
- `L_L1`：配對後相對 cxcywh 的絕對誤差總和，再除以本 batch 的真值框數。保留遠距離修正訊號。
- `L_GIoU`：配對後 `1−GIoU` 的平均，約束框的重疊與範圍。
- `L_NWD`：將框表示為像素座標 `z=(cx,cy,w/2,h/2)`，計算 `mean(1−exp(−||z_pred−z_gt||₂/C))`。中心和寬高必須先分別乘輸入 W/H。

NWD 參考 [A Normalized Gaussian Wasserstein Distance for Tiny Object Detection](https://arxiv.org/abs/2110.13389) 與[作者實作](https://github.com/jwwangchn/NWD)。本專案採用 `C = --bbox-nwd-scale × sqrt(H²+W²)`，預設 scale=0.05，在 768×576 輸入時 C=48 像素（舊 256×192 為 16 像素）；這是為保持等比例縮放一致而採用的設定，不是原論文固定常數的完整重現。權重與尺度只是待驗證的起點，沒有真實效果改善的保證。

小框移動幾個像素就可能完全不重疊。NWD 在附近、不重疊時仍提供距離訊號，補充重疊指標；距離非常遠時指數項會飽和，因此保留 L1 與 GIoU。Hungarian 成本同時使用 `−p(class) + 5×L1 − 2×GIoU + 2×NWD_loss`，避免配對與回歸採用不同的幾何目標。框 loss 僅在配對真值上計算，不把所有 query 拉向同一位置。

以下兩次使用相同資料、模型、解析度與 seed，比較是否值得保留 NWD；每次都建立新 run：

```powershell
# 原版 loss 對照
python tools/train_astt_bbox.py --bbox-nwd-weight 0 --epochs 30 --device auto
# 新的混合 loss
python tools/train_astt_bbox.py --bbox-nwd-weight 2 --bbox-nwd-scale 0.05 --epochs 30 --device auto
```

其餘權重可用 `--bbox-class-weight`、`--bbox-l1-weight`、`--bbox-giou-weight`、`--bbox-background-weight` 調整，必須為有限正數；NWD 權重可為零，scale 必須為有限正數。直接呼叫 `BBoxSetCriterion` 的既有 API 預設 NWD 權重仍為 0，以保留相容性；訓練入口的新 run 預設為 2。

`config.json` 保存 `loss_config`，checkpoint 的 `args` 保存各權重、尺度與輸入尺寸。續訓和獨立驗證均恢復 checkpoint 的 loss 設定，覆蓋 CLI 的 loss 參數；舊 checkpoint 缺少設定時按原版 CE+L1+GIoU（NWD=0）處理。變更 loss 應另開新實驗，不能用 `--resume` 改變既有實驗目標。

`metrics.jsonl` 額外記錄未乘權重的 `bbox_nwd`；關閉時為 0。現有 epoch loss 是依 batch 大小加權的平均，框項在無正框 batch 為 0，所以驗證集負樣本多時平均框 loss 會偏低；它不是「只針對正框」的誤差統計。batch 內按真值框數正規化也使平均 loss 受 batch 分組影響；獨立驗證若要重現訓練時的 val loss，請使用相同 `--batch-size`（驗證入口不自動恢復此參數）。不同 loss 設定的總 loss 不可直接比較優劣，應比較完整 val AP50、召回率與無框窗口誤報率。NWD 不更改 AP50 的 IoU≥0.5 判準，也不是注意力 mask 或時序平滑監督。現有資料沒有物體 track ID，先不加入可能把移動目標拉回舊位置的跨幀平滑項。

## 輸出與續訓

每次建立新的 `output/astt_bbox/training/astt_bbox_<時間>_<ID>/`，包含：

- `config.json`：模型、參數、資料摘要、標註檔 fingerprint 與縮放方式。
- `metrics.jsonl`：每個 epoch 的訓練／驗證 loss 和驗證指標。
- `last.pt`：最後完整完成的 epoch，包含模型與 optimizer。
- `best.pt`：val AP50 最高的模型；同 AP50 時取較低 val loss。val 無正樣本時改以 loss 選擇。

```powershell
python tools/train_astt_bbox.py --data data/astt_bbox/prepared/astt_bbox_v1 --resume "output/astt_bbox/training/你的run/last.pt" --epochs 100 --device auto
```

`--epochs 100` 表示總計完成至第 100 個 epoch。續訓恢復模型、optimizer、輸入尺寸、抽樣、batch、學習率、seed 與小規模樣本限制等訓練設定；每 epoch 重新設定抽樣／模型隨機種子。同機同套件版本盡可能重現，但 GPU 運算未承諾 bitwise 一致。續訓要求標註檔 fingerprint 相同（不包含圖片位元組）；不要修改匯出資料。續訓寫入新的 run，保留原 run 的最佳權重，不覆寫來源。Ctrl+C 後可使用最後已完整完成的 `last.pt`；第一個 epoch 未完成則沒有可續訓 checkpoint。

## 單獨驗證

```powershell
python tools/train_astt_bbox.py --data data/astt_bbox/prepared/astt_bbox_v1 --evaluate "output/astt_bbox/training/你的run/best.pt" --device auto
```

驗證使用 checkpoint 的架構與輸入尺寸，結果寫入全新的 `output/astt_bbox/results/astt_bbox_<時間>_<ID>/metrics.json`。可用 `--confidence 0.3` 改變 Precision/Recall 的信心門檻。AP50 使用全部 queries 的信心排序、IoU≥0.5、一對一貪婪配對及 101 點插值；不執行 NMS，重複預測會造成 precision 損失。Precision/Recall 的預設信心門檻為 0.5。

`negative_window_false_alarm_rate` 是無框歷史窗口中至少有一個預測框的比例，**不是每分鐘誤報事件數**。`matched_iou` 是門檻內成功配對的平均 IoU。`model_windows_per_second`／`model_ms_per_window` 僅量測 batch 模型前向傳播（CUDA 已同步，包含首批暖機開銷）；`pipeline_windows_per_second` 包含讀圖、前向、loss、匹配與統計，不代表部署端到端 FPS。相鄰窗口重疊；本資料沒有獨立 test。尚未提供事件延遲、GPS 定位誤差或即時 adapter。

## 小規模檢查

已訓練 checkpoint 可用 [影片推論工具](ASTT_BBOX_INFERENCE.md) 批次處理影片，輸出原尺寸 ROI 框影片與逐幀分數。訓練尚未完成時可使用已完成 epoch 的權重，但必須保留其 epoch 與驗證結果，不能當作最終效果。

```powershell
python tools/train_astt_bbox.py --epochs 1 --batch-size 2 --max-train-samples 16 --max-val-samples 16 --device auto
```

限制樣本時用 seed 固定的隨機子集，盡可能保留有框和無框窗口。這只能確認流程，不能據此報告正式效能；正式訓練不要加 `--max-train-samples`／`--max-val-samples`。本工具檢查來源群組不跨 train/val、連續歷史索引、略過狀態、圖片路徑、單類標籤、框座標與 query 容量；不重新切分資料，也不覆寫資料集。
