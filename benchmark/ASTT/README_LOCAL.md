# ASTT for Boom（本機整理版）

正式入口是 `train.py`、`test.py`、`scripts/extract_frames.py`。
原 repo、README、所有 ablation 和歷史訓練檔案均保留；原始檔仍含作者的舊路徑與 CUDA 寫法，僅作研究參考，請勿使用它們啟動本機流程。新的正式流程沒有固定使用者名稱、特定影片或 checkpoint 路徑，也不依賴 timm / torchvision。

上游：[Tungufm/ASTT](https://github.com/Tungufm/ASTT)，基準 commit `1d05305a94fb47b33897c7aa085c5351bcbbc8f9`。目前作為一般原始碼目錄納入 Boom Git，已移除內層 `.git`；上游 README 和歷史模型／實驗程式保留。快速入口與檔案分組見 [START_HERE.md](START_HERE.md)，完整設定見 [RUN_SETTINGS.md](../../doc/RUN_SETTINGS.md)。

## 模型與論文對照

已閱讀 *Transformer-Based Spatio-Temporal Unsupervised Traffic Anomaly Detection in Aerial Videos*（IEEE TCSVT 2024，DOI [10.1109/TCSVT.2024.3376399](https://doi.org/10.1109/TCSVT.2024.3376399)）的 Section III、Fig. 2–3、Eq. 6–13，以及 Section IV-C；本機來源為 Downloads 中使用者指定名稱的 PDF。

`model.py` 是以 `model_ste_tte.py` 的分解流程與 decoder 為基礎整理的 STE + TTE + CSA：

1. 四幀依時間先後送入**共享** spatial transformer，取每幀 spatial CLS。
2. 加 temporal CLS 與 temporal position embedding，送入 temporal transformer，保留全部 5 tokens。
3. temporal CLS 經 Wa1 alignment 成 query；將 aligned CLS 與其餘 temporal tokens 組成完整 K/V context。
4. 注意力聚合為單一 CLS 向量，再做 alignment 與 residual，送入 decoder 預測下一幀。

Eq. 11 原印刷最後一行 `zt' = (zt,c + zt,c ⊙ att) ⊙ Wa2` 的 token 軸 reduction 不明確，且定義了 V 卻未在最後一行明確使用；照字面 broadcast 會得到 N+1 tokens，與正文和 Fig. 2 所述「enhanced temporal CLS」不一致。因此本版採取維度明確的解讀（每個 head 分別計算，最後 concat）：

```text
original = temporal_tokens[:, :1]                 # B,1,D
aligned = Wa1(original)                          # B,1,D
context = concat(aligned, temporal_tokens[:,1:]) # B,N+1,D
Q = Wq(aligned); K = Wk(context); V = Wv(context)
A = softmax(Q Kᵀ / sqrt(D / heads))              # B,heads,1,N+1
enhanced = Wa2(aligned + concat_heads(A @ V))     # B,1,D; Eq.11 interpretation
fused = original + enhanced                      # Eq.12 outer skip
```

原 repo CrossAttention 的 `proj` 只有輸出投影，缺少 Wa1 和完整 skip；原 STE+TTE forward 在 CSA 前就切掉其他 tokens，且反向 prepend 幀順序，不能機械取消註解。本版增加兩個 alignment、內部與外部 skip，維持 chronological sequence。這是可運作的論文架構適配與公式解讀，**不是已取得作者確認的逐字公式重現**；舊 checkpoint 不保證相容。

Encoder 改用 PyTorch pre-norm GELU transformer blocks，與上游 encoder 的 norm→attention→skip、norm→MLP→skip 流程一致；修正上游 dropout=0 時的 None 呼叫。Decoder 保留 dense→256×16×16、兩組 convolution blocks、2×/2×/4× transpose convolution 與 Tanh generator。預設影像 256、patch 32、embedding 768、8 heads、STE/TTE 各 12 layers（Section IV-C）；也能指定較小模型做 smoke test。預設模型約 226.7M parameters，正式訓練顯存需求須依設備評估，可先降低 batch size。

## 哪些是論文，哪些是擴充

論文方法：STE/TTE/CSA、normal-only future-frame prediction、MSE/L2 訓練、frame-level anomaly score 與 AUROC。AdamW、OneCycleLR 依論文 implementation details。

Boom 工程修正：per-video windows、不跨缺失編號、CPU/CUDA 選擇、相對路徑、train/val/test 分離、checkpoint/resume、配置保存。

Boom **定位擴充**：channel-mean squared-error spatial maps、threshold/morphology/connected components bbox、IoU temporal merge/debounce、ECC 配準、MP4 抽幀。這些 bbox 只代表高預測誤差的**候選變化區域**，不代表已辨識煙霧、煙源或地面 GPS；尚未接入 Boom 的即時 runner，也未驗證烟霧定位精度。

## 安裝（PowerShell）

```powershell
conda activate boom
cd C:\Users\roger\IDEA\Boom\benchmark\ASTT
python -m pip install -r requirements.txt
python smoke_test.py
```

可另建 Python 3.10–3.13 環境。GPU 請先依 PyTorch 官方安裝方式選擇符合硬體的 torch build；`requirements.txt` 不固定 CUDA wheel。保留的上游 ablation 若要另行研究，還需 matching torchvision 與 timm，且那些腳本尚未整理成正式本機入口。

`Path(__file__).resolve().parents[2]` 自 ASTT 根目錄檔案推導 Boom；抽幀工具位於 scripts，使用 `parents[3]`。因此程式碼沒有硬編碼 `roger`，移動整個 Boom 目錄仍可使用。

## 資料格式

```text
Boom/data/astt/
  train/<normal_video_name>/frame_00000000.jpg ...
  val/frames/<val_video_name>/frame_00000000.jpg ...
  val/test_frame_masks/<val_video_name>.npy
  test/frames/<test_video_name>/frame_00000000.jpg ...
  test/test_frame_masks/<test_video_name>.npy
```

建立的是空根目錄，不生成正式訓練資料。Train 必須由人工確認為正常、無烟霧或其他目標異常的連續片段組成；**缺少 label 不代表自動判定正常**。若提供 `train/test_frame_masks/<video>.npy`，loader 會排除 input+target 五幀中任何標註為 1 的窗口；仍須確認所有未標註影片為正常。

每個直接子目錄為一支影片／連續片段，按自然數字順序排列 jpg/jpeg/png。至少 5 幀；L 幀可產生 L−4 個 windows，缺幀編號窗口會被排除。不同影片永不拼接。片段有剪接時，必須拆成不同資料夾；不要把兩段不連續片段重編號後拼在同一資料夾。

OpenCV BGR，float32，[-1,1]，input `(4,3,H,W)`，target `(3,H,W)`；縮放至正方形 model resolution。若需要保留 aspect ratio 須額外設計一致的 resize/pad 協定，本版沿用 upstream resize。

Labels 為一維長度 L 的 `.npy`，每個元素只能 0（normal）或 1（anomaly），與該影片**抽幀後自然排序**逐一對齊。不能直接使用原 MP4 的不同 FPS labels，不能使用 pixel-mask `.npy` 冒充 frame labels。Loader 嚴格檢查長度／binary values，缺少 val/test labels 仍可推論，但不計該影片的 AUC。

Train 與 val 必須来自不同原影片，程式拒絕相同影片目錄名稱或共享檔案。Test 也應由使用者獨立留存；程式不會自行以 test 當 validation，也不會自動切分／搬移既有資料。複製影片後重新命名仍可能洩漏，需人工管理來源。

## 抽幀

```powershell
python scripts/extract_frames.py ..\..\data\video\normal.mp4 --split train --name normal_001 --fps 10
python scripts/extract_frames.py ..\..\data\video\val.mp4 --split val --name val_001
python scripts/extract_frames.py ..\..\data\video\test.mp4 --split test --name test_001
```

預設 10 FPS，順序 decode，不複製影格提高 FPS。输出資料夾已存在則拒絕覆寫；保存 `frames.csv`（filename/source_frame_index/time_seconds），供 label 重採樣和時間轉換。此工具假設固定 FPS 的 MP4；變動 FPS／損壞 timestamp 需要另外驗證時間對應。若 decode 中途失敗會保留已輸出內容，請檢查後使用新的名稱重試。

## 訓練與 validation

```powershell
python train.py --experiment-name baseline_001 --epochs 30 --batch-size 2 --device auto
```

預設從零開始（checkpoint=None）。使用 normal train，只在 val 驗證；不讀 test。預設以 val 全部可評分 target 的平均 MSE 選 best；若 val 含異常，此值會包含異常，建議在全標註的 val 改用 `--selection auc`。AUC 選模必須 val 所有影片都有 labels，且可評分幀含 0、1 兩類。

```powershell
python train.py --experiment-name auc_001 --selection auc --epochs 30 --batch-size 2
python train.py --resume ..\..\output\astt\baseline_001\checkpoints\last.pth --experiment-name baseline_resume --epochs 30 --batch-size 2
python train.py --checkpoint ..\..\output\astt\baseline_001\checkpoints\best.pth --experiment-name finetune_001 --epochs 10 --batch-size 2
```

`--resume` 恢復 model、optimizer、OneCycle scheduler、epoch、best score 與 Torch RNG；總 epochs、每 epoch steps、selection 和 registration 必須相同。它是完成原排程，不是任意延長 OneCycle；旁邊須保留同實驗的 `best.pth`，續訓新目錄也會保留此前最佳模型。`--checkpoint` 只初始化 model，開始新的 optimizer／排程；與 resume 互斥。使用本版存出的 checkpoint，拒絕自動猜測舊權重格式。

每次用新的 experiment name，已存在目錄會拒絕覆寫；預設名稱包含時間。若中途中止，保留已完成 epoch 的 checkpoint。CLI 可覆寫 `--data-dir D:\datasets --output-dir D:\results`，它們是 data/output **根目錄**，程式會再附加 `astt/`。

## 測試／推論

```powershell
python test.py --checkpoint ..\..\output\astt\baseline_001\checkpoints\best.pth --experiment-name test_001 --device auto --batch-size 2
python test.py --split val --checkpoint ..\..\output\astt\baseline_001\checkpoints\best.pth --experiment-name val_review
```

Test 預設 `--split test`，必須明確給 checkpoint，不會以隨機權重推論。Inference 的模型尺寸由 checkpoint 讀取。

Scalar score 是原始 MSE，值越高越異常，不在每影片 min-max 正規化。AUC 僅計算有 label 且有預測的 target；頭四幀和跨缺幀窗口不評分。`metrics.json` 有 pooled frame AUROC、各影片 AUROC、labelled_frames、缺 label 影片、MSE、處理耗時和 FPS。單一類別 AUC 是 null；不同場景 pooled score 的尺度可能不同，不等於所有 benchmark 的正規化評估協定。

## 輸出

```text
Boom/output/astt/<experiment_name>/
  checkpoints/best.pth, last.pth  # training 存出，test 不複製大型 checkpoint
  scores/<video>.npy             # 長度 L；不可評分位置為 NaN
  heatmaps/<video>/<frame_index>.npy   # 原始 float32 H×W channel-mean error
  heatmaps/<video>/<frame_index>.png   # 固定 [0,4] 顏色尺度
  visualizations/<video>/<frame_index>.jpg # error overlay + ID + pending/candidate
  events.csv
  config.json
  metrics.json
  frame_index.json              # 各 video 完整原檔名順序，index 為 zero-based
  history.json                  # training only，各 epoch train MSE / val metrics
```

Training 最後用最佳 checkpoint 輸出 **val** scores/heatmaps/events；testing 輸出所選 split。每個評分幀保存 heatmap 與 visualization，長影片可能占用較多磁碟。

Error map `E[y,x] = mean_channel((prediction−target)^2)`；scalar score = E.mean。BBox 使用絕對 `--threshold`（預設 0.1）、`--morph-kernel`（預設 3，odd）open+close、`--min-area`（預設 25 model pixels）connected components。PNG 色階僅顯示用途，threshold 使用原始 error。

`events.csv` 欄位：video/event_id/start_frame/end_frame/confirmed_frame/hits/x1/y1/x2/y2/peak_score/status。每事件一列；status 固定 candidate。BBox 是整段事件 bbox 聯集，**座標位於 model resolution**（預設 256×256，x2/y2 exclusive），不是原 1920×1080 的像素座標，也不是 GPS。原圖座標可按各圖 W/H 比例換算；frame_index.json 对应原檔名，抽幀 frames.csv 對應來源時間。

`--min-event-frames 3` 要求至少連續 3 個評分幀命中才能確認；`--event-iou 0.3` 為相鄰 bbox 關聯門檻；`--event-max-gap 2` 容忍漏檢 2 幀仍保持同事件。確認後持續烟塵不每幀新增事件。超過 gap、IoU 不足、分裂或大幅 camera motion 仍可能形成新事件；單純 IoU 不是專業煙霧 tracker。參數以**抽幀後幀數**計，10 FPS 下 3 幀約 0.3 秒觀察窗口（首次命中到第三次命中約 0.2 秒）。

```powershell
python test.py --checkpoint ..\..\output\astt\baseline_001\checkpoints\best.pth --experiment-name localized_001 --threshold 0.15 --min-area 40 --min-event-frames 5 --event-iou 0.3 --event-max-gap 2
```

Threshold 請以正常 validation 的 error 分布及誤報/漏報調整，不能用 test labels 調參再把同一 test 結果作無偏比較。高誤差也可能来自雲霧、光線變化、粉塵、未建模場景或相機運動，不保證是煙。

## 可選 UAV camera motion 配準

`--registration` 預設關閉。以 ECC Euclidean（translation + rotation）將較早的 input frames 對齊到**最後一個 input**；target 保留原幀，沒有以未來 target 配準造成作弊。失敗時 warning 並回退原圖。Train/inference 必須同樣設置，checkpoint 會檢查。

```powershell
python train.py --registration --experiment-name registered_001 --epochs 30 --batch-size 2
python test.py --registration --checkpoint ..\..\output\astt\registered_001\checkpoints\best.pth --experiment-name registered_test
```

這只穩定歷史 context，不能消除最後 input→future target 的相機運動；對 parallax、尺度變化、透視、rolling shutter、大面積煙霧／動態前景仍有限，邊界反射填補也可能產生誤差。ECC 增加 CPU latency，不宜直接聲稱可以機載即時。需要在 UAV 資料比較 registration on/off、誤報、漏報、事件延遲、FPS 和定位誤差。

## 已執行的驗證與限制

2026-10-01，Windows，既有 conda `boom`：Python 3.10.20、torch 2.13.0+cu130、OpenCV 5.0.0、NumPy 2.2.6；測試選 CPU，未安裝／修改這個環境的套件。

- imports；train/test/extract CLI `--help`。
- 兩影片暫存圖像：窗口不跨 boundary、不跨缺幀、label 長度檢查及 normal-only 排除。
- tiny model 隨機 forward `(2,4,3,32,32)→(2,3,32,32)`，CSA identity-projection 公式核對與 backward/update。
- **完整預設模型** CPU forward `(1,4,3,256,256)→(1,3,256,256)`，有限數值。
- 真正微型 train/val/test CLI、checkpoint 載入與輸出保存。
- 模擬第二 epoch 中斷後 resume，保留此前 best。
- temporal debounce、持續 bbox 只產生一事件、threshold/morphology；AUC ties 和單類別。
- 暫存 MP4 20 FPS→10 FPS 抽幀、10 個 frames 與 manifest。

Fixtures 只位於系統暫存目錄並在測試結束清除，正式 data 沒有假資料。**未做真實資料完整 GPU 訓練，也沒有論文 AUC 重現或煙霧效果評估。** 小模型 smoke 結果和 FPS 沒有效能／準確率意義。
