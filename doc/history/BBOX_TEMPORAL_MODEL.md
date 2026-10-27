# Boom BBox 時序模型：一般物體影像位置

> 歷史文件：保留當時的方法、指令與驗證紀錄；其中舊程式可能已移除，請勿當成目前操作指引。目前操作請見 [ST²](../ST2.md) 與 [文件首頁](../README.md)。

> 歷史文件：2026-10-22 已移除直接 BBox 回歸模型與其入口。下列程式路徑／指令僅供舊實驗追溯，不能再執行；目前流程見 [Smoke Heatmap](SMOKE_HEATMAP.md)。資料、權重與輸出仍保留。

本頁為先前「直接回歸 BBox」的模型，保留供對照。使用 BBox 監督關注區域、最後顯示 MSE heatmap 的新流程請見 [SMOKE_HEATMAP.md](SMOKE_HEATMAP.md)，兩者 checkpoint 不共用。

這是一個獨立於 ASTT 的有監督研究模型，目標是用 BBox 標註學習一般物體的位置，供「找出落地後可見物體」的影像定位實驗使用。Boom 的荒野煙霧監測主軸及既有 motion／YOLO 流程維持原樣。

目前完成架構、BBox 訓練損失、參數查看入口、[時間區段與手動逐幀 BBox 標註 UI](SMOKE_HEATMAP_DATASET.md)，以及新版 ASTT_BBOX 的資料 loader、訓練／驗證與續訓入口。訓練方式見 [ASTT_BBOX 訓練](ASTT_BBOX_TRAINING.md)。已提供[獨立影片框疊圖入口](ASTT_BBOX_INFERENCE.md)；尚未提供注意力 mask 監督 loss 或即時流程適配器；不能宣稱已能辨識物體、已判定落地、已改善誤報或已達即時速度。

## 與 ASTT 的分工

| 項目 | ASTT | 新模型 |
|---|---|---|
| 程式 | benchmark/ASTT/ | src/boom/models/ |
| 學習目標 | 正常片段的下一幀預測 | 最後一個輸入影格的物體類別與 BBox |
| 標註 | 二元影格標籤；正常窗口訓練 | 每個目標物體的類別及 BBox；含正負影格 |
| 框來源 | 預測誤差圖的門檻處理 | 可學習的 BBox 回歸分支 |
| 時間資訊 | 四幀預測未來第五幀 | 預設八幀歷史窗口，含目前幀，預測第八幀的框 |
| 依賴關係 | 自有資料工具、核心與 scripts | 不匯入 ASTT，不共用 checkpoint 格式 |

ASTT 工具及標記介面已恢復至其本機專案。依後續指示，ASTT 資料與結果也回到 `benchmark/ASTT/data/`、`benchmark/ASTT/output/`，保留 `output/astt/training/`、`results/`、`annotations/` 分層；與本頁的新 BBox 模型分開管理。

## 架構概念與實作

```text
截至目前的 T 幀 RGB 影像
    → 共享 CNN：保留 1/8 尺寸的空間特徵
    → 每個空間格點沿時間軸做 Transformer 融合
    → 最後時間 token ＋二維位置編碼
    → 可學習的物體 queries ＋ Transformer decoder
    → 物體類別／背景 ＋ BBox 中心、寬、高
```

`BBoxTemporalModel` 保留空間格點，而非將整張畫面壓成一個 CLS 再重建。CNN 與時序融合是本機研究骨架；物體 queries 與集合預測的設計參考 [DETR 官方實作](https://github.com/facebookresearch/detr)，不是 ASTT 原論文方法或 DETR 的完整重現。

預設 embedding 128、4 heads、1 層時序 encoder、2 層物體 decoder、16 個 queries。所有輸入時間都在輸出影格的當下或以前，推論不用未來影格，也不要求提供人工 BBox。特徵格點的時間融合尚無運動對齊；相機移動或物體跨格點會影響融合品質，必須在實際資料驗證。

### 輸入與輸出

| 資料 | 格式 |
|---|---|
| 輸入 | float RGB，範圍 [0,1]，形狀 `(B,T,3,H,W)` |
| 尺寸 | H／W 至少 16 且可被 8 整除；同一 batch 相同尺寸 |
| 類別 | `class_logits` 為 `(B,Q,C+1)`；最後一類是背景 |
| 框 | `boxes_cxcywh` 為 `(B,Q,4)`；相對輸入影像尺寸的中心與寬高 |
| 評分影格 | 輸入窗口的最後一幀 |

框轉成 xyxy 後可能超出輸入邊界。訓練損失保留原預測座標；未來顯示時再裁切到有效影像範圍。若使用等比縮放及 padding，必須保存映射並去除 padding，不能把相對模型影像的框直接當作原影片座標。

BBox 中心可作為影像上的物體位置標記，但不等於實際接觸地面的點或地理座標。模型只預測物體框，尚未實作「已落地」的狀態判斷；這需要明確的狀態標註與另外評估。物體完全遮蔽時不把未知位置標成確定結果。

## BBox 如何進入訓練

`BBoxSetCriterion` 接收每張最後影格的 `labels` 與 `boxes`，將 predictions 與 ground truth 一對一配對，再計算分類、L1 框回歸及 GIoU 損失；可額外啟用小目標 NWD 距離，並同步加入配對成本。訓練入口的新實驗預設啟用，既有 API 與舊權重維持原版設定，詳見 [loss 設計與對照指令](ASTT_BBOX_TRAINING.md#小目標-bbox-loss)。配對用 SciPy 的 Hungarian solver；沒有配到真實物體的 query 以背景分類訓練。這個集合預測／配對方向參考 [DETR 官方說明](https://github.com/facebookresearch/detr)。

```python
from boom.models.bbox_temporal import BBoxTemporalConfig, BBoxTemporalModel
from boom.models.bbox_loss import BBoxSetCriterion

config = BBoxTemporalConfig(num_classes=1, num_frames=8, num_queries=16)
model = BBoxTemporalModel(config)
criterion = BBoxSetCriterion(config.num_classes)

# clips: 真實的 RGB 歷史窗口；targets: 最後影格的人工 BBox。
outputs = model(clips)
losses = criterion(outputs, targets)
losses['loss'].backward()
```

每個 target 的契約：

- `labels`：長度 N 的 int64 tensor，類別 ID 為 0～C−1。
- `boxes`：N×4 的 float tensor，依序是相對輸入影像的 cx、cy、w、h；框須有正面積並位於影像內。
- 已完整檢查、確定無目標的影格：明確傳入空 labels `(0,)` 及空 boxes `(0,4)`。
- 未檢查或遮蔽到無法判斷的影格：暫時排除監督，不能用空標註冒充背景。
- 一張圖的物體數不能超過 queries；目前超過時會報錯，不會靜默丟棄標註。

模型不會僅靠既有 ASTT 時間區段 `.npy` 學到 BBox。新的資料需要可見一般物體的實際框與類別；這次沒有轉換或重用既有砲彈資料集。

## 新資料與工具規劃

標註、匯出與新版訓練使用 `astt_bbox` 資料／輸出名稱：

```text
src/boom/models/
    bbox_temporal.py          # 已完成：架構、設定與框轉換
    bbox_loss.py              # 已完成：分類與 BBox 監督
tools/bbox_model.py           # 已完成：參數／架構查看
tools/prepare_astt_bbox_dataset.py # 已完成：時間軌、同位套用、逐幀微調與匯出
tools/train_astt_bbox.py      # 訓練、驗證、續訓入口
src/boom/datasets/astt_bbox_training.py # 歷史窗口與單類框 loader
src/boom/models/bbox_metrics.py # IoU 0.5 的 Precision/Recall/AP50
data/astt_bbox/prepared/       # 版本資料集
output/astt_bbox/annotations/  # 手動儲存標註專案
output/astt_bbox/training/     # 獨立 run、best.pt、last.pt、訓練紀錄
output/astt_bbox/results/      # 獨立驗證結果
```

標註工具獨立於原版 ASTT，保存來源 ID、群組、原幀率時間及逐幀框；時間軌顯示區段與有框／缺框位置，可將模板同位套用後再直接拖曳微調。匯出時保存原圖尺寸。UI 不要求物體類別，也不設框或來源的已檢查狀態；區段內每幀須有框或明確略過，區段外視為無目標。未儲存修改會攔住關閉或切換專案。

依目前標註需求，使用者在有物體的區段逐幀標記關注區域 BBox。這些區域可用來產生 ROI 注意力 mask；未必是物體緊密外框，也不代表已判定落地。samples 的類別 ID 固定 0 供單類框監督使用，UI 不標記物體種類。直接約束注意力的 loss 仍需後續實作，不能將區域標註完成等同於注意力監督已接上。

train／val／test 依原始影片或同一次錄製的來源群組切分；相鄰幀、不同解析度版本及同來源裁切片段不可跨組。歷史窗口不得跨影片、剪接或缺幀。先保存原始時間，不強迫把低幀率影片重複成 30 FPS。

loader、訓練 runner 與獨立影片推論工具已接上；影片推論使用 tools/infer_astt_bbox.py。正式即時流程仍需量測延遲與設備負載，目前不將這個模型註冊為可選偵測後端。

## 驗證與使用

架構查看入口從 Boom 根目錄執行：

```powershell
conda activate boom
python tools/bbox_model.py --help
python tools/bbox_model.py --classes 1 --frames 8 --queries 16
```

這個入口只顯示設定與參數量，不會對影片產生未訓練的辨識結果。架構及損失分別需要 PyTorch 與 SciPy；缺少時可使用 `python -m pip install -e ".[bbox]"`。現有環境若已具備套件，不需重裝。

本次臨時驗證涵蓋模型載入、不同大小輸入、BBox 輸出形狀、有監督 loss 與反向傳播、空負樣本、標註排列不變性及錯誤標註檢查；程式和測試資料位於 `tmp/` 並在完成後清理。這些不代表真實辨識效能。

正式資料應比較單幀偵測與時序模型的 precision／recall、AP、每分鐘誤報、可見物體首次被找到的延遲、BBox／影像位置誤差及端到端 FPS。位置指標需約定是中心、框重疊或其他影像點，並另行記錄遮蔽案例。沒有地理真值時不報告 GPS 定位精度。
