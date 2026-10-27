# Smoke Heatmap：BBox 監督的煙霧目標變化

> 歷史文件：保留當時的方法、指令與驗證紀錄；其中舊程式可能已移除，請勿當成目前操作指引。目前操作請見 [ST²](../ST2.md) 與 [文件首頁](../README.md)。

> 2026-10-27 更新：本頁保留早期研究架構與命令說明；現行訓練實作已移至src/st2，請以 [ST2.md](../ST2.md) 與 [ST2_DATASET.md](../ST2_DATASET.md) 為準。舊命令與研究網路已移除，下文僅保留歷史紀錄；目前只使用ST²。

模型相對原 ASTT 的差異見 [MODEL_EVOLUTION.md](../MODEL_EVOLUTION.md)，每次變更見 [MODEL_CHANGELOG.md](../MODEL_CHANGELOG.md)。v2.0 已完成訓練；v2.1 是推論顯示／可選防護。2026-10-27 核對實際紀錄：v2.2 camera-aware 已完成 30 epochs，並以第 7 輪 best.pt 完成後續影片推論；下文 2026-10-22 的待訓練敘述保留作為當日歷史。新增 v3.0 `transformer` 為無卷積研究版本，尚未正式訓練。下列既有架構說明為 baseline，各新版差異見後續章節。

使用者的資料定義：框聯集內是想找的目標變化，框外是干擾；沒有框的已標註影格全部視為干擾。ignore 影格不作負樣本。這個模型保留下一幀影像預測及 MSE，再用 BBox 教獨立的關注分支篩選變化，不直接回歸 BBox。

這是 Boom 的獨立研究模型，不是原 ASTT 的完整重現；不更動 `benchmark/ASTT/origin/`、原 ASTT、motion、YOLO 或先前 BBox 實驗資料／輸出。新格式為 `boom.roi_heatmap.checkpoint.v1`，不相容先前的 BBox 回歸 checkpoint，必須重新訓練。

## 完整執行：訓練後自動推論

純 Transformer 模型正式命名為 **ST² v1.0（Smoke-Targeted Spatiotemporal Transformer，煙霧目標導向時空 Transformer）**，架構、相容性與驗證詳見 [ST2.md](../ST2.md)，以 `--model-variant transformer` 選用；以下既有指令繼續使用 camera-aware。

從 Boom 根目錄的 boom 環境執行。依賴可用 `python -m pip install -e ".[smoke-heatmap]"` 安裝，包含 PyTorch、SciPy、tqdm 與 TensorBoard：

```powershell
conda activate boom
python tools/train_smoke_heatmap.py --model-variant camera-aware --epochs 30 --history-frames 8 --height 576 --width 768 --batch-size 4 --device cuda --infer-source "data/video/1440-1080/0603"
```

新實驗位於 `output/smoke_heatmap/training/smoke_heatmap_<時間>_<ID>/`。每輪完成後保存 `last.pt`、`best.pt` 與 `metrics.jsonl`。`config.json` 保存參數、資料摘要、標註 fingerprint、模型及 loss 定義。`progress.json` 保存目前 batch 與 ETA，`pipeline.json` 保存訓練／推論／完成／失敗階段。

新實驗預設為 `camera-aware`；如需重現 v2.0，明確指定 `--model-variant baseline --camera-augmentation 0 --motion-loss-weight 0`。不能以 `--resume` 舊 baseline 權重升級架構；續訓會恢復原權重的 variant，缺少 variant 的舊檔視為 baseline。

資料載入預設 `--workers 4`，使用 4 個程序平行解碼／縮圖及 DataLoader 預取，CUDA 時啟用 pinned memory；需要單程序可設 `--workers 0`。2026-10-22 在本機 RTX 5080、約 63.35 GiB RAM、20 邏輯核心，實際 768×576／9 幀／batch 4、連續 24 個訓練窗口測量：略過第一批後，接續 5 批讀取耗時 workers=0 約 3.47 秒、2 約 2.20 秒、4 約 1.29 秒。第一批（含程序啟動）分別約 0.84／5.02／9.43 秒。資料已在 GPU 時，camera-aware FP32、開啟相機增強，每批訓練更新穩態約 0.15 秒。這是小樣本、檔案快取及當時負載下的讀取／計算分開量測，不是完整 epoch 加速保證；每次重新建立 loader 仍需啟動程序。本次只改預設 workers，不改 batch、尺寸、精度、loss 或模型。RAM 不必吃滿；保留系統空間，避免換頁反而變慢。

只有最後一輪訓練及驗證成功、權重保存完成後，才載入 **best.pt** 自動推論；不使用尚未完整訓練一輪的權重。`--infer-source` 可為影片或第一層含影片的資料夾，支援 MP4/MOV/AVI/MKV/M4V，忽略照片、不遞迴。0603 目前包含 14 支影片。訓練失敗或中斷不會自動進入推論，已完成權重保留。主機或程序中斷後可續訓到同一個總 epoch：

```powershell
python tools/train_smoke_heatmap.py --resume "output/smoke_heatmap/training/你的run/last.pt" --epochs 30 --device cuda --infer-source "data/video/1440-1080/0603"
```

續訓恢復架構、尺寸、batch、抽樣、seed、lr 及 optimizer，要求同一資料標註 fingerprint，另建新 run 保留來源。若要在續訓完成後自動推論，需再次提供 `--infer-source`。每 epoch 重新設定抽樣／dropout 隨機種子；未承諾跨硬體 bitwise 一致。

## 資料與模型

`SmokeHeatmapDataset` 使用既有 BBox 匯出資料，不改寫圖片或標註。從有連續、互相重疊有效窗口的來源重建 9 幀：前 8 幀為 predictor 的輸入，第 9 幀為當前評分影格及其 BBox mask。這與舊 BBox 模型「含當前幀的 8 幀、預測第 8 幀框」不同。時間順序不跨影片、缺幀或 ignore，來源群組不跨 train/val；不補幀。

輸入為 RGB [0,1]、H=576/W=768，Pillow bilinear 直接縮放，框的相對座標轉成同尺寸二元 mask，重疊框取聯集。預設 batch 4 表示 4 個窗口，各有 8 張歷史圖及 1 張當前圖。單批 GPU 預檢在 RTX 5080、FP32、實際資料通過前向、反向及 optimizer 更新，峰值配置顯存約 5.68 GiB；不是完整訓練峰值或效果保證。

架構：

```text
前 8 幀 -> 共享 CNN (1/8 空間格點) -> 各格點的時間 Transformer -> 空間 decoder
        -> 最後歷史幀 + 有界殘差修正 -> 預測當前正常畫面
        -> 與實際當前 RGB 畫面比較 -> 原始 MSE 圖 E

最後歷史幀 + 實際當前幀 + detached E -> CNN encoder/decoder -> 關注圖 A
目標變化圖 H = E * [0.05 + 0.95*A]
```

predictor 預設 embedding=128、4 heads、1 層時間 encoder。預測畫面採前一幀持續性基準加上最大幅度 0.25 的學習修正，並裁至 [0,1]。它不接觸當前真實影格；關注分支在當前畫面到達後可讀取當前影格，因此沒有借用未來畫面。推論不需要人工 BBox。

## 損失及梯度隔離

```text
L = L_normal_prediction_MSE + L_balanced_BCE + L_positive_Dice
```

- predictor 僅使用整個「歷史＋當前」都標為 normal 的窗口學習 MSE。只要任何一幀出現目標 ROI，就不對該窗口計算正常畫面預測損失，避免把目標學成必須重建的正常變化。
- 關注分支在全部有效窗口學習：框內 mask=1、框外=0。正像素與負像素各取平均 BCE，再平均存在的類別，避免小框被大量背景像素淹沒。無目標影格仍有負像素 BCE，要求整張低反應。
- 有目標的影格另計算 Dice，無目標影格不計 Dice。
- 關注分支輸入的誤差圖 detach；關注 loss 不更新 predictor，也不藉由人為放大影像誤差來製造目標熱圖。
- 所有權重的初始係數為 1；ROI floor=0.05 及混合 loss 均是待驗證設定。不可把分類／mask 的 supervised loss 當作 MSE。

原始誤差 `E=mean_RGB((predicted-current)^2)` 保留真實 RGB [0,1] 尺度；加權圖 `H` 是經模型篩選的誤差，必須標明 **weighted MSE**，不能當成未修改 MSE。兩者分開保存／顯示。若原始誤差為零，加權不會憑空生成反應；低誤差的目標仍可能漏掉。

## 驗證與選權重

train 預設以有放回抽樣取得約 50% 有框窗口，epoch 長度與原 train 窗口數相同；沒有新增獨立事件。val 保持完整分布及時間順序。比較原始／加權圖的「最高 0.1% 像素均值」逐幀分數，用當前幀是否有框計算 ROC AUC；相同分數用平均排名。`best.pt` 以加權圖的 val frame AUC 最高選擇，同分取較低 gate loss；val 缺一類時退回 gate loss。

另保存 attention frame AUC 與 attention≥0.5 對 BBox mask 的像素 precision/recall/IoU。這些是相對使用者 ROI 標註的指標，不是另有像素級分割真值、事件延遲或 GPS 定位誤差。epoch loss 為 batch 加權均值；改變 batch／正負抽樣會影響其數值，應以同一 val 設定比較分數與誤報。沒有獨立 test，不承諾加權圖必定優於原始誤差。

## 推論輸出

自動推論放在同一訓練 run 的 `inference/heatmap_<時間>_<ID>/`。也可單獨執行：

```powershell
python tools/infer_smoke_heatmap.py --checkpoint "output/smoke_heatmap/training/你的run/best.pt" --source "data/video/1440-1080/0603" --device cuda
```

每次先複製 checkpoint 並保存 SHA-256。輸出各影片的 `annotated.mp4` 為 1920×900：原圖、原始 MSE、目標加權 MSE 疊圖，下方畫最近 10 秒的 raw/weighted 最高 0.1% 像素分數曲線。模型分數計算早於畫面縮放，原始影片不修改；播放幀率沿用來源 FPS，不保留音訊。可變幀率 PTS 尚未保留。

前 8 幀只暖機，CSV 分數留空，第 9 幀開始評分，不用零值冒充暖機分數。`frame_scores.csv` 每幀保存原始均值、加權均值、兩個 top 分數與關注圖最大值，影格編號從 0 開始。圖的固定色階預設 0～0.02，`--heatmap-max` 只改顏色上限、不改模型、分數或資料；兩張熱圖採相同色階。

全流程的 `summary.json`、`progress.json` 明確保存完成／失敗狀態。可選 `--save-heatmaps` 保存每幀 `.npz`，內含 raw_mse／weighted_mse／attention 原始 float32；預設不逐幀保存大陣列，以免完整影片占用大量磁碟。這是 ROI 變化分數，不直接輸出煙霧機率、事件確認或地理座標。

小規模流程驗證使用 `--epochs 1 --max-train-samples 16 --max-val-samples 16 --inference-max-frames 16`，輸出需指定在 `tmp/`，完成後清理；有限幀推論會標為 `limited`，不能冒充完整結果。正式訓練不使用這些限制。

## v2.1：固定 0～1 分數（不重訓）

2026-10-22 起，顯示與曲線使用所有影片共用的固定換算：

```text
score_01 = clip(log1p(score / 0.0001) / log1p(0.02 / 0.0001), 0, 1)
```

`--score-max 0.02` 決定對應 1.00 的原始分數，`--score-knee 0.0001` 決定對數曲線靈敏度；兩者寫入 summary，不從目前影片的最大／最小值估計、不讀未來畫面。raw、weighted、最終分數都用相同映射，以保留可比性；例如 0→0、0.0004→約0.30、0.02及以上→1。這是數值映射，不是目標機率或統計校準。不同權重／影像尺度可能仍需用獨立 validation 選擇合適尺度，不能僅靠 normalize 保證模型分布相同。

UI 縱軸固定 0.00／0.50／1.00、主分數顯示兩位小數，原始 top MSE 仍顯示於下一行。CSV 原有欄位保留，新增 `raw_score_01`、`learned_weighted_score_01`、`target_score_01`。熱圖像素本身仍採原始／加權 MSE 色階，沒有把每幀熱圖各自拉滿。

## v2.1：可選推論運動檢查（不是新訓練模型）

預設 `--motion-mode off`。若要比較額外運動檢查，明確指定：

```powershell
python tools/infer_smoke_heatmap.py --checkpoint "output/roi_heatmap/training/roi_heatmap_20261021_154749_df620258/best.pt" --source "data/video/1440-1080/0603/DJI_009_V.MP4" --motion-mode guard --device cuda
```

LK 前後一致性追蹤搭配 RANSAC 相似變換估計前一幀到當前幀的背景運動，可靠時檢查 warp 後的局部灰階殘差，產生 soft support，最終圖為 `learned_weighted_mse * support`。不修改神經網路輸入或已訓練權重，不能說模型因此學會載具運動。方法 API 見 [OpenCV tracking](https://docs.opencv.org/4.6.0/dc/d6b/group__video__track.html)。

預設旋轉上限 60 度／秒、中心移動上限 0.75 影像對角線／秒、局部殘差最低尺度 0.015；可由 `--motion-rotation-limit`、`--motion-translation-limit`、`--motion-residual-floor` 調整。估計還檢查重疊比例、RANSAC inlier 與縮放。快速轉鏡或不可靠配準時，`target_score_01`／`guarded_top_score` 留空，曲線不跨空值連線，畫面顯示 N/A；不是「沒有目標」。恢復後需連續 8 張可靠圖更新歷史。原始／learned-weighted 分數與診斷圖仍保留，NPZ 增加 `displayed_mse` 和有效性旗標。可靠時也可能壓掉微弱煙霧或慢變目標，須另測召回；不適合直接當成已證實改善的正式門檻。

這是針對使用者回報的 DJI_007_V 約9.24秒、DJI_009_V 17秒後所做的可選對照；主方案改以後續的運動感知訓練為方向，不用整段丟棄轉鏡畫面取代模型學習。

## v2.2：背景運動與完整時序關注（已實作，待正式重訓）

`--model-variant camera-aware` 保留上述正常畫面 predictor，再增加可學習的 `BackgroundMotion`，並擴充 gate。兩條新路徑都在第 9 幀到達後運作，predictor 仍只能讀前 8 幀。

```text
8 張歷史 + 當前圖 -> 8 組相鄰影格 -> 共享 CNN -> 各組平移、旋轉、縮放
                  -> 可微分 warp -> 補償後各組 MSE / 背景 photometric loss

完整 9 幀 RGB + detached 原始 MSE + detached 8 組補償 MSE
             + detached 運動參數均值/RMS -> gate -> 關注 A
H = 原始預測 MSE * [0.05 + 0.95*A]
```

gate 預設輸入 44 通道：27 個 RGB 通道、1 張原始 MSE、8 張補償 MSE、8 個全局運動統計。這是依時間順序的密集卷積融合，不是新增一個時間 Transformer。背景運動網路在縮小的影像對上估計 4 個參數，以相似變換處理整體平移、旋轉、縮放；使用符合長寬比的取樣矩陣。不是深度或完整 3D 運動模型，視差、滾動快門、動態樹葉、模糊及大幅轉鏡仍可能失效。

訓練時 loader 額外提供全部 9 幀 ROI mask。每組背景配準 loss 排除「變換後的前幀 ROI」與「當前幀 ROI」聯集及其一像素鄰域，並排除取樣邊界，避免要求煙霧也跟著背景完全對齊。背景完全被 ROI 遮住的影格對不計 photometric loss。BBox 是區域弱監督，框內仍可能含背景；此方法不需要煙霧個體編號／track ID。

```text
L = L_normal_prediction_MSE + L_balanced_BCE + L_positive_Dice
  + 0.2 * (L_background_photometric + 0.001 * L_motion_regularization)
```

photometric loss 是有效背景上 RGB 的平滑絕對差 `sqrt(diff²+1e-6)` 均值，不是最終 MSE heatmap。regularization 是有界運動參數平方均值；係數由 `--motion-loss-weight` 控制。gate 接收的 motion 特徵全部 detach，因此 gate loss 不會把背景運動變成任意放大目標的工具；motion loss 也不更新 predictor 或 gate。這讓背景分支以配準學習共同運動，再讓 ROI 監督學習哪些局部變化值得關注；不能在正式評估前認定它已能辨識所有載具移動。

預設 `--camera-augmentation 0.5`：每個訓練窗口有 50% 機率套用沿時間平滑變化的平移／旋轉與裁切縮放，同時變換全部影格與各幀 mask。不是每張獨立亂移；影像使用 bilinear、mask 使用 nearest。驗證與推論不施加此增強。原始匯出資料不修改；ignore／不連續窗口仍排除，無框的有效影格仍作為干擾監督。真實轉鏡與雲霧等 hard negatives 的品質及覆蓋仍影響學習效果。

舊 checkpoint 缺少 variant 時自動載入 baseline，原始／加權分數保持一致。新 variant 新增 motion 與不同 gate 輸入，必須另開訓練；不以舊權重冒充新版。推論依 checkpoint 自動建立對應架構，不需要人工框，預設關閉額外 `--motion-mode guard`，0～1 固定顯示繼續保留。新版尚未整合至 `artillery/online/`。

2026-10-22 驗證完成：合成已知平移的背景配準 loss 從約 0.09207 降至 0.00293；ROI 排除、無背景、連續增強對齊、梯度隔離及 predictor 因果性測試通過；小資料完成訓練→續訓→自動影片推論，舊 checkpoint 恢復與真實舊影片分數一致（誤差 <1e-7）。另以實際資料、768×576、8 張歷史＋當前、batch 4、FP32、啟用增強，在 RTX 5080 完成一次前向／反向／optimizer 更新，峰值配置顯存約 7.18 GiB。這不是完整 30 epochs 峰值，也不是煙霧效果驗證。臨時驗證及產物已清理；尚無 v2.2 正式訓練權重或全片推論結果。

## 2026-10-22 命名與檔案整理

方法名稱統一為 **Smoke Heatmap**，套件為 `boom.smoke_heatmap`。模型類別為 `SmokeHeatmapConfig/Model/Criterion`，資料集為 `SmokeHeatmapDataset`，底層連續窗口讀取為 `SmokeSequenceDataset`；不再使用名為 astt_bbox_training 的模組。

```powershell
python tools/prepare_smoke_heatmap_dataset.py
python tools/train_smoke_heatmap.py --model-variant camera-aware --epochs 30 --workers 4 --device cuda --infer-source "data/video/1440-1080/0603"
python tools/infer_smoke_heatmap.py --checkpoint "output/smoke_heatmap/training/你的run/best.pt" --source "data/video/1440-1080/0603" --device cuda
```

新訓練／推論／標註輸出統一放 `output/smoke_heatmap/`，新標註資料預設匯出 `data/smoke_heatmap/prepared/`、名稱 smoke_heatmap。既有資料與輸出不搬動：訓練預設仍指向 `data/astt_bbox/prepared/astt_bbox_v1`；新匯出的資料必須以 `--data` 指定路徑。

為了讀取舊標註／權重，JSON schema `boom.astt_bbox.v1/v2/v3`、checkpoint 格式 `boom.roi_heatmap.checkpoint.v1` 與 model/postprocess 版本識別仍保留，這些是序列化相容性識別，不是程式入口名稱。state_dict 的參數鍵與計算不變，不因更名要求重訓。舊 heatmap 權重可用新推論入口直接載入；resume 使用新訓練入口，仍需符合原資料指紋與架構。

移除的是已停用的 `train_astt_bbox.py`、`infer_astt_bbox.py`、`bbox_model.py` 與 bbox_temporal/loss/metrics 三個模型模組；仍使用的資料與工具邏輯已搬入新套件。原 train/infer_roi_heatmap 與 prepare_astt_bbox_dataset 入口已更名，不保留同功能的舊入口。舊 BBox 回歸權重本身保留，但本次整理後不再提供其執行入口。

## 訓練紀錄、TensorBoard 與終端機進度

每個新 run 保存 `training.log`（帶時間的進度／每輪摘要／錯誤）、`metrics.jsonl`（每輪完整數值）、`progress.json`（最新 batch 與 ETA）、`pipeline.json`（流程狀態），並預設寫入 `tensorboard/events.out.tfevents.*`。TensorBoard 已安裝於本機 boom 環境；其他環境可安裝上述 smoke-heatmap 依賴。指定 `--no-tensorboard` 時只停用事件紀錄，保留文字及 JSON 紀錄。

在另一個終端機啟動檢視，不影響正在訓練的程序：

```powershell
conda activate boom
python -m tensorboard.main --logdir output/smoke_heatmap/training --port 6006
```

瀏覽器開啟 `http://localhost:6006`。Scalars 提供 train/val 的總 loss、正常畫面預測 MSE、BCE、Dice、motion photometric／regularization、驗證 raw/weighted/attention AUC、ROI precision/recall/IoU、耗時及學習率。`train_step/*` 在每個訓練階段第一批、約每5秒及最後一批記錄當批 loss；`train/*`、`val/*` 為完整 epoch 統計。缺少正／負類而未定義的 AUC 不寫成0。Text 保存設定，這版不逐批寫 heatmap 影像，減少額外 I/O。

續訓另建 run、事件步數延續原 epoch／batch 編號；選取不同 Runs 可對照，不覆寫先前事件。事件定期 flush 並在每輪、結束／例外時關閉保存。metrics.jsonl 仍是每輪數值的可重讀紀錄，不依賴 TensorBoard 才能取用。

互動終端機中 train／val 各使用同一行 tqdm 進度條，顯示 epoch、完成比例、batch、平均 loss、耗時、速度及 ETA；階段結束清掉暫時進度條，每輪只保留一行結果摘要。stdout/stderr 重導向檔案或非互動環境時不畫動態條，只保留摘要；詳細 batch 狀態繼續寫 training.log／progress.json。

舊 run 只有 metrics.jsonl 時可轉換，原檔、權重及設定不修改，轉出獨立且不覆寫的 tensorboard_import_<時間>_<ID> 子目錄：

```powershell
python tools/export_smoke_heatmap_tensorboard.py --run "output/roi_heatmap/training/roi_heatmap_20261021_154749_df620258"
python -m tensorboard.main --logdir output/roi_heatmap/training --port 6006
```

轉換只包含原本已保存的完整 epoch 指標，不能還原舊程序沒有保存的 batch 曲線；前28輪與續訓29～30輪若分屬不同 run，需各自轉換查看。這些顯示與紀錄改動不更動模型／loss／訓練設定，也不代表已啟動正式訓練。
