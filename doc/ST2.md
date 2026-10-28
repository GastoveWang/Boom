# ST² v2：正常畫面預測＋LoFTR／RANSAC 背景補償

更新日期：2026-10-28。保留 ST² 的正常影格 Predictor 與框監督 Gate，移除 `ST2Motion`／`PairMotionEncoder`，背景配準統一使用凍結的戶外 LoFTR＋RANSAC。架構標記為 `st2.v2.0-loftr-ransac`。

Boom 是一套用於飛行中無人機的荒野煙霧偵測與定位系統。ST² 是獨立辨識研究套件，目前輸出候選熱圖與分數，尚未接入 Boom 即時事件確認或 GPS 定位。

## 資料流

```text
歷史 8 幀 → ST2Predictor → 正常預測畫面 → 與當前影格比較 → 原始 MSE E
歷史＋當前的 8 組相鄰影格 → 戶外 LoFTR → RANSAC 相似變換
    → warp 前幀到後幀 → 補償 MSE、有效遮罩、配準品質
完整 9 幀＋E＋補償資訊 → ST2Gate → attention A
H = detached(E) × (0.05 + 0.95 × A)
```

Predictor 仍只讀歷史，沒有當前影格洩漏；本版沒有先對齊 Predictor 的歷史影像，也沒有改用補償 MSE 直接取代最終熱圖的 E。Gate 的補償輸入與原始誤差皆隔離梯度。配準品質差表示補償不確定，不等同有煙霧；補償後殘差也可能來自視差、樹葉、遮擋或曝光。

## Predictor 與 Gate

- 預設 8 張歷史、RGB [0,1]、768×576；patch16、embedding128、4 heads、dropout0.1、2 組空間／時間 block、6×6 空間視窗。
- Predictor 保留 `clip(last_history + 0.25*tanh(delta),0,1)`；以位置編碼、交替視窗空間注意力、同格點時間注意力及線性 patch decoder 生成影像。
- Gate 的序列 encoder 保留。誤差 embedding 從 9 通道擴為 17 通道：原始 MSE 1＋補償 MSE 8＋有效遮罩 8。八項配準品質沿影格對取平均後，經線性層加入 decoder query。
- Predictor／Gate 仍無卷積；LoFTR 使用 CNN＋Transformer，因此完整系統不能稱為純 Transformer。

## LoFTR＋RANSAC 配準

實作：`src/st2/models/registration.py`。依賴固定為 `kornia==0.8.2`，使用戶外預訓練 LoFTR；權重凍結、始終 eval，不計算配準反向梯度。

1. 相鄰兩幀轉灰階，最長邊最多 320，尺寸調整為 8 的倍數，最短邊至少 32。匹配與品質門檻在縮圖座標計算。
2. 灰階標準差低於 0.01 的影格對直接判為不可靠，避免平坦畫面的虛假匹配。
3. LoFTR 輸出兩幀對應點與 confidence；保留有限值及 confidence≥0.2 的點。
4. `cv2.estimateAffinePartial2D(current_points, previous_points, RANSAC)` 估計平移、旋轉、等比例縮放。門檻 3 像素、最多 2000 次、confidence0.995、refine10。
5. 接受條件：至少 12 內點、內點比例≥0.25、兩幀內點凸包面積各≥畫面5%、內點中位重投影誤差≤3像素、取樣尺度0.5～2、有效重疊≥30%。這些是初始設定，尚未依煙霧驗證集校準。
6. 依縮圖比例與半像素 resize 座標轉換，將矩陣換回模型輸入解析度；用 `grid_sample` 雙線性取樣對齊。輸出矩陣為 3×3，但它表示相似變換，沒有透視自由度。

矩陣方向明確為「當前／輸出像素中心 → 前幀／來源像素中心」。殘差在每個影格對的後幀座標，不是全部累積到最新幀。取樣越界像素不參與殘差；失敗輸出零殘差、全 false 遮罩、accepted=0，以及 identity 矩陣占位，不能將 identity 解讀成已成功估到零運動。

八項品質順序：accepted、inlier_ratio、normalized_reprojection_error、previous_coverage、current_coverage、overlap、mean_inlier_confidence、inlier_support。誤差以 RANSAC 門檻正規化，support 為 min(內點數/100,1)。accepted 才是是否成功的判斷；其他指標在拒絕時可能仍有診斷值。

配準不讀真實 ROI，也不依賴推論時不存在的標註；RANSAC 假設主要匹配屬於背景。大面積動態物體、重複紋理及山林視差仍可能使此假設失效。沒有 ST2Motion 回退；失敗資訊交給 Gate。

為限制顯存，每個 batch 內的相鄰影格對逐組匹配。目前沒有跨窗口快取，避免打亂訓練／多影片時混用狀態；代價是重複窗口會重算，不能據此宣稱已達機載即時 FPS。

## Loss 與權重相容性

總 loss = normal-window prediction MSE＋balanced BCE＋positive Dice。移除原運動背景 photometric loss、運動參數 regularization 與 `--motion-loss-weight`。保留 50% 訓練相機軌跡增強、框弱監督、normal／ignore 定義、top-0.1% 分數與固定對數顯示；框不是煙霧像素真值。

Gate 輸入已改變，新版本需重新訓練。舊 ST² v1／camera-aware／baseline 權重會明確拒絕，不靜默轉換；既有權重、資料與輸出保留原地。完整 v2 checkpoint 保存凍結 LoFTR state_dict 與全部配準設定，因此續訓與推論不需要下載或依賴原本的 LoFTR 檔案。容器仍用 `st2.checkpoint.v1`，以 `model_config.registration_backend=loftr-ransac` 區分新架構。

## 執行

```powershell
conda activate boom
python -m pip install -e ".[st2]"
python src/st2 download-weights
python train_st2.py
```

LoFTR 權重放在 `models/pretrained/registration/loftr_outdoor.ckpt`，受 Git 忽略；下載命令保留既有檔案不覆寫。新訓練可用 `--loftr-weights` 指定戶外權重，`--registration-max-side` 調整匹配解析度；其他配準參數定義於 ST2Config，皆隨 checkpoint 保存。

```powershell
python src/st2 train --epochs 30 --device cuda --registration-max-side 320 --infer-source "data/video/1440-1080/0603"
python train_st2.py --resume "output/st2/你的日期時間/training/last.pt" --epochs 30
python src/st2 infer --checkpoint "output/st2/你的日期時間/training/best.pt" --source "data/video/1440-1080/0603" --device cuda
python src/st2 visualize
```

## 推論診斷與 Boom 接入

CSV 新增最新影格對的 `registration_status`（warming_up／aligned／uncertain）、inlier_ratio、overlap 與正規化重投影誤差。配準品質不是煙霧信心，也不自動將原始模型候選分數設為零。`--save-heatmaps` 額外保存配準狀態。

`ST2Runtime.process_frame()` 保留暖機／原始 MSE／加權 MSE／attention／score，另外回傳最新配準狀態、八項品質與矩陣。影格跳號或來源尺寸改變會重新暖機。尚未提供 `run_realtime.py st2`；Boom 的 motion／YOLO 與定位不變。

## 驗證與限制

2026-10-28 使用實際戶外 LoFTR 權重驗證 RANSAC 外點排除、矩陣方向、縮圖比例、warp／無效邊界、平坦影格失敗、前向／反向、梯度隔離及完整 checkpoint 離線重載。實際森林照片在320×240人工右移12像素，估得逆向平移約(-12.114,-0.027)；有效區域補償 MSE約0.00002288，未補償全圖MSE約0.01902。這是受控幾何驗證，兩個均值的比較範圍不同，不是煙霧效能指標。

實際資料 train／val 各4窗口、64×96、batch2、embedding32、patch8、layers1、window3、CUDA，完成1輪與續訓至第2輪，兩次均自動推論真實影片10幀；已核對前8幀暖機、配準CSV、輸出層級、MP4解碼與Runtime跳號重置。小樣本AUC不作效能結論。預設768×576實際窗口前向通過，八組配準均accepted；預設總參數14,964,080，其中可訓練3,402,624、凍結LoFTR 11,561,456。

尚未正式30輪訓練、校準煙霧誤報／漏報、量測穩態端到端FPS或地面定位誤差。此版不是完整3D補償，也未解決Predictor歷史未對齊或原始MSE熱圖基底的限制。

測試程式與產物位於 `tmp/loftr_validation_20261028/`，驗證完成清理；長期保留正式預訓練LoFTR權重及架構文件。舊版完整說明與驗證見 [ST² v1歷史快照](history/ST2_V1.md)，模型演進見 [MODEL_EVOLUTION.md](MODEL_EVOLUTION.md)。

## 訓練輸出整理（2026-10-27）

每次執行建立 `output/st2/YYYYMMDD_HHMMSS_微秒/`，日期時間來自執行電腦；微秒用於避免同秒啟動時撞名。`--output` 可更換這些日期時間資料夾的父目錄。既有輸出不搬移、不覆寫。

```text
output/st2/20261027_153000_123456/
├── training/
│   ├── best.pt          # 驗證集指標最佳；自動推論使用
│   ├── last.pt          # 最新完成一輪的權重，含 optimizer，可續訓
│   ├── final.pt         # 全部指定輪次成功完成的最終權重
│   ├── training.log     # 訓練、驗證與流程錯誤紀錄
│   ├── tensorboard/     # loss、驗證指標、學習率與設定
│   ├── config.json      # 模型／訓練設定、資料指紋、數量與環境版本
│   ├── metrics.jsonl    # 每輪 train／val 指標
│   ├── progress.json    # 最新 batch 進度
│   ├── pipeline.json    # 訓練／推論流程狀態
│   └── summary.json     # 完成輪次、最後指標與權重位置
└── inference/
    ├── checkpoint.pt     # 本次推論使用的權重快照
    ├── summary.json      # 所有影片的推論摘要
    ├── progress.json     # 推論進度
    ├── 001_影片名稱/     # 影片、CSV 及該影片摘要
    └── 002_影片名稱/
```

訓練後自動推論的影片資料夾直接位於 `inference/`。獨立執行 `python src/st2 infer` 時，仍在指定輸出目錄建立獨立的 `st2_inference_<時間>_<ID>/`，避免覆寫已有推論。

`final.pt` 與完成當下的 `last.pt` 保存同一輪模型；前者表示訓練成功結束，後者每輪更新。訓練失敗或中斷時不產生 `final.pt`；`last.pt` 保留最近成功保存的一輪。訓練完成後，即使後續推論失敗仍保留 final.pt。`best.pt` 可能來自較早的輪次。續訓會建立新的日期時間資料夾，也接受舊輸出中的權重路徑。未指定推論來源時 inference 目錄留空；根目錄入口仍預設完成後推論 0603 影片。`--no-tensorboard` 可停用 TensorBoard。

```powershell
python train_st2.py
python train_st2.py --resume "output/st2/你的日期時間/training/last.pt" --epochs 30
tensorboard --logdir output/st2
```
