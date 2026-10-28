# ST² v3：不使用框資訊的預測與運動補償

更新日期：2026-10-28。依使用者要求，先不考慮框資訊。現行模型保留Transformer Predictor＋凍結戶外LoFTR／RANSAC，沒有ST2Motion、Gate、ROI attention、BCE／Dice或框加權分數。架構標記 `st2.v3.0-box-free`，仍使用ST2作為套件名稱。

## 目前輸出什麼

```text
歷史8幀 → Predictor → 預測當前畫面 → 與實際當前幀比較 → raw_mse
最後歷史幀＋實際當前幀 → LoFTR → RANSAC相似矩陣 → warp前幀
    → 與實際當前幀比較 → compensated_mse＋有效遮罩＋配準品質
```

這是兩種獨立診斷：`raw_mse`是模型預測誤差；`compensated_mse`是最新相鄰「實際影格」的補償後差異。補償殘差不使用Predictor輸出，兩者沒有相乘或融合成煙霧信心。本版也沒有先對齊Predictor歷史輸入；目的在先檢查哪些變化可由相機運動解釋，剩餘殘差不能直接認定為煙霧。

Predictor保留8歷史、RGB[0,1]、768×576、patch16／embedding128／4 heads／dropout0.1／2組空間時間block／window6，以及 `clip(last_history+0.25*tanh(delta),0,1)`。當前影格不進Predictor，維持因果性。預設總參數13,016,560，可訓練Predictor1,455,104，凍結LoFTR11,561,456。LoFTR含CNN，因此完整系統不是純Transformer。

## 「不使用框資訊」的範圍

- 不讀框位置／尺寸／數量、target.labels或ROI mask；資料缺少target欄位也能訓練，既有框檔案不改動。
- 不再依框做正負平衡取樣，不算ROI IoU、框導出的frame AUC，不以AUC選權重。
- 不建立Gate或學習框注意力，不輸出attention／weighted_mse。
- **仍使用整幀normal／roi／ignore狀態。**訓練與驗證只接受整段9幀皆normal的窗口；roi與ignore窗口不當成正常樣本。這是有整幀正常資訊的異常偵測，不是完全無標註學習。
- 新版資料fingerprint只涵蓋來源分組、窗口、影格編號／檔名與狀態；框變更不影響此訓練的fingerprint。影像仍遵循原有資料路徑與連續窗口安全檢查。

訓練總loss只有正常窗口的全圖預測MSE，best.pt依最低正常驗證reconstruction MSE選擇。保留訓練相機軌跡增強，影像同步轉換但不建立任何框mask。新增缺框／不合法框內容忽略與fingerprint不受框變更影響的驗證。

標註UI、既有框、資料與輸出保留，供後續研究；不會因停用框監督而刪除。既有標註工具的匯出規則保留，本次直接使用已匯出資料的正常窗口，無須重畫或重新匯出框。

## LoFTR＋RANSAC

實作在 `src/st2/models/registration.py`，使用固定Kornia0.8.2戶外預訓練權重。LoFTR始終eval、權重凍結，不參與反向梯度。

Gate移除後不再需要8組運動特徵，模型每個窗口只匹配最後歷史幀與當前幀一次。配準模組仍支援獨立輸入多幀，正式模型僅使用一對。

預設匹配長邊最多320，尺寸為8倍數、最短邊至少32；灰階std<0.01拒絕；匹配confidence≥0.2。RANSAC以 `estimateAffinePartial2D(current,previous)` 估平移、旋轉、等比例縮放，閾值3縮圖像素、2000次、confidence0.995、refine10。接受至少12內點、內點比≥0.25、兩幀內點凸包面積各≥5%、中位誤差≤3、取樣尺度0.5～2、有效重疊≥30%。這些初始門檻尚未校準煙霧效能。

矩陣為3×3相似矩陣，方向為「當前輸出像素中心→前幀來源像素中心」，縮圖回原尺寸考慮半像素及x/y比例，再以雙線性取樣warp。不是具有透視自由度的homography或完整3D運動。品質八項順序仍為accepted、inlier_ratio、normalized_reprojection_error、previous_coverage、current_coverage、overlap、mean_inlier_confidence、inlier_support。

無效邊界不計入補償殘差或分數。失敗的內部張量為零殘差、false遮罩、accepted=0及identity占位；CSV／Runtime的補償分數為空／None，畫面顯示配準不可用，不將其當作零變化。RANSAC仍可能受視差、樹葉、大面積動態物體或重複紋理影響。

## 輸出與顯示

畫面分為原影片、原始預測MSE、LoFTR／RANSAC補償殘差疊圖。補償失敗時顯示不可用，無效邊界保留原圖，曲線留空。

CSV分開記錄 `raw_mse_mean`／`raw_top_score`／`raw_score_01` 與 `compensated_mse_mean`／`compensated_top_score`／`compensated_score_01`，附最新配準品質。補償分數只取有效像素最高0.1%的平均；原始預測分數仍取全圖最高0.1%。維持固定對數0～1顯示，這些分數不是機率，也不能用舊Gate加權分數的門檻直接解讀。

`--save-heatmaps`保存raw_mse、compensated_mse、compensated_valid、compensated_score_valid與registration_status，不再有attention或weighted_mse。可選舊motion guard保留作額外對照，預設off；不使用任何框。

`ST2Runtime.process_frame()`回傳兩種MSE、有效遮罩、raw_score_01、補償score_01與配準品質／矩陣。配準失敗仍可回傳原始預測分數，補償score_01為None；影格跳號或來源尺寸變動會重新暖機。尚未接入Boom即時確認／GPS，不能用殘差直接產生已確認煙霧事件。

## 執行與權重

```powershell
conda activate boom
python -m pip install -e ".[st2]"
python src/st2 download-weights
python train_st2.py
```

此機LoFTR權重已存在 `models/pretrained/registration/loftr_outdoor.ckpt`。新訓練可用 `--loftr-weights` 指定；匹配大小以 `--registration-max-side` 調整。移除 `--gate-floor`、`--positive-fraction` 與舊運動loss選項。

```powershell
python src/st2 train --epochs 30 --device cuda --infer-source "data/video/1440-1080/0603"
python train_st2.py --resume "output/st2/你的日期時間/training/last.pt" --epochs 30
python src/st2 infer --checkpoint "output/st2/你的日期時間/training/best.pt" --source "data/video/1440-1080/0603" --save-heatmaps
python src/st2 visualize
```

舊v1／v2 Gate模型不相容，需要重新訓練，不自動丟棄舊權重中的Gate後假裝續訓。新checkpoint仍用 `st2.checkpoint.v1` 容器，以 `model_config.supervision=normal-frames-only` 區分架構，完整保存凍結LoFTR；續訓／推論不需外部權重或下載。

## 驗證與限制

2026-10-28已驗證缺失／惡意格式框欄位不影響資料讀取，正常窗口篩選、框變更不影響fingerprint、無Gate模型輸出、MSE-only loss、Predictor反向、LoFTR凍結、單對配準、因果性、完整checkpoint重載、舊Gate權重拒絕、失敗分數None與有效像素top-score。

實際資料train／val各4窗口、64×96、batch2、embedding32／patch8／layers1／window3、CUDA完成1輪及續訓到第2輪，兩次均對真實影片自動推論10幀；核對前8幀暖機、CSV不含ROI欄位、MP4解碼、Runtime空白畫面失敗與跳號重置。獨立CPU推論、guard、NPZ與新版顯示通過，並人工檢查渲染畫面。預設768×576實際窗口前向通過，最新一對配準accepted；三張架構圖匯出通過。

這些是流程驗證，尚未正式30輪訓練、校準異常或煙霧門檻，未證明誤報／漏報改善、偵測延遲、端到端FPS或定位誤差。暫存驗證位於 `tmp/st2_box_free_20261028/`，完成後清理；原框標註及所有既有輸出保留。

舊版方法見 [v1快照](history/ST2_V1.md)、[v2 Gate版快照](history/ST2_V2.md)；目前三圖見 [視覺化](ST2_VISUALIZATION.md)，完整變更見 [MODEL_CHANGELOG.md](MODEL_CHANGELOG.md)。

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
