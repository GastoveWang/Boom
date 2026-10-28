# ST² 與 ASTT 的架構差異及模型演進

2026-10-28 新增獨立比較套件 `benchmark/HSTforU/`，固定官方 commit `9548473306bbb89895b077275e8503de79653029`，保留完整快照並包裝 Ped2／Avenue 模型。此次沒有修改 ST² 架構、loss、訓練／推論設定或既有權重，也沒有以 HSTforU 取代 ST²。與 HSTforU 的差異及驗證邊界另見 [HSTforU 比較](HSTFORU_COMPARISON.md)。

本頁用於回答「目前模型相對原本模型改了什麼」。後續逐次修改見 [MODEL_CHANGELOG.md](MODEL_CHANGELOG.md)；目前使用方式見 [ST²](ST2.md)。記錄以實際原始碼、run 設定及輸出為準，不將推測寫成實驗結論。

## 目前狀態（2026-10-27）

目前只保留 `src/st2/` 的純 Transformer ST²；Boom 維持 motion／YOLO 與原有執行、顯示、事件及定位流程。下文按日期保留早期模型與搬移紀錄，涉及的舊入口可能已刪除。

2026-10-27 訓練輸出改為 `output/st2/<日期時間>/training/` 與同層 `inference/`，新增訓練成功結束的 `final.pt` 與完成摘要；每輪 `last.pt`、最佳 `best.pt`、log、TensorBoard、設定與指標集中於 training。此次只調整實驗保存方式，不改相對 ASTT 的架構、loss、訓練參數或權重格式，既有 ST² 權重可繼續載入／續訓，無須因此重訓。

2026-10-28 訓練後自動推論改為各影片目錄直接位於該次的 `inference/`，移除中間一層推論 run 目錄；模型、推論運算與相對 ASTT 的架構差異不變。

| 項目 | 本機 ASTT 比較模型 | 目前 ST² |
|---|---|---|
| 空間／時間主幹 | STE → 每幀 CLS → TTE → CSA | 保留 patch 格點的交替空間視窗／時間注意力 |
| 正常畫面解碼 | 融合向量＋CNN decoder | Transformer query decoder＋線性 RGB patches |
| 背景運動與目標關注 | 核心模型無此兩分支 | Transformer 運動估計＋框監督關注 |
| 模型狀態 | 歷史比較模型，程式在 benchmark/ASTT | 流程驗證完成，尚未正式30輪訓練或接入即時事件 |

ST²保留下一幀預測的研究思路，但不是ASTT完整架構。操作見 [ST2.md](ST2.md)，完整歷史見 [MODEL_CHANGELOG.md](MODEL_CHANGELOG.md)。

## 對照基準

「原版」在本頁指本機可執行的 `benchmark/ASTT/src/model.py` 與實驗 `astt_artillery_shell_v1_001/config.json`。該版源自固定上游快照，官方未修改檔案在 `benchmark/ASTT/origin/`；本頁的參數是本機實驗設定，不聲稱都是原論文預設值。

| 項目 | 原本本機 ASTT | 直接 BBox 模型（保留對照） | 已訓練 ROI heatmap baseline（v2.0） |
|---|---|---|---|
| 入口 | `benchmark/ASTT/scripts/train.py`、`inference.py` | `tools/train_astt_bbox.py`、`infer_astt_bbox.py` | `tools/train_roi_heatmap.py`、`infer_roi_heatmap.py` |
| 任務 | 正常片段下一幀預測，用誤差找異常 | 單類 ROI 框位置與信心 | 正常畫面預測＋BBox 監督的目標變化篩選 |
| 時間 | 前 4 幀預測第 5 幀 | 最後版本含當前幀的 8 幀，預測第 8 幀框 | 前 8 幀預測第 9 幀；當前真實第 9 幀到達後評分 |
| 尺寸 W×H | 256×256 | 最初 256×192，後改為 768×576 | 768×576 |
| 影像尺度 | OpenCV BGR，[-1,1] | Pillow bilinear RGB，[0,1] | Pillow bilinear RGB，[0,1] |
| 空間特徵 | patch=32、STE，每幀壓成 CLS | CNN 保留 1/8 空間格點 | CNN 保留 1/8 空間格點 |
| 時序與解碼 | 12 層 STE、12 層 TTE、CSA、影像 decoder | 各格點時間 Transformer＋16 個 object queries / decoder | 各格點 1 層時間 Transformer＋空間影像 decoder |
| 特徵維度／heads | 768／8 | 128／4 | 128／4 |
| 預測畫面 | 從融合向量生成完整下一幀 | 沒有影像預測輸出 | 上一幀持續性基準＋幅度上限 0.25 的學習殘差，再裁到 [0,1] |
| BBox 用途 | 無逐幀框監督 | Hungarian 配對、分類與框回歸 | 框聯集轉二元 mask，框內目標、框外干擾 |
| 正常預測訓練 | 正常歷史與目標窗口 | 不適用 | 9 幀全為 normal 才計算預測 MSE |
| 其他監督 | 無目標 ROI 關注分支 | CE＋5L1＋2GIoU；後加入 2NWD | 像素正負平衡 BCE＋正樣本 Dice |
| 梯度隔離 | 不適用 | 不適用 | 關注分支讀取 detached 誤差，不能靠 ROI loss 修改 predictor 製造誤差 |
| 最終圖 | 原始像素預測 MSE | 預測 BBox，不能直接產生真正 MSE | 原始 E 與 H=E×[0.05+0.95A] 分開顯示與記錄 |
| frame score | 全圖 MSE 平均 | query 分數與框的 AP50 | 最高 0.1% 像素的平均誤差 |
| 最佳權重 | 本機該實驗以 val AUC 選擇 | val AP50，同分較低 loss | val 加權 frame AUC，同分較低 gate loss |
| 影片時間 | 本機版統一 30 FPS，可能取樣或重複 | 來源原生 FPS | 來源原生 FPS，前 8 幀暖機、不補幀 |
| checkpoint | 原 ASTT 格式 `.pth` | `boom.astt_bbox.checkpoint.v1` | `boom.roi_heatmap.checkpoint.v1` |

這是換了研究架構與監督目標，不是僅在原 ASTT 加一項 loss。原版 BGR [-1,1] 與新版 RGB [0,1] 的 MSE 尺度不同，不能把兩個模型的絕對 MSE 當成同尺度效能比較。新模型與先前兩種權重都不相容，資料／輸出各自保留。

## 目前已訓練實驗

- 初始訓練：`output/roi_heatmap/training/roi_heatmap_20261018_172637_92fc4611/`。
- 第 28 輪後中斷，續訓：`output/roi_heatmap/training/roi_heatmap_20261021_154749_df620258/`，總共完成至第 30 輪。
- 保留的最佳權重是第 7 輪。該輪完整 val 的原始 frame AUC 約 0.60925、加權 frame AUC 約 0.70882；ROI mask IoU 約 0.03382。這些是舊推論流程的指標，不是相機運動檢查後的效果。
- 完整 0603 推論：`inference/heatmap_20261021_204148_88c3bdc2/`，14 支影片、42,435 幀。這些既有影片不因後續程式修改而改變。

目前關注分支的 ROI 定位及泛化仍有不足；完成 30 輪不等於所有干擾已排除。新增運動檢查的效果需以有標註的新對照評估，不能把分數降低直接當成誤報改善。

## 2026-10-22 的推論改動（權重不變）

1. 固定對數換算成 0～1 分數，所有影片／raw／weighted 使用相同參數，曲線縱軸固定，原始 MSE 繼續保留。不是每影片 min-max，也不是機率校準。
2. 在模型之外，以前後影格的 LK 特徵追蹤與 RANSAC 相似變換估計背景運動；檢查補償後的局部變化，作為額外的軟權重。模型輸入及 checkpoint 不變，原始 learned-weighted 圖仍保留。
3. 快速轉鏡、配準失敗或歷史恢復期間，目標分數留空、曲線斷線、畫面標示不可用；不填成 0，不聲稱沒有目標。原始誤差與診斷熱圖仍可檢查。
4. 不重新訓練。下一步若仍有葉片、粉塵、雲霧、模糊等誤報，需要使用者確認的干擾片段與目標片段做 hard-negative／召回率評估，再決定增補訓練。先不自動更改真值或啟動另一輪長時間訓練。

使用者後續明確希望改訓練模式，因此運動檢查保留為 `--motion-mode guard` 的可選對照、預設關閉；不是解決載具運動的主要模型方案。

## 2026-10-22 的 v2.2 模型改動（已實作，未正式重訓）

| 項目 | v2.0 baseline | v2.2 camera-aware |
|---|---|---|
| predictor | 前 8 幀預測當前正常畫面 | 保留，沒有讀入當前答案 |
| gate 時序輸入 | 最後歷史 RGB、當前 RGB、原始 MSE | 完整 9 幀 RGB、原始 MSE、8 組補償 MSE、運動參數統計，共 44 通道 |
| 背景運動 | 無顯式分支 | 相鄰影格共享 CNN，估計平移／旋轉／縮放，再可微分配準 |
| 新增監督 | 無 | 框外背景 photometric loss＋小幅參數正則化；兩幀 ROI 聯集排除 |
| 資料增強 | 無相機軌跡增強 | 訓練窗口 50% 機率連續相機變換，全部影格及 mask 同步 |
| 標註需求 | 逐幀 BBox／normal／ignore | 沿用既有標註，額外讀歷史 mask，不需個體 ID |
| 最終 heatmap | 原始 MSE 乘學習關注權重 | 保留相同公式；額外 guard 預設 off |
| 訓練狀態 | 已完成 30 epochs | 僅小規模流程及實際資料單批 GPU 更新通過 |

新實驗預設 camera-aware；舊檔自動恢復 baseline，新版需要另開訓練，不能直接轉換舊完整權重。測試顯示實際尺寸 batch 4 的峰值配置顯存約 7.18 GiB，尚未測得正式全流程峰值。還不能宣稱 DJI_009_V 17 秒後的誤報已改善；需用固定驗證資料比較靜止、有移動無目標、有移動且有目標的誤報／召回、延遲與 FPS。執行及 loss 詳見 [SMOKE_HEATMAP.md](history/SMOKE_HEATMAP.md)。

同日的訓練速度調整：預設平行載入從 workers=0 改為 4，模型、loss、精度、尺寸及 batch 不變，舊權重可繼續使用。實際資料 5 批純讀取約 3.47→1.29 秒，尚未量測完整 epoch 加速；不以 RAM 是否吃滿判定 GPU 使用效率。

## 2026-10-22 統一名稱：Smoke Heatmap

目前方法正式使用 **Smoke Heatmap** 名稱。上表中的 ASTT、直接 BBox 與 v2.0 比較保留為歷史紀錄；現行程式集中於 `src/boom/smoke_heatmap/`，入口改為 `tools/train_smoke_heatmap.py`、`tools/infer_smoke_heatmap.py`、`tools/prepare_smoke_heatmap_dataset.py`。資料載入已更名為 sequence.py／dataset.py，不再依賴舊 BBox 訓練工具。

舊直接 BBox 回歸程式及專用入口已刪除，相關標註／資料／權重／結果保留。新輸出使用 output/smoke_heatmap，新資料匯出使用 data/smoke_heatmap/prepared；目前訓練預設仍讀既有 data/astt_bbox/prepared/astt_bbox_v1。相容性 schema／checkpoint format 保持原識別，既有 heatmap 權重可在新入口載入，並已驗證搬移前後 baseline／camera-aware 數值完全一致。更名不改模型、loss 或偵測能力，不需要因此重訓；v2.2 仍未完成正式30 epochs。檔案分工及目前指令見 [SMOKE_HEATMAP.md](history/SMOKE_HEATMAP.md)。

同日加入訓練觀測：既有 JSON 紀錄繼續保留，新增 training.log 與預設 TensorBoard 事件（loss、AUC、ROI 指標、耗時、學習率），並將終端機逐批換行改為互動單行進度條／每輪摘要。舊 metrics.jsonl 可匯入獨立事件資料夾；沒有改變模型或權重格式，不能將可視化改動視為偵測效果改善。

## 2026-10-27：v3.0 純 Transformer 分支

保留原ASTT的「正常歷史預測下一幀，用誤差找候選變化」研究思路，以及v2.2的BBox關注監督與背景配準，將三條網路的可學習影像處理改為無卷積實作。這是新架構，需要重新訓練；不直接繼承原ASTT／v2.2完整權重，不是單純替原ASTT新增loss。

| 項目 | 本機原 ASTT | v2.2 camera-aware | v3.0 transformer |
|---|---|---|---|
| 空間處理 | patch＋STE，取每幀CLS | CNN保留1/8格點 | 線性patch16，保留全部格點；局部視窗attention |
| 時間處理 | 每幀CLS進TTE＋CSA | 各格點1層時間Transformer | predictor/gate各2組交替空間／時間block |
| 預測解碼 | 單向量→全連接＋CNN影像decoder | CNN decoder＋前幀有界殘差 | 時間query cross-attention＋空間attention＋線性RGB patches＋有界殘差 |
| 運動分支 | 核心無 | 影格對CNN估相似變換 | 影格對patch tokens＋全局motion token估相同參數 |
| 目標關注 | 無框監督gate | 完整9幀等44通道CNN | 完整9幀時空tokens＋誤差patch/運動統計query＋Transformer decoder |
| 最終熱圖 | 原始預測MSE | MSE×學習關注權重 | 沿用相同MSE與加權定義 |

新版不使用全時空全域attention，也沒有任意跨位置的稠密對應保證；時間attention與decoder cross-attention仍按格點處理，空間訊息透過交替偏移視窗傳遞。背景分支仍只有全局平移／旋轉／縮放，不等於3D相機運動或光流。標註仍是BBox弱監督。

新架構完成GPU前反向／梯度隔離／checkpoint相容性／真實資料小訓練→續訓→影片推論驗證。預設模型約358.7萬參數，真實尺寸batch4的單次更新allocated峰值約7.13GiB；尚未正式訓練或評估煙霧效果，不推論優於v2.2。方法、限制及執行見 [SMOKE_HEATMAP_TRANSFORMER.md](history/SMOKE_HEATMAP_TRANSFORMER.md)。

同日校正目前v2.2狀態：實際run `smoke_heatmap_20261022_182638_27a266c1` 已完成30輪與影片推論，使用第7輪最佳權重；當輪val raw AUC=0.60939、weighted AUC=0.76319、ROI mask IoU=0.04359。這是既有v2.2紀錄，不是本次v3.0的結果，且不代表煙霧分類正確率或像素真值IoU。

## 2026-10-27：模型正式名稱 ST²

上述v3.0純Transformer架構正式命名為 **Smoke-Targeted Spatiotemporal Transformer（ST² v1.0，煙霧目標導向時空 Transformer）**，程式用ST2/st2。主實作更名為 `src/boom/smoke_heatmap/st2.py`，完整文件改為 [ST2.md](ST2.md)，原匯入與文件連結保留轉接。此次只有命名，運算、state_dict及版本相容識別不變，不需因更名重訓。

架構關係上，ST²保留ASTT的下一幀預測研究思路，但沒有沿用完整STE→每幀CLS→TTE→CSA→CNN decoder主幹；時空交替attention較接近TimeSformer設計思路，並加入局部偏移視窗、背景運動及目標關注。ST²為Boom自訂架構，不應描述成ASTT、TimeSformer或Segmenter的官方重現；命名不代表已證實辨識效果。

## 2026-10-27：ST²與Boom執行環境分離

現行研究實作移至`src/st2/`，主架構為`architecture.py`，訓練／推論分別為training.py／inference.py。Boom的motion／YOLO、runner、顯示、事件與定位維持原本實作；boom.smoke_heatmap僅留舊匯入轉接。資料與結果改用data/st2、output/st2，訓練新入口train_st2.py預設純Transformer。

此為程式分工及命名整理，不改ASTT或ST²的運算差異、標註監督與loss。新容器／版本名稱st2.checkpoint.v1／st2.v1.0仍對應同一組模型權重鍵；舊schema可讀，資料指紋與既有權重一致。舊BBox歷史實驗不能因output目錄更名就視為ST²結果。

新增純模型ST2Runtime供未來Boom detector adapter接入，但沒有擅自把變化熱圖確認成煙霧事件；事件門檻、確認規則、即時端到端延遲與GPS定位尚待整合驗證。搬移對照與驗證見 [ST2_MIGRATION.md](history/ST2_MIGRATION.md)。

## 2026-10-27：現行程式僅保留ST²

依使用者後續要求，已移除Boom內smoke_heatmap轉接目錄、四個舊工具入口、ST²內baseline/camera-aware網路及舊路徑自動映射。現行程式只提供ST²三分支Transformer，模型／loss命名為ST2Config、ST2Model、ST2Criterion。Boom原有辨識／顯示／事件／定位程式保持原樣；benchmark/ASTT原比較實作仍獨立保留。

此清理不改ST²運算，清理前後相同state_dict的全部前向輸出逐值一致，不需因刪除舊支線而重訓。舊baseline/camera-aware權重執行支援已取消，但權重與歷史輸出完整留在原處；186個研究輸出檔的清單／大小／時間及116個紀錄雜湊一致。先前各節描述的相容入口與歷史模型支援僅代表當時狀態，現況以 [ST2.md](ST2.md) 為準。

## 2026-10-27：ST²子目錄與統一命令

使用者決定保留Boom名稱；ST²改按models、data、engine、common與cli分工。工具入口全部移入src/st2/cli，平常以`python src/st2 train`／`infer`／`prepare`／`export-tensorboard`執行。模型主體現在為models/network.py與models/transformer.py，純逐幀介面為engine/runtime.py。

此次只整理檔案、匯入與命令入口，不改相對ASTT的架構差異、模型計算或監督方式。同權重輸出逐值一致；資料、權重與歷史輸出不改動，Boom原本執行功能保持原樣。現行結構見 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。

同日新增根目錄`train_st2.py`，預設訓練成功後以best.pt推論0603影片資料夾；模型架構與訓練參數不變。此為一鍵流程入口，套件內`python src/st2 train`仍維持可選推論。詳見 [ST2.md](ST2.md)。

同日新增`python src/st2 visualize`，以torchview匯出目前模型與三分支的真實forward圖。此功能只追蹤既有模型，不改相對ASTT的架構或訓練方法；它不是反向圖或attention權重圖。圖片與操作見 [ST2_VISUALIZATION.md](ST2_VISUALIZATION.md)。

同日依使用者要求移除資料路徑的prepared外層，現行資料為`data/st2/st2_v1/`，標註新匯出也直接放data/st2。資料內容及fingerprint不變，這是目錄整理，不改模型或監督方式；Git只保留零位元.gitkeep資料夾結構，不上傳資料集。
