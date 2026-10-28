# ST² v1.0：煙霧目標導向時空 Transformer

> 目前程式只保留ST²；舊Smoke Heatmap轉接、baseline/camera-aware網路與舊路徑映射已移除。所有既有輸出維持原處，沒有因程式清理而改名或刪除。

模型全名為 **Smoke-Targeted Spatiotemporal Transformer**，論文／文件簡稱 **ST²**，程式與檔名使用 **ST2 / st2**。Boom 是無人機荒野煙霧偵測與定位系統，ST² 是其中的辨識研究模型。

> 我們提出煙霧目標導向時空 Transformer（Smoke-Targeted Spatiotemporal Transformer, ST²），結合正常影格預測、背景運動估計與區域監督的目標關注機制，產生荒野煙霧候選變化熱圖。

Smoke-Targeted 表示框標註監督的目標關注；Spatiotemporal 表示空間／時間注意力；Transformer 表示三條可學習分支皆採無卷積架構。名稱尚未查核是否與既有論文或專案重名。

ST² v1.0 的初期工作名稱是 Smoke Heatmap v3.0；目前已獨立為 ST² 套件。程式主體為 `src/st2/models/transformer.py`，包含 `ST2Predictor`、`ST2Motion`、`ST2Gate`；模型建立與 loss 由 `src/st2/models/network.py` 負責，訓練與推論由同套件負責，與 Boom 執行環境分離。

目前只支援 `transformer` 模型架構。這是 Boom 自行實作的架構，參考分離時空注意力與密集 patch 解碼思路，不是 TimeSformer 或 Segmenter 的官方重現，也沒有匯入其權重。

## 架構及預設值

所有可學習的影像特徵處理使用線性投影、Transformer attention、MLP 與 LayerNorm，模型內沒有卷積／反卷積。切塊、拼圖、雙線性幾何配準及 MSE 是明確的數學操作，仍會保留。

- 8 張歷史、RGB [0,1]、768×576；patch=16，每張 48×36=1,728 個 tokens。
- embedding=128、heads=4、dropout=0.1、前饋維度=512；預測與 gate 各有 2 組空間／時間 block。
- 空間使用 6×6 patch 視窗，交替使用未移位與偏移半個視窗的分組。偏移以 padding 實現，不環繞；padding keys 被遮罩。相鄰視窗隨層交換資訊，不是一次全圖注意力。
- 時間注意力在各格點的歷史 tokens 上執行；二維正弦位置編碼＋可學時間編碼保留位置及順序。不把每張圖壓成單一 CLS。
- 完整預設模型共有 3,587,204 個參數。

### 正常畫面 predictor

歷史 RGB → 線性 patch embedding → 時空 encoder → 每個格點以最後歷史特徵＋可學 query，交叉注意歷史同格點記憶 → 空間視窗 attention → 線性輸出每塊 RGB → 拼回影像。

`predicted = clip(last_history + 0.25*tanh(delta), 0, 1)`；依舊只用歷史，不讀當前答案。decoder 的 cross-attention 在同格點時間記憶上進行，跨位置資訊由空間 block 傳遞，並非全域任意對應。

### 背景運動

完整 9 幀拆成 8 組相鄰 RGB 對（6 通道），每對先縮至 80×64，使用固定 patch=8，得到 80 個 pair tokens。加入全局 motion token 與 2 層 Transformer encoder，再由線性層輸出 4 個相似變換參數。

沿用 v2.2 的有界平移／旋轉／縮放、符合長寬比的矩陣、可微分 warp、配準誤差及 ROI 排除 loss。幾何運算直接實作在 `ST2Motion.forward`，已移除舊 CNN 運動類別。最後線性 head 初始化為零，從 identity 開始。這不是光流、深度或完整 3D 運動估計。

### 目標 gate

完整 9 幀 → 獨立的線性 patch embedding＋時空 encoder。原始 MSE＋8 張 motion MSE 經另一個線性 patch embedding；4 個運動參數的時間均值與 RMS 經線性層，共同加入當前位置 query。Transformer decoder 查詢 9 幀記憶，再經空間 attention、線性 patch logits 與拼圖，得到像素 logits；Sigmoid 得到 A。

`E = mean_RGB((predicted-current)^2)`；`H = detached(E)*(0.05+0.95*A)`。輸出鍵、固定 0～1 顯示、top-0.1% 逐幀分數與可選 guard 均沿用。gate 的所有 predictor／motion 輸入 detach，三條分支維持獨立監督梯度。

## 訓練與相容性

### 根目錄一鍵訓練與推論

```powershell
conda activate boom
python train_st2.py
```

根目錄入口直接呼叫現有ST²引擎，預設資料`data/st2/st2_v1`、30輪、batch4、768×576、device=auto。訓練與驗證成功、權重保存完成後，載入best.pt對`data/video/1440-1080/0603`內影片推論。輸出仍在`output/st2/<日期時間>/training/`，自動推論放在同一日期時間資料夾的inference子目錄，與training並列。

```powershell
# 更換推論來源
python train_st2.py --infer-source "data/video/你的影片.mp4"
# 續訓至總共30輪，完成後自動推論
python train_st2.py --resume "output/st2/你的日期時間/training/last.pt" --epochs 30
```

根目錄入口只是設定預設推論來源；`python src/st2 train`原本的可選推論行為不變。所有ST²訓練參數均可使用，`--help`查看完整選項。推論來源先驗證，來源無效不啟動訓練；訓練失敗或中斷不會進入推論。既有模型與輸出不覆寫。

### 引擎設定與權重

標註定義、資料切分、9 幀連續窗口、正常窗口限定預測 MSE、平衡 BCE、正樣本 Dice、背景 loss 係數 0.2、motion regularization 係數 0.001、相機增強機率 0.5 均不變。BBox 是區域弱監督，不是煙霧像素真值。

ST² 架構仍需重新訓練，不能把 v2.0/v2.2 的權重直接當作 ST²。2026-10-27 獨立套件整理不改模型參數、loss 或分數，已存在的同架構 ST² 權重無須因搬移重訓。

目前主程式位於 `src/st2/`，不依賴 `boom`。新 checkpoint 使用 `st2.checkpoint.v1`，架構標記為 `st2.v1.0`；訓練／推論摘要保存 `model_name=ST2` 與完整名稱。載入器接受新容器，以及舊容器中variant明確為transformer的同架構ST²權重；state_dict鍵不變。baseline/camera-aware及缺少variant的歷史模型會明確拒絕，不再提供其網路程式。

新入口 `src/st2/cli/train.py` 預設 `variant=transformer`，資料為 `data/st2/st2_v1`，輸出至 `output/st2/<日期時間>/training/`。標註新匯出名預設 st2，新標註 schema 為 `st2.annotations.v1`，仍支援舊 boom.astt_bbox.v1/v2/v3。歷史 JSON 與權重不改寫。

支援 `--patch-size`、`--transformer-layers`、`--window-size`；訓練尺寸需同時符合資料載入器的8倍數及patch整除。續訓恢復來源架構設定，推論尺寸由checkpoint取得。舊tools、boom.smoke_heatmap轉接與舊路徑自動映射已移除，只使用現行ST²入口與實際路徑。模型／loss類別為ST2Config、ST2Model、ST2Criterion；資料集為ST2Dataset／ST2SequenceDataset。

```powershell
conda activate boom
python src/st2 train --model-variant transformer --epochs 30 --history-frames 8 --height 576 --width 768 --batch-size 4 --workers 4 --device cuda --infer-source "data/video/1440-1080/0603"
```

正式輸出仍在 `output/st2/<日期時間>/training/`。單獨推論與續訓使用 ST² 入口，依 checkpoint 恢復設定：

```powershell
python src/st2 infer --checkpoint "output/st2/你的日期時間/training/best.pt" --source "data/video/1440-1080/0603" --device cuda
python src/st2 train --resume "output/st2/你的日期時間/training/last.pt" --epochs 30 --device cuda
```

## 驗證與限制

現行ST²已驗證無卷積、predictor因果性、熱圖公式、三分支梯度隔離、ROI排除、padding前反向、patch=8、checkpoint往返，以及清理前後相同權重的輸出逐值一致。舊baseline/camera-aware網路已移除；其早期驗證紀錄只保留在歷史文件。

RTX 5080 實際資料 768×576、8 歷史＋當前、batch=4、FP32：用 2 個不同窗口各重複一次，完成單次前向／反向／AdamW 更新，峰值 allocated 顯存約 7.13 GiB，單次約 0.89 秒（不含解碼及增強、包含首次 GPU 計算開銷）。不是穩態 FPS、完整訓練峰值或公平效能比較。

所有流程驗證產物均在專案tmp下建立並已清理，摘要保留於本文與 [模型變更紀錄](MODEL_CHANGELOG.md)。既有模型、資料與輸出保持原樣。

完整流程另以實際train/val各16窗口、64×96、batch2、embedding32、patch8/layers1/window3完成第1輪與另run續訓至第2輪；兩次均從best.pt自動推論12幀合成影片。已核對續訓恢復自訂架構、版本與motion資訊、前8幀暖機CSV空分數／後4幀評分，以及12幀MP4解碼。此流程只驗證可執行性，小樣本AUC不作辨識效果結論。

尚未正式30輪訓練或進行煙霧效能評估，也未接入即時 runner／GPS 定位。視窗與 patch 可能限制小煙霧細節及大幅位移的對應；純 Transformer 不保證更準、更快或自動解決相機移動。後續需固定資料比較誤報、召回、延遲、像素／區域定位及端到端 FPS；最終定位仍需另外量測地理座標誤差。

## 與 Boom 的分工

- `src/st2/`：模型、訓練、標註／資料集、權重載入與模型推論。
- `src/boom/`：原有 motion／YOLO、影片／相機 runner、顯示、事件紀錄與定位；已移除smoke_heatmap子目錄。
- `src/st2/engine/runtime.py`：提供 `ST2Runtime(checkpoint).process_frame(frame_bgr, frame_index)`，回傳暖機狀態、raw/weighted MSE、關注圖與分數，不產生確認事件或地理座標。影格跳號或來源尺寸變動會重新暖機。

Boom 的 `BaseSmokeDetector` 負責顯示與事件契約，ST²Runtime 是純模型輸出契約；後續接入時在 detection 寫薄適配層，將模型結果交給原有顯示／事件流程。ST²熱圖尚未具備驗證過的事件門檻、確認規則與來源定位，這次沒有擅自改動motion/YOLO或新增確認事件，因此現在仍不能用 `run_realtime.py st2`。

資料與歷史目錄對照及後續清理見 [ST2_MIGRATION.md](history/ST2_MIGRATION.md)，標註操作見 [ST2_DATASET.md](ST2_DATASET.md)。舊歷史輸出中的原模型名稱不代表已用ST²重訓。

## 套件目錄與命令入口

模型運算圖可使用 `python src/st2 visualize` 匯出，支援完整模型與三條分支；現成圖與說明見 [ST2_VISUALIZATION.md](ST2_VISUALIZATION.md)。

ST²入口已收進 `src/st2/cli/`，不再使用tools中的ST²腳本；平常從專案根目錄執行 `python src/st2 train`、`infer`、`prepare` 或 `export-tensorboard`。各命令可加`--help`。

模型在`models/`，資料與標註在`data/`，訓練及推論引擎在`engine/`，共用I/O在`common/`。完整結構見 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。資料集、既有權重與輸出路徑不變。

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
