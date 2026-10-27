# ST² 獨立訓練與資料命名整理（2026-10-27）

> 歷史文件：保留當時的方法、指令與驗證紀錄；其中舊程式可能已移除，請勿當成目前操作指引。目前操作請見 [ST²](../ST2.md) 與 [文件首頁](../README.md)。

> 後續清理更新：使用者要求移除全部舊研究程式，現已刪除Smoke Heatmap相容目錄、四個舊工具入口、舊路徑映射及baseline/camera-aware網路。以下搬移表與驗證是當時的歷史紀錄；目前只使用src/st2與ST²工具。此次清理沒有搬移、改名、改寫或刪除任何既有輸出資料。

使用者要求：ST²訓練與資料準備獨立於Boom；Boom保留原有辨識、顯示、事件與定位，後續以模型適配介面替換模型。

## 新位置

| 原位置 | 新位置 | 處理 |
|---|---|---|
| `src/boom/smoke_heatmap/` 研究實作 | `src/st2/` | 主體移出，舊目錄只留匯入轉接 |
| `src/boom/smoke_heatmap/st2.py` | `src/st2/architecture.py` | 純Transformer三分支 |
| `tools/train_smoke_heatmap.py` | `tools/train_st2.py` → `src/st2/training.py` | 新入口預設ST²；舊入口保留轉接 |
| `tools/infer_smoke_heatmap.py` | `tools/infer_st2.py` → `src/st2/inference.py` | ST²命名的影片推論 |
| `tools/prepare_smoke_heatmap_dataset.py` | `tools/prepare_st2_dataset.py` → `src/st2/prepare_dataset.py` | ST²標註與版本化匯出 |
| `tools/export_smoke_heatmap_tensorboard.py` | `tools/export_st2_tensorboard.py` → `src/st2/export_tensorboard.py` | 訓練紀錄轉TensorBoard |
| `data/astt_bbox/` | `data/st2/` | 同磁碟重新命名，完整保留檔案 |
| `prepared/astt_bbox_v1/` | `prepared/st2_v1/` | 資料集內容、分組與標註不變 |
| `output/astt_bbox/` | `output/st2/` | 保留64個歷史輸出檔及子目錄 |
| `doc/SMOKE_HEATMAP_DATASET.md` | `doc/ST2_DATASET.md` | 舊文件保留導向 |

原 `output/smoke_heatmap/`、`output/roi_heatmap/`、ASTT benchmark、來源影片與模型權重不搬動。output/st2內原本的BBox歷史實驗仍是原模型，不能因目錄更名就視為ST²訓練結果；其既有子目錄與JSON內容不改寫。

## 新訓練／輸出命名

```text
data/st2/prepared/st2_v1/               # 已搬移的現有資料
data/st2/prepared/st2_vN/               # 新匯出，不覆寫既有版本
output/st2/annotations/st2_<時間>.json
output/st2/training/st2_<時間>_<ID>/
  config.json                         # model_name=ST2，model_version=st2.v1.0
  best.pt                             # format=st2.checkpoint.v1
  last.pt
  training.log
  metrics.jsonl
  progress.json
  pipeline.json
  tensorboard/
  inference/st2_inference_<時間>_<ID>/
    summary.json
    checkpoint.pt
    <影片目錄>/annotated.mp4           # 畫面標題含ST2
    <影片目錄>/frame_scores.csv
output/st2/inference/                 # 獨立影片推論的預設父目錄
```

舊 `boom.roi_heatmap.checkpoint.v1` 與 `boom.astt_bbox.v1/v2/v3` 標註仍能讀取。新ST²checkpoint使用`st2.checkpoint.v1`，標註使用`st2.annotations.v1`；只改序列化名稱，不改權重鍵／運算／loss，現有同架構權重不需因搬移重訓。

本專案的ST²載入器對已不存在的 `data/astt_bbox/...`、`output/astt_bbox/...` 路徑會唯讀解析到新位置，不建立symlink，也不改寫歷史檔案。歷史JSON中的絕對路徑仍是當時紀錄；其他外部工具若直接依該字串讀檔，需要自行改成新路徑。

## Boom 的模型替換邊界

`src/st2/runtime.py` 提供獨立的 `ST2Runtime`，只接收影格、管理8幀歷史，回傳熱圖與分數；不讀相機、不畫Boom UI、不確認事件、不估GPS。

Boom原有 `src/boom/interfaces/detector.py` 的 `BaseSmokeDetector` 是執行端契約。未來新增adapter時，將ST2Runtime的輸出轉入原本顯示／候選事件流程，即可讓runner繼續管理影片、相機、事件與定位。這次完成的是訓練分離與模型端介面，尚未設定煙霧事件門檻與確認規則，也未新增`run_realtime.py st2`入口。

## 搬移與驗證證據

- 搬移前後核對45,098個資料檔的相對清單、大小與修改時間（只映射已更名的目錄）；64個歷史輸出檔也完整一致。
- 67個JSON／JSONL／NPY／CSV資料描述檔的SHA-256逐一一致。未改影像、標註、權重或原始來源；未覆蓋目的地。
- 現有dataset fingerprint與既有v2.2 best.pt相符；仍可讀取32,354個train窗口，val為12,571個，分組不變。
- canonical st2 package不依賴boom，舊boom.smoke_heatmap模組可匯入同一實作；舊camera-aware權重嚴格載入及前向成功。
- 真實資料train/val各16窗口、64×96、batch2、embedding32、patch8/layers1/window3完成新入口訓練1輪→另run續訓至第2輪→自動影片推論；核對ST²名稱、格式與st2目錄前綴。
- 逐幀runtime與直接model結果一致；8幀暖機後才評分，跳號或尺寸改變重新暖機。新舊checkpoint容器皆可讀。
- 合成標註成功匯出st2_v1、載入9幀train/val窗口，再匯出st2_v2保留v1；新schema儲存／重讀成功。Boom原motion factory建立成功。
- 驗證程式與所有臨時權重、合成影片／匯出均位於`tmp/st2_separation_20261027/`，完成後清理。這些是流程驗證，未正式訓練30輪，也不代表煙霧效果改善。

實際使用指令見 [ST2.md](../ST2.md)，標註見 [ST2_DATASET.md](../ST2_DATASET.md)。
