# ASTT_BBOX 影片推論

> 歷史文件：保留當時的方法、指令與驗證紀錄；其中舊程式可能已移除，請勿當成目前操作指引。目前操作請見 [ST²](../ST2.md) 與 [文件首頁](../README.md)。

> 歷史文件：2026-10-22 已移除直接 BBox 回歸模型與其入口。下列程式路徑／指令僅供舊實驗追溯，不能再執行；目前流程見 [Smoke Heatmap](SMOKE_HEATMAP.md)。資料、權重與輸出仍保留。

從 Boom 根目錄使用已訓練的 `boom.astt_bbox.checkpoint.v1` 權重推論。這是新版 BBox 時序模型的獨立影片入口，不是原版 ASTT，也不經過 motion／YOLO runner。

```powershell
conda activate boom
python tools/infer_astt_bbox.py --source "data/video/1440-1080/0603" --checkpoint "output/astt_bbox/training/astt_bbox_20261017_235728_d7e27d82/best.pt" --device cuda
```

`--source` 可指定單支影片或資料夾。資料夾只處理第一層 `.mp4/.mov/.avi/.mkv/.m4v`（副檔名不分大小寫），不遞迴；靜態照片列入略過清單，不用重複圖片補足歷史。原始影片不修改。

模型架構、歷史幀數及 H/W 全部由 checkpoint 恢復。RGB 轉換、Pillow bilinear 直接縮放及 [0,1] 正規化與訓練相同。使用連續原幀、不降採樣、不補幀；每個窗口預測最後一幀的 ROI 框。8 幀模型的前 7 幀只顯示 WARMUP，不推測缺少歷史的框；窗口不跨影片。

每次建立新的 `output/astt_bbox/results/inference_<時間>_<ID>/`：

- `checkpoint.pt`：本次啟動時複製的權重快照。訓練更新來源 `best.pt` 不會改變正在執行的推論。
- `summary.json`：來源、權重 SHA-256、epoch、模型設定、門檻、略過檔案及各影片結果。
- `progress.json`：目前影片、影格進度及完成數；每約 5 秒或影片結束更新。
- `<序號>_<影片名>/overlay.mp4`：保留原尺寸及 FPS，含前段暖機畫面；畫出達門檻的 ROI 候選框。`q` 是 query 編號，不是跨幀事件 ID。
- `<序號>_<影片名>/predictions.jsonl`：每個已輸出影格一筆，含全部 queries 的分數、原始相對 cxcywh、裁切至原影片邊界的像素 xyxy，以及達門檻且有面積的候選框。影格編號從 0 開始，時間為 frame_index/FPS。
- `<序號>_<影片名>/summary.json`：完成／失敗狀態、輸出影格數、候選框數、最高信心與處理時間。

預設 `--confidence 0.5`、`--batch-size 4`。框表示單類 ROI 候選區域，不代表已確認煙霧、落地事件或地理位置。保留所有 queries、不做 NMS，與現有評估相同；多個 query 可能重複框到同一物體。沒有真值時不輸出 AP、召回率或定位精度。

可以用 `--confidence 0.3` 改變顯示門檻；請在比較時保留門檻資訊。若最高信心仍低於門檻，影片只顯示零個候選框，不用人工作框。`predictions.jsonl` 仍保留低分 queries 供檢查。

只檢查前 16 幀：

```powershell
python tools/infer_astt_bbox.py --source "data/video/1440-1080/0603/DJI_001_V.MP4" --checkpoint "output/astt_bbox/training/astt_bbox_20261017_235728_d7e27d82/best.pt" --max-frames 16 --output tmp/inference_check --device cuda
```

有限幀執行會明確記錄 `limited`；正式處理不要加 `--max-frames`。臨時檢查的產物需按專案 tmp 規則清理。影片無法開啟、尺寸中途改變或明顯提前結束時標為失敗，保留已完成產物並繼續其他影片，最後以非零狀態結束。

若來源為固定幀率，輸出時間沿用該 FPS；可變幀率 PTS 尚未保留。輸出影片以 OpenCV mp4v 編碼，不保留音訊。`pipeline_frames_per_second` 包含解碼、縮放、推論、JSON 與影片輸出，不是單獨模型 FPS，也不代表無人機端即時效能。
