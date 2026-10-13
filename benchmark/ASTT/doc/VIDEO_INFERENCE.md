<!-- 檔案功能：以 ASTT 訓練權重直接推論新影片的操作與輸出說明。 -->

# 用權重跑新影片

不需先標記影片、切 train／val 或產生 `.npy`，也不會重新訓練。從 Boom 根目錄執行（將影片路徑換成你的檔案）：

```bat
conda activate boom
python benchmark/ASTT/inference.py --video "C:\videos\new_video.mp4" --checkpoint benchmark/ASTT/output/astt/astt_artillery_shell_v1_001/checkpoints/best.pth --device cuda
```

預設會建立新的 `benchmark/ASTT/output/astt/video_<時間>/`。模型尺寸與歷史幀數從 checkpoint 讀取。預設 batch size 8，影片與標記匯出共用固定 30 FPS 時間軸：高幀率取樣、低幀率重複原始影格，無需手動設定 FPS。影片來源不會被修改，不會保存抽出的原始圖片。

加入 `--display` 可在處理期間預覽；按 Q 或 Esc 提前結束，保留已處理部分。預覽以實際推論速度更新，輸出的影片仍為 30 FPS。若只想先確認能跑，加 `--max-frames 300`，只處理轉換後前十秒。`--experiment-name new_video_001` 可指定名稱，已存在的輸出目錄會拒絕覆寫。正式跑完整影片不需 `--max-frames`。

例如：

```bat
python benchmark/ASTT/inference.py --video "C:\videos\new_video.mp4" --checkpoint benchmark/ASTT/output/astt/astt_artillery_shell_v1_001/checkpoints/best.pth --device cuda --display --experiment-name new_video_001
```

## 輸出

| 檔案 | 內容 |
|---|---|
| `annotated.mp4` | 30 FPS、1920×900 成果影片：原畫面＋熱圖＋候選框疊圖，下方顯示分數曲線；不保留原音訊 |
| `frame_scores.csv` | 30 FPS 影格索引／時間、原始影格索引／時間、MSE 分數與狀態 |
| `scores.npy` | 每個已輸出影格的分數；歷史不足的開頭影格為 NaN |
| `events.csv` | 達到連續命中條件的候選事件、開始／結束／確認時間與框座標 |
| `metrics.json` | 來源與輸出 FPS、處理幀數、速度及是否已處理完整影片；沒有人工標籤時 AUC 為 null |
| `config.json` | 使用的權重路徑與推論參數 |
| `progress.json` | 處理進度、剩餘時間估計及 Completed／Stopped／Failed 狀態 |

成果影片預設就會顯示所有資訊，不需額外參數。左側是未疊加的原始畫面，中間是像素預測誤差熱圖，右側是熱圖疊上原畫面、候選框與事件編號。上方顯示目前時間、狀態與當前幀 MSE，下方曲線顯示最近十秒的逐幀 MSE，開頭不足十秒時顯示目前所有時間；暖機期不會用零分冒充已評分。各影像面板保持原影片比例，影片原始檔不會被修改。

熱圖以固定 0～0.25 色階呈現，超過上限飽和為紅色；色階在同次推論中不會隨每幀重新正規化。`--heatmap-max 0.5` 可更改顯示上限，**只影響顏色、不影響原始誤差數值、scores 或候選判斷**。曲線 Y 軸隨最近十秒的分數範圍調整，軸旁顯示實際 MSE 數值；像素的 `--threshold` 不作為整幀 score 的分類閾值。

`--save-heatmaps` 只控制是否額外保存 `heatmaps/<影格索引>.npy` 與 `.png`，不控制成果影片是否顯示熱圖；PNG 使用相同顯示色階，NPY 保留原始誤差。長影片可能占用較多空間。此入口的事件框座標以**原始影片像素**為準，x2／y2 為 exclusive；`metrics.json` 分別記錄來源與成果影片尺寸。

預設歷史四幀，因此最前面四幀只顯示 warming up，從第五幀開始預測與評分。灰框／pending 是尚未達到連續命中條件的候選；黃框／candidate 是已達到條件的候選。**這些是高預測誤差的變化區域，不是已確認煙霧的分類結果**；score 是 MSE，並非煙霧機率。

`--threshold` 預設 0.1、`--min-area` 預設 25（模型像素面積）、`--min-event-frames` 預設 3。若訓練使用 `--registration`，推論也必須加上它；本次 `astt_artillery_shell_v1_001` 沒有使用配準。時間換算使用固定原始 FPS；VFR 影片請先轉為固定幀率。

程式分工：`inference.py` 是 CLI 入口；`src/video_inference.py` 負責串流、推論與影片輸出；模型與事件合併沿用既有 `detection/model.py` 和 `localization/candidates.py`。
