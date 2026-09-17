# 地圖與 GPS 照片初始化／VO

`src/localization/` 提供 runner 可共用的地圖與定位模組。煙霧 detector、
參數、事件確認與追蹤規則均未修改。目前接入 offline 共用 runner，尚未接入 online runner。

模組分工與匯入方式見 [src/localization/README.md](src/localization/README.md)。

## 執行

0603 的 GPS 照片與影片，使用 optical-flow 煙霧偵測：

```powershell
C:\Users\roger\anaconda3\envs\wildfire_seg\python.exe artillery/offline/src/pipelines/optical_flow_pipeline.py `
  --reference-image data/0603/DJI_001_P.JPG `
  --video data/0603/DJI_001_V.MP4 `
  --allow-gps-seed `
  --display
```

`--allow-gps-seed` 是**明確允許暫估模式**：照片對地圖匹配失敗時，仍可用照片 GPS、
相對高度、雲台姿態與照片對影片匹配啟動 VO。畫面標記 `VO_GPS_ESTIMATE`，
紀錄 `gps_metadata_only`，並非已驗證的地圖視覺定位。

移除 `--allow-gps-seed` 可使用嚴格模式：照片對地圖匹配未通過，就不輸出煙霧地理座標，
但影片與煙霧偵測仍執行。`--display` 可省略，只輸出影片與紀錄。

可用 `--start-frame 720 --max-frames 360` 驗證約 12–18 秒的煙霧片段。
從非零影格啟動會直接匹配參考照片與該影格，並重新建立 detector 狀態；
其事件編號與前置歷史不會與從零開始處理整片完全相同。

地圖預設使用專案根目錄下的 `asset/maps`，不受執行時的工作目錄影響；無人機圖示預設為 `asset/pictures/drone.png`。
可用 `--map-dir` 指定其他地圖目錄。若不使用地圖與 VO，加入 `--no-map`，此時不強制提供初始化照片。

## 流程與資料契約

指定 `--map-dir asset/maps` 時，程式啟動時以 rasterio 建立 TIFF metadata 與地理網格索引。
每個 maps 第一層資料夾視為一個 submap。地圖顯示依初始化 GPS 選最近的 submap，
只完整載入該資料夾內全部 TIFF，按 `--map-mpp` 合成並快取，不受定位參數限制。
只有定位時，才依照片 GPS、公尺搜尋半徑與 TIFF 地理 bounds 選出附近候選，預設最多
9 張，匹配各候選的局部 ROI；没有附近地圖就回報定位失敗，完整地圖仍保留。
submap 以資料夾分組；遠近使用 TIFF 地理 bounds 判定，不使用資料夾名稱推測。
完整 API、CLI 參數、品質門檻、debug 與限制見
[localization/README.md](src/localization/README.md#粗定位--附近地圖匹配)。

1. 讀 GPS 照片 EXIF 與 DJI XMP，建立 GPS 起點、近似內參與地面平面。
2. 載入最近 submap 內所有 GeoTIFF 的完整地理範圍，依各圖 CRS／affine 合成完整 UI 地圖，
   保留 alpha/nodata 有效區域；定位另外依粗 GPS 選附近 ROI，使用 pyproj 做座標轉換。
3. 照片與選中 submap 內各 ROI 做 SuperPoint + LightGlue／RANSAC 匹配，
   依 inlier 數／比例、重投影誤差與空間分布驗證，再以 PnP／平面 IPPE 驗證相機姿態，
   得到正北 0°、順時針增加的無人機朝向（相機與機頭一致）。直接匹配失敗時保留依照片姿態校正斜拍的流程，
   從完整地圖裁出附近候選 ROI，使用相同幾何品質門檻，不匹配全部地圖。
4. 照片與影片首格視覺匹配；在 0.9–1.2 範圍搜尋影片相較照片的焦距比例，
   用有效匹配支持度選擇並固定，處理額外影片裁切。
5. 獨立的 KLT 光流追蹤地面特徵，經前後向一致性檢查及 PnP/RANSAC 估算相機位置。
   定期補點並每 5 秒嘗試地圖校正。這不是煙霧 detector 裡的 optical flow。
6. 按影格保存投影與位置。事件使用自己的 `frame_idx`，而不是確認影格；
   optical-flow 煙霧 detector 的縮圖像素在 runner 換算回原始相機像素後才投影。
7. 越過地平線、地圖外、超過 1,800 公尺地面範圍或缺少定位的事件，不新增座標。

地圖顯示 `asset/pictures/drone.png`、估計軌跡與紅色疑似煙霧源標點。
目前假設**雲台與機頭方向一致**，方向扇形統一代表無人機朝向；相機位置近似機體位置，尚未校正桿臂偏移。
追蹤失效時保留最後位置並標示 LOST，停止新增地理座標，嘗試與參考照片重新匹配。

## 假設與限制

- 這是**地面平面約束 VO**，不是無約束的完整 3D SLAM。沒有 DEM 或實測相機校正。
- 照片 `RelativeAltitude` 是相對起飛點高度，預設把它近似為對當地地面高度；
  不能使用海拔直接替代。若有實測對地高度，使用 `--ground-height <公尺>`。
- 照片焦距由 35 mm 等效焦距估算；未修正鏡頭畸變。影片假設中心裁切，焦距在初始化後固定。
- 地圖使用 GPS 起點附近的局部 east/north 近似，適用本次數公里範圍，不適用大範圍地圖。
- 單一 GPS 起點本身沒有 VO 公尺尺度；尺度依賴上述高度、平面與相機內參假設。
- 嚴格視覺定位模式在 15 秒沒有有效地圖／參考校正後停止可信追蹤。
  暫估模式仍允許 VO 推進，但會漂移，始終標示未經地圖驗證。
- 重投影誤差是影像幾何品質，**不等於地理定位公尺誤差**。
- 既有 optical-flow 候選事件與其信心值保持原樣，不能把它們直接視為已確認煙霧真值。

## 輸出

地圖模式輸出固定為 1920×1080：左側主地圖寬 1180，右上影片寬 740（維持比例），右下保留事件座標圖卡。
影片事件框、地圖標點與圖卡統一使用 `S<事件編號>` 與固定配色；事件文字於影片縮放後繪製。
圖卡優先顯示最新事件，包含 WGS84、TWD97 與時間；無法定位時座標顯示「—」，不建立地圖標點。
畫面隱藏信心與定位狀態，原始紀錄仍保留。左上台灣小地圖框出主視窗範圍，右上指北針固定北向上。
無人機旁的扇形與角度統一表示無人機朝向（假設雲台與機頭同向）；扇形為方向示意，並非精確視野。
地圖保留歷史標點，影片框與圖卡仍依 `--box-hold-sec`、`--panel-hold-sec` 到期隱藏。

每次執行仍建立不覆寫的 `output/offline/<video>_optical_flow[_NN]/`：

- `output_<video>.mp4`：偵測畫面與地圖。
- `ui_preview.jpg`：第一格 UI。
- `map_preview.jpg`：具地理參考的地圖預覽。
- `reference_ground_projection.jpg`：啟用 `--map-debug` 且執行斜拍 fallback 時的診斷圖。
- `map_match_*/`：僅 `--map-debug` 啟用時產生 UAV／ROI／matches／inliers／footprint 與品質紀錄。
- `localization.jsonl`：初始化結果、影片每格狀態／座標／匹配品質與煙霧事件投影。
- `region_selection.json`：各區域經緯度範圍、照片 GPS、估計拍攝中心與選區理由。
- `*_impact_coordinates.txt`：事件座標與 `coordinate_status`；失效時 `coordinate=unavailable`。

地圖快取放在 `output/offline/_map_cache/`，以來源路徑、大小、修改時間、GPS 起點與解析度
建立選中 submap 的完整地圖快取鍵。`--map-mpp` 控制地圖預覽解析度，預設 0.75 公尺／像素；
`--map-max-image-size` 只控制定位 API 的 frame／ROI 解析度，不影響完整地圖載入。

## 驗證

```powershell
python -m unittest discover -s tests -p test_map_localization.py -v
python tests/validate_localization_video.py --video data/0603/DJI_001_V.MP4 --reference-image data/0603/DJI_001_P.JPG --allow-gps-seed
```

第二個指令逐格驗證全片 VO，不執行／修改煙霧 detector，另外產生唯一目錄、
`summary.json` 與 `trajectory.jpg`。GPS 暫估的成功追蹤率不能代表地圖匹配成功率。

先前 SIFT 版本的 0603 資料照片對地圖匹配未通過品質門檻。當時照片對影片匹配可用，
因此 GPS 暫估模式可驗證 VO 與標點流程；**地圖視覺校正與真實定位誤差仍未驗證**。

目前預設 matcher 已改為 SuperPoint + LightGlue，不能沿用上述舊版結果評估新 matcher。
實際 CUDA 模型與合成朝向測試已通過；真實飛行地面真值驗證仍待進行。
