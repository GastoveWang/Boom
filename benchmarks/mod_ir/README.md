# MOD-IR Smoke Detector

這個資料夾依據論文 **MOD-IR: moving objects detection from UAV-captured video
sequences based on image registration** 實作無人機影片煙霧偵測。

## 演算法流程

1. 使用 ORB 擷取相鄰幀特徵，並以 Hamming KNN 配對。
2. 使用 RANSAC 排除前景與錯誤配對，再估計 Homography。
3. 將前一幀配準到目前幀，消除無人機相機的主要 ego-motion。
4. 計算灰階絕對幀差，使用 Kapur maximum entropy 自動選擇閾值。
5. 以形態學操作清除雜訊。
6. 使用 Quick-shift 色彩區域分割修整前景範圍。
7. 以煙霧的低飽和度、柔和邊緣、面積及時間持續性過濾候選區域。
8. 將事件中心投影為 TWD97/WGS84 落點，並輸出右側事件資訊 UI。

與論文不同的是，第 7 步是本專案為「煙霧」辨識加入的規則。MOD-IR 原文是
一般移動物體分割，不會直接判斷物體是否為煙霧。

## 安裝

在專案根目錄執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r mod_ir_smoke/requirements.txt
```

若沒有安裝 `scikit-image`，程式仍可執行，但區域後處理會自動改用 OpenCV
mean-shift，而不是論文使用的 Quick-shift。

## 執行

自動處理 `data/` 中所有影片：

```bash
python3 mod_ir_smoke/detect_smoke.py
```

指定影片：

```bash
python3 mod_ir_smoke/detect_smoke.py \
  --video data/DJI_0331.mp4 \
  --output-dir output/mod_ir_smoke
```

先測試前 120 幀：

```bash
python3 mod_ir_smoke/detect_smoke.py --max-frames 120
```

顯示即時視窗：

```bash
python3 mod_ir_smoke/detect_smoke.py --display
```

## 輸出

每支影片會產生：

```text
output/mod_ir_smoke/<影片名稱>/
├── annotated.mp4       # 偵測框、遮罩與狀態
├── detections.csv      # 每個已確認煙霧事件
└── snapshots/          # 事件確認畫面
```

重要參數可透過 `python3 mod_ir_smoke/detect_smoke.py --help` 查看。4K 影片預設
縮小為 30% 處理並輸出縮小後的標註影片，以控制 ORB 與 Quick-shift 的運算量。

## 落點座標與 UI

若 `data/` 有對應的 DJI 定位照片，可執行：

```bash
python3 mod_ir_smoke/detect_smoke.py \
  --video data/DJI_0331.mp4 \
  --reference-image data/DJI_0331_P.JPG
```

也可以手動提供完整位置與姿態：

```bash
python3 mod_ir_smoke/detect_smoke.py \
  --video data/DJI_0331.mp4 \
  --drone-lon 121.5406 --drone-lat 25.0134 --drone-alt 100 \
  --drone-yaw 0 --drone-pitch -45 --drone-roll 0
```

六個手動參數必須全部提供。沒有定位資料時，偵測與右側 UI 仍會運作，但座標欄位
會顯示 unavailable，CSV 座標欄位則保持空白。
