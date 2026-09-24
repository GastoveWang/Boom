# YOLO 與 Motion 加速

## 執行設備

YOLO 的 `--device auto` 會在可用時選擇 CUDA；也可明確指定 `--device cuda` 或 `--device cpu`。YOLO `.pt` 的 `--precision fp16` 僅在 CUDA 上啟用；ONNX 精度由匯出模型決定。

Motion 使用 OpenCV 光流與影像變化分析。此機的 OpenCV 沒有可用的 CUDA 裝置，因此 Motion 仍在 CPU 執行。相機取像、影片編碼及桌面顯示也主要使用 CPU。

## 已做的延遲改善

- 即時相機只保留最新取像影格；顯示、取像和偵測分別執行，避免較慢的推論卡住視窗事件。
- YOLO 路線不執行 Motion 光流。可用 `--processing-width` 與 `--yolo-imgsz` 調整畫面處理量。
- 離線輸出使用有界背景編碼佇列，保存每個處理影格；`--writer-queue 0` 可改成同步寫入。
- 離線與即時預覽可用 `--display-width` 限制寬度，輸出影片及偵測座標保持原解析度。
- Motion 快取歷史影格的 ORB 特徵，避免重複計算。

## 執行範例

```powershell
conda activate boom

python run_offline.py motion --video data/video/example.mp4 --no-map --display
python run_offline.py yolo --video data/video/example.mp4 --no-map --device auto --display

python run_realtime.py motion --display
python run_realtime.py yolo --yolo-model-path models/pretrained/yolo26m.pt --yolo-target-classes person --processing-width 960 --yolo-imgsz 512 --display
```

## 本機觀察

2026-09-24 在 RTX 5080 與 Spinnaker 相機測試即時 YOLO。使用 `yolo26m.pt`、`--processing-width 960` 和 `--yolo-imgsz 512`，暖機後每 30 個處理影格的平均速度約 30 FPS。這是當時設備與場景的觀察值；相機設定、場景和模型大小都會影響速度。降低輸入尺寸也可能漏掉遠處的小目標。

代表性荒野影片仍需分別量測誤報、漏報、偵測延遲、處理 FPS 與定位誤差。
