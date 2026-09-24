# Boom Online

Boom 是一套用於飛行中無人機的荒野煙霧偵測與定位系統，目標是在影像中即時標示煙霧區域，並估算疑似煙霧源的地面座標。

## 環境與相機檢查

這台 Windows 電腦的 `boom` Conda 環境已安裝原廠 `spinnaker-python 4.2.0.88`，
使用本機 Spinnaker SDK 附的 Python 3.10／Windows x64 wheel。
已確認 SDK 可載入，且與 RTX 5080 的 PyTorch CUDA 推論共存。
2026-09-24 已用實體 Spinnaker 相機驗證 motion 與 YOLO 即時流程。

在專案根目錄執行：

```powershell
conda activate boom
# 僅列出 SDK 版本、相機型號與序號；未找到相機時回傳代碼 2
python tools/check_camera.py
# 接上 USB3 / GigE 相機，關閉 SpinView 後測試 30 格取像
python tools/check_camera.py --frames 30
# 多台相機時指定序號
python tools/check_camera.py --serial YOUR_CAMERA_SERIAL --frames 30
```

Windows 的 PySpin 與 PyTorch 可能受 DLL 載入順序影響；Boom 的 SDK 載入函式
會先載入已安裝的 PyTorch，再載入 PySpin。不需改名 DLL 或修改系統 PATH。
自行寫測試程式時也請先 `import torch`，再 `import PySpin`。
Jetson 需另安裝與 JetPack、Python、ARM 架構相容的 SDK、PySpin、PyTorch 與 OpenCV；
此 Windows 安裝不能直接移植到 Jetson。

## 執行

```powershell
# Motion：只執行光流／影像變化偵測（預設模式）
python run_realtime.py motion --display
# YOLO：只執行 YOLO 推論
python run_realtime.py yolo --display

# YOLO26m 一般物件模型：辨識人
python run_realtime.py yolo --yolo-model-path models/pretrained/yolo26m.pt --yolo-target-classes person --processing-width 960 --yolo-imgsz 512 --display
# 指定相機與曝光，測試 300 個處理影格
python run_realtime.py motion --serial YOUR_CAMERA_SERIAL --exposure-us 3000 --display --max-frames 300
```

`--device auto` 是預設值：YOLO 在 CUDA 後端可用時使用 GPU，否則使用 CPU。
Motion 的 OpenCV 光流、相機取像與桌面顯示仍在 CPU 執行。
`--precision fp16` 為可選 CUDA 半精度，預設保留 FP32；不保證每種硬體上都更快。
`--detector motion` 或 `--detector yolo` 仍可作為舊版參數使用。
YOLO 路線固定關閉光流補償；模型內建的框追蹤與事件編號仍會運作。
`--display-width 1280` 限制預覽寬度；`0` 使用完整尺寸。
`--processing-width 960` 保留完整視野並縮小處理影格；`0` 保留相機原始解析度。
`--yolo-imgsz 512` 可降低推論延遲；遠處小人物可能需要改回 `640`。
一般物件模型需用 `--yolo-target-classes` 指定模型內的類別名稱，例如 `person`。
此機使用 `yolo26m.pt` 啟動較快；`yolo26m.onnx` 亦可使用，但首次推論需要較長的後端初始化。

## 顯示執行緒

`--display` 自動啟用以下分工，offline 也使用同一套顯示機制：

1. 主執行緒：建立視窗、顯示影像、處理按鍵及關閉視窗。
2. 處理執行緒：模型載入、煙霧推論、標註、定位與事件保存。
3. 相機取像執行緒：連續取像，只保留最新影格。

顯示信箱只保留最新完成的標註畫面，不累積過時預覽。按 `Q`、`Esc`、`Ctrl+C`
或關閉視窗會請求停止，等待正在處理的影格與必要寫檔結束，再釋放資源。
推論變慢時視窗仍可回應，但標註畫面的更新率仍受推論速度限制；
不把舊偵測框貼到較新的原始畫面上假裝已完成偵測。
不加 `--display` 則保持無視窗同步處理，仍有獨立的相機取像執行緒。

## 輸出與效能

預設輸出位於 `output/realtime/YYYYMMDD_HHMMSS/`，可用 `--output-dir` 指定。
包含 `runtime.log`、`events.jsonl`、`detections.csv`、`impact_coordinates.txt`
及 `snapshots/` 事件完整畫面／裁切影像；不保存整段影片。

日誌包含 `CAMERA CHECK open=OK`、`first_frame=OK`、處理 FPS、延遲與跳格數。
`--camera-fps` 是要求的取像速度，不是保證的偵測速度。
延遲由取得並轉換影格後開始計算，不包含完整曝光／傳輸時間。
事件截圖仍在處理執行緒同步保存；寫檔慢時推論會等候，但不阻塞視窗事件處理。
完整加速選項與測試結果見 [加速說明](ACCELERATION.md)。

## 定位

可用 `--drone-lon`、`--drone-lat`、`--drone-alt`、`--drone-yaw`、`--drone-pitch`、
`--drone-roll` 提供暫時固定姿態。未提供時仍保存事件，座標記為 unavailable。
正式飛行仍需串接時間同步的飛控遙測及相機內參；目前不保證飛行定位精度。
