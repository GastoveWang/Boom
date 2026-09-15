# Boom Online

程式分工見 [artillery/README.md](../README.md)。即時 pipeline 直接使用
`artillery/common/` 的設定與座標運算，與離線入口分開。

Boom Online 是 Jetson 上的即時荒野煙霧偵測程式。它使用
FLIR／Teledyne SpinView 相機的 Spinnaker SDK（PySpin）取得畫面，
偵測演算法則沿用專案的 `artillery/offline/` detector。

## 專案結構

```text
artillery/online/
├── pyproject.toml
├── README.md
├── requirements-jetson.txt
├── src/
│   ├── realtime_runner.py
│   ├── boom_online/
│   │   ├── cli.py / configuration.py
│   │   ├── pipeline.py
│   │   ├── positioning.py / storage.py
│   │   └── logging_setup.py / display.py
│   ├── spin_camera.py
│   └── spinnaker_camera.py
└── output/
    └── YYYYMMDD_HHMMSS/
```

程式碼全部位於 `src/`；`output/` 與 `src/` 位於同一層。

## 安裝

先安裝與 JetPack、Ubuntu、Python、aarch64 相符的原廠 Spinnaker SDK
與 PySpin。使用 SpinView 確認相機可以取像後，執行本程式前需關閉 SpinView。

在 repository 根目錄執行：

```bash
python3 -m pip install -r artillery/online/requirements-jetson.txt
python3 -m pip install -e artillery/online
```

Jetson 優先使用 JetPack 內建的 OpenCV，不要用一般 pip wheel 覆蓋。
PySpin 也必須使用原廠 SDK 提供的版本。

## 執行

安裝後可在任意位置使用：

```bash
boom-online
```

也可以在 repository 根目錄使用 `python -m artillery.online`。

或者在 repository 根目錄執行：

```bash
PYTHONPATH=artillery/online/src python3 -m realtime_runner
```

指定相機：

```bash
boom-online \
  --serial YOUR_CAMERA_SERIAL \
  --width 1920 \
  --height 1080 \
  --camera-fps 30 \
  --exposure-us 3000 \
  --gain-db 6
```

需要顯示即時畫面時才加入：

```bash
boom-online --display
```

沒有 `--display` 時不建立視窗，但偵測、log 與事件截圖會持續運作。
程式不保存整段影片。

## 輸出

每次啟動會依時間建立 session：

```text
artillery/online/output/YYYYMMDD_HHMMSS/
├── runtime.log
├── events.jsonl
├── detections.csv
├── impact_coordinates.txt
└── snapshots/
    ├── event_..._full.jpg
    └── event_..._crop.jpg
```

`runtime.log` 包含：

- `CAMERA CHECK starting`
- `CAMERA CHECK open=OK`
- `CAMERA CHECK first_frame=OK`
- 定期 `camera_status=OK` heartbeat
- 相機逾時的 `camera_status=TIMEOUT`
- 相機或 SDK 錯誤的 `camera_status=FAILED`
- 處理 FPS、延遲、跳過的 frame 數與事件資訊

## 暫時定位輸入

```bash
boom-online \
  --drone-lon 121.5406 \
  --drone-lat 25.0134 \
  --drone-alt 100 \
  --drone-yaw 0 \
  --drone-pitch -45 \
  --drone-roll 0
```

沒有提供 GPS／姿態時，事件仍會保存，
`impact_coordinates.txt` 會記錄 `coordinate=unavailable`。
正式飛行仍應串接時間同步的飛控遙測與相機內參。
