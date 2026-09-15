# Artillery 程式分工

Boom 是一套用於飛行中無人機的荒野煙霧偵測與定位系統，目標是在影像中即時標示煙霧區域，並估算疑似煙霧源的地面座標。

## 模組責任

| 位置 | 責任 |
| --- | --- |
| `common/defaults.py` | 現行預設值、路徑與偵測參數 |
| `common/detector_factory.py` | 建立偵測器與其設定，讓 runner 不必知道各模型建構細節 |
| `common/coordinates.py` | 既有靜態相機投影、TWD97／WGS84 運算 |
| `common/events.py` | 共用座標／事件資料結構 |
| `offline/src/detectors/` | 煙霧分割、光流偵測與融合事件判定 |
| `offline/src/pidnet_model/` | PIDNet 網路結構 |
| `offline/src/localization/` | 照片初始化、地圖、VO、定位品質與地圖 UI |
| `offline/src/pipelines/` | 參數解析、驗證及各方法入口 |
| `offline/src/runtime/` | 影片來源、流程串接、事件資料整理、儲存與進度 |
| `offline/src/ui/` | 偵測疊圖、原座標面板與最終畫面組合 |
| `online/src/boom_online/` | 即時設定、pipeline、定位銜接、事件儲存、日誌與疊圖 |
| `online/src/spin_camera.py`、`spinnaker_camera.py` | Spinnaker 取像、相機資源生命週期與最新影格緩衝 |

偵測器和模型保留原匯入位置及演算法。即時流程直接使用共用設定／座標與偵測器，
不再匯入離線 pipeline。照片／VO 模組維持 `offline/src/localization/`，
目前只接入離線 runner；這次重整沒有新增即時 VO 或真實飛控遙測。

## 離線 pipeline

```text
pipelines/optical_flow_pipeline.py / pidnet_pipeline.py / optical_pidnet_pipeline.py
  → main.py：共用啟動函式
  → cli.py：解析與驗證參數
  → runtime/runner.py：串接單一處理流程
      inputs.py             解析輸入、讀取影格
      localization/         先更新當格定位
      common/detector_factory.py → detectors/  偵測煙霧
      runtime/events.py     依事件影格定位、整理 UI 顯示期限
      ui/composition.py     合成偵測圖、地圖／座標面板
      runtime/output.py     保存影片與事件座標
      runtime/progress.py   顯示執行進度
```

`runtime/events.py` 的顯示期限只影響 UI，不干預 detector 的事件追蹤／確認。
`runtime/reference.py` 保留舊靜態照片模式；VO 的照片資訊由 `localization/photo.py` 處理，
兩者高度與姿態政策不同，沒有為了重整而混用。

要用哪種辨識方法，就執行對應入口；每個入口固定自己的偵測器：

```powershell
python artillery/offline/src/pipelines/optical_flow_pipeline.py --video data/0603/DJI_001_V.MP4
python artillery/offline/src/pipelines/pidnet_pipeline.py --video data/0603/DJI_001_V.MP4 --device auto
python artillery/offline/src/pipelines/optical_pidnet_pipeline.py --video data/0603/DJI_001_V.MP4 --device auto
```

GPS 照片、地圖與 VO 暫估模式：

```powershell
python artillery/offline/src/pipelines/optical_flow_pipeline.py `
  --reference-image data/0603/DJI_001_P.JPG `
  --video data/0603/DJI_001_V.MP4 `
  --map-dir asset/maps --allow-gps-seed --display
```

`--allow-gps-seed` 明確允許未通過地圖視覺匹配的暫估定位，並不代表精確地理位置。
完整限制見 [offline/MAP_LOCALIZATION.md](offline/MAP_LOCALIZATION.md)。

## 即時 pipeline

`online/src/realtime_runner.py` 保留為入口與舊公開函式的轉接層。
`boom_online/cli.py` 解析 CLI，`pipeline.py` 串接相機、偵測與結果輸出；
`configuration.py`、`positioning.py`、`storage.py`、`logging_setup.py`、`display.py`
各自處理設定、靜態定位、事件證據、日誌與 UI。

```powershell
python -m artillery.online --help
python -m artillery.online --display
```

取像需要 Spinnaker/PySpin 與實體相機；`--help` 和模組載入不需要 SDK。
原本的 `boom-online` 與 `PYTHONPATH=artillery/online/src python -m realtime_runner` 仍可使用。
安裝仍採 repository 內的 editable install；共用程式與 detector 由同一份 repository 提供。

## 相容性與驗證

- 原三個 `*_pipeline.py` 都轉到相同 `pipelines/main.py`，維持各自預設演算法。
- 各方法有自己的 `main(argv=None)`，命令列與程式呼叫都固定使用該方法。
- `pipelines/map_localization.py` 保留既有定位匯入轉接。
- 原模型、資料與輸出目錄不搬動；新執行結果仍建立獨立目錄。
- 預設值改在 `common/defaults.py` 維護，執行時優先使用 CLI。

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

測試涵蓋定位幾何、事件影格／像素、輸出目錄不覆寫、流程呼叫順序、
來源資源釋放與模擬相機事件儲存。真實 Jetson／相機效能需在設備上另行驗證。
