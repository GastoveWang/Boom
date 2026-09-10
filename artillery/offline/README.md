# Boom Offline

Boom 是一套用於飛行中無人機的砲擊事件偵測與定位系統，目標是在影像中即時標示砲擊點，並估算其地面座標。

目前離線主流程預設使用 `model/sam_sup_pidnet_s.pt`。PIDNet-S 負責分割煙霧，事件 tracker 則只回報剛出現的煙霧；影片一開始已存在、已追蹤超過 1 秒，或已回報過的煙霧都不會再次建立事件。

## 目錄

```text
artillery/offline/
├── model/                         # 模型權重（不放程式）
├── src/
│   ├── detectors/
│   │   ├── pidnet_smoke_detector.py       # PIDNet 分割與新煙霧追蹤
│   │   ├── optical_flow_smoke_detector.py # 傳統差分與光流偵測
│   │   └── optical_pidnet_fusion_detector.py # 光流候選與 PIDNet 確認
│   ├── pidnet_model/                       # 內建 PIDNet 網路架構
│   └── pipelines/
│       ├── pidnet_pipeline.py             # PIDNet 算法執行入口
│       ├── optical_flow_pipeline.py       # 光流算法執行入口
│       ├── optical_pidnet_pipeline.py     # 光流 → PIDNet 融合入口
│       └── shared_pipeline.py             # 共用 UI、座標、影片與輸出
└── requirements.txt
```

`src` 依角色整理：`pipelines/` 放執行入口，`detectors/` 放算法實作，`pidnet_model/` 放 PIDNet 網路架構。原本的 `pidnet_pipeline.py` 與 `optical_flow_pipeline.py` 保持獨立；`optical_pidnet_pipeline.py` 是額外的兩階段融合入口，只安排先後與配對，不把兩套 detector 寫在一起。三者共用 UI、座標、影片 I/O 與輸出邏輯。

PIDNet 網路架構已放在 `src/pidnet_model/`，權重放在 `model/`，因此 `artillery/` 不依賴外部的 `wildfire-real-time-segmentation/` 也能載入模型。PIDNet 架構來源的 MIT 授權保留在 `src/pidnet_model/LICENSE`。

## 新煙霧事件規則

1. 啟動後先學習 `0.5` 秒，這段時間看得到的煙霧標為「已存在」。
2. 同一次 PIDNet 推論以較低的 `0.45` 門檻追蹤初期淡煙；弱訊號只保存候選時間和位置，不會直接回報事件。
3. 煙霧達到正式 `0.80` 門檻並連續看到至少 2 次後，才確認為候選砲擊事件。
4. 確認時回查相符的淡煙 track，事件時間與落點使用最早弱訊號影格，而不是煙霧變濃後的確認影格。
5. 淡煙從首次出現至正式確認不得超過 `1.0` 秒；持續更久才變明顯的煙霧視為舊煙，不回報。
6. 已存在和已確認煙霧會寫入長期空間記憶，避免煙團持續擴散時重複報點。
7. 落點像素採最早煙霧框的下緣中心；輸出同時保存事件時間、確認時間與確認延遲。
8. 確認事件包含 confidence。它是模型機率、持續觀測與初期增長的組合，應視為候選事件信心，不是絕對真值。

早期候選與正式確認都來自 PIDNet 機率圖；這項功能沒有呼叫或合併 `optical_flow_pipeline.py`。

如果影片在砲擊發生後才開始，該煙霧會在 warm-up 被當成既有煙霧；這是「不回報既有煙霧」需求下的刻意取捨。

## 執行

目前電腦已有 `wildfire_seg` Conda 環境時：

```powershell
C:\Users\roger\anaconda3\envs\wildfire_seg\python.exe artillery\offline\src\pipelines\pidnet_pipeline.py `
  --video data\your_video.mp4 `
  --device cpu
```

有 CUDA 的環境可改成 `--device cuda`；`--device auto` 會自動選擇。

傳統光流算法：

```powershell
python artillery\offline\src\pipelines\optical_flow_pipeline.py `
  --video data\your_video.mp4
```

Optical Flow 初始候選、PIDNet 二次確認：

```powershell
python artillery\offline\src\pipelines\optical_pidnet_pipeline.py `
  --video data\your_video.mp4 `
  --device auto `
  --fusion-window-sec 1.0 `
  --fusion-match-distance 220
```

融合流程只在 PIDNet 於 1 秒內、且煙塵位置與 Optical Flow 候選相符時建立事件。事件時間、紅框與座標採用 Optical Flow 第一次候選的位置；PIDNet 的時間記為確認時間。黃色框表示正在等待 PIDNet 的 Optical Flow 候選。

常用調整：

```powershell
python artillery\offline\src\pipelines\pidnet_pipeline.py `
  --video data\your_video.mp4 `
  --early-seg-threshold 0.45 `
  --seg-threshold 0.80 `
  --new-smoke-window-sec 1.0 `
  --warmup-sec 0.5 `
  --box-hold-sec 1.5 `
  --panel-hold-sec 10
```

- `--seg-threshold` 越高，分割較保守，通常降低誤報但可能漏掉淡煙。
- `--early-seg-threshold` 只控制初期淡煙候選，必須低於正式門檻；降低可以更早保留落點，但太低可能增加弱訊號干擾。
- `--model-width`、`--model-height` 控制推論解析度；降低可換取較低延遲。
- `--inference-stride` 控制每隔幾格推論一次；即時落點建議維持 `1`。
- `--warmup-sec` 是建立既有煙霧的時間。
- `--new-smoke-window-sec` 是允許確認落點的最大時間，預設即需求中的 1 秒。
- `--box-hold-sec` 控制紅框存在時間。
- `--panel-hold-sec` 控制座標卡片存在時間。
- 光流算法請直接使用 `optical_flow_pipeline.py`，避免混用 PIDNet 參數。

輸出路徑固定在 `output/offline/`。程式會依目前執行的 pipeline，把方法名稱標示在影片結果資料夾後方，不會另外建立方法子資料夾。例如：

```text
output/offline/
├── DJI_0070_pidnet/
│   ├── output_DJI_0070.mp4
│   └── DJI_0070_impact_coordinates.txt
├── DJI_0070_optical_flow/
│   ├── output_DJI_0070.mp4
│   └── DJI_0070_impact_coordinates.txt
└── DJI_0070_optical_pidnet/
    ├── output_DJI_0070.mp4
    └── DJI_0070_impact_coordinates.txt
```

若同方法的結果資料夾已存在，程式會建立如 `DJI_0070_pidnet_02`、`DJI_0070_pidnet_03` 的新資料夾，不會覆蓋舊結果。若提供 DJI 參考照片，座標計算使用其 GPS/XMP；否則沿用程式中的預設姿態與座標。現階段仍是假設單一靜態姿態，正式飛行應由每一影格的即時 telemetry 取代。
