# Boom 專案結構

Boom 是一套用於飛行中無人機的荒野煙霧偵測與定位系統，目標是在影像中即時標示煙霧區域，並估算疑似煙霧源的地面座標。

## 偵測流程

| 流程 | 方法 | 離線影片 | 即時相機 |
|---|---|---:|---:|
| `motion` | 影像變化與光流分析 | ✓ | ✓ |
| `yolo` | YOLO 物件偵測 | ✓ | ✓ |

兩種偵測器共用事件、顯示、輸出與定位程式；YOLO 流程不執行光流。`--device auto` 會讓可用的 YOLO GPU 後端自動使用 CUDA。Motion 的 OpenCV 運算仍使用 CPU。

## 主要目錄

```text
Boom/
├── run_offline.py                  # 離線影片入口
├── run_realtime.py                 # Spinnaker 相機入口
├── configs/default.yaml           # 系統設定
├── doc/                            # Boom 文件
├── benchmark/                      # 碩論比較用的其他專案
├── src/boom/
│   ├── detection/
│   │   ├── optical_flow/           # Motion 偵測器
│   │   ├── yolo/                   # YOLO 偵測器
│   │   └── factory.py              # 偵測器建立
│   ├── pipelines/                  # 離線 CLI 與流程入口
│   ├── online/                     # 即時相機、顯示與事件儲存
│   ├── runtime/                    # 離線讀取、輸出與事件處理
│   ├── localization/               # 地圖比對、視覺里程與座標估計
│   ├── core/                       # 事件及座標資料結構
│   ├── config/                     # 預設值與 YAML 載入
│   ├── interfaces/                 # 共用介面
│   └── ui/                         # 標註與畫面合成
├── models/                          # 模型權重與轉出的 ONNX；舊檔案保留
├── data/                            # 影片、圖片與 YOLO 資料集
├── tools/                           # YOLO 訓練、資料切分與相機檢查
├── output/
│   ├── offline/                   # 離線結果
│   └── realtime/                  # 即時結果
└── tests/                           # 驗證程式
```

## 執行

使用 `.pt` 模型需安裝 `python -m pip install -e ".[ml]"`；ONNX 推論可另選 `.[onnx]` 或 `.[onnx-gpu]`。

```powershell
conda activate boom

python run_offline.py motion --video data/video/example.mp4 --no-map --display
python run_offline.py yolo --video data/video/example.mp4 --no-map --display
python run_offline.py yolo --video data/video/example.mp4 --no-map --yolo-model-path models/pretrained/yolo26m.pt --yolo-target-classes person

python run_realtime.py motion --display
python run_realtime.py yolo --display
```

YOLO 模型可用 `--yolo-model-path` 指定。即時辨識一般物件時，可加上 `--yolo-target-classes person`。詳見 [即時流程說明](ONLINE.md) 與 [工具說明](TOOLS.md)。
