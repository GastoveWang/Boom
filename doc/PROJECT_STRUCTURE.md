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
├── train_st2.py                    # ST² 訓練完成後自動以 best.pt 推論
├── configs/                        # 分類設定（見 configs/README.md）
├── doc/                            # Boom 文件
├── benchmark/                      # 碩論比較用的其他專案
│   ├── ASTT/                       # ASTT 比較模型
│   └── HSTforU/                    # 官方快照 origin＋獨立 hstforu 套件
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
├── src/st2/                       # ST² 獨立模型、訓練、標註與推論
├── models/                          # 模型權重（不納入 Git）
│   ├── pretrained/detector/       # YOLO 預訓練模型
│   ├── pretrained/map_matching/   # EDM 地圖比對模型
│   └── checkpoints/               # YOLO ONNX 模型
├── data/                            # 影片、圖片與 YOLO 資料集
├── tools/                           # YOLO 訓練、資料切分與相機檢查
├── output/
│   ├── offline/                   # 離線結果
│   └── realtime/                  # 即時結果
└── tmp/                             # 臨時驗證，完成後立即清理
```

## 執行

使用 `.pt` 模型需安裝 `python -m pip install -e ".[ml]"`；ONNX 推論可另選 `.[onnx]` 或 `.[onnx-gpu]`。

```powershell
conda activate boom

python run_offline.py motion --video data/video/example.mp4 --no-map --display
python run_offline.py yolo --video data/video/example.mp4 --no-map --display
python run_offline.py yolo --video data/video/example.mp4 --no-map --yolo-model-path models/pretrained/detector/yolo26m.pt --yolo-target-classes person

python run_realtime.py motion --display
python run_realtime.py yolo --display
```

YOLO 模型可用 `--yolo-model-path` 指定。即時辨識一般物件時，可加上 `--yolo-target-classes person`。詳見 [即時流程說明](ONLINE.md) 與 [工具說明](TOOLS.md)。

## ST² 與 Boom 分離

`src/boom/` 保留原有motion、YOLO、runner、畫面顯示、事件與定位。`src/st2/`獨立負責ST²的訓練及模型推論，不依賴boom。Boom名稱保留；tools中已不再放ST²入口，所有ST²命令由本套件提供。

```text
src/st2/
  __main__.py           # 統一命令入口：python src/st2 <command>
  cli/                  # train、infer、prepare、export_tensorboard、visualize
  models/
    network.py          # ST2Config、ST2Model、ST2Criterion
    transformer.py      # 純 Transformer 三分支、patch與attention
    motion.py           # 幾何配準、背景loss與相機增強
  data/
    dataset.py          # 8幀歷史＋當前影格與9幀ROI mask
    sequence.py         # 來源分組、連續窗口與fingerprint
    annotations.py      # 逐幀標註與資料集匯出
    editor.py           # 影片標註UI
  engine/
    train.py            # 訓練、驗證與續訓
    infer.py            # 影片熱圖、分數與輸出
    runtime.py          # 純模型逐幀介面，供未來Boom adapter
    postprocess.py      # 分數換算與可選運動guard
    view.py             # 研究推論畫面
    tensorboard.py      # 既有metrics匯入TensorBoard
    visualize.py        # torchview實際forward圖，SVG／DOT匯出
  common/
    utils.py            # 前處理、checkpoint、JSON與抽樣
```

目前資料為 `data/st2/st2_v1/`；新訓練輸出 `output/st2/<日期時間>/training/`，自動推論的各影片目錄直接位於同一日期時間資料夾的 `inference/`。獨立推論預設 `output/st2/inference/`，仍建立獨立 run 目錄。標註新匯出為 `data/st2/st2_vN/`。

`data/astt_bbox`與`output/astt_bbox`已依使用者要求更名，檔案內容保留；原smoke_heatmap/roi_heatmap歷史run不移動、不改寫。舊Smoke Heatmap工具、匯入轉接及舊研究網路均已移除，統一使用ST²入口。移動清單見 [ST2_MIGRATION.md](history/ST2_MIGRATION.md)。

```powershell
conda activate boom
python src/st2 prepare
python src/st2 train --epochs 30 --device cuda --infer-source "data/video/1440-1080/0603"
```

目前只提供ST²（transformer），每個checkpoint保存實際架構與模型名稱；baseline/camera-aware執行程式已移除，既有輸出保留原地。v2.2已完成30輪，新ST²僅完成流程驗證，尚未正式重訓。

模型替換以Boom的BaseSmokeDetector作為執行端邊界，ST2Runtime提供純熱圖／分數。此次尚未建立把熱圖變成已確認煙霧事件的adapter，因此不新增即時偵測器選項、不改既有motion/YOLO行為。架構與指令見 [ST2.md](ST2.md)，資料見 [ST2_DATASET.md](ST2_DATASET.md)。
