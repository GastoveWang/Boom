# Boom 資料準備與 YOLO 訓練工具鏈 (tools/)

本目錄包含 Boom 專案的資料工程、資料集切分、自動標註與 YOLO 訓練工具。

---

## 1. 影片擷取圖片 (`tools/video_to_frames.py`)

在已安裝 OpenCV 的環境中，於專案根目錄執行：

```powershell
# 預設將每一幀保存為 PNG，保持原始解析度
python tools/video_to_frames.py "data/raw_video/example.mp4"

# 每 0.5 秒擷取一張，指定新的輸出資料夾
python tools/video_to_frames.py "data/raw_video/example.mp4" --interval 0.5 --output "data/frames/example"

# 如需 JPG，可指定格式
python tools/video_to_frames.py "data/raw_video/example.mp4" --format jpg
```

* 預設保存到 `data/extracted_frames/<影片名稱>/`，目錄已存在時自動加 `_2`、`_3` 等流水號。

---

## 2. 資料集版本管理與 7:3 類別平衡切分 (`tools/split_dataset.py`)

使用含 NumPy、SciPy >= 1.9、PyYAML 與 Tkinter 的 Python 環境啟動桌面 GUI：

```powershell
python tools/split_dataset.py
```

### 操作步驟
1. 按「加入資料夾」加入本次新增的資料（可重複加入多個來源）。
2. 選擇共用 `classes.txt`（每行一個類別名稱）。
3. 選擇輸出位置與資料集名稱（預設 `dataset`）。
4. 按「預檢」查看 train/val 分配與預計輸出路徑。
5. 按「開始切分」建立新版本（例如第一次 `dataset_v1/`，下次自動讀取並合併產出 `dataset_v2/`）。

---

## 3. YOLO 訓練與模型匯出 (`tools/train_yolo.py`)

在具備 Ultralytics (YOLO) 與 PyTorch 的環境中執行：

```powershell
python -m pip install ultralytics onnx
python tools/train_yolo.py
```

* 程式會自動搜尋 `models/` 與 `label/model/` 根目錄與所有子資料夾中的預訓練權重。
* 訓練完成後自動將 `best.pt` 轉換匯出為 ONNX 格式，保存至 `models/checkpoints/`。

---

> **向後相容說明**：
> 歷史路徑（如 `python label/split_dataset.py` 與 `python label/train_yolo_labeling.py`）仍保留相容轉接，舊指令均可正常運作。
