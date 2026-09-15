# 影片擷取圖片

在已安裝 OpenCV（`python -m pip install opencv-python`）的環境中，於專案根目錄執行：

```powershell
# 預設將每一幀保存為 PNG，保持原始解析度
python label/video_to_images.py "data/raw_video/example.mp4"

# 每 0.5 秒擷取一張，指定新的輸出資料夾
python label/video_to_images.py "data/raw_video/example.mp4" --interval 0.5 --output "data/frames/example"

# 如需 JPG，可指定格式
python label/video_to_images.py "data/raw_video/example.mp4" --format jpg
```

預設保存到 `data/extracted_frames/<影片名稱>/`，目錄已存在時自動加 `_2`、`_3` 等流水號。
使用 `--output` 時，指定目錄必須尚不存在。檔名包含影片名稱與從 0 起算的原始影格編號。
指定大於 0 的間隔時，依影片回報的 FPS 換算，適合固定幀率影片；可變幀率影片的時間間隔為近似值。
讀取失敗時停止擷取（OpenCV 無法可靠區分影片結束與中途解碼失敗）。

# YOLO 訓練與模型匯出

在具備 Ultralytics 與訓練所需 CUDA / PyTorch 的環境中，於專案根目錄執行：

```powershell
python -m pip install onnx
python label/train_yolo_labeling.py
```

`MODEL_NAME` 填入不含副檔名的權重名稱，例如 `detect_v1_yolo26m`。
程式會搜尋 `label/model/` 根目錄與所有子資料夾中的同名 `.pt`，包含 `pretrained/`、`trained/`。
若有多份同名模型，會列出路徑並停止，請改用唯一檔名。找不到時沿用原本的載入／下載流程，保存至 `pretrained/`。
訓練使用 `.pt` 權重，搜尋不會選取 `.onnx`。

訓練完成後，程式使用 [Ultralytics 匯出功能](https://docs.ultralytics.com/modes/export/)
將 `best.pt` 轉為 ONNX，以 `<DATASET_NAME>_<MODEL_NAME>.onnx` 保存至 `label/model/trained/`。
同名 ONNX 已存在時，改用 `_2`、`_3` 等流水號，避免覆蓋已保存模型。
轉換使用的暫存 `.pt` 會自動清除，原始訓練權重仍保留在 `label/runs/`。
ONNX 使用與訓練相同的 `IMAGE_SIZE`（預設 640），固定輸入尺寸、batch 1，
在 CPU 匯出且不簡化圖結構。匯出失敗會回報錯誤，原始訓練權重仍保留。
首次匯出時，Ultralytics 可能需要安裝額外的匯出相依套件。

# 資料集版本管理與 7:3 切分

UI 與切分功能整合在 **`label/split_dataset.py` 一支程式**，不再需要 `split_dataset_ui.py`。
使用有 NumPy、SciPy >= 1.9、PyYAML 與 Tkinter 的 Python 環境啟動：

```powershell
python label/split_dataset.py
```

## 操作

1. 按「加入資料夾」加入本次新增的資料，可重複加入多個來源。
2. 選擇共用 `classes.txt`。每行一個類別名稱，依序對應 class 0、1、2；不讀來源 `data.yaml`。
3. 選擇輸出位置與資料集名稱（預設 `dataset`）。
4. 按「預檢」查看上一版、新增與總圖片數、train/val 分配及預計輸出路徑。
5. 按「開始切分」建立新版本。成功後清空本次新增清單，方便下一次操作。

同一個輸出位置與資料集名稱視為同一系列：第一次生成 `dataset_v1/`；
下次自動讀取 `dataset_v1/`，合併本次新增資料後重新切分成 `dataset_v2/`。
之後使用版本編號最大的資料夾繼續累加。更換資料集名稱或輸出位置可建立另一個系列。
**來源清單只需加入新增資料，不要再次選取上一版。** 舊版與原始資料都保留。
上一版有 `classes.txt` 時，類別名稱與順序必須與本次選取的類別檔完全一致。
舊版沒有 `classes.txt` 時，直接使用 UI 選取的類別檔解讀標註，不讀取舊版 `data.yaml`，也不修改舊版。

## 輸出結構

```text
dataset_v2/
  images/train/image_00000001.png
  images/val/image_00000002.png
  labels/train/image_00000001.txt
  labels/val/image_00000002.txt
  classes.txt
  data.yaml
  split_report.json
```

所有來源合併成一個資料集，train/val 內不再按來源分子資料夾。
圖片統一編號、保留原始圖片格式，同名圖片不會覆蓋；標註同步改名。
`data.yaml` 自動填入新版路徑、train/val 與類別名稱；報告記錄來源、圖片數量、分配與版本。
UI 中的圖片總數不包含標註檔。每一版重新分配 train/val，不固定沿用上一版的分配。

## 切分與驗證規則

- 支援 `images/` 與 `labels/` 分開的 YOLO 資料集，也支援圖片和同名 `.txt` 在根目錄同層的資料夾。
- 類別以選取的 `classes.txt` 為準，不允許空白行、重複類別名稱或不在類別範圍內的標註。
- train:val 固定 7:3，無 test 集。整體張數取最接近 70% 的整數；每個類別的含類別圖片數、框數與背景圖片數，均限制在 70% 的向下／向上取整範圍。
- 缺少標註的圖片沿用原本規則視為背景，在輸出補空白標註；預檢會顯示數量。
- 完全相同內容的圖片保留，但綁在同一集合；相同圖片的標註衝突或孤立標註會報錯。若重複加入舊資料，也會計入張數。
- 多類別共現或重複圖片綁定使比例無法滿足時報錯，不擅自放寬規則。
- 預設 seed=42。先在暫存資料夾完成輸出與配對驗證，成功才建立正式版本；失敗保留暫存結果並顯示位置，不覆蓋舊版。

這是按影像分層切分，並未按來源影片分組；相鄰影片影格仍可能落在不同集合。
