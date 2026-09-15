# YOLO 資料集切分

在專案根目錄執行（使用已有 SciPy、NumPy、PyYAML 的 `wildfire_seg` 環境）：

```powershell
python label/split_dataset.py data/yolo_datasets/detect_v1 --dry-run
python label/split_dataset.py data/yolo_datasets/detect_v1
```

只需提供資料夾路徑，預設原地更新該資料集；固定 train:val = 7:3，無 test 集。
程式先在旁邊的暫存目錄完成切分及配對驗證，成功後才替換原資料夾。
完整原資料保留於同層 `<資料夾>.backup_<時間>_<識別碼>`；替換失敗時嘗試恢復原目錄。
準備階段失敗不會更動原資料，暫存目錄保留供檢查。`--dry-run` 完全不寫入。
重複執行會各自保留備份，不會累積 `train/train/` 巢狀目錄。
其他附加檔案會保留；舊的 train/val/test 清單改由新的 `data.yaml` 目錄設定取代。
也可沿用 `--source 路徑`；若加 `--output 新目錄`，則只輸出新資料集、不更新來源。
明確指定的獨立輸出目錄存在時仍拒絕覆蓋。

程式掃描 `images/`、`labels/` 下相同相對路徑的影像與 YOLO detection `.txt`，
讀取 `data.yaml` 的類別名稱。合併現有子資料夾後重新切分，保留相對路徑避免同名衝突。
不依賴匯出時可能已失效的 `train.txt`。依目前資料集設定，缺少標註的影像視為無物件負樣本，
在重建的資料集補上空白 `.txt`，並列入報告的 `filled_empty_labels`。
這是補空白標註，不是模型自動辨識／畫框。原有空白標註也視為負樣本，統一按 7:3 切分。
內容相同的影像保留，但綁定在同一集合以避免洩漏。孤立標註、無效框或相同影像的標註衝突會報錯。

整體影像數採最接近 70% 的整數；**每個類別的含類別影像數及標註框數**，
都限制在 70% 的向下／向上取整範圍內，負樣本也一樣。
同張影像的所有 label 永遠一起移動，避免 train/val 重疊。
只有筆數可整除且類別共現條件容許時，才可能精確 7:3；共現衝突時報錯，
不會為了湊比例複製影像或刪除框。使用 SciPy >= 1.9 的整數最佳化求解。
預設 seed=42，固定來源與相同套件環境可重現切分。

輸出包含 `images/train/`、`images/val/`、`labels/train/`、`labels/val/`、
可直接供 YOLO 訓練的 `data.yaml`，及逐類比例／每張分配／未標註清單的 `split_report.json`。
原地更新 `detect_v1` 後，可直接沿用訓練程式的 `DATASET_NAME = "detect_v1"`。

這是按影像分層隨機切分，未按來源影片分組；相鄰影片影格可能落在不同集合。
