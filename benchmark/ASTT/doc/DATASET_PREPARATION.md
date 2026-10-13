<!-- 檔案功能：影片異常時間區段標記、自動分組與逐幀 .npy 的操作說明。 -->

# ASTT 影片標記與資料集準備

標記畫面與匯出資料固定使用 **30 FPS**，只標記異常的開始、結束區段，工具會將其餘影格視為正常，並自動產生 ASTT 可讀取的 train／val 與逐幀 `.npy`。這是本工具標記專案的明確規則，不會把既有 LabelMe 缺少 JSON 的圖片自動判定正常。

## 啟動

在 Boom 根目錄執行：

```powershell
conda activate boom
python benchmark/ASTT/tools/prepare_dataset.py
```

工具、介面與資料準備邏輯都放在 ASTT 裡。也可在 `benchmark/ASTT/` 目錄執行 `python tools/prepare_dataset.py`；預設標記紀錄與資料集位置由程式推導，兩種啟動方式相同。

使用現有 NumPy、OpenCV 與 Python 內建 Tkinter，無需新增套件。程式分工：`tools/prepare_dataset.py` 是入口；`src/ui/annotation_app.py` 負責播放與操作；`src/annotations.py` 負責標籤、分組與安全匯出。

## 標記一支影片

1. 按「加入影片」，可同時選擇多支 MP4。現有原影片在 `data/video/1440-1080/`。也可「加入影格資料夾」，選 `data/picture/` 一次加入直接子資料夾，或只選一支影片的圖片資料夾。
2. 播放／拖動進度到異常開始的位置，按「異常開始（S）」；到最後一個異常影格，按「異常結束（E）」。兩端影格都包含在標記內，可標記多個區段。
3. 紅色時間軸顯示已標記異常，黃色線是等待終點的起點，藍色線是目前影格。重疊或相鄰區段自動合併。雙擊區段可跳回起點；選取後可刪除重新標記。
4. 看完後按「完成此來源標記」。整支正常的影片不用畫區段，直接按完成即可。每支來源都要完成，才能匯出。

**若異常代表煙霧，請標記煙霧從出現到消失的整個區段，包含持續存在的煙霧；不要只標記它剛開始變化的幾幀。** 雲霧、粉塵與純相機移動可保留為正常干擾，前提是你確認其中沒有要偵測的煙霧事件。這裡做的是時間標籤，不是框選煙霧範圍。

| 快捷鍵 | 操作 |
|---|---|
| 空白鍵 | 播放／暫停 |
| ←／→ | 上一幀／下一幀 |
| Shift＋←／→ | 移動一秒（30 幀） |
| S／E | 異常起點／終點 |

影片加入後立即建立 30 FPS 時間軸，標記畫面與匯出共用同一個影格對應表，無需設定 FPS。每個時間點使用當時最新的原始影格：60 FPS 會約每兩幀取一幀，15 FPS 會每幀重複兩次，30 FPS 原樣使用；不做影像插值。低 FPS 補幀不會增加原影片的動作細節。

時間、逐幀按鍵、起訖標記都以 **轉換後的 30 FPS 影格** 為準。播放可調速度，每次顯示下一個時間軸影格；處理較慢時放慢播放，不跳過時間軸影格。GUI 顯示從第 1 幀開始，JSON 的區段索引從 0 開始。標記單幀代表轉換後的一幀，即使相鄰幀使用相同原始畫面，也能各自標記。

圖片資料夾沒有影片 FPS 資訊，工具將其視為**已經是 30 FPS 的影格序列**，按自然數字順序保留所有圖片與原檔名；不推測原始影格率。若圖片來自其他 FPS，請改加入原影片，才能自動統一。此處影片時間依 OpenCV 讀到的固定 FPS 計算；可變幀率（VFR）影片應先轉為固定幀率影片再標記。

標記變更自動保存至 `output/astt/annotations/annotations_<時間>.json`；可另存新檔，或下次按「載入標記專案」繼續。只有完成的起訖區段會保存，等待終點的起點不會保存。修改區段或開始新增區段後，需重新按完成。

新 JSON 為 version 2，記錄 `annotation_fps: 30`、30 FPS 區段、轉換後 `frame_count`、原始 `source_frame_count` 與原始 `fps`。舊版 version 1 的標記載入後會依原始影格對應轉換成 30 FPS 區段，所有來源回到「待標記」；請檢查區段並重新按完成。僅載入不會修改舊 JSON，可用「另存標記專案」保存新版。

## 自動分組

至少加入 **兩個獨立來源群組**，分成 train／val。預設比例為 **7:3**，依群組數近似分配；不同影片長度會讓實際幀數比例不同。十個群組會分成七個 train、三個 val；只有兩個群組時各一個。此工具不建立獨立 test。

「原影片群組」預設是來源名稱。同一原影片的裁切片段、不同解析度版本，或影片與它的抽幀資料夾，必須填相同群組名稱；同群組完整留在同一組，避免相鄰畫面跨 train／val。工具不會辨識重新命名的重複影片，請自行確認來源關係。

按「預覽分組」查看各來源分配、影格數、異常數與可用正常窗口。分組會優先留正常來源給 train，並優先讓 val 有正常和異常。實際能否做到取決於來源資料；若可評分影格只有一類，工具會提示 AUROC 無法計算。分組使用固定 seed 2026，同一來源集合與標記可重現。

Train 保留影格與標籤，ASTT loader **只使用歷史四幀＋下一幀全部正常的窗口**；不會把異常影格移除後拼接原本不相鄰的畫面。匯出要求至少有一個正常訓練窗口；val 也要有連續五幀可評分窗口。上述檢查對應目前 ASTT 預設 `--num-frames 4`。

## 產生資料集與 .npy

按「產生資料集與 .npy」。預設輸出到 `data/astt/prepared/artillery-shell_v1/`，再次匯出建立 `artillery-shell_v2/`，不覆寫先前版本、原影片、原圖片或既有 `data/astt/`。

影片順序解碼，依標記時的同一對應表輸出 30 FPS 圖片序列：高幀率取樣、低幀率重複影格。每個 `.npy` 與輸出圖片順序一一對齊，完整保留你在 30 FPS 時間軸上標記的區段、單幀與兩端標籤。輸出幀數為 `ceil(原始幀數 × 30 / 原始 FPS)`，時長誤差小於一個輸出影格。圖片資料夾保留原編號缺幀資訊，ASTT loader 不會跨缺幀拼接窗口。

```text
data/astt/prepared/artillery-shell_v1/
├── annotations.json             # 本次來源與人工區段快照
├── split_report.json            # 分配、比例、統計與警告
├── README.md                    # 此版本可直接使用的訓練／測試指令
└── astt/
    ├── train/
    │   ├── <影片名稱>/           # 圖片與 frames.csv
    │   └── test_frame_masks/<影片名稱>.npy
    └── val/
        ├── frames/<影片名稱>/
        └── test_frame_masks/<影片名稱>.npy
```

`.npy` 是一維 `uint8` 陣列，長度等於該來源輸出的圖片數；0 正常、1 異常。它不是像素分割遮罩。資料夾名稱 `test_frame_masks` 是 ASTT loader 的標籤格式名稱，train 與 val 都使用它，不代表另外分出 test。

`frames.csv` 記錄 `filename`、原始 `source_frame_index`、30 FPS 的 `time_seconds`、`label` 與原始 `source_time_seconds`。第 k 張輸出圖片的時間為 `k / 30` 秒；低 FPS 時原始索引可重複，高 FPS 時原始索引會跳號。圖片資料夾的原始秒數留空。`split_report.json` 記錄 `frame_policy: fixed_30fps`、`output_fps: 30` 及各來源原始 FPS。

匯出在背景進行，可取消；失敗或取消會清除本次未完成的新版本。來源讀取失敗會報錯，避免產出長度錯誤的標籤。請保留原始來源位置，後續載入標記專案仍需原始檔案。

## 接到完整模型

匯出完成後看該版本的 `README.md`。`--data-dir` 必須指向 **包含 astt 的版本目錄**，例如 `data/astt/prepared/artillery-shell_v1`。從 Boom 根目錄執行：

```powershell
python benchmark/ASTT/scripts/train.py --data-dir "data/astt/prepared/artillery-shell_v1" --output-dir output --selection auc --experiment-name astt_artillery_shell_001 --device cuda --batch-size 8 --epochs 30
python benchmark/ASTT/scripts/test.py --split val --data-dir "data/astt/prepared/artillery-shell_v1" --output-dir output --checkpoint output/astt/training/astt_artillery_shell_001/checkpoints/best.pth --experiment-name astt_artillery_shell_val_001 --device cuda --batch-size 8
```

`--selection auc` 要求 val 可評分幀含兩類；不符合時匯出的 README 會使用 `--selection mse`，請以該文件為準。模型維持完整 STE＋TTE＋CSA 預設值。每次訓練使用新的 experiment name。訓練結束已會用最佳權重輸出 val 的熱圖、候選事件與指標；也可用上面 `test.py --split val` 重新產生。這些是 validation 結果。

完成標記不等於模型已訓練；煙霧效果仍要以實際 val 的誤報、漏報、事件延遲與處理速度判斷。

## 已有專案的命令列操作

```powershell
# 只預覽，不產生資料集
python benchmark/ASTT/tools/prepare_dataset.py --project "output/astt/annotations/你的標記專案.json" --preview

# 使用保存的標記建立下一個新版本
python benchmark/ASTT/tools/prepare_dataset.py --project "output/astt/annotations/你的標記專案.json" --export --output-parent data/astt/prepared --name "artillery-shell" --ratios 7 3 --seed 2026

# 載入專案繼續標記
python benchmark/ASTT/tools/prepare_dataset.py --project "output/astt/annotations/你的標記專案.json"

```
