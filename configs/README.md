# 設定檔分類

| 檔案 | 內容 |
| --- | --- |
| `system.yaml` | 專案執行與資源路徑 |
| `ui.yaml` | 顯示、標記與字型 |
| `pipeline.yaml` | 影格處理範圍與頻率 |
| `detector.yaml` | 偵測器選擇 |
| `detector-yolo.yaml` | YOLO 偵測參數 |
| `detector-motion.yaml` | 影像變化偵測參數 |
| `matching.yaml` | 地圖特徵匹配 |
| `localization.yaml` | 定位與預設無人機姿態 |
| `default.yaml` | 個人常用覆寫值；預設為空 |

`load_config()` 會依上表順序讀取分類檔，再合併 `default.yaml`。指定自訂 YAML 時，最後合併自訂檔。相同鍵以後讀入者為準，巢狀設定會逐層合併。

例如只想切換偵測器，在 `default.yaml` 寫入：

```yaml
detector:
  backend: yolo
```

YOLO 預設模型路徑與目標類別由 `detector-yolo.yaml` 的 `model_path`、`target_classes` 決定，離線與即時入口都會使用；命令列的 `--yolo-model-path`、`--yolo-target-classes` 可以暫時覆蓋。程式中的 `src/boom/config/defaults.py` 仍提供其他既有 Python 常數；修改其他偵測參數前，請確認使用的執行流程有讀取對應 YAML 值。

## 地圖匹配

`matching.yaml` 的 `matching.matcher_backend` 預設為 `edm`。離線 CLI 會讀取這個檔案，命令列的 `--map-matcher-backend` 可以暫時覆蓋。直接建立 `MapMatchingConfig()` 時也預設使用 EDM。

EDM 所需參數都在 `matching.yaml` 的 `matching` 區塊：

| 參數 | 用途 |
| --- | --- |
| `edm_model_path` | 官方固定 640×480 ONNX 模型路徑，相對於執行目錄 |
| `edm_conf_threshold` | 粗匹配信心門檻 |
| `edm_sigma_threshold` | 精細匹配分數門檻 |
| `device` | `auto`、`cpu` 或 `cuda`；`auto` 優先使用 CUDA |

模型放在 `models/pretrained/edm_w640_h480_topk1680.onnx`，或用 `--map-edm-model-path` 指定其他位置。模型權重不納入 Git。原本的 SuperPoint + LightGlue 可用 `--map-matcher-backend superpoint_lightglue` 啟用，門檻由 `lightglue_filter_threshold` 控制。
