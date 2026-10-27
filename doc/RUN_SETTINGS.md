# 執行入口與完整設定

> 2026-10-27 更新：ST²統一入口為`python src/st2 train`／`infer`／`prepare`／`export-tensorboard`；實作與CLI均收在src/st2。資料為data/st2/prepared/st2_v1、輸出output/st2，完整設定見 [ST2.md](ST2.md)。Boom名稱及motion／YOLO設定不變。

從 Boom 根目錄執行。以下 CLI 表由目前 argparse 定義（ASTT 位於 benchmark/ASTT/src/config.py）產生；絕對路徑以 `<Boom>` 表示專案根。ASTT 的資料／輸出根會再附加 astt/；修改 configs/*.yaml 不會改 ASTT 設定。

## 常用入口

```powershell
conda activate boom
python run_offline.py motion --video data/video/example.mp4 --no-map --display
python run_offline.py yolo --video data/video/example.mp4 --no-map --display
python run_realtime.py motion --display
python run_realtime.py yolo --display
python tools/check_camera.py --frames 30
python tools/video_to_frames.py data/video/example.mp4 --interval 0.5
python tools/split_dataset.py
python tools/train_yolo.py
python src/st2 train --epochs 30 --device cuda
python src/st2 prepare
python benchmark/ASTT/scripts/train.py --experiment-name astt_first --epochs 30 --batch-size 1
python benchmark/ASTT/tools/prepare_dataset.py
python benchmark/ASTT/scripts/test.py --split val --checkpoint benchmark/ASTT/output/astt/training/astt_first/checkpoints/best.pth --experiment-name astt_val --batch-size 1
```

影片／模型路徑為示例，須替換成現有檔案。ASTT 訓練前準備正常 train 與獨立 val；測試需本機版權重。即時相機需原廠 PySpin SDK。YOLO 訓練的模型、資料集與超參數由互動提示選擇，細節見 [TOOLS.md](TOOLS.md)。

## Boom YAML 設定

ST² 使用 `src/st2/cli/train.py` 訓練／續訓、`src/st2/cli/infer.py` 推論，主體位於 `src/st2/`。設定由 CLI 與 checkpoint 決定，不由既有 Boom YAML 控制，詳見 [模型與操作](ST2.md)。舊 Smoke Heatmap 工具已移除，只使用現行ST²入口。

ST²標註 UI 使用 `src/st2/cli/prepare.py`，新標註及版本資料分別位於 `output/st2/annotations/`、`data/st2/prepared/`。既有 astt_bbox 資料已更名為 `data/st2/prepared/st2_v1`；參數及操作見 [標註工具](ST2_DATASET.md)，與原版 ASTT 的入口及設定分開。

讀取順序與作用見 [configs/README.md](../configs/README.md)。CLI 值可覆蓋對應設定；非所有 YAML 值都由每個 runner 使用，請以 CLI 和 loader 實作為準。保留目前使用者設定，這裡只列出內容。

### default.yaml

```yaml
# Boom configuration entry point.
# Defaults are grouped by purpose in the YAML files beside this one.
# Add only values you want to override here; nested keys are merged.
# A custom YAML passed to load_config() takes precedence over this file.
{}
```

### detector-motion.yaml

```yaml
detector:
  motion:
    diff_interval_sec: 0.12
    diff_threshold: 18
    min_blob_area: 15.0
    min_confirm_area: 35.0
    min_growth_ratio: 1.3
    open_kernel_size: 3
    close_kernel_size: 7
    close_iterations: 1
    min_residual_polarity_ratio: 0.60
    min_residual_signed_coherence: 0.24
    min_candidate_fill_ratio: 0.16
    max_initial_motion_history_overlap: 0.55
    small_min_blob_area: 4.0
    small_diff_floor: 8.0
    small_noise_window_size: 31
    small_noise_sigma: 2.2
    small_min_signal_to_noise: 2.2
    small_min_growth_ratio: 1.35
    small_max_center_shift_ratio: 1.0
    small_confirmation_hits_required: 3
    small_min_cumulative_signal: 700.0
    max_distributed_candidates: 8
    min_candidate_field_span_ratio: 0.30
    confirmation_hits_required: 2
    stop_growth_ratio: 1.08
    max_event_age_sec: 3.0
    confirmed_box_hold_sec: 1.5
    track_match_distance: 200.0
    trajectory_match_distance: 270.0
    recent_confirmed_match_sec: 10.0
    post_event_memory_max_sec: 600.0
    post_event_memory_idle_sec: 600.0
    post_event_spatial_history_sec: 600.0
    post_event_match_distance: 110.0
    post_event_trajectory_distance: 80.0
    post_event_retrigger_min_area: 70.0
    post_event_retrigger_min_growth_ratio: 4.5
    max_missed_frames: 6
    min_dimension_growth_ratio: 1.08
    max_center_shift_ratio: 0.65
    split_blob_area: 2500.0
    enable_radial_flow_check: true
```

### detector-yolo.yaml

```yaml
detector:
  yolo:
    model_path: "models/checkpoints/yolo26m.onnx"
    input_size: 640
    conf_threshold: 0.25
    iou_threshold: 0.45
    target_classes: ["person"]
    confirmation_hits: 2
```

### detector.yaml

```yaml
detector:
  backend: "motion" # "motion", "yolo"
```

### localization.yaml

```yaml
localization:
  mpp: 0.75
  map_span_m: 1100.0
  allow_gps_seed: false
  default_drone_pose:
    yaw: 0.0
    pitch: 20.0
    roll: 0.0
```

### matching.yaml

```yaml
matching:
  search_radius_m: 1000.0
  max_candidates: 9
  roi_size_m: 1000.0
  max_image_size: 1600
  max_features: 2048
  matcher_backend: "edm" # "edm" (default), "superpoint_lightglue", "sift"
  device: "auto"
  lightglue_filter_threshold: 0.1
  edm_model_path: "models/pretrained/edm_w640_h480_topk1680.onnx" # Official fixed 640x480 ONNX model
  edm_conf_threshold: 0.2 # Coarse match confidence threshold
  edm_sigma_threshold: 0.000001 # Fine match score threshold
  ratio_threshold: 0.70
  ransac_threshold_px: 3.0
  min_inliers: 16
  min_inlier_ratio: 0.35
  max_reprojection_error_px: 2.5
  min_coverage: 0.02
  cache_entries: 9
  debug: false
```

### pipeline.yaml

```yaml
pipeline:
  start_frame: 0
  max_frames: -1
  downscale: 0.7
  stride_frames: 2
```

### system.yaml

```yaml
system:
  project_name: "Boom"
  seed: 42
  device: "auto" # "auto", "cpu", "cuda"
  output_root: "output"
  log_level: "INFO"

paths:
  map_root: "asset/maps"
  drone_icon: "asset/pictures/drone.png"
```

### ui.yaml

```yaml
ui:
  mode: "client"
  display_scale: 0.5
  display_seconds: 10.0
  no_display: true
  panel_width: 500
  show_candidate_boxes: true
  draw_debug_tiles: false
  font_paths:
    - "/System/Library/Fonts/PingFang.ttc"
    - "/System/Library/Fonts/STHeiti Medium.ttc"
    - "C:/Windows/Fonts/msjh.ttc"
    - "C:/Windows/Fonts/msjhbd.ttc"
    - "C:/Windows/Fonts/mingliu.ttc"
```

## Boom 離線 CLI

| 參數 | 預設值 | 用途／限制 |
|---|---|---|
| `--video` | `None` | Input video; defaults to the first video under data/. |
| `--reference-image` | `None` | DJI image containing GPS/XMP metadata; required unless --no-map is used. |
| `--map-dir` | `<Boom>\asset\maps` | Georeferenced RGB/grayscale GeoTIFF directory (default: C:\Users\roger\IDEA\Boom\asset\maps). |
| `--no-map` | `<Boom>\asset\maps` | Disable map initialization and VO; use the legacy coordinate panel. |
| `--map-mpp` | `0.75` | Map preview metres per pixel (default: 0.75). |
| `--map-matcher-backend` | `edm` | map matcher backend；可選 superpoint_lightglue, edm, sift |
| `--map-device` | `auto` | map device；可選 auto, cpu, cuda |
| `--map-lightglue-filter-threshold` | `0.1` | map lightglue filter threshold |
| `--map-edm-model-path` | `models/pretrained/edm_w640_h480_topk1680.onnx` | map edm model path |
| `--map-edm-conf-threshold` | `0.2` | map edm conf threshold |
| `--map-edm-sigma-threshold` | `1e-06` | map edm sigma threshold |
| `--map-search-radius-m` | `1000.0` | map search radius m |
| `--map-max-candidates` | `9` | map max candidates |
| `--map-roi-size-m` | `1000.0` | map roi size m |
| `--map-max-image-size` | `1600` | map max image size |
| `--map-max-features` | `2048` | map max features |
| `--map-ratio-threshold` | `0.7` | map ratio threshold |
| `--map-ransac-threshold-px` | `3.0` | map ransac threshold px |
| `--map-min-inliers` | `16` | map min inliers |
| `--map-min-inlier-ratio` | `0.35` | map min inlier ratio |
| `--map-max-reprojection-error-px` | `2.5` | map max reprojection error px |
| `--map-min-coverage` | `0.02` | map min coverage |
| `--map-cache-entries` | `9` | map cache entries |
| `--map-debug` | `False` | 額外保存 UAV／ROI／inliers／footprint 診斷；匹配圖與數據預設保存。 |
| `--ground-height` | `None` | Camera height above local ground in metres; default: photo RelativeAltitude approximation. |
| `--allow-gps-seed` | `False` | Allow explicitly provisional GPS-photo VO if photo/map visual matching fails. |
| `--drone-icon` | `<Boom>\asset\pictures\drone.png` | drone icon |
| `--precision` | `fp32` | Neural inference precision; opt-in fp16 for CUDA (CPU uses fp32).；可選 fp32, fp16 |
| `--writer-queue` | `2` | Bounded background encoding queue; 0 disables it. No frames dropped. |
| `--display-width` | `1280` | Maximum preview width; 0 uses full resolution. Saved video is unchanged. |
| `--display` | `False` | Show the processed video and map window. |
| `--start-frame` | `0` | start frame |
| `--max-frames` | `-1` | max frames |
| `--report-every` | `60` | Print progress every N frames (default: 60). |
| `--detector` | `motion` | Select optical flow or YOLO.；可選 motion, yolo |
| `--device` | `auto` | YOLO inference device (default: auto).；可選 auto, cpu, cuda |
| `--inference-stride` | `1` | inference stride |
| `--yolo-model-path` | `<Boom>\models\checkpoints\yolo26m.onnx` | Path to YOLO ONNX/PT model (default: C:\Users\roger\IDEA\Boom\models\checkpoints\yolo26m.onnx). |
| `--yolo-conf` | `0.5` | YOLO detection confidence threshold (default: 0.5). |
| `--yolo-iou` | `0.45` | YOLO NMS IoU threshold (default: 0.45). |
| `--yolo-imgsz` | `640` | YOLO input image size (default: 640). |
| `--yolo-target-classes` | `None` | Model class names to detect, e.g. person car; defaults to boom. |
| `--yolo-track-match-distance` | `140.0` | YOLO tracking match distance in pixels (default: 140.0). |
| `--yolo-max-missed-sec` | `2.0` | YOLO tracking max missed tolerance in seconds (default: 2.0). |
| `--box-hold-sec` | `1.5` | box hold sec |
| `--panel-hold-sec` | `10.0` | panel hold sec |

## Boom 即時 CLI

| 參數 | 預設值 | 用途／限制 |
|---|---|---|
| `pipeline` | `None` | Detection pipeline: optical-flow motion or YOLO (default: motion).；可選 motion, yolo |
| `--serial` | `None` | Camera serial; uses the first camera by default. |
| `--width` | `None` | Optional camera ROI width. |
| `--height` | `None` | Optional camera ROI height. |
| `--offset-x` | `None` | offset x |
| `--offset-y` | `None` | offset y |
| `--camera-fps` | `30.0` | camera fps |
| `--exposure-us` | `None` | exposure us |
| `--gain-db` | `None` | gain db |
| `--camera-timeout-ms` | `1000` | camera timeout ms |
| `--detector` | `None` | Compatibility alias for the pipeline name.；可選 motion, yolo |
| `--device` | `auto` | auto uses CUDA for YOLO when available; motion uses CPU.；可選 auto, cpu, cuda |
| `--precision` | `fp32` | precision；可選 fp32, fp16 |
| `--inference-stride` | `1` | inference stride |
| `--processing-width` | `0` | Resize full camera frame to this width before detection; 0 keeps native resolution. |
| `--yolo-model-path` | `<Boom>\models\checkpoints\yolo26m.onnx` | yolo model path |
| `--yolo-imgsz` | `640` | yolo imgsz |
| `--yolo-target-classes` | `None` | Model class names to detect, e.g. person car; defaults to the smoke event class. |
| `--downscale` | `None` | downscale |
| `--stride` | `None` | stride |
| `--diff-threshold` | `None` | diff threshold |
| `--min-blob-area` | `None` | min blob area |
| `--min-growth-ratio` | `None` | min growth ratio |
| `--disable-flow-check` | `False` | Disable the original radial optical-flow validation. |
| `--debug-tiles` | `False` | debug tiles |
| `--output-dir` | `<Boom>\output\realtime` | output dir |
| `--display` | `False` | Show the live window. Headless logging/output remains active without it. |
| `--display-width` | `1280` | display width |
| `--max-frames` | `-1` | max frames |
| `--report-every` | `30` | Write a heartbeat log every N processed frames. |
| `--drone-lon` | `None` | drone lon |
| `--drone-lat` | `None` | drone lat |
| `--drone-alt` | `None` | drone alt |
| `--drone-yaw` | `0.0` | drone yaw |
| `--drone-pitch` | `-45.0` | drone pitch |
| `--drone-roll` | `0.0` | drone roll |

## ST² 訓練 CLI

根目錄`python train_st2.py`會預設加上推論來源`data/video/1440-1080/0603`，訓練成功後自動以best.pt推論；參數仍共用下表。`python src/st2 train`的infer_source預設為None，維持可選推論。

入口：`python src/st2 train`。以下預設值直接由目前parser讀取；續訓時部分值會從checkpoint恢復。ST²推論與資料標註指令見 [ST2.md](ST2.md) 及 [ST2_DATASET.md](ST2_DATASET.md)。

| 參數 | 預設值 | 用途／限制 |
|---|---|---|
| `--data` | `<Boom>\data\st2\prepared\st2_v1` | data |
| `--output` | `<Boom>\output\st2\training` | output |
| `--epochs` | `30` | epochs |
| `--history-frames` | `8` | Past inputs; current scoring frame is separate |
| `--height` | `576` | height |
| `--width` | `768` | width |
| `--batch-size` | `4` | batch size |
| `--embedding-dim` | `128` | embedding dim |
| `--model-variant` | `transformer` | model variant；可選 transformer |
| `--patch-size` | `16` | Transformer-only RGB patch side |
| `--transformer-layers` | `2` | Transformer-only space/time block count |
| `--window-size` | `6` | Transformer-only spatial window side in patches |
| `--camera-augmentation` | `0.5` | Training-only coherent camera trajectory probability |
| `--motion-loss-weight` | `0.2` | motion loss weight |
| `--gate-floor` | `0.05` | gate floor |
| `--lr` | `0.0001` | lr |
| `--positive-fraction` | `0.5` | positive fraction |
| `--workers` | `4` | Parallel image loaders; use 0 for serial loading |
| `--device` | `auto` | device |
| `--seed` | `2026` | seed |
| `--max-train-samples` | `0` | max train samples |
| `--max-val-samples` | `0` | max val samples |
| `--resume` | `None` | resume |
| `--infer-source` | `None` | Run full-folder heatmap inference after final epoch succeeds |
| `--inference-max-frames` | `0` | Limited inference only for preflight |
| `--heatmap-max` | `0.02` | heatmap max |
| `--no-tensorboard` | `False` | Disable TensorBoard event logging |

## ASTT 訓練 CLI

| 參數 | 預設值 | 用途／限制 |
|---|---|---|
| `--data-dir` | `<Boom>\benchmark\ASTT\data` | Root containing astt/ |
| `--output-dir` | `<Boom>\benchmark\ASTT\output` | output dir |
| `--experiment-name` | `20261001_201320_522097` | experiment name |
| `--device` | `auto` | device；可選 auto, cpu, cuda |
| `--batch-size` | `8` | batch size |
| `--num-workers` | `0` | 0 is safest on Windows |
| `--registration` | `False` | ECC align past inputs to latest input |
| `--checkpoint` | `None` | Initialize model / required for inference |
| `--seed` | `2026` | seed |
| `--threshold` | `0.1` | Absolute channel-mean squared error in [-1,1] image space |
| `--min-area` | `25` | Minimum component area at model resolution |
| `--morph-kernel` | `3` | morph kernel |
| `--min-event-frames` | `3` | min event frames |
| `--event-iou` | `0.3` | event iou |
| `--event-max-gap` | `2` | event max gap |
| `--resume` | `None` | Restore model, optimizer, scheduler and RNG |
| `--epochs` | `30` | Total target epochs, including resumed epochs |
| `--lr` | `0.0001` | lr |
| `--weight-decay` | `0.0001` | weight decay |
| `--image-size` | `256` | image size |
| `--patch-size` | `32` | patch size |
| `--emb-dim` | `768` | emb dim |
| `--mlp-dim` | `3072` | mlp dim |
| `--num-heads` | `8` | num heads |
| `--spatial-layers` | `12` | spatial layers |
| `--temporal-layers` | `12` | temporal layers |
| `--num-frames` | `4` | num frames |
| `--dropout-rate` | `0.1` | dropout rate |
| `--selection` | `mse` | Best checkpoint criterion, val only；可選 mse, auc |

## ASTT 測試 CLI

| 參數 | 預設值 | 用途／限制 |
|---|---|---|
| `--data-dir` | `<Boom>\benchmark\ASTT\data` | Root containing astt/ |
| `--output-dir` | `<Boom>\benchmark\ASTT\output` | output dir |
| `--experiment-name` | `20261001_201320_523098` | experiment name |
| `--device` | `auto` | device；可選 auto, cpu, cuda |
| `--batch-size` | `8` | batch size |
| `--num-workers` | `0` | 0 is safest on Windows |
| `--registration` | `False` | ECC align past inputs to latest input |
| `--checkpoint` | `None` | Initialize model / required for inference |
| `--seed` | `2026` | seed |
| `--threshold` | `0.1` | Absolute channel-mean squared error in [-1,1] image space |
| `--min-area` | `25` | Minimum component area at model resolution |
| `--morph-kernel` | `3` | morph kernel |
| `--min-event-frames` | `3` | min event frames |
| `--event-iou` | `0.3` | event iou |
| `--event-max-gap` | `2` | event max gap |
| `--split` | `test` | split；可選 val, test |

## tools/check_camera.py 設定

| 參數 | 預設／形式 | 用途／限制 |
|---|---|---|
| `--serial` | `未指定` | --serial |
| `--frames` | `0` | 0 lists cameras; positive number also tests acquisition. |

## tools/video_to_frames.py 設定

| 參數 | 預設／形式 | 用途／限制 |
|---|---|---|
| `video` | `未指定` | 輸入影片路徑 |
| `--output` | `未指定` | 輸出至指定的新資料夾 |
| `--interval` | `0.0` | 每隔幾秒擷取一張（預設 0，擷取每一幀） |
| `--format` | `'png'` | 圖片格式（預設 png）; choices=('jpg', 'png') |

## tools/split_dataset.py 設定

| 參數 | 預設／形式 | 用途／限制 |
|---|---|---|
| `--source, -s` | `未指定; nargs='+'` | 來源資料夾路徑（可指定多個） |
| `--classes, -c` | `未指定` | 共用 classes.txt 類別檔路徑 |
| `--output, -o` | `未指定` | 輸出上層資料夾路徑（搭配 --name 產生版本資料夾） |
| `--name, -n` | `'dataset'` | 資料集名稱（預設：dataset） |
| `--seed` | `42` | 隨機種子（預設：42） |
| `--dry-run` | `未指定; action='store_true'` | 僅預檢分配，不寫入檔案 |
| `--in-place` | `未指定; action='store_true'` | 針對單一現有資料集進行原地重新切分（保留備份） |

## tools/benchmark_matchers.py 設定

| 參數 | 預設／形式 | 用途／限制 |
|---|---|---|
| `image0` | `未指定` | image0 |
| `image1` | `未指定` | image1 |
| `--edm-model-path` | `未指定` | --edm-model-path |
| `--device` | `'auto'` | --device; choices=('auto', 'cpu', 'cuda') |
| `--warmup` | `3` | --warmup |
| `--repeats` | `20` | --repeats |
| `--ransac-threshold-px` | `3.0` | --ransac-threshold-px |

## tools/train_yolo.py 設定

| 參數 | 預設／形式 | 用途／限制 |
|---|---|---|
| `--export` | `False; nargs='?'` | 單獨將指定的 best.pt 轉換為 ONNX 放入 models/checkpoints/（若不指定路徑則自動尋找最新 best.pt） |

YOLO 訓練不是以 argparse 設定全部超參數；模型與版本資料集由互動選單挑選，訓練參數定義在 main() 的 model.train() 呼叫：

```python
model.train(data=str(DATA_YAML), epochs=100, imgsz=IMAGE_SIZE, batch=-1, device=0, patience=20, optimizer='auto', workers=8, project=str(RUNS_DIR), name=EXPERIMENT_NAME, exist_ok=True)
```

## ASTT 抽幀設定

| 參數 | 預設 | 用途 |
|---|---|---|
| video | 必填 | 來源影片路徑。 |
| --fps | 10 | 抽幀 FPS，不得高於來源 FPS。 |
| --name | 影片 stem | 片段資料夾名稱，已存在則拒絕覆寫。 |
| --split | train | train、val 或 test。 |
| --data-dir | <Boom>/benchmark/ASTT/data | 根目錄，會附加 astt/<split>。 |

模型設定由 checkpoint 載入時，以權重內 model_config 為準；registration 必須與訓練一致。resume 要保持原總 epochs、每 epoch steps 和選模方式。threshold/min-area 等只影響候選區域，不是煙霧分類信心。更多格式和限制見 [ASTT 本機說明](../benchmark/ASTT/doc/README_LOCAL.md)。
