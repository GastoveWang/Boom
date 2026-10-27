# 檔案使用盤點（2026-10-22）

> 歷史文件：保留當時的方法、指令與驗證紀錄；其中舊程式可能已移除，請勿當成目前操作指引。目前操作請見 [ST²](../ST2.md) 與 [文件首頁](../README.md)。

本次只盤點，沒有刪除程式、資料、權重或輸出，也沒有改變訓練流程。

範圍為 Boom 自有的 src/boom、tools、根目錄兩個執行入口、configs、doc；共檢查87個 Python 檔案，從12個工具／主入口追查靜態匯入，並人工核對相容入口、抽象介面、設定及文件。benchmark 與 X-AnyLabeling 等第三方專案不作廢棄判定；資料／權重／輸出不做全面內容掃描。

沒有完整的歷史執行頻率紀錄，不能由修改時間或無匯入直接證明「長年沒用」。以下區分沒有內部直接呼叫、尚未接上、已停用歷史文件與可選工具；外部腳本或人工直接執行仍可能使用相容入口。5個程式候選在Git最近變更日期為2026-10-01，這不代表最後使用日期。

## 一、目前沒有主流程直接呼叫的舊入口（3個）

| 檔案 | 用處與判斷依據 | 處置判斷 |
|---|---|---|
| [src/boom/pipelines/map_localization.py](../../src/boom/pipelines/map_localization.py) | 重新匯出 GeoMap、MapLocalizer、PhotoPose 等定位符號，保留舊定位匯入介面；未找到其他自有程式直接匯入它，現行定位由 boom.localization 提供。 | 相容層，不是新的定位演算法；外部舊腳本可能使用。專案規則要求保留既有 artillery 匯入相容性，不能只憑無內部引用刪除。 |
| [src/boom/pipelines/optical_flow_pipeline.py](../../src/boom/pipelines/optical_flow_pipeline.py) | 舊 motion 專用啟動入口，固定 detector=motion，再交給 pipelines.main。現行 run_offline.py 已直接呼叫 pipelines.main。 | 可列入入口整併，但檔案本身仍可直接執行，不能說功能失效。 |
| [src/boom/online/runner.py](../../src/boom/online/runner.py) | 匯出即時 main、run、參數、定位與儲存等舊符號；run_realtime.py 已直接呼叫 online.cli，未找到內部直接匯入 runner。 | 相容入口，可能服務外部腳本；不因目前 heatmap 未使用就刪除。 |

## 二、定義存在，但沒有實作端使用的抽象介面（2個）

| 檔案 | 用處與判斷依據 | 處置判斷 |
|---|---|---|
| [src/boom/interfaces/sensor.py](../../src/boom/interfaces/sensor.py) | BaseFrameSource 定義 read_frame、fps、resolution、close。全專案自有Python搜尋僅見定義及 interfaces/__init__.py 匯出，未見實作或呼叫；實際使用 VideoSource／SpinCamera 自身介面。 | 未落地的共用介面候選。若決定移除，需同步修改 interfaces/__init__.py；若要統一離線／即時影格來源，可保留供整合。 |
| [src/boom/interfaces/matcher.py](../../src/boom/interfaces/matcher.py) | BaseFeatureMatcher 定義通用 match 介面，同樣僅定義／匯出，現行 EDM、SIFT、SuperPointLightGlue 後端沒有繼承它。 | 未落地的共用介面候選；可選擇接上實作或連同匯出整理。 |

這兩個檔案仍由 interfaces 套件初始化匯入，不能直接刪檔而保留原 __init__.py。相對地，interfaces/localizer.py 的 BaseLocalizer 已被 MapLocalizer 繼承，interfaces/detector.py 也有實際用途，不應混列為廢檔。

## 三、已停用模型的歷史文件（3個）

| 檔案 | 用途 | 狀態 |
|---|---|---|
| [doc/ASTT_BBOX_TRAINING.md](ASTT_BBOX_TRAINING.md) | 舊直接 BBox 回歸模型的訓練、loss、續訓與驗證操作。 | 入口已移除，文件開頭已標示歷史；不能照舊指令執行。 |
| [doc/ASTT_BBOX_INFERENCE.md](ASTT_BBOX_INFERENCE.md) | 舊框模型的影片推論與輸出格式。 | 舊推論程式已移除，供舊結果追溯。 |
| [doc/BBOX_TEMPORAL_MODEL.md](BBOX_TEMPORAL_MODEL.md) | 舊 CNN＋時間 Transformer＋query 框回歸架構及資料契約。 | 架構已退役，供原版／舊版比較。 |

這些文件不是訓練必要檔案，但與使用者要求保留模型演進紀錄有關。適合移入歷史文件區，或保留目前歷史標示，不建議直接抹除方法紀錄。

## 四、舊程式留下的快取與檢查結果

| 路徑 | 用途與發現 | 處置判斷 |
|---|---|---|
| src/boom/models/__pycache__/ | 4個舊 __init__／bbox_temporal／bbox_loss／bbox_metrics 的 .pyc，對應原始碼已移除。 | Python 編譯快取，不是模型權重；可列清理候選。 |
| src/boom/datasets/__pycache__/ | 5個舊 astt_bbox／astt_bbox_training／__init__ 的 .pyc，含Python3.10與3.13版本，原始碼已搬入新套件。 | 舊編譯快取，不是資料集；可列清理候選。 |
| tmp/camera_check/20260922_235032/ | 存有 runtime.log、impact_coordinates.txt、events.jsonl、detections.csv，是既有相機／事件測試輸出。 | 可能有實測追溯價值，來源不是本次建立，保留供使用者確認。 |
| pyproject.toml 的 bbox 選用依賴 | 只列 torch／scipy，為舊BBox環境安裝分組；新 smoke-heatmap 已包含完整訓練／圖表依賴。 | 不是獨立檔案，也不執行舊模型；可視為舊安裝介面，需確認既有安裝指令是否仍依賴。 |

## 五、現在 heatmap 訓練沒用，但仍有獨立用途

| 檔案／功能 | 用途 | 保留理由 |
|---|---|---|
| [tools/check_camera.py](../../tools/check_camera.py) | 列出Spinnaker相機、SDK，並可做有限幀取像驗證。 | 即時無人機部署／硬體檢查。 |
| [tools/benchmark_matchers.py](../../tools/benchmark_matchers.py) | 同影像對比較EDM與SuperPoint/LightGlue的速度及幾何一致性。 | 地圖定位方法評估。 |
| [tools/train_yolo.py](../../tools/train_yolo.py) | YOLO訓練及ONNX匯出。 | YOLO仍是離線／即時可選偵測器。 |
| [tools/split_dataset.py](../../tools/split_dataset.py) | YOLO資料分組與版本匯出。 | 與YOLO訓練搭配使用。 |
| [tools/video_to_frames.py](../../tools/video_to_frames.py) | 依時間間隔抽取原始解析度影格。 | 通用資料準備工具。 |
| [src/boom/online/spinnaker_camera.py](../../src/boom/online/spinnaker_camera.py) 與 spin_camera.py | 前者提供公開相機類別及open流程，後者提供基礎相機／背景取像邏輯。 | online.pipeline、storage及check_camera有實際匯入，兩者不是可隨意刪除的重複檔。 |
| `src/boom/smoke_heatmap/postprocess.py`（原路徑已移除） | 固定0～1分數＋可選CameraMotionGuard。 | 此列是當時的盤點；目前實作見 [src/st2/postprocess.py](../../src/st2/engine/postprocess.py)，不能以歷史結論判定舊程式仍存在。 |
| configs/default.yaml | 空的個人設定覆寫入口。 | config.loader仍會讀取它，空內容不等於廢檔。 |
| configs/*.yaml、localization、detection、runtime、online | 離線／即時、YOLO／motion及地面定位所需。 | 未參與新模型訓練，但仍支援專案的煙霧偵測與定位目標。 |

目前 smoke_heatmap 套件及 train／infer／prepare／export_tensorboard 四個入口均有現行用途。舊 data/astt_bbox/prepared/astt_bbox_v1 仍為預設訓練資料，output/roi_heatmap 的舊權重仍可透過新入口推論，不列為廢棄資料。

若進一步整理，優先檢查舊 .pyc 快取，再由使用者確認兩個未接上的介面與三個相容入口；歷史文件宜歸檔，資料／權重／相機事件紀錄不因這次盤點自動刪除。
