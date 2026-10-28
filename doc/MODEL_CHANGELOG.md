# 模型與推論變更紀錄

## 2026-10-28 — 新增 HSTforU 比較套件

- 需求：將指定的官方 HSTforU repo 放進 benchmark，先整理獨立套件並說明與目前 ST² 的差異，不接資料或啟動訓練。
- 修改前後：新增 `benchmark/HSTforU/origin/` 官方完整工作樹與 `UPSTREAM.json`（commit `9548473306bbb89895b077275e8503de79653029`、321 檔 SHA-256）；`src/hstforu/` 複製模型／必要 helpers，僅修改套件匯入命名空間，加入 Ped2／Avenue API、doctor、forward 及嚴格 checkpoint 載入。官方 forward、loss、設定與分數不變；ST² 模型及推論未改動。
- 檔案／參數：`benchmark/HSTforU/pyproject.toml`、`src/hstforu/`、README、`doc/HSTFORU_COMPARISON.md`、文件導覽及 dev 的 benchmark 忽略例外。模型固定 4 張歷史；依賴 timm0.6.13／MMCV1.7.2；沒有新增煙霧門檻。
- 權重與重訓：兩個模型家族不相容，不能把 ST² 權重套入 HSTforU。封裝不改官方 state_dict 鍵，不需因封裝重訓；尚未取得官方訓練權重驗證。ST² 既有權重不變。
- 驗證：CPU、batch1、四張256×256，Ped2／Avenue 前向皆輸出有限的 `[1,3,256,256]`；實測參數137,615,680／148,969,536。Ped2 官方／封裝相同種子下輸出逐值一致且 state_dict 鍵一致；ST² 預設參數實測3,587,204。官方321檔SHA-256一致，套件 wheel 建置成功。
- 輸出位置：驗證及暫時依賴均在 `tmp/hstforu_setup_20261028/`，完成後清理；長期產物是套件及比較文件。未修改現有 Python 環境，未訓練、未量測煙霧準確率／FPS／定位誤差。
- 指令：先依 `benchmark/HSTforU/README.md` 建立獨立環境並安裝，再執行 `python -m hstforu doctor`、`python -m hstforu forward --variant ped2 --device cpu`。完整原版訓練／AUC 評估流程尚未驗證，不能把前向通過當成效能重現。

> 本檔依時間保留完整歷史；舊指令與路徑不一定仍可執行。目前操作請看 [ST²](ST2.md)，舊方法文件已歸入 [history](history/README.md)。

後續每次修改模型、loss、資料解讀、訓練參數、分數、門檻或推論顯示，都在本檔新增一筆，不覆寫舊紀錄。每筆包含：日期／版本、需求、修改前後、涉及檔案、權重／資料相容性、執行指令、驗證與尚未確定的效果。相對原 ASTT 的完整比較見 [MODEL_EVOLUTION.md](MODEL_EVOLUTION.md)。

## 2026-10-22 — v2.1：固定分數尺度與相機運動檢查

使用者後續澄清：希望改的是模型的時序訓練，而不是僅在推論扣分。因此運動檢查改為明確指定 `--motion-mode guard` 才啟用的對照，預設 `off`；0～1 分數與固定縱軸仍啟用。**v2.1 沒有學習載具運動，不能描述為已完成相機運動感知訓練。**

- 需求：數值統一 0.00～1.00；快速轉鏡與非目標變化不要被當作可靠目標反應；建立持續變更紀錄。
- 原本：raw/weighted 的 top-0.1% MSE 直接顯示，曲線縱軸依最近 10 秒最大值縮放；沒有獨立相機運動檢查。
- 現在：固定 `clip(log1p(s/0.0001)/log1p(0.02/0.0001),0,1)`，CSV 保留原分數並新增 normalized 分數，UI 固定 0～1、顯示兩位小數。不進行每影片或未來視窗正規化。
- 相機檢查：LK 前後一致性追蹤＋RANSAC 相似變換、重疊／inlier 品質檢查；可靠時使用補償後局部殘差的軟 support 乘上 learned-weighted 圖。不對原始模型輸入做額外 warp。
- 快速轉鏡：預設旋轉超過 60 度／秒或中心位移超過 0.75 影像對角線／秒等狀況，標記不可用。恢復期間等待 8 個可靠影格。不可用分數為 CSV 空值、圖上 N/A，不是 0。
- 原始誤差仍可見；未可靠配準時右側保留原 learned-weighted 診斷圖，並明確標示不可用，不能依其亮度確認事件。
- 相關檔案：`src/boom/models/heatmap_postprocess.py`、`tools/infer_roi_heatmap.py`、`src/boom/ui/roi_heatmap_view.py`、本檔與比較文件；AGENTS.md 新增持續記錄規則。
- 相容性：不改神經網路架構或權重、不重新訓練。`--motion-mode off` 可關閉額外運動檢查，仍保留固定分數顯示。舊輸出不覆寫。
- 驗證：固定換算端點／單調性；靜止圖、已知平移、局部目標、快速旋轉與恢復；實際權重的 MP4/CSV/NPZ 端到端測試。關閉運動檢查後，真實影片前 16 幀的原始／加權 MSE 與舊輸出一致，誤差小於 1e-7；所有影格仍輸出，不可用分數留空。臨時測試位於 tmp，完成後清理。
- 使用者回報對照：DJI_007_V 約 9.24 秒尖峰；DJI_009_V 約 17 秒之後。實際結果及路徑於完成對照後補記，尚不宣稱誤報率或召回率已改善。
- DJI_009_V 完整可選防護對照已輸出於 `output/roi_heatmap/inference_reviews/20261022/heatmap_20261022_164622_0f680ae9/`，處理全部 1,567 幀；這是推論後處理實驗，不是新訓練模型，不能代替下列時序學習方案。
- 執行方式見 [SMOKE_HEATMAP.md](history/SMOKE_HEATMAP.md) 的 v2.1 節。這次分數參數／運動門檻是可調的實驗起點，不是機率校準或安全閾值。

## 2026-10-18～21 — v2.0：BBox 監督的 MSE heatmap

- 需求修正：BBox 用來指出要關注的變化，並非最終要回歸物體框。框內為目標、框外為干擾。
- 架構：以 8 張歷史圖預測第 9 張正常畫面；新增獨立 ROI 關注 encoder/decoder；raw MSE 與 `E×[0.05+0.95A]` 分開輸出。
- Loss：整個 9 幀皆正常的窗口才計算預測 MSE；全部有效窗口訓練平衡 BCE＋正樣本 Dice。ROI loss 不更新 predictor。
- 資料：沿用 `astt_bbox_v1`，重建連續 9 幀窗口，保留來源分組，排除 ignore／缺幀。8 幀歷史＋1 張當前圖；不是舊版含當前的 8 幀。
- 設定：768×576、batch 4、30 epochs、embedding 128、4 heads、1 層時序 encoder。初始 run 第 28 輪後中斷，2026-10-21 從 last.pt 續訓至第 30 輪，保留第 7 輪最佳權重。
- 訓練後自動推論 0603 的 14 支影片，全部完成；三欄 heatmap 與曲線，CSV 保存逐幀分數。格式改成 `boom.roi_heatmap.checkpoint.v1`，不能載入舊 BBox 權重。
- 驗證：無當前幀偷看、梯度隔離、資料時間順序、模型續訓及訓練後自動推論、小樣本影片完整解碼、實際 GPU batch 4 更新；詳見 [SMOKE_HEATMAP.md](history/SMOKE_HEATMAP.md)。
- 結果／限制：第 7 輪 val weighted frame AUC 約 0.70882；仍有轉鏡及其他干擾，未提供事件延遲／GPS 誤差，不保證泛化。

## 2026-10-17～18 — v1.x：直接 BBox 實驗（保留對照）

- 最初為 CNN＋時間 Transformer＋query decoder，輸出單類框與背景；CE/L1/GIoU 使用 Hungarian 配對。
- 2026-10-17 加入可選 NWD 配對／回歸損失，保留權重 0 的原版對照。新訓練預設混合係數為 1/5/2/2。
- 尺寸由 256×192 提高至 768×576；後將 50 epochs 改成 30、歷史由 4 改成含當前的 8 幀，loader 從既有匯出重建窗口。
- 先前 4 幀、256×192 實驗跑完 50 輪但跨影片效果極差；只記錄觀察，不認定新增框無效。後續 8 幀訓練未完整跑完時曾用已保存權重推論，輸出與新 heatmap 實驗分開。
- v1.x checkpoint 與原 ASTT、v2.0 不相容；這階段不是注意力 mask 監督，亦不產生真正影像預測 MSE。

## 2026-10-22 — v2.2 設計：學習載具運動與局部目標（尚未實作／尚未訓練）

- 使用者澄清：要改訓練模式，讓模型理解整段前後關係中的背景共同運動，而不是只在推論階段遮掉快速轉鏡。
- 現況缺口：predictor 使用前 8 幀，但目前 gate 只直接讀最後一張歷史圖、當前圖及 MSE；沒有被訓練成顯式的背景運動模型。
- 建議模型：增加以完整歷史＋當前圖估計全局背景運動／時序關注的分支，分開背景共同移動與補償後局部變化。背景配準的學習損失忽略目標 ROI，避免把目標自身移動也當成相機運動去消除。當前圖只能在到達後供運動估計及評分，不能餵給下一幀 predictor 偷看答案。
- 建議資料：對整個「8幀歷史＋當前」施加連續一致的平移／旋轉／縮放增強；當前 BBox mask 必須跟隨同一變換。加入真實轉鏡、模糊及無目標變化片段。轉鏡同時有目標的畫面仍以框內正樣本、框外干擾訓練，不能把所有轉鏡整幀標成負樣本。
- 驗證設計：分開比較靜止、有載具運動但無目標、載具運動且有目標三類，檢查誤報、召回、偵測延遲與速度。先做小規模消融，不以訓練 loss 下降或推論分數被壓低宣稱學會分辨。
- 狀態：這筆是設計紀錄；目前已訓練權重仍是 v2.0，沒有啟動 v2.2 的長時間訓練。v2.1 normalize 可繼續使用，推論運動 guard 只保留為預設關閉的可選對照。

## 2026-10-22 — v2.2 實作：背景運動學習與完整窗口關注（未正式重訓）

- 需求：使用者要求先修改模型，從訓練學習載具共同移動與局部目標的差異；保留 MSE heatmap，沿用現有框，不要求煙霧 ID。
- 修改前 → 後：gate 只讀最後歷史／當前兩幀 → 順序融合全部 9 幀 RGB、原始 MSE、8 組運動補償 MSE 與運動統計。另增獨立背景運動 CNN，對每組相鄰影格估計 4 參數相似變換；predictor 仍只讀前 8 幀。
- loss：保留正常窗口 MSE＋平衡 BCE＋正樣本 Dice，新增 `0.2*(框外背景 photometric + 0.001*運動參數平方均值)`。排除 warp 後的前幀 ROI 與當前 ROI 聯集及邊界；gate 與 predictor／motion 的梯度隔離。補償特徵供 gate 判讀，最終圖仍是原始預測 MSE 乘關注權重。
- 資料：同一匯出資料額外讀 9 幀 mask；訓練時 50% 機率套用連續相機軌跡，影像與所有 mask 同步。驗證及推論不增強，原資料不修改。沒有把轉鏡且含煙霧的影格整張改成負樣本。
- 檔案：新增 `src/boom/models/camera_motion.py`；修改 `src/boom/models/roi_heatmap.py`、`src/boom/datasets/roi_heatmap.py`、`tools/train_roi_heatmap.py`、`tools/infer_roi_heatmap.py`，以及架構／方法／本紀錄文件。
- 參數：新訓練預設 `--model-variant camera-aware --camera-augmentation 0.5 --motion-loss-weight 0.2`，仍是 8 張歷史、768×576、batch 4、30 epochs。baseline 消融使用 `--model-variant baseline --camera-augmentation 0 --motion-loss-weight 0`。
- 相容性：checkpoint 格式仍為 v1，variant 區分架構。缺少 variant 的舊檔自動視為 baseline；resume 恢復原架構／增強／loss 設定，不能藉舊檔續訓冒充新版。best 權重沿續訓保留時，比較訓練設定而非 run 路徑，並正規化舊 model_config。新架構需另開訓練；不覆蓋舊權重、影片、輸出或標註。
- 執行：`python tools/train_roi_heatmap.py --model-variant camera-aware --epochs 30 --device cuda --infer-source "data/video/1440-1080/0603"`。正式輸出另建 `output/roi_heatmap/training/roi_heatmap_<時間>_<ID>/`；訓練完成後才以 best.pt 自動推論。本次未啟動正式 30 epochs。
- 驗證：合成已知平移學習 loss 約 0.09207→0.00293；真實 MSE、predictor 因果性、三分支梯度隔離、ROI 排除、無背景、同步增強及歷史 ROI 載入均通過。小資料完成新架構訓練／續訓／自動推論，舊檔恢復及多次續訓保留既有最佳權重通過；實際舊 checkpoint 前 16 幀分數與既有 CSV 差異 <1e-7。實際資料 RTX 5080 FP32、768×576、8+1 幀、batch 4、開啟增強，單批前向／反向／optimizer 更新通過，峰值配置顯存約 7.18 GiB。臨時程式及產物放 tmp 並於完成後清理。
- 限制：以上是功能／可訓練性驗證，不是煙霧效果驗證；尚無新版正式權重／全片結果。2D 相似變換無法完整解決視差、滾動快門與模糊；現有框為弱監督，仍需實際轉鏡 hard negatives 及有移動且有目標的召回對照。尚未接入 `artillery/online/`，未提供事件定位。

## 2026-10-22 — v2.2 訓練資料載入加速

- 需求：使用者希望訓練更快，觀察上次 RAM 未吃滿。
- 修改前 → 後：`tools/train_roi_heatmap.py` 的預設 `--workers 0` → `--workers 4`，平行讀取／解碼／縮圖並預取下一批；沿用原有 CUDA pinned memory。不以吃滿 RAM 為目標。
- 參數／相容性：仍為 768×576、8+1 幀、batch 4、FP32，架構、loss、抽樣及 30 epochs 不變。權重格式不變，不因載入加速要求重訓；原續訓允許指定 workers，未變更權重恢復規則。可用 `--workers 0` 恢復單程序。
- 驗證：本機約 63.35 GiB RAM、20 邏輯核心、RTX 5080；同一實際訓練資料前 24 窗口、batch 4，workers=0/2/4 第一批含啟動分別約 0.84/5.02/9.43 秒，後續 5 批約 3.47/2.20/1.29 秒。分開量測 GPU resident batch、camera-aware FP32、相機增強及完整梯度更新，略過暖機後平均約 0.150 秒／批。4 程序實際讀取及 GPU 更新通過，臨時程式於 tmp 建立並清理。
- 限制：這是連續小樣本及當時檔案快取下的量測，不能當作完整訓練快 2.7 倍；目前每 epoch 仍重新建立 loader，會有程序啟動時間。未加入 RAM 影像快取／AMP／更大 batch，也未啟動正式訓練。
- 執行／輸出：`python tools/train_roi_heatmap.py --model-variant camera-aware --epochs 30 --workers 4 --device cuda --infer-source "data/video/1440-1080/0603"`。正式結果另建 `output/roi_heatmap/training/` 的 run，既有模型與輸出不覆寫；本次只有效能預檢，沒有新的正式權重。使用說明及整體差異同步更新於 SMOKE_HEATMAP.md、MODEL_EVOLUTION.md。

## 2026-10-22 — Smoke Heatmap 統一命名與移除停用的 BBox 程式

- 需求：使用者要求統一方法名稱及所有相關檔案，刪除已不用的舊程式。
- 修改前 → 後：分散的 roi_heatmap 模型／資料／UI、astt_bbox_training 資料載入與舊 BBox 入口的共用函式，集中於 `src/boom/smoke_heatmap/`。名稱統一 **Smoke Heatmap**；模型／資料類別改為 SmokeHeatmapConfig/Model/Criterion/Dataset、SmokeSequenceDataset、SmokeProject/Source/AnnotationApp。
- 新套件：model.py、motion.py、dataset.py、sequence.py、annotations.py、annotation_app.py、postprocess.py、view.py、utils.py。utils.py 保留原本前處理、JSON、抽樣及 checkpoint 保存邏輯，model.py 保留仍使用的 CNN block，避免對停用 BBox 模組的依賴。
- 新入口：`tools/train_smoke_heatmap.py`、`tools/infer_smoke_heatmap.py`、`tools/prepare_smoke_heatmap_dataset.py`。原 heatmap／prepare_astt_bbox 入口移動更名，不保留重複入口。方法及標註文件改為 `doc/SMOKE_HEATMAP.md`、`doc/SMOKE_HEATMAP_DATASET.md`。
- 已刪除：`tools/train_astt_bbox.py`、`tools/infer_astt_bbox.py`、`tools/bbox_model.py`、`src/boom/models/bbox_temporal.py`、`bbox_loss.py`、`bbox_metrics.py`，以及已無內容用途的 models/datasets 套件初始化檔。刪除前已遷出仍被 heatmap 使用的函式；目前不再提供直接 BBox 回歸的訓練／推論功能。相關舊文件標為歷史文件、保留實驗追溯。
- 資料與輸出：新 run 與標註寫入 `output/smoke_heatmap/`，新匯出預設 `data/smoke_heatmap/prepared/`。既有 `data/astt_bbox/`、`output/astt_bbox/`、`output/roi_heatmap/` 及全部權重／影片原地保留，不搬動、不改寫；訓練預設繼續讀現有 astt_bbox_v1，新資料可用 --data 指定。
- 相容性：JSON schema boom.astt_bbox.v1/v2/v3、checkpoint format boom.roi_heatmap.checkpoint.v1、既有模型版本識別不變，保存為序列化相容性欄位。state_dict 參數名稱及運算不變；不因更名要求重訓。原版 benchmark/ASTT 及 artillery 匯入路徑未修改。
- 設定：8 張歷史＋當前、768×576、batch 4、30 epochs、workers 4、camera-aware 架構及 loss 全部不變。只更名與整理，不能宣稱偵測效果改善。
- 驗證：搬移前後 baseline／camera-aware 的全部輸出 torch.equal、state_dict strict 載入通過；實際舊標註載入與新空專案保存／重載通過；真實舊 heatmap checkpoint 前16幀推論的 weighted score 與既有 CSV 差異 <1e-7。新入口使用實際訓練／驗證資料、32×40小尺寸、16窗口、Windows workers=2 完成1輪訓練、保存及自動16幀推論，再續訓至2輪並自動推論通過。這是功能驗證，非正式訓練；所有本次臨時權重／影片／程式位於 tmp 並清理。
- 指令：`python tools/train_smoke_heatmap.py --model-variant camera-aware --epochs 30 --workers 4 --device cuda --infer-source "data/video/1440-1080/0603"`。本次沒有啟動正式30輪；舊實驗的完整路徑仍保留在前述歷史紀錄，新的操作以 SMOKE_HEATMAP.md 為準。

## 2026-10-22 — TensorBoard、文字 log 與單行訓練進度條

- 需求：保留訓練 log，以 TensorBoard 呈現數值，避免終端機每批換行持續拉長。
- 修改前 → 後：保留 metrics.jsonl／progress.json／pipeline.json，新增 run/training.log 及預設 run/tensorboard/ 事件。互動終端機以 tqdm 在同一行顯示 epoch、batch、百分比、平均 loss、速度與 ETA，train／val 階段完成清除暫時條，每輪只留一行摘要；重導向時停用動態條，詳細進度仍寫檔。
- TensorBoard：每 epoch 記錄 train/val 所有有限數值，包括總 loss、預測 MSE、BCE、Dice、motion loss、驗證 raw/weighted/attention AUC、ROI precision/recall/IoU、耗時與學習率；另在訓練首批、約每5秒及末批記錄 train_step loss。None／未定義的 AUC 不寫為0。設定保存為 Text；未加入逐批影像輸出。
- 續訓：每次另建 run，epoch 與 global batch step 延續，保持舊事件不覆寫；每輪 flush，結束／例外關閉 writer／文字 handler。新增 --no-tensorboard 可關閉事件，但保留文字／JSON。
- 舊紀錄：新增 tools/export_smoke_heatmap_tensorboard.py --run <既有run>，驗證 JSONL 後寫獨立 tensorboard_import_<時間>_<ID>，原紀錄與權重不變；只還原已保存的 epoch 指標，無法還原未保存的 batch loss。
- 涉及檔案：tools/train_smoke_heatmap.py、上述 export 工具、pyproject.toml 的 smoke-heatmap 選用依賴、方法／架構／索引／差異與本紀錄。已在本機 boom 環境安裝 TensorBoard 2.21.0；tqdm 已存在。
- 相容性／參數：不改模型、loss、抽樣、尺寸、batch、精度、workers 或30 epochs，不要求因紀錄功能重訓；原 checkpoint 可續訓，舊 JSONL 可匯入。新增依賴為 tqdm／tensorboard；其他環境可執行 python -m pip install -e ".[smoke-heatmap]"。
- 驗證：實際資料小尺寸16窗口完成訓練與續訓，EventAccumulator 讀取事件並核對 JSONL loss、AUC／ROI／學習率 tag、續訓 epoch/global step；舊紀錄匯入與未定義 AUC 排除通過。模擬互動終端機驗證 carriage-return 單行條，非互動輸出只有摘要；--no-tensorboard 模式仍保留文字／JSON。驗證程式、權重與事件位於 tmp，完成後清理；不是正式30 epochs 訓練。
- 執行／輸出：訓練指令不變，事件在 output/smoke_heatmap/training/<run>/tensorboard/；另一終端機執行 `python -m tensorboard.main --logdir output/smoke_heatmap/training --port 6006`，瀏覽 http://localhost:6006。完整說明與舊紀錄指令見 SMOKE_HEATMAP.md。本次未啟動正式訓練或常駐檢視服務。

## 2026-10-27：v3.0 純 Transformer 三分支（實作與流程驗證）

- 需求：保留使用者的正常畫面預測→差異→煙霧目標關注，以及相機背景運動建模，將可學習的 CNN 特徵及解碼改為 Transformer。
- 修改前→後：v2.2 使用 CNN predictor encoder/decoder、CNN motion 與44通道 CNN gate；新增獨立 `transformer` variant，使用線性 patch embedding、交替局部空間／時間 attention、query cross-attention decoder、線性 patch 組圖。motion 使用相鄰影格對 tokens＋全局 motion token。完整模型不含卷積／反卷積，不是上述論文的官方重現。
- 檔案：新增 `src/boom/smoke_heatmap/transformer.py`；更新 `model.py`、`tools/train_smoke_heatmap.py`、`tools/infer_smoke_heatmap.py` 及方法／結構／演進文件。新版本識別 `smoke_heatmap.v3.0-transformer`。
- 設定：history=8、patch_size=16、embedding=128、heads=4、transformer_layers=2（predictor/gate 各自）、window_size=6（交替半視窗偏移）、motion 內部固定80×64／patch8／2層。預設總參數3,587,204。訓練入口仍預設 camera-aware，新版需 `--model-variant transformer`。
- 保留：BBox 弱監督及 normal/ignore 定義、原始 MSE、0.25有界殘差、正常窗口限定預測loss、平衡BCE＋正樣本Dice、0.2背景loss＋0.001運動正則、0.5相機增強、梯度隔離、H=E*(0.05+0.95A)、top-0.1%分數、固定對數顯示及預設關閉guard。
- 權重／重訓：舊 baseline/camera-aware state_dict不變，按舊variant恢復；新增config欄位有相容預設。checkpoint容器格式不變，新variant必須從頭訓練，不能resume舊權重升級。續訓恢復來源patch/layers/window，推論自動建立來源架構。
- 驗證：無卷積檢查、predictor不讀當前幀、H公式、三分支梯度隔離、有限loss/梯度/AdamW更新、ROI全遮蔽不計背景loss、單格點與不整視窗padding前反向、patch8、checkpoint往返數值一致、實際舊baseline/camera-aware best.pt嚴格載入並與原公式輸出逐值一致。
- 實際尺寸：RTX5080／FP32／768×576／8歷史＋當前／batch4，以2個實際窗口各重複一次；完成單次前反向與更新。峰值allocated約7.1256GiB，計算約0.885秒（不含讀取與增強，含首次GPU計算開銷）。不作穩態FPS或與v2.2的公平速度比較。
- 完整流程：實際資料限制train/val各16窗口、64×96、batch2、embedding32、patch8/layers1/window3，完成1輪→另run續訓到第2輪→各自best.pt推論12幀合成影片。確認自訂架構恢復、前8幀暖機CSV空分數、後4幀有效分數、模型版本／motion metadata、12幀MP4可解碼。小樣本AUC不作效能結論。
- 驗證輸出：`tmp/transformer_validation_20261027/`（含驗證程式、合成影片及小訓練權重），完成後全部清理；摘要保留本紀錄及 `doc/SMOKE_HEATMAP_TRANSFORMER.md`。正式輸出仍為 `output/smoke_heatmap/training/<新run>/`，未啟動正式30輪、不覆寫既有資料／模型／輸出。
- 執行：`python tools/train_smoke_heatmap.py --model-variant transformer --epochs 30 --history-frames 8 --height 576 --width 768 --batch-size 4 --workers 4 --device cuda --infer-source "data/video/1440-1080/0603"`。
- 限制：patch與局部視窗可能限制薄煙及大位移；沒有證實煙霧誤報／召回改善，仍需固定資料驗證延遲、端到端FPS及定位。未接入即時runner或GPS。
- 文件狀態校正：實際v2.2 run `smoke_heatmap_20261022_182638_27a266c1/pipeline.json`顯示30輪及推論完成，推論摘要使用第7輪best.pt；更新現況摘要，保留2026-10-22當日「未正式訓練」歷史。

## 2026-10-27：正式命名 ST² v1.0 與檔名整理

- 需求：使用者採用 ST² 名稱，希望寫入 Markdown 並將相關檔名改為易讀名稱。
- 名稱：**Smoke-Targeted Spatiotemporal Transformer（ST²）／煙霧目標導向時空 Transformer**；程式與檔名用 ST2/st2，目前名稱版本 v1.0，對應先前 Smoke Heatmap v3.0-transformer 架構。Boom 為系統，ST²為其中研究模型；尚未查核重名。
- 修改前→後：`src/boom/smoke_heatmap/transformer.py` 主實作移至 `st2.py`，三分支類別改為 `ST2Predictor/ST2Motion/ST2Gate`；舊檔僅保留相容匯入。`doc/SMOKE_HEATMAP_TRANSFORMER.md` 完整內容移至 `doc/ST2.md`，舊頁保留導向。更新文件首頁、專案結構、方法說明與演進紀錄。
- 涉及檔案／參數：`model.py` 依新類別建立分支；新增唯讀model_name/model_display_name，train/infer摘要保存ST2與完整名稱。共用訓練／推論入口仍為train_smoke_heatmap.py／infer_smoke_heatmap.py。架構、loss、標註、分數、門檻、模型參數與預設值均未改。
- 權重相容性：state_dict鍵、`variant=transformer`、`smoke_heatmap.v3.0-transformer`及checkpoint容器格式保留。改名不要求重訓，既有資料、模型及輸出原地保存；原v2.2升級到ST²仍需另開訓練。
- 驗證：以改名前類別定義與相同state_dict核對全部前向輸出逐值一致；新舊匯入同一類別；小尺寸合成batch完成訓練更新，checkpoint→10幀影片推論成功，8幀暖機／2幀評分及ST2摘要名稱正確；實際舊baseline/camera-aware best.pt嚴格載入與前向成功，文件連結檢查通過。
- 驗證輸出：僅本次建立的`tmp/st2_naming_20261027/`，完成後清理。正式輸出路徑維持`output/smoke_heatmap/`；沒有重訓或宣稱效果改善。執行仍用 `python tools/train_smoke_heatmap.py --model-variant transformer --epochs 30 --device cuda`，完整指令見ST2.md。

## 2026-10-27：ST²獨立套件、資料與訓練輸出命名

- 需求：使用者要求astt_bbox改名st2，未來訓練結果也用st2；src下獨立訓練資料夾，Boom保持原有辨識／顯示，以模型介面替換。
- 修改前→後：研究實作從`src/boom/smoke_heatmap/`移至`src/st2/`（無boom依賴），舊目錄只留匯入相容轉接。原st2.py改為architecture.py。訓練／推論／標註／TensorBoard工具主體移入套件，tools新增train_st2.py、infer_st2.py、prepare_st2_dataset.py、export_st2_tensorboard.py薄入口，舊工具轉接。
- 資料／輸出：`data/astt_bbox`→`data/st2`，`prepared/astt_bbox_v1`→`prepared/st2_v1`，`output/astt_bbox`→`output/st2`，均同磁碟rename且目的地不存在。保留45,098資料檔與64歷史輸出檔的清單／大小／時間、67資料描述檔SHA256一致。舊smoke_heatmap/roi_heatmap訓練run不搬動，歷史JSON與權重不改寫。
- 設定：新入口預設transformer，data= data/st2/prepared/st2_v1，output=output/st2/training，run前綴st2_，inference前綴st2_inference_，影片標題顯示ST2。新格式st2.checkpoint.v1、model_version=st2.v1.0、model_name=ST2，新標註st2.annotations.v1；模型運算、loss、分數及權重鍵不改。
- 相容／重訓：新舊heatmap容器與舊標註schema均可讀；唯讀解析已搬移的舊astt_bbox路徑。資料fingerprint與實際v2.2權重一致，32,354train／12,571val不變。同架構權重不需因搬移重訓；從camera-aware升級ST²仍需重新訓練。
- 模型端介面：新增`src/st2/runtime.py`，ST2Runtime接BGR影格回傳暖機／MSE／attention／分數；跳號或尺寸變動重啟暖機。沒有確認事件／GPS／Boom UI；BaseSmokeDetector adapter尚待門檻與事件規則驗證，這次不新增run_realtime.py st2、不改motion/YOLO。
- 驗證：獨立套件不匯入boom、舊模組別名、新舊權重、實際資料讀取／指紋、標註新舊schema／新匯出st2_v1/v2、runtime與直接模型結果一致及重置、真實資料16窗口小訓練→續訓→12幀影片推論的目錄／metadata／格式、原Boom motion factory均通過。
- 輸出：臨時驗證於`tmp/st2_separation_20261027/`，完成後清理；完整搬移及驗證摘要保存`doc/ST2_MIGRATION.md`。正式訓練未啟動，不將小樣本AUC或分數當成已改善煙霧效果。
- 指令：`python tools/train_st2.py --epochs 30 --device cuda --infer-source "data/video/1440-1080/0603"`；設定與輸出詳見ST2.md。新增pip依賴別名`.[st2]`，原別名保留。

## 2026-10-27：移除舊研究程式，只保留現行ST²

- 需求：使用者明確要求「之前的輸出資料不要改，但是程式碼全部都不要了，全部都用現在新的，Boom只留下實作部分」。因此只清理舊研究程式，不搬移或改動任何既有輸出。
- 修改前→後：刪除`src/boom/smoke_heatmap/`全部匯入轉接，以及tools下train_smoke_heatmap.py、infer_smoke_heatmap.py、prepare_smoke_heatmap_dataset.py、export_smoke_heatmap_tensorboard.py。移除ST²套件內的NormalFramePredictor、TargetChangeGate及CNN BackgroundMotion舊網路；只保留目前Transformer三分支。ST2Motion直接持有相同幾何運算，不再繼承舊CNN類別。
- 現行命名：ST2Config／ST2Model／ST2Criterion、ST2Dataset／ST2SequenceDataset、render_st2；移除舊SmokeHeatmap類別名稱及src/st2/paths.py舊路徑映射。工具僅使用train_st2.py、infer_st2.py、prepare_st2_dataset.py、export_st2_tensorboard.py，移除smoke-heatmap安裝別名，更新文件及索引產生器。
- 參數／權重：ST²計算、權重鍵、loss、標註解讀、分數正規化、門檻及預設訓練參數未改；相同ST²權重不需因清理重訓。只接受明確variant=transformer的架構；既有同架構舊容器仍能載入，baseline/camera-aware或缺variant的歷史權重會明確拒絕，不再提供執行支援。
- 輸出保護：output/st2、output/smoke_heatmap、output/roi_heatmap保留原位；186個檔案的相對清單／大小／修改時間與116個JSON／JSONL／CSV雜湊清理前後一致。其他既有資料、模型、來源影片及ASTT benchmark不修改。過去紀錄中的舊入口僅作歷史參考，不代表仍可執行。
- 驗證：以清理前ST²定義建立參考模型，嚴格載入相同state_dict後全部前向輸出逐值一致；無卷積、梯度隔離、有限梯度與optimizer更新通過；runtime暖機／推論及同架構新舊容器可讀，舊架構明確拒絕。
- 流程：實際資料train/val各16窗口、64×96、batch2、embedding32、patch8/layers1/window3，完成第1輪→另run續訓至第2輪→12幀影片推論（8幀暖機、4幀評分），ST²名稱與格式正確；原Boom motion factory建立成功。沒有改motion/YOLO、runner、顯示、事件或定位實作。
- 臨時驗證：只在`tmp/st2_current_only_20261027/`保存參考源碼、測試、權重與影片，完成後清理。正式30輪訓練未啟動，沒有以小樣本指標宣稱煙霧效果改善。
- 指令：`python tools/train_st2.py --epochs 30 --device cuda`；操作見doc/ST2.md。Boom與ST²的未來事件adapter仍未接入，本次不增加即時ST²偵測入口。

## 2026-10-27：ST²按職責整理，全部入口收進套件

- 需求：使用者確認保留Boom名稱，只整理ST²；tools中的ST²工具應收進ST²資料夾，讓目錄容易閱讀。
- 修改前→後：src/st2根目錄的平鋪模組整理成models/、data/、engine/、common/、cli/。模型設定／loss在models/network.py，Transformer在models/transformer.py；data含資料集、標註與editor；engine含訓練、影片推論、runtime、顯示、分數及TensorBoard；common保存共用I/O。
- 入口：移除tools中的train_st2.py、infer_st2.py、prepare_st2_dataset.py、export_st2_tensorboard.py，收進src/st2/cli。新增src/st2/__main__.py統一入口：`python src/st2 train`／`infer`／`prepare`／`export-tensorboard`。單獨CLI腳本與安裝後的`python -m st2`亦可使用。不再保留重複工具轉接。
- 涉及設定：修正搬到子目錄後的專案根目錄解析、模組匯入及新匯出資料README中的指令；同步修改現行文件／索引與索引產生器。訓練參數、資料分組、模型運算、loss、分數、門檻、state_dict鍵與checkpoint格式均不變，同架構權重不需重訓。
- 保留範圍：Boom與src/boom名稱及實作不變，既有data、models、output不搬動、不改寫。186個歷史研究輸出檔清單／大小／時間及116個紀錄雜湊與整理前一致。
- 驗證：整理前模型state_dict嚴格載入新路徑，全部前向輸出逐值一致；runtime暖機及推論、標註匯出與資料窗口、四種CLI help、原Boom motion factory成功。
- 流程驗證：實際train/val各16窗口、64×96、batch2、embedding32、patch8/layers1/window3，統一入口完成訓練1輪→另run續訓至2輪→12幀自動影片推論；另測獨立infer及export-tensorboard。名稱、權重架構恢復與輸出路徑正確，小樣本指標不作煙霧效果結論。
- 臨時產物：參考源碼、驗證程式、模型與影片只放在tmp/st2_layout_20261027，完成後清理。未啟動正式30輪訓練。
- 現行指令：`python src/st2 train --epochs 30 --device cuda`。完整結構見doc/PROJECT_STRUCTURE.md及src/st2/README.md。

## 2026-10-27：根目錄訓練＋推論入口

- 需求：使用者希望在專案根目錄直接執行完整訓練與inference。
- 修改前→後：新增`train_st2.py`，呼叫現有ST²引擎並預設推論來源為`data/video/1440-1080/0603`。engine/train.py的main新增可選argv及default_infer_source參數；套件原命令的預設infer_source仍為None。
- 設定：根目錄入口沿用30輪、batch4、768×576、device=auto、data/st2/prepared/st2_v1；所有CLI可覆寫，包含--infer-source與--resume。訓練／驗證成功並保存權重後才載入best.pt推論，不使用last.pt代替最佳模型。來源無效先失敗，訓練失敗／中斷不進推論。
- 相容／重訓：只增加執行入口與根目錄入口的推論預設，不改模型、loss、訓練參數、標註、分數、門檻、state_dict或checkpoint格式，不需因入口變更重訓。既有資料與輸出不改寫，新run仍在output/st2/training/st2_<時間>_<ID>。
- 驗證：確認root main無參數會傳入預設來源，自訂argv及resume原樣轉交。以實際train/val各16窗口、64×96、batch2、embedding32、patch8/layers1/window3完成1輪→best.pt→12幀合成影片推論，核對8幀暖機與4幀評分；CLI來源覆寫正確，無效來源不建立訓練run。語法及文件連結檢查通過。
- 輸出：驗證程式、權重與影片僅在tmp/st2_root_entry_20261027，完成後清理。未啟動正式30輪，也不以小樣本指標宣稱效果改善。根目錄、模型、設定與索引文件均已更新。
- 指令：`conda activate boom` 後執行 `python train_st2.py`；更換來源加 `--infer-source "data/video/你的影片.mp4"`。

## 2026-10-27：加入torchview模型運算圖

- 需求：使用者明確指定torchview，希望將模型視覺化加入專案。
- 新功能：新增engine/visualize.py與cli/visualize.py，統一入口`python src/st2 visualize`；實際追蹤完整ST²及predictor／motion／gate三分支的forward，顯示張量尺寸與運算連線，輸出SVG、DOT及summary.json。預設depth2、batch1、768×576、meta、四張圖，支援checkpoint、單一分支、CPU與深度／方向設定。
- 環境：本機torchview 0.2.7與Python graphviz已存在，dot在boom環境Library/bin；自動尋找可執行檔，只暫時設定本程序PATH。新增可選依賴`.[st2-visualization]`，未安裝／更動本機套件。SVG使用內建字型度量以避免Linux字型名稱造成的跨平台裁切；不提供平台點陣圖渲染。
- 模型／參數：不修改架構、loss、標註、訓練設定、門檻或推論分數；使用獨立模型，隨機初始化只用於無checkpoint時的架構展示。追蹤期間暫關融合MHA fastpath，完成／例外均恢復RNG與原設定。同架構權重無需重訓。
- 驗證：完整預設架構四分支meta追蹤與SVG/DOT成功；自訂history3／patch8／embedding32的checkpoint嚴格載入、參數量與輸入尺寸正確；CPU DOT、重複匯出不覆蓋、缺dot保留DOT且partial、模擬追蹤失敗為failed、RNG／MHA／PATH恢復、權重SHA256不變均通過。這不是反向梯度圖、attention權重或煙霧效果評估。
- 產物：正式本次新圖在`output/st2/visualization/st2_torchview_20261027_152849_adf6abb7/`，四張SVG文件快照在`doc/diagrams/`，操作見`doc/ST2_VISUALIZATION.md`。臨時驗證與點陣圖檢查僅在tmp/st2_torchview_20261027，完成後清理。既有資料、模型與輸出不改動，未啟動訓練。
- 指令：`python src/st2 visualize`；例如 `--checkpoint "output/st2/training/你的run/best.pt" --part model --depth 3`。

## 2026-10-27：ST²資料集移除prepared外層並推送全部更新

- 需求：使用者要求將st2_v1移出prepared，全部程式與文件推送，但資料只保留資料夾，不上傳內容。
- 修改前→後：`data/st2/prepared/st2_v1`以同磁碟rename搬至`data/st2/st2_v1`；移走空的.gitkeep後，prepared僅在確認空目錄時移除。訓練預設、標註匯出父目錄、根目錄入口與現行文件改為平坦路徑，內部常數PREPARED_ROOT更名DATASET_ROOT。新的匯出為data/st2/st2_v2等，不再建立prepared。
- 資料完整性：45,097個原資料集檔案的相對清單、大小、修改時間搬移前後一致；67個JSON／JSONL／CSV／NPY雜湊一致。資料標註與來源不改寫，dataset fingerprint與既有v2.2權重一致。
- Git範圍：data僅加入data/st2、st2_v1及train／val的零位元.gitkeep；不提交影像、標註、描述檔、資料集README或來源ID資料夾。權重與實驗輸出維持忽略。本次依「全部」授權一併保存前次torchview與既存ASTT路徑／文件修改；dev推送，不更新master。
- 參數與相容性：只改資料根目錄及匯出位置，模型計算、loss、標註解讀、抽樣、分數、門檻及權重鍵不變，同架構權重不需重訓。續訓使用新的預設資料位置；歷史JSON路徑保留原紀錄，不自動改寫。
- 驗證：新預設路徑、32,354train／12,571val窗口、資料指紋、真實正負窗口前向／反向及optimizer更新、平坦st2_v1/v2版本匯出、零位元標記均通過。ASTT官方18檔快照校驗與train/test入口help通過。
- 臨時產物：tmp/st2_flat_data_20261027下完成搬移清單、檢查、合成影片及匯出，完成後清理；既有輸出不改動，未啟動正式訓練。現行指令仍為`python train_st2.py`。

## 後續紀錄格式

```text
日期／版本：
需求／觀察：
修改前 → 修改後：
涉及檔案與設定：
資料／checkpoint 相容性、是否需重訓：
執行指令與輸出位置：
驗證方法、實測結果：
限制、未完成及下一步：
```


## 2026-10-27 — ST² 日期時間實驗目錄與最終權重

- 需求：輸出集中在 st2，以日期時間分次；training／inference 分開，訓練權重、log、TensorBoard 與基本資訊集中保存。
- 修改前：`output/st2/training/st2_<時間>_<ID>/` 混合訓練檔案與 inference 子目錄，只保存 best.pt／last.pt。
- 修改後：`output/st2/YYYYMMDD_HHMMSS_微秒/training/` 保存 best.pt、每輪更新的 last.pt、全部指定輪次完成後的 final.pt、training.log、tensorboard、config.json、metrics.jsonl、progress.json、pipeline.json，以及新增 summary.json。推論放在同層 inference，保留原有子目錄、影片、CSV 與摘要格式。`--output` 表示日期時間目錄的父目錄。獨立 infer 命令預設與內容不變。
- 檔案／參數：`src/st2/engine/train.py`、README、ST²套件README、PROJECT_STRUCTURE、ST2、ST2_VISUALIZATION、TOOLS、MODEL_EVOLUTION；預設output由output/st2/training改為output/st2，其他模型與訓練參數不變。
- 權重相容性／重訓：模型架構、loss、標註、分數、門檻、state_dict與checkpoint格式不變，既有ST²權重仍可載入／續訓，不需因此重訓。final.pt與完成當下last.pt的模型張量一致；保留最佳best.pt供自動推論。訓練未完成時沒有final.pt，推論失敗則保留已完成訓練的final.pt。既有資料、模型及輸出不搬移、不覆寫。
- 指令：`python train_st2.py`；續訓 `python train_st2.py --resume "output/st2/你的日期時間/training/last.pt" --epochs 30`；TensorBoard `tensorboard --logdir output/st2`。
- 驗證：實際資料train／val各16窗口、64×96、batch2、embedding32、patch8、layers1、window3、CPU完成1輪；核對三個權重、final／last逐張量一致、TensorBoard event、設定／log／指標／摘要與同層inference影片／CSV。以best.pt自動推論真實影片10幀後，從last.pt另開日期時間目錄續訓至第2輪，確認原權重內容不變、無推論來源的inference留空；根目錄--help成功。
- 驗證輸出：僅使用 `tmp/st2_layout_20261027_check/`，驗證後清理；不啟動正式訓練，不據此宣稱辨識效果改善。

## 2026-10-27 — 預訓練模型目錄路徑對齊

- 需求：核對根目錄 models 新增子資料夾後的模型路徑。原 EDM 預設指向 pretrained 根目錄、YOLO 訓練工具下載位置也在根目錄；現分別指向 `models/pretrained/map_matching/` 與 `models/pretrained/detector/`，YOLO 遞迴搜尋既有權重維持不變。修正 YOLO 匯出函式獨立使用時的預設 checkpoints 路徑為專案根目錄下 `models/checkpoints/`。
- 檔案／參數：`configs/matching.yaml`、`src/boom/config/defaults.py`、`tools/train_yolo.py` 與相關文件；兩個新目錄各加入 `.gitkeep`。沒有改動模型、loss、偵測參數、權重內容或格式，不需重訓；既有自訂路徑可繼續指定。
- 驗證：讀取合併設定並檢查實際檔案，使用 CPU ONNX Runtime 載入 EDM 與預設 YOLO ONNX，確認 YOLO 訓練工具能找到 detector 子目錄的權重；未執行完整訓練或影片偵測。原模型與輸出保留原地，模型檔仍受 Git 忽略。

## 2026-10-28 — 訓練後推論移除多餘目錄

- 需求：訓練後的 `inference/` 直接放各影片結果，不再多一層 `st2_inference_<時間>_<ID>/`。
- 修改前後：自動推論各影片資料夾改為 `output/st2/<日期時間>/inference/001_<影片名稱>/` 等，權重快照、推論摘要與進度也移到 inference 根目錄；訓練 pipeline 記錄該實際路徑。直接輸出時若目錄非空會拒絕執行，避免覆蓋。獨立推論仍建立獨立的時間／ID 目錄。
- 檔案／參數：`src/st2/engine/infer.py` 新增內部 `direct_output` 選項、`src/st2/engine/train.py` 啟用，以及 `doc/ST2.md`、`doc/PROJECT_STRUCTURE.md`、`doc/MODEL_EVOLUTION.md`。模型、loss、訓練參數、分數、顯示、checkpoint 格式與影片輸出內容不變；既有權重相容，無須重訓，舊輸出不移動。
- 驗證：以實際 train／val 各 16 窗口、CPU、64×96、batch 2 完成一輪，接著對真實影片推論 10 幀，核對影片目錄直接位於 inference、MP4／CSV、摘要／權重快照及 pipeline 路徑；確認非空目錄被拒絕且內容未變、獨立推論維持原有子目錄。驗證輸出位於 `tmp/st2_direct_check_20261028/`，完成後清理；不據此宣稱辨識效果改善。
- 指令：`python train_st2.py`。
