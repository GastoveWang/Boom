# Boom 問題盤點與可套用論文（2026-10-26）

> 研究快照：內容代表2026-10-26的文獻與專案查核狀態，不代表目前模型或程式狀態。目前操作請見 [ST²](../ST2.md) 與 [文件首頁](../README.md)。

本次只做程式、文件、既有訓練紀錄與文獻的閱讀核對；沒有改模型、重訓、安裝套件、執行外部程式或改寫既有輸出。沒有逐片重新觀看全部推論影片，因此文中區分「先前回報的現象」「紀錄中的量測」與「架構上的推論」。

## 1. 搜尋範圍與查核標準

- 以 ASTT 的下一幀預測／誤差定位為背景，聚焦移動相機的背景運動、時序融合、煙霧弱監督分割與機載部署。
- 先搜尋 2023-10-26 至 2026-10-26，另擴至 2021-10-26，納入直接對應目前問題的較早期刊工作。會議論文另列工程背景，不算 JCR 期刊推薦。
- 已使用 Consensus 尋找候選，再回作者實作、原文與出版社核對。Consensus 的年份有 online-first／正式卷期混用，不能直接照抄。
- JCR：透過使用者提供的網址，在現有 Chrome 瀏覽器讀取 Clarivate 正式期刊頁；採 **2026 發布版本、2025 指標資料年**，頁面資料更新日期為 **2026-10-14**，查詢日期 **2026-10-26**。頁面下拉選單的 `JCR Year 2025` 是指標資料年。
- [Clarivate 2026 發布公告](https://ir.clarivate.com/news-events/press-releases/news-details/2026/Clarivate-Releases-Journal-Citation-Reports-2026/default.aspx)確認此版本使用 2025 資料。本報告不以 SJR、CiteScore 或 JCI 分區代替 JIF 分區。
- 正式推薦須同時具備：主題相關、直接 JCR Q1 證據、實質程式及明確開源授權。目前確認 **1 篇**；已有符合條件的 Q1，因此不混入 Q2。

## 2. 專案目前真正的狀態

### 文件比實際訓練落後

`MODEL_EVOLUTION.md`、`SMOKE_HEATMAP.md` 仍記載 camera-aware 未正式重訓；但以下 run 已完成 30 epochs，並完成 14 支影片推論：

`output/smoke_heatmap/training/smoke_heatmap_20261022_182638_27a266c1/`

核對檔案：`config.json`、`pipeline.json`、`metrics.jsonl`，以及 `inference/heatmap_20261023_152847_1156b794/summary.json`。正式設定為 8 張歷史＋當前、768×576、batch 4、camera-aware、相機增強 0.5、motion loss weight 0.2。最佳權重選在第 7 輪：

| 同一最佳輪的驗證指標 | 數值 |
|---|---:|
| 原始誤差 frame ROC AUC | 0.609390 |
| 加權誤差 frame ROC AUC | 0.763189 |
| attention frame ROC AUC | 0.793901 |
| attention≥0.5 對框聯集的 pixel IoU | 0.043588 |
| 相同定義的 pixel precision | 0.049430 |
| 相同定義的 pixel recall | 0.269454 |

這些不是「煙霧準確率」。ROI 真值是矩形聯集，並非精細煙霧像素；frame label 只表示有無關注框。現有資料只有 train/val，沒有獨立 test。相鄰窗口高度重疊，窗口數不等於獨立事件數。

### 最近問題與程式上的可能原因

1. **轉鏡／背景變化反應**：歷史回報包括 DJI_007_V 約 9.24 秒、DJI_009_V 約 17 秒之後；見 `MODEL_CHANGELOG.md`。不能把舊回報直接當成新版仍失敗的證明，需要重看新版相同時段。
2. **運動模型表達有限**：`motion.py::BackgroundMotion` 只學每對影格的平移、旋轉、縮放，共 4 參數；沒有稠密光流、深度或分層運動。山林視差、近處樹葉和遠處背景不一定符合單一變換。
3. **predictor 的時序特徵未對齊**：`model.py::NormalFramePredictor` 對同一空間格點沿時間融合，且沒有用 motion 分支先對齊歷史特徵。相機移動時，同一格點可能已是不同場景內容。這是程式閱讀推論，尚未經消融確認其影響大小。
4. **最終分數受原始誤差限制**：`H = E × (0.05 + 0.95A)`。即使 A 接近 0，仍保留約 5% 的 E；反過來，薄煙或緩慢目標若 E 很小，A 很高也不能把 H 拉高。最佳輪 attention AUC 高於 weighted AUC，值得先比較兩者；不能僅憑 AUC 差異認定煙霧事件效果更好。
5. **框被當成完整前景**：`dataset.py::boxes_to_mask` 把矩形內全部設為 1，`SmokeHeatmapCriterion` 再做像素 BCE/Dice。適合學「關注區域」，若要學煙霧實際輪廓，框內背景會形成有雜訊的監督。
6. **語意與評估缺口**：目前標註格式是無物種類別的 ROI，不能由名稱 Smoke Heatmap 判定模型已學會煙霧語意；需要核對正樣本內容。區段外預設為無目標，也需要確認未漏標。事件誤報／小時、首次偵測延遲、完整推論 FPS 與定位誤差尚不能由上述 AUC 推出。

## 3. 通過三條件的 Q1 期刊論文

| 論文／年份／DOI | 主題相關性 | 期刊／ISSN | JCR 年度／類別／JIF 分區及證據 | 程式碼／來源／授權 | 核心方法與可重現限制 |
|---|---|---|---|---|---|
| **EM-Driven Unsupervised Learning for Efficient Motion Segmentation**；Meunier、Badoual、Bouthemy；正式卷期 2023-04，45(4):4462–4473；2022 online-first／預印本。[DOI](https://doi.org/10.1109/TPAMI.2022.3198480)、[作者全文](https://arxiv.org/html/2201.02074v3) | 最貼近「共同背景運動與局部變化如何分開」；可當目前四參數運動分支的研究對照 | IEEE TPAMI；0162-8828／1939-3539 | 2026 版／2025 資料：Artificial Intelligence **Q1，4/210**；Electrical & Electronic Engineering **Q1，4/370**。依電腦視覺主題採 AI 類別。[直接 JCR 頁](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=IEEE%20T%20PATTERN%20ANAL&year=2025&fromPage=%2Fjcr%2Fsearch-results) | [作者實作](https://github.com/Etienne-Meunier-Inria/EM-Flow-Segmentation)，AGPL-3.0；含訓練、推論、模型及 coherence loss，非空倉庫 | 用光流及分段 affine／quadratic 運動模型設計 EM 訓練；不是煙霧語意模型。README 提供 `.flo`、CSV split、外部權重入口；舊環境 Python 3.7.7／torch 1.8.1／Lightning 1.2.8，需獨立評估與現有環境的相容性。光流估算成本必須計入端到端速度。 |

已閱讀作者 HTML 全文的問題設定與方法，並核對實作目錄、模型程式、README 的資料與環境要求；沒有下載權重或驗證其實驗可重現。AGPL-3.0 為儲存庫明示授權，整合時須保留並遵循其條件。它提供運動分組，不能把獨立運動直接等同煙霧，也不保證完全排除視差。

Consensus 對應記錄 [1]：[EM-Driven Unsupervised Learning for Efficient Motion Segmentation](https://consensus.app/papers/emdriven-unsupervised-learning-for-efficient-motion-meunier-badoual/500f721a238b519aa353753b2b1bef39/?utm_source=chatgpt)，作者 E. Meunier、A. Badoual、P. Bouthemy；TPAMI；查詢時引用數 53。其紀錄年份 2022 與正式卷期 2023 需區分。

## 4. 相關 Q1 期刊候選：程式／授權未過關，不能列為正式推薦

以下期刊分區皆已在 JCR 直接核對；「期刊是 Q1」不代表下列程式已符合開源及重現條件。

| 論文／DOI | 正式年份、期刊／ISSN | 可參考的部分 | 程式核對結果／排除原因 |
|---|---|---|---|
| [Learning Motion and Temporal Cues for Unsupervised Video Object Segmentation（MTNet）](https://doi.org/10.1109/TNNLS.2024.3418980) | 2024 online-first；2025-05 正式卷期；TNNLS，2162-237X／2162-2388 | 外觀與運動特徵融合、跨影格 temporal transformer；比直接堆疊 RGB 更有結構 | [作者程式](https://github.com/hy0523/MTNet)有 train/test、資料與權重入口，但未找到專案層明確開源授權。Python 3.8.15／torch 1.10；預訓練及資料另取。不能直接標成「已確認開源」。 |
| [Prototype-based scatter learning for smoke segmentation（PSL）](https://doi.org/10.1016/j.patcog.2025.112605) | 正式卷期 2026-04；Pattern Recognition，0031-3203／1873-5142 | 用多原型與特徵判別改善煙霧外觀表達，適合研究煙霧／背景相似 | [作者程式](https://github.com/LujianYao/psl)有核心實作、訓練設定與資料入口；未找到專案層明確授權。README 要求 Linux、Python 3.8、torch 1.8、MMCV；不是現有 BBox 的直接替代 loss。 |
| [EGNL-FAT: An Edge-Guided Non-Local network with Frequency-Aware transformer for smoke segmentation](https://doi.org/10.1016/j.eswa.2025.127621) | 出版社頁日期 2025-06-25；ESWA，0957-4174／1873-6793 | 煙霧邊界、頻域與外觀特徵，FSS 森林場景資料可作候選資料來源 | [作者程式](https://github.com/yitccc/smoke-segmentation/tree/master)有 lib、train/test 及 FSS 目錄；未找到專案層明確授權，安裝資訊亦有限。不是移動相機分離方案。 |
| [BoXFire: Geometry-aware box-supervised fusion for accurate fire and smoke segmentation](https://doi.org/10.1016/j.aei.2026.104671) | 正式卷期 2026-09；Advanced Engineering Informatics，1474-0346／1873-5320 | 主題很合適：只有框標註的火煙分割、box-to-mask 生成 | [指定作者倉庫](https://github.com/ayazkhan789/BoxFire)實際頂層僅 README 與圖片。雖 README 聲明 Apache-2.0 並稱將發布程式，未取得實質對應實作，安裝指令還指向別的專案；不能列成可直接套用。 |
| [DenSiSeg](https://doi.org/10.1016/j.patcog.2025.112517) | 正式卷期 2026-04；Pattern Recognition | 半透明煙霧密度、背景原型與軟對比學習 | [作者倉庫](https://github.com/Caoyichao/DenSiSeg)目前只有 README／結果圖片，仍寫 code coming soon；排除。 |

MTNet 已讀作者 HTML 全文的方法及限制；PSL、EGNL-FAT、BoXFire、DenSiSeg 主要取得出版社摘要／預覽及作者倉庫，不能視為完整全文評讀。此處不把各文的資料集成績外推為 Boom 可達效果。

Consensus 對應記錄 [2]：[Learning Motion and Temporal Cues for Unsupervised Video Object Segmentation](https://consensus.app/papers/learning-motion-and-temporal-cues-for-unsupervised-video-zhuge-gu/4facdabd7011593f862a4d34c7ae9499/?utm_source=chatgpt)，Zhuge、Gu、Zhang、Qi、Lu；TNNLS；查詢時引用數 18。注意其 UVOS 的「unsupervised」主要指推論無人工初始提示，不等同完全不使用 mask 訓練；原文 loss 使用訓練 mask。

### 直接 JCR 查核明細

下表全部為 2026 發布版本的 2025 JIF 資料，查詢 2026-10-26。與本研究最直接相關者採 COMPUTER SCIENCE, ARTIFICIAL INTELLIGENCE，不選不相關類別充作依據。

| 期刊／JIF | 取得的全部學科類別：排名、JIF 分區 | 直接來源 |
|---|---|---|
| TPAMI／20.4 | AI 4/210 Q1；Electrical & Electronic Engineering 4/370 Q1 | [JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=IEEE%20T%20PATTERN%20ANAL&year=2025&fromPage=%2Fjcr%2Fsearch-results) |
| TNNLS／9.7 | AI 25/210 Q1；Hardware & Architecture 2/60 Q1；Theory & Methods 8/146 Q1；Electrical & Electronic Engineering 22/370 Q1 | [JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=IEEE%20T%20NEUR%20NET%20LEAR&year=2025&fromPage=%2Fjcr%2Fsearch-results) |
| Pattern Recognition／9.1 | AI 29/210 Q1；Electrical & Electronic Engineering 26/370 Q1 | [JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=PATTERN%20RECOGN&year=2025&fromPage=/jcr/search-results) |
| ESWA／9.4 | AI 28/210 Q1；Electrical & Electronic Engineering 23/370 Q1；Operations Research & Management Science 6/109 Q1 | [JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=EXPERT%20SYST%20APPL&year=2025&fromPage=/jcr/search-results) |
| Advanced Engineering Informatics／11.5 | AI 17/210 Q1；Engineering, Multidisciplinary 4/179 Q1 | [JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=ADV%20ENG%20INFORM&year=2025&fromPage=/jcr/search-results) |

## 5. 工程參考：與專案很合適，但屬會議，不算 JCR 期刊推薦

### 優先閱讀：Detecting Wildfires on UAVs with Real-Time Segmentation Trained by Larger Teacher Models

Pesonen 等，WACV 2025，5166–5176；2024-08 預印本。[正式論文](https://openaccess.thecvf.com/content/WACV2025/papers/Pesonen_Detecting_Wildfires_on_UAVs_with_Real-Time_Segmentation_Trained_by_Larger_WACV_2025_paper.pdf)、[作者 HTML 全文](https://arxiv.org/html/2408.10843v3)、[官方實作](https://gitlab.com/fgi_nls/public/wildfire-real-time-segmentation)、[MIT LICENSE](https://gitlab.com/fgi_nls/public/wildfire-real-time-segmentation/-/raw/main/LICENSE)。

把框交給 SAM 等教師產生偽 mask，再訓練輕量 PIDNet。與 Boom 的現有框、煙霧區域顯示、有限機載算力直接相合。作者報告約 25 FPS（Jetson Orin NX）及 63.3% mIoU，是指定實驗條件下的結果，不是 Boom 的速度承諾。它也沒有替 Boom 解決相機姿態下的煙霧源定位。

實際倉庫有 `generate_pseudo_labels.py`、`train.py`、`eval.py`、`infer.py` 與 models/datasets；README 提供資料與 checkpoint 入口，Python 3.8 及 requirements，另需 SAM ViT-H 權重。指令入口包括 `python generate_pseudo_labels.py`、`python train.py --data-dir custom_data`。本次僅閱讀，未執行。移植時優先借用 box→pseudo-mask→student 的訓練方案，保留現有 predictor 當診斷對照，不必一開始整套換掉。

配套 [Boreal Forest Fire 資料論文](https://doi.org/10.1038/s41597-025-05634-0)（Scientific Data, 2025）值得作外部資料候選：有 UAV 影像、影片和分割資料；並非所有 mask 都是純人工逐像素真值。此資料論文的 JCR 尚未在本次核對，不列入正式 Q1 清單。

### 框弱監督參考：Mask-Free Video Instance Segmentation

Ke 等，CVPR 2023。[作者全文](https://arxiv.org/html/2303.15904v1)、[官方實作](https://github.com/SysCV/MaskFreeVis)，儲存庫明示 Apache-2.0。

重點是框投影／像素關係與跨影格 TK-Loss，而非把框內每個像素都當真正前景。已讀 HTML 方法。可借用訓練約束的想法，但完整實例模型用序列框配對，Boom 沒有 track ID，不能宣稱直接相容；煙霧半透明、快速變形也可能破壞 patch 對應。作者程式依賴 Detectron2、torch 1.9 與額外 CUDA 編譯，提供資料準備與模型下載，完整重現成本較高。

### 煙霧語意基準：FoSp: Focus and Separation Network for Early Smoke Segmentation

Yao 等，AAAI 2024，38(7):6621–6629。[正式論文與 DOI](https://doi.org/10.1609/aaai.v38i7.28484)、[作者實作](https://github.com/LujianYao/FoSp)、[MIT LICENSE](https://github.com/LujianYao/FoSp/blob/main/LICENSE)。

以多尺度聚焦與煙霧／背景分離處理小而透明的早期煙霧，可作單幀煙霧語意基準。實作含 mmseg、config、train/test 與資料入口；設定有外部 SegFormer 預訓練 checkpoint，不能把它當成完全不需外部權重。只有現有框不足以直接重現原本的像素分割監督。本次讀正式摘要及程式設定，未完整核對所有預訓練資產可下載性。

## 6. 建議如何套用：先確認瓶頸，再做最小消融

以下是針對 Boom 的研究建議，不是已驗證改善。暫定方向：**面向移動無人機的背景運動分離與框弱監督煙霧時序分割**。

1. **先用既有最佳權重，對照 E、A、H 三種訊號。** 固定相同驗證片段，單獨評估 attention、原始誤差及現行乘積的事件誤報與召回；先確認瓶頸是否真在乘法評分。A 是 ROI 關注輸出，不能直接稱煙霧機率。
2. **以 EM-driven 作運動對照。** 先比較目前單一 similarity 模型與光流分組，再決定是否需要稠密／分層背景特徵；維持煙霧語意分支，不能以運動區域直接當煙霧。光流使用相鄰已到達影格，避免引入未來影格。
3. **以 WACV 2025 的方案修正監督。** 僅在確認為煙霧的既有框上產生偽 mask，保留原框與版本。抽取部分人工像素真值檢查薄煙漏標；雲、霧、粉塵、葉片與轉鏡片段作明確干擾資料。SAM 輸出不能自動升格為真值。
4. **再做對齊後的時序一致性。** 參考 MTNet 的融合概念與 MaskFreeVIS 的時間約束；使用過去到當前的因果窗口，並對遮擋／錯誤匹配降低約束。先做語意 mask，保留 raw MSE 作診斷與輔助；每次只改一項。

建議實驗組：現行 v2.2、現行 attention-only 評分、背景分離、偽 mask 監督、背景分離＋偽 mask。固定來源群組切分及獨立測試；不得讓同一飛行的近鄰片段跨組。

必測三類：相機穩定、有相機運動且無煙、有相機運動且有煙；另標雲／霧／粉塵／葉片／模糊。至少報告每小時誤報事件數、固定誤報率下召回、首次偵測延遲、人工像素真值 IoU、端到端 FPS／顯存；有地理真值才報定位誤差。運動補償失敗時輸出候選／不可用與信心，不以零分冒充無煙。

資源：現有 RTX 5080 可作本地實驗資源；教師產生標籤可離線做，小模型才放機載。光流與教師模型額外資源尚未量測，不估算訓練時數。難度以偽 mask 支線中等、運動分組及因果時間融合中高為合理起點。

新穎性補查已找到 BoXFire 的框監督火煙分割，以及 WACV 2025 的框引導教師蒸餾，所以「使用框」或「加 SAM」本身不能當作新穎貢獻。可檢驗的研究價值應落在移動相機、煙霧／干擾辨別、因果時序與機載成本的共同評估；仍需進一步系統檢索，不能保證尚無人研究。

最值得先讀：期刊基礎選 **EM-driven**；工程方案選 **WACV 2025 UAV smoke segmentation**。MTNet 適合先學融合設計，取得授權證據前不直接複製整合其程式。
