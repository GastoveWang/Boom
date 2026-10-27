# Boom：Transformer 論文補充篩選（2026-10-26）

> 研究快照：內容代表2026-10-26的文獻與專案查核狀態，不代表目前模型或程式狀態。目前操作請見 [ST²](../ST2.md) 與 [文件首頁](../README.md)。

本次延續 find-paper 條件，搜尋近三年（2023-10-26～2026-10-26），並對關鍵工作擴查近五年。目標是延續 ASTT／Smoke Heatmap，處理相機移動、煙霧／背景相似、時序定位與機載成本。不是只要標題含 attention 就算 Transformer；下列方法都實際使用 Transformer 模組，但多數仍與 CNN 混合。

已透過 Consensus 找候選，回出版社、作者原文及實際程式倉庫核對。本次只做閱讀查核，沒有下載或執行外部程式、安裝套件、改模型或重訓。

JCR 採 **2026 發布版本／2025 JIF 資料年**，資料更新日 2026-10-14、查詢日 2026-10-26。沿用本日直接核對的 TPAMI、TNNLS、ESWA，並透過使用者的 Chrome JCR 頁新增核對 TII、IoT Journal 與 Applied Intelligence。分區來自 **Rank by Journal Impact Factor**，不是 JCI 或 SJR。

## 1. 有 Q1 證據及明確開源對應實作的期刊工作

| 論文／年份／DOI | 主題相關性 | 期刊／ISSN | JCR 年度／類別／JIF 分區及證據 | 程式碼／來源／授權 | 核心方法與可重現限制 |
|---|---|---|---|---|---|
| **Scalable Video Object Segmentation With Identification Mechanism**；Yang 等；2024，46(9):6247–6262；預印本始於 2022。[DOI](https://doi.org/10.1109/TPAMI.2024.3383592)、[作者全文](https://arxiv.org/html/2203.11442v6) | 跨影格保留區域、減少 mask 閃爍，研究長短期記憶與精度／速度取捨；並非煙霧偵測器 | TPAMI；0162-8828／1939-3539 | 2026 版、2025 資料：Artificial Intelligence **Q1，4/210**；Electrical & Electronic Engineering **Q1，4/370**。採 AI 類別。[JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=IEEE%20T%20PATTERN%20ANAL&year=2025&fromPage=%2Fjcr%2Fsearch-results) | 論文指定的 [AOT-Benchmark](https://github.com/yoxu515/aot-benchmark)，[BSD-3-Clause](https://github.com/yoxu515/aot-benchmark/blob/main/LICENSE)；有 networks、train/eval、模型入口及 Transformer 實作 | 論文包括 AOT 與可伸縮 AOST。已確認文中 AOT 系列的實質開源實作；**尚未逐一確認 AOST 新增功能及各權重完整重現**。半監督 VOS 需要初始 mask，必須由自動煙霧候選產生，不能直接替代從零偵測。 |

已讀作者 HTML 全文的問題設定、LSTT／S-LSTT 方法與限制。原文使用長短期 Transformer attention 和物體 ID 表達，不是僅有 CNN 注意力。對 Boom 可先研究「當前區域查詢過去記憶」；沿用語意煙霧分支產生候選，避免把任意物體追蹤當成煙霧辨識。

程式環境入口：Python 3、torch≥1.7、torchvision、OpenCV、Pillow，以及額外 Pytorch Correlation extension；README 提供 `train_eval.sh`、`inference.sh`、MODEL_ZOO、DAVIS／YouTube-VOS 資料準備。部分自訂資料說明仍標待補。硬體速度不可直接外推至 Boom；外部預訓練與資料各有授權及取得條件。本次沒有測試下載或執行。

## 2. Q1 期刊閱讀候選：架構相關，但授權尚未通過

以下四篇中的前兩篇為本次新增；MTNet、EGNL-FAT 已在前份報告提過，這次明確整理其 Transformer 用途。候選不能混成「已確認開源 Q1 推薦」。

| 論文／年份／DOI | Transformer 放在哪裡／對 Boom 的用途 | 期刊與 JCR | 作者程式與限制 |
|---|---|---|---|
| **SmokeSeger: A Transformer-CNN Coupled Model for Urban Scene Smoke Segmentation**；Jing、Meng、Hou；2023 線上、2024 正式卷期 20(2):1385–1396。[DOI](https://doi.org/10.1109/TII.2023.3271441) | Transformer 分支看全局，CNN 分支保留局部細節。適合作煙霧語意分割基準，減少純誤差圖對背景變化的依賴；屬單幀方法，沒有直接解決相機運動 | TII；1551-3203／1941-0050。2025 JIF：Computer Science, Interdisciplinary Applications **Q1，19/185**；其餘類別見下表 | [官方倉庫](https://github.com/VisAcademic/SmokeSeger)有 train/predict/val、PaddleSeg 程式及設定，**未找到專案層明確開源授權**。PaddleSeg 2.4.2，資料與權重由百度網盤提供；與 Boom 的 PyTorch 不直接相容。城市煙霧／排氣不等於荒野煙霧域。 |
| **Real-Time Smoke Detection With Split Top-K Transformer and Adaptive Dark Channel Prior in Foggy Environments（STKformer）**；Yu 等；2024 線上、2025 正式卷期 12:6943–6960。[DOI](https://doi.org/10.1109/JIOT.2024.3492347) | Split top-k attention 選擇有用關聯，結合可學暗通道先驗處理煙／霧混淆。可參考 ROI 語意分支，**其提供流程以分類為主，不能當成現成像素分割器** | IoT Journal；2327-4662。2025 JIF：Electrical & Electronic Engineering **Q1，30/370**；其餘類別見下表 | [官方倉庫](https://github.com/Jiongze-Yu/STKformer)有 models/main/evaluate、資料及權重入口，**未找到專案層明確開源授權**。README 含 ImageNet 預訓練、煙霧微調及 TensorRT；作者速度數字並非 Boom 端到端速度。 |
| **Learning Motion and Temporal Cues for Unsupervised Video Object Segmentation（MTNet）**；Zhuge 等；2024 線上、2025 正式卷期。[DOI](https://doi.org/10.1109/TNNLS.2024.3418980)、[作者全文](https://arxiv.org/html/2501.07806v1) | 外觀／光流融合＋Mixed Temporal Transformer。適合借鑑目前 gate 的時序融合；不是煙霧專用，也不等同無標註訓練 | TNNLS；2162-237X／2162-2388。AI **Q1，25/210**；其餘見下表 | [官方倉庫](https://github.com/hy0523/MTNet)含 train/test 與權重、資料入口，**未找到專案層明確開源授權**。Python 3.8.15／torch 1.10；原訓練使用 mask，不能直接套現有 BBox 當像素真值。 |
| **EGNL-FAT: An Edge-Guided Non-Local network with Frequency-Aware transformer for smoke segmentation**；Fu 等；2025。[DOI](https://doi.org/10.1016/j.eswa.2025.127621) | 頻域 Transformer 配合邊緣特徵與跨域融合，適合煙霧邊界與森林外觀支線；不是時序或相機補償模型 | ESWA；0957-4174／1873-6793。AI **Q1，28/210**；其餘見下表 | [官方倉庫](https://github.com/yitccc/smoke-segmentation/tree/master)有 lib、train/test、FSS 資料入口，**未找到專案層明確開源授權**；安裝資訊有限，需額外檢查依賴及像素標註需求。 |

閱讀深度：SmokeSeger 主要為摘要與作者 README；STKformer 為作者摘要、架構說明及訓練／評估入口；MTNet 為可取得的作者 HTML 全文；EGNL-FAT 為出版社摘要／預覽與程式。未讀到的消融或細節不補寫成事實。主幹套件的授權不能自動視為作者新增模型的授權。

## 3. 很接近 ASTT，但目前排除於正式推薦：HSTforU

**HSTforU: anomaly detection in aerial and ground-based videos with hierarchical spatio-temporal transformer for U-net**；Le、Jin、Kim；Applied Intelligence 55, 261，正式刊登 **2025-01-03**。[出版社全文／DOI](https://doi.org/10.1007/s10489-024-06042-4)、[作者程式](https://github.com/vt-le/HSTforU)。

它以多階段 Pyramid Transformer 產生多尺度特徵，接階層時空 Transformer，再以 CNN decoder 預測下一幀。與 ASTT 的「正常影片訓練→未來影格→誤差找異常」很接近，且評估包含 Drone-Anomaly。可以閱讀其多尺度 skip connections 與時空注意力設計。

**排除原因**：直接查得 Applied Intelligence 的 2025 JIF AI 類別為 **Q2，102/210**，不是 Q1；本次已有 Q1 工作，因此不混入正式 Q1 推薦。作者倉庫有模型、訓練及權重入口，但亦未找到專案層明確授權。論文的 CC BY-NC-ND 授權不等同程式開源授權。

已讀出版社 HTML 方法。值得注意：它的 temporal attention 仍對同一空間位置做跨幀關聯，不能因有 aerial 實驗就說已解決飛行中的背景運動；同樣需要真實轉鏡、視差、薄煙及干擾資料評估。README 以 Python≥3.7 與 requirements 為環境入口；沒有實際執行驗證。

## 4. JCR 直接查核明細

下列均為 **2026 發布版／2025 指標資料年**，不是發表年度分區。TII 採 Computer Science, Interdisciplinary Applications，IoT Journal 採 Electrical & Electronic Engineering，其他採 AI；選擇理由是視覺學習／機載系統主題，不選不相干類別的最高分區。

| 期刊／ISSN／JIF | 取得的全部 JIF 類別、排名、分區 | 正式 JCR 證據 |
|---|---|---|
| TPAMI／0162-8828，1939-3539／20.4 | AI 4/210 Q1；Electrical & Electronic Engineering 4/370 Q1 | [JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=IEEE%20T%20PATTERN%20ANAL&year=2025&fromPage=%2Fjcr%2Fsearch-results) |
| TII／1551-3203，1941-0050／9.8 | Automation & Control Systems 8/88 Q1；Computer Science, Interdisciplinary Applications 19/185 Q1；Engineering, Industrial 7/70 Q1 | [JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=IEEE%20T%20IND%20INFORM&year=2025&fromPage=%2Fjcr%2Fsearch-results) |
| IoT Journal／2327-4662／8.7 | Computer Science, Information Systems 16/267 Q1；Electrical & Electronic Engineering 30/370 Q1；Telecommunications 12/127 Q1 | [JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=IEEE%20INTERNET%20THINGS&year=2025&fromPage=%2Fjcr%2Fsearch-results) |
| TNNLS／2162-237X，2162-2388／9.7 | AI 25/210 Q1；Hardware & Architecture 2/60 Q1；Theory & Methods 8/146 Q1；Electrical & Electronic Engineering 22/370 Q1 | [JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=IEEE%20T%20NEUR%20NET%20LEAR&year=2025&fromPage=%2Fjcr%2Fsearch-results) |
| ESWA／0957-4174，1873-6793／9.4 | AI 28/210 Q1；Electrical & Electronic Engineering 23/370 Q1；Operations Research & Management Science 6/109 Q1 | [JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=EXPERT%20SYST%20APPL&year=2025&fromPage=%2Fjcr%2Fsearch-results) |
| Applied Intelligence／0924-669X，1573-7497／3.5 | AI 102/210 Q2 | [JCR](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=APPL%20INTELL&year=2025&fromPage=%2Fjcr%2Fsearch-results) |

## 5. 怎麼選與怎麼套用

- **要保留預測誤差研究主線**：HSTforU 在方法上最接近，但須接受它目前不符合本次正式 Q1＋開源條件；先讀設計，不把它當成完成相機運動解法。
- **要讓煙霧關注圖利用跨影格資訊**：讀 MTNet 的融合，再看 AOT／AOST 的長短期記憶。AOT 可以自動候選 mask 初始化，不需讓使用者在飛行中手動畫框；不過這是需要設計並驗證的適配，不是原論文已提供煙霧自動偵測。
- **要減少煙／霧／背景混淆**：讀 SmokeSeger 與 STKformer。前者可作語意 mask 基準；後者先作候選區域分類／驗證支線，不能直接拿整圖分類分數估煙霧源座標。
- **要精細煙霧輪廓**：EGNL-FAT 可作外觀對照，但現有 BBox 不能當成真實像素 mask，先補人工像素驗證集或可信偽標註。

適合 Boom 的最小路線是假設「外觀與運動分開編碼，再以當前特徵查詢歷史對齊特徵」，並與原本 44 通道 gate 比較。先保留 predictor，只更換 gate 融合，避免同時改多個瓶頸。對純歷史記憶保證因果性；不把未來影格放入當前偵測。

實驗至少固定：現行 v2.2、雙流時序融合、記憶傳播、語意分支；再做個別與組合消融。比較每小時誤報、固定誤報率下召回、首次偵測延遲、人工像素 IoU、端到端 FPS／顯存。分相機穩定、移動無煙、移動有煙及雲霧粉塵等案例，不能以某篇道路或城市資料的 AUC 當成 Boom 的煙霧能力。

本次還找到 TAFormer（TGRS，目標感知影片預測）與 Mask2Anomaly（TPAMI，mask Transformer 異常分割），但前者未確認對應實質作者程式，後者取得的倉庫主要標示 ICCV 2023 前作且未找到專案層授權；不為湊數列為可直接重現的正式推薦。

## 6. Consensus 對應查詢紀錄

下列引用數僅為查詢時資料，不用於推薦排序：

- [1] Le、Jin、Kim，2025，Applied Intelligence，17 citations：[HSTforU](https://consensus.app/papers/hstforu-anomaly-detection-in-aerial-and-groundbased-le-jin/e1989b1f1c6a58b389a1ceff4bff59d8/?utm_source=chatgpt)。
- [2] Jing、Meng、Hou，2024，TII，29 citations：[SmokeSeger](https://consensus.app/papers/smokeseger-a-transformercnn-coupled-model-for-urban-scene-jing-meng/fbe606f6ef2851d79d36101ee40dc7fe/?utm_source=chatgpt)。
- [3] Yu 等，2025，IoT Journal，4 citations：[STKformer](https://consensus.app/papers/realtime-smoke-detection-with-split-topk-transformer-and-yu-huang/f6221d99bcec5bc981141f9d9010403e/?utm_source=chatgpt)。

正式推薦目前新增 1 篇期刊工作（AOT／AOST 文中的 AOT 對應實作）；另提供 4 篇 Q1 方法閱讀候選及 1 篇 Q2 排除說明。沒有把無授權倉庫、前作實作或未核對完整的新模組包裝成全部可直接套用。
