> 本文比較的是2026-10-28替換前的ST² v1。其後ST² v2已改為LoFTR＋RANSAC，完整系統含CNN；現行架構與參數見 [ST²](ST2.md)。下表保留當時比較紀錄。

# HSTforU 與目前 ST² 的差異

日期：2026-10-28。比較以固定官方 commit `9548473306bbb89895b077275e8503de79653029` 的程式，對照目前 `src/st2/`，不是把論文指標直接套用到荒野煙霧。
來源：[HSTforU 官方 repo](https://github.com/vt-le/HSTforU/tree/9548473306bbb89895b077275e8503de79653029)、本機 [官方快照](../benchmark/HSTforU/origin/README.md)、[ST² 文件](ST2.md)。

## 主要差異

兩者共有「使用歷史影格預測下一幀，再比較當前真實畫面」的思路。HSTforU 是多尺度、含卷積的時空 Transformer／U-net；ST² 是自行實作的三分支純 Transformer，另外學習背景運動和煙霧目標關注。ST² 不是 HSTforU 的改名或官方延伸版，本次也沒有以 HSTforU 替換 ST²。

| 項目 | HSTforU（官方預設 Ped2） | 目前 ST² v1.0 |
|---|---|---|
| 任務 | 一般影片異常偵測；無煙霧專屬 ROI 監督 | 荒野煙霧候選變化熱圖，含框標註監督 |
| 歷史窗口 | 4 幀預測第 5 幀；decoder 通道數固定乘 4 | 8 幀預測第 9 幀；motion/gate 讀完整 9 幀 |
| 預設尺寸 W×H | 256×256 | 768×576 |
| 影像前處理 | RGB、Pillow/torchvision bicubic、[-1,1] | RGB、Pillow bilinear、[0,1] |
| 空間 encoder | 每幀 PVTv2-B1；四層尺度、通道 64/128/320/512 | patch16、embedding128，保留單尺度格點；局部 6×6 視窗，交替偏移 |
| 時序 | 四層尺度各自使用時間及空間 attention；時間特徵加回串接的歷史特徵 | predictor/gate 各 2 組交替空間／時間 attention |
| 空間注意力範圍 | PVT 空間縮減 attention，跨該層空間位置；不是 ST² 的局部視窗 | 跨位置訊息經相鄰移位視窗傳遞，時間 attention 按格點處理 |
| 卷積 | 有；重疊 patch Conv2d、深度卷積、空間縮減卷積及 decoder | 可學習分支使用線性層與 attention，沒有卷積／反卷積 |
| decoder | 多尺度 U-net skip connections、反卷積上採樣及 CNN | 時間 query cross-attention、空間 attention、線性 RGB patches |
| 預測限制 | decoder 直接輸出 RGB，無額外殘差裁切 | `clip(last_history + 0.25*tanh(delta),0,1)` |
| 背景相機運動 | 沒有獨立幾何估計／背景配準 loss；時空特徵並不等於相機補償 | Transformer 估平移、旋轉、縮放，配準相鄰影格，框外 photometric loss |
| 目標分支 | 無 BBox 關注 gate | 框聯集為弱監督 mask；完整 9 幀＋誤差＋motion 統計輸出 attention |
| 訓練 loss | intensity MSE＋gradient loss＋(1−MS-SSIM)/2＋sqrt(MSE+eps)，等權相加 | normal 窗口 MSE＋平衡 BCE＋正樣本 Dice＋0.2×(背景 loss＋0.001×motion regularization) |
| 標註與訓練資料 | 官方正常訓練影格目錄；標籤用於異常評估，模型無煙霧框 loss | normal/ROI/ignore；9 幀皆正常才訓練 predictor；gate 使用框監督 |
| 原始誤差 | 評估時將 output/target 從 [-1,1] 轉回 [0,1]，計算全圖平均 MSE | RGB [0,1] 像素 MSE，另保留加權圖 |
| 異常／正常分數 | MSE→PSNR；每支完整影片 min-max，分數高代表正常；ROC pos_label=0 | 誤差最高 0.1% 像素平均；固定對數換算，分數高代表較大候選變化 |
| 最終熱圖 | 官方標準 test.py 著重 frame PSNR/AUC，無煙霧關注加權 | `H=detach(E)×(0.05+0.95×A)`，raw、attention、weighted 分開保留 |
| 預設參數量 | 137,615,680（實際模型計數） | 3,587,204（實際模型計數） |
| 權重 | HSTforU state_dict；Ped2/Avenue 主幹不同 | ST² 自有 checkpoint，與 HSTforU 不相容 |
| 即時／定位 | 官方影格資料集訓練／評估入口；本次未接 Boom | 有 ST2Runtime，但尚未接入 Boom 即時事件確認及 GPS 定位 |

HSTforU 的原始 RGB MSE 如果依官方 eval 換回 [0,1]，可與相同前處理下的 MSE 有相同單位；但兩者尺寸、插值、時序窗口與模型不同，不能用絕對誤差值高低判定誰更好。HSTforU 的每影片 min-max 需要全段統計，不能直接作為無人機因果串流分數。ST² 固定換算則不依賴未來影格，仍不是校準的煙霧機率。

## 對荒野煙霧的含義

HSTforU 的金字塔與 skip connections 提供多尺度表徵，值得作為結構對照；是否更能辨認小煙霧必須實驗。一般異常偵測沒有直接教模型區分煙霧、雲霧、粉塵和相機移動。

ST² 的框監督及背景運動分支是為目標／干擾區分而設計，但框不是像素真值，全局相似變換也無法涵蓋視差、地形和完整 3D 運動。不能因加入這兩分支就宣稱煙霧誤報已改善。

HSTforU Ped2 參數約為 ST² 的 38.36 倍。參數量不能直接換算 FPS、顯存或電力；ST² 使用更高解析度與更長歷史，仍需相同設備上的實測。

## 此次完成與驗證邊界

已將官方完整工作樹保存於 `benchmark/HSTforU/origin/`，加入可獨立安裝的 `hstforu` 套件與 `doctor`／`forward` 入口。封裝只調整命名空間及新增外層 API，不改官方 forward、loss、架構或分數；嚴格權重載入是外層 API，尚未取得官方訓練權重驗證。ST² 程式及現有資料／模型／輸出未變更。

本次沒有訓練、接資料或產生煙霧效能結論。官方 README 的預訓練權重表目前沒有下載連結；requirements.txt 與部分 README 範例路徑也不符合 repo 實際內容。原訓練入口使用 CUDA/distributed，原 AUC 程式仍含 NumPy 已移除的 `np.float/np.int`；不把「模型前向通過」寫成「完整原版評估通過」。ShanghaiTech／Drone 保留原碼，未包裝其評估流程。

驗證環境：Python3.10、PyTorch2.13.0+cu130、torchvision0.28.0+cu130、NumPy2.2.6、timm0.6.13、MMCV1.7.2、einops0.8.2。CPU、eval、batch1、4×RGB256×256。驗證結果與執行方式見 [套件 README](../benchmark/HSTforU/README.md)。臨時依賴、輸出和驗證程式都在專案 tmp，結束後刪除。

Ped2 與 Avenue 都成功輸出有限值 `[1,3,256,256]`。Ped2 的官方／封裝程式使用相同種子時，輸出逐值完全一致且 state_dict 鍵相同。Avenue 實測參數為 148,969,536；上表與 38.36 倍比較專指 Ped2，不能混用版本。

下一階段公平評估需固定影片與來源切分，HSTforU 僅以正常訓練窗口學習，ST² 保留其 ROI 弱監督並清楚揭露監督差別。分開比較原始設定與相同解析度／時間跨度的設定，記錄誤報、漏報、偵測延遲、端到端 FPS、記憶體、區域定位與 GPS 誤差；BBox IoU 不應被稱為煙霧像素真值 IoU。
