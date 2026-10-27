<!-- 檔案功能：ASTT 本機專案總覽與文件入口。 -->

# ASTT for Boom

ASTT 是用連續影格預測下一幀的異常偵測方法；本機版用於驗證荒野煙霧資料上的效果。正式模型為 **STE＋TTE＋CSA**，採用「入口、平坦 src 核心、tools、doc」的傳統結構，UI 集中在 src/ui。

`origin/` 是官方固定版本的完整原始碼，禁止修改；`src/` 是本機適配與擴充版本。正式入口集中在 `scripts/`，資料工具在 `tools/`，操作文件在 `doc/`。正式程式不匯入 origin。

- [快速開始](doc/START_HERE.md)：安裝、訓練與測試指令。
- [影片標記與自動資料準備](doc/DATASET_PREPARATION.md)：標記與匯出統一 30 FPS、標異常時間區段、依 7:3 分 train／val 與生成 `.npy`。
- [用權重跑新影片](doc/VIDEO_INFERENCE.md)：inference.py 直接讀影片，輸出 30 FPS 標記 MP4 與候選事件。
- [專案結構與程式資料流](doc/PROJECT_STRUCTURE.md)：各目錄與檔案做什麼。
- [執行設定](doc/RUN_SETTINGS.md)：正式參數在哪裡改。
- [本機方法與限制](doc/README_LOCAL.md)：模型公式解讀、資料格式、事件與輸出。
- [官方原始碼與保護方式](doc/ORIGIN.md)：固定版本、來源和校驗指令。
- [作者原始 README](origin/README.md)：論文、作者與引用，內容未修改。

ASTT 資料與輸出位於 benchmark/ASTT 本機專案內。標記工具建立 `benchmark/ASTT/data/prepared/<名稱>_vN/astt/`，使用版本目錄作為 `--data-dir`。訓練權重與紀錄放在 `benchmark/ASTT/output/astt/training/<experiment-name>/`；最後的資料集評估與影片推論結果放在 `benchmark/ASTT/output/astt/results/<experiment-name>/`；人工標記放在 `benchmark/ASTT/output/astt/annotations/`。詳見 [資料與輸出結構](doc/PROJECT_STRUCTURE.md#資料與輸出位置)。候選框代表高預測誤差區域，尚未接入 Boom 的即時相機與 GPS 定位。
