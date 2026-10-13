<!-- 檔案功能：ASTT 本機專案總覽與文件入口。 -->

# ASTT for Boom

ASTT 是用連續影格預測下一幀的異常偵測方法；本機版用於驗證荒野煙霧資料上的效果。正式模型為 **STE＋TTE＋CSA**，目錄採用與 Boom 相同的「入口、原始碼、工具、測試、文件」分工。

- [快速開始](START_HERE.md)：安裝、訓練與測試指令。
- [影片標記與自動資料準備](doc/DATASET_PREPARATION.md)：標記與匯出統一 30 FPS、標異常時間區段、依 7:3 分 train／val 與生成 `.npy`。
- [用權重跑新影片](doc/VIDEO_INFERENCE.md)：inference.py 直接讀影片，輸出 30 FPS 標記 MP4 與候選事件。
- [專案結構與 Boom 對照](doc/PROJECT_STRUCTURE.md)：各目錄與檔案做什麼。
- [執行設定](doc/RUN_SETTINGS.md)：正式參數在哪裡改。
- [本機方法與限制](doc/README_LOCAL.md)：模型公式解讀、資料格式、事件與輸出。
- [上游歷史參考](legacy/README.md)：作者模型、消融與舊工具。
- [作者原始 README](legacy/README_UPSTREAM.md)：論文、作者與引用。

手動資料預設在 Boom 的 `data/astt/`；標記工具建立 `benchmark/ASTT/data/prepared/<名稱>_vN/astt/`，使用版本目錄作為 `--data-dir`。結果放在 `output/astt/<experiment-name>/`。候選框代表高預測誤差區域，尚未接入 Boom 的即時相機與 GPS 定位。
