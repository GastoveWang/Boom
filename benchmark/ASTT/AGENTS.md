# ASTT 維護原則

- `origin/` 是 Tungufm/ASTT 固定提交的官方原始碼快照。不得修改、格式化、加註解、重命名、生成快取或在其中寫入資料；來源和 SHA-256 清單放在 `doc/ORIGIN_MANIFEST.json`。
- 功能修改只做在本機版 `src/`、入口及 `tools/`；採用平坦模組，只有 UI 使用子套件，不為單一檔案新增資料夾。
- 正式入口集中於 `scripts/train.py`、`scripts/test.py`、`scripts/inference.py`；資料工具在 `tools/`，操作文件在 `doc/`。根目錄只保留 README、依賴清單、套件標記及維護／Git 設定。
- 舊 `legacy/`、根目錄轉接程式與 `scripts/extract_frames.py` 已移除；新程式直接匯入 `src/`，不要重建重複入口。正式流程不匯入 `origin/`。
- 保留 CLI、checkpoint state_dict、資料與輸出格式；未經使用者指示不移動或覆寫資料、權重和輸出。
- 依使用者指示，ASTT 資料及輸出回到本機專案：資料根為 `benchmark/ASTT/data/`，版本為 `data/prepared/`，手動資料為 `data/astt/<split>/`。輸出根為 `benchmark/ASTT/output/`，其下 `astt/training/` 保存訓練、`astt/results/` 保存評估／推論、`astt/annotations/` 保存人工標記，保留原有 cache 與搬移紀錄。不要將 ASTT 資料重新移到 Boom 根目錄的 data/output，也不與新 BBox 模型混用。
- 所有新增測試程式、測試資料與產物一律放在 Boom 根目錄的 `tmp/`，執行完成後立即刪除，不保留 ASTT 的 `tests/`。`scripts/test.py` 是正式資料集評估功能，不是臨時測試程式。
- 更新後執行 `python benchmark/ASTT/tools/verify_origin.py` 與對應入口載入檢查；需要其他驗證時依 tmp 規則執行與清理。所有提交與推送使用 `dev`，不更新 `master`。
