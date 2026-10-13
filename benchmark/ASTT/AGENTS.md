# ASTT 維護原則

- `origin/` 是 Tungufm/ASTT 固定提交的官方原始碼快照。不得修改、格式化、加註解、重命名、生成快取或在其中寫入資料；來源和 SHA-256 清單放在 `doc/ORIGIN_MANIFEST.json`。
- 功能修改只做在本機版 `src/`、入口及 `tools/`；採用平坦模組，只有 UI 使用子套件，不為單一檔案新增資料夾。
- 正式入口集中於 `scripts/train.py`、`scripts/test.py`、`scripts/inference.py`；資料工具在 `tools/`，測試在 `tests/`，操作文件在 `doc/`。根目錄只保留 README、依賴清單、套件標記及維護／Git 設定。
- 舊 `legacy/`、根目錄轉接程式與 `scripts/extract_frames.py` 已移除；新程式直接匯入 `src/`，不要重建重複入口。正式流程不匯入 `origin/`。
- 保留 CLI、checkpoint state_dict、資料與輸出格式；不移動或覆寫使用者資料、權重和輸出。
- 更新後執行 `python benchmark/ASTT/tools/verify_origin.py`、既有 unittest 與 `python benchmark/ASTT/tests/smoke_test.py`。所有提交與推送使用 `dev`，不更新 `master`。
