# ASTT 維護原則

- `origin/` 是 Tungufm/ASTT 固定提交的官方原始碼快照。不得修改、格式化、加註解、重命名、生成快取或在其中寫入資料；來源和 SHA-256 清單放在 `doc/ORIGIN_MANIFEST.json`。
- 功能修改只做在本機版 `src/`、入口及 `tools/`；採用平坦模組，只有 UI 使用子套件，不為單一檔案新增資料夾。
- `legacy/` 是先前加過本機註解／匯入調整的歷史參考，不能宣稱它是官方未修改版本。正式流程不匯入 `origin/` 或 `legacy/`。
- 根目錄 `model.py`、`dataset.py`、`runner.py`、`localization.py` 及 `scripts/extract_frames.py` 是舊匯入／指令相容層，核心實作在 `src/`。
- 保留 CLI、checkpoint state_dict、資料與輸出格式；不移動或覆寫使用者資料、權重和輸出。
- 更新後執行 `python benchmark/ASTT/tools/verify_origin.py`、既有 unittest 與 `smoke_test.py`。所有提交與推送使用 `dev`，不更新 `master`。
