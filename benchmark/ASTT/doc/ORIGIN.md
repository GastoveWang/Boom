# 官方原始碼快照

- 來源：[Tungufm/ASTT](https://github.com/Tungufm/ASTT)
- 固定版本：[1d05305a94fb47b33897c7aa085c5351bcbbc8f9](https://github.com/Tungufm/ASTT/tree/1d05305a94fb47b33897c7aa085c5351bcbbc8f9)
- 路徑：`benchmark/ASTT/origin/`
- 清單：[ORIGIN_MANIFEST.json](ORIGIN_MANIFEST.json)，記錄所有 18 個檔案的 SHA-256。

快照依固定提交的 Git tree 列出全部追蹤檔案，再以 `git cat-file blob` 讀取原始位元內容，保留原始檔名且不包含巢狀 `.git`。不採用 Windows checkout 的自動換行轉換。`.gitattributes` 禁用快照的換行轉換。作者 README 與引用完整保留在 [origin/README.md](../origin/README.md)。

不要修改或格式化 `origin/`，也不要向其中增加用途註解、資料、權重、快取或輸出。用途說明放在本文件或結構文件。正式訓練、測試及影片推論使用外層入口和 `src/`；原作者腳本保留其原有路徑與依賴，沒有在快照中修補。

```powershell
python benchmark/ASTT/tools/verify_origin.py
```

校驗會檢查缺檔、多出檔案及任何位元變更。`tools/update_file_index.py` 已排除對快照的原始碼註記。
