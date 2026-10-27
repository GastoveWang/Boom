# Boom

Boom 是一套用於飛行中無人機的荒野煙霧偵測與定位系統，目標是在影像中即時標示煙霧區域，並估算疑似煙霧源的地面座標。

目前 `src/boom/` 負責辨識執行、顯示、事件與定位；`src/st2/` 獨立負責 ST² 模型、訓練與研究推論。ST² 已完成流程驗證，尚未正式訓練或接入 Boom 即時事件流程。

訓練完成後自動推論，從根目錄直接執行：

```powershell
conda activate boom
python train_st2.py
```

預設訓練30輪，資料為`data/st2/st2_v1`，完成後以best.pt推論`data/video/1440-1080/0603`。可加`--infer-source`指定其他影片或資料夾、`--resume`續訓，完整參數用`python train_st2.py --help`查看。

- [文件首頁與閱讀順序](doc/README.md)
- [專案結構與 Boom 執行](doc/PROJECT_STRUCTURE.md)
- [ST² 架構、訓練與推論](doc/ST2.md)
- [ST² 資料標註與匯出](doc/ST2_DATASET.md)
- [模型變更紀錄](doc/MODEL_CHANGELOG.md)
