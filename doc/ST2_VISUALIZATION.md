# ST² 模型視覺化：torchview

更新日期：2026-10-27。此功能使用 [torchview 官方實作](https://github.com/mert-kurttutan/torchview) 追蹤 ST² 實際的 forward，顯示模組、張量尺寸及運算連線。

## 直接查看專案內的圖

以下為目前預設架構的實際 torchview 圖：8 張歷史＋當前影格、768×576、batch1、depth2。沒有使用訓練權重；只展示架構與張量流。

| 圖 | 內容 |
|---|---|
| [完整模型](diagrams/ST2_MODEL.svg) | ST2Predictor、ST2Motion、ST2Gate 與誤差加權 |
| [正常畫面預測分支](diagrams/ST2_PREDICTOR.svg) | 歷史影格→時空編碼→預測畫面 |
| [背景運動分支](diagrams/ST2_MOTION.svg) | 相鄰影格對→運動參數→配準誤差 |
| [目標關注分支](diagrams/ST2_GATE.svg) | 完整序列＋誤差＋運動統計→關注 logits |

SVG可用瀏覽器開啟並放大。這是實際運算圖，會包含reshape、detach及誤差計算，因此比概念架構圖更細。

## 產生自己的圖

```powershell
conda activate boom
python src/st2 visualize
```

預設同時匯出完整模型與三條分支，產物保存在新建的`output/st2/visualization/st2_torchview_<時間>_<ID>/`：

```text
model.svg / model.dot
predictor.svg / predictor.dot
motion.svg / motion.dot
gate.svg / gate.dot
summary.json
```

`summary.json`記錄模型設定、工具版本、輸入尺寸、參數量、checkpoint來源及各圖完成／失敗狀態。每次建立獨立資料夾，不覆寫已有輸出。

```powershell
# 只看完整模型，使用較淺的模組層級
python src/st2 visualize --part model --depth 1
# 展開更多層，改成由左向右
python src/st2 visualize --part predictor --depth 3 --direction LR
# 依已訓練checkpoint的架構與影像尺寸畫圖
python src/st2 visualize --checkpoint "output/st2/你的日期時間/training/best.pt"
# CPU實際前向，或只輸出DOT原始圖
python src/st2 visualize --device cpu --height 64 --width 96 --format dot
```

預設使用`meta`裝置，只依張量形狀執行追蹤，不配置完整影像activation；模型初始化、checkpoint讀取與圖結構仍會占用CPU記憶體。沒有checkpoint時使用目前預設ST²架構，不代表模型已訓練。提供checkpoint時嚴格載入權重，舊baseline/camera-aware模型會被拒絕。

## 安裝及 Graphviz

```powershell
python -m pip install -e ".[st2-visualization]"
```

torchview依賴Python graphviz套件，匯出SVG還需要Graphviz的`dot`執行檔，只有pip安裝Python套件並不保證dot存在。程式會尋找PATH、目前Python環境的Library/bin，以及Windows常見安裝位置；只在渲染子程序期間調整本程序的PATH，不修改系統設定。

本機boom環境已確認有torchview 0.2.7、Python graphviz，以及環境Library/bin內的dot.exe。若其他環境缺dot，仍保留DOT原始圖，SVG未完成會明確標為partial並傳回非零結束碼；安裝方式參考[torchview官方說明](https://github.com/mert-kurttutan/torchview#installation)。

## 行為與限制

- 圖是forward追蹤，不是反向梯度圖、煙霧熱圖或attention權重圖；圖上有detach也不等於完整描述訓練梯度監督。
- 深度為1較適合總覽；深度越大圖越複雜。三條分支可分開檢視。
- 視覺化建立獨立模型，不使用正在訓練的實例；不改模型計算、loss或分數，不需要重訓。追蹤時暫時關閉融合MHA fastpath，完成或失敗都恢復原設定與RNG。
- SVG與DOT為本功能提供的格式，避免依賴平台點陣字型渲染；圖中輸入尺寸、參數及工具版本可在summary核對。

已驗證預設尺寸四張圖、checkpoint自訂history／patch、CPU與meta追蹤、權重不變、RNG／MHA／PATH恢復、缺dot及追蹤失敗的狀態記錄。測試產物在tmp下建立後清理；文件SVG快照與本次新建正式視覺化輸出保留。模型訓練與Boom執行流程未改動。
