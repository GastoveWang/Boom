# ST² v2 架構與運算圖

更新日期：2026-10-28。ST2Motion已移除，現行為Transformer Predictor／Gate＋凍結LoFTR／RANSAC。

## 現行架構

以下圖使用預設8歷史＋當前、768×576、batch1。沒有使用訓練權重；用來展示架構。

| 圖 | 種類與內容 |
|---|---|
| [完整模型](diagrams/ST2_V2_MODEL.svg) | 明確繪製的資料流示意，標示非可微分配準邊界 |
| [配準模組](diagrams/ST2_V2_REGISTRATION.svg) | LoFTR→RANSAC→warp、有效遮罩與殘差的流程示意 |
| [Predictor](diagrams/ST2_V2_PREDICTOR.svg) | torchview實際forward追蹤 |
| [Gate](diagrams/ST2_V2_GATE.svg) | torchview實際forward追蹤；含殘差／有效遮罩／品質輸入 |

OpenCV RANSAC包含CPU陣列與資料依賴判斷，不能用meta張量追蹤。完整模型與registration圖因此明確標為pipeline schematic，不能稱為完整模型的實際forward計算圖；Predictor／Gate仍使用實際torchview追蹤。圖上參數量包含凍結LoFTR，不等於可訓練參數量。

## 匯出

```powershell
conda activate boom
python -m pip install -e ".[st2-visualization]"
python src/st2 visualize
python src/st2 visualize --part registration
python src/st2 visualize --part predictor --depth 3 --direction LR
python src/st2 visualize --checkpoint "output/st2/你的日期時間/training/best.pt"
python src/st2 visualize --device cpu --height 64 --width 96 --format dot
```

`--part`支援all／model／predictor／registration／gate；不再有motion選項。預設meta裝置，Predictor／Gate也可用CPU。架構圖不執行LoFTR匹配，不需下載預訓練權重。提供checkpoint時只接受現行v2完整權重，嚴格載入，不轉換舊ST2Motion權重。

每次建立新目錄 `output/st2/visualization/st2_torchview_<時間>_<ID>/`，包含各部分SVG／DOT與summary.json；各圖的graph_kind分別記錄是示意圖或實際追蹤。原有檔案不覆寫。

渲染SVG需Graphviz的dot執行檔；缺dot時保留DOT並回報partial。圖不是反向梯度圖或attention熱圖。匯出在獨立模型執行，保留RNG、MHA fastpath與PATH狀態。

## 歷史快照

舊檔 [ST2_MODEL.svg](diagrams/ST2_MODEL.svg)、[ST2_MOTION.svg](diagrams/ST2_MOTION.svg)、[ST2_PREDICTOR.svg](diagrams/ST2_PREDICTOR.svg)、[ST2_GATE.svg](diagrams/ST2_GATE.svg) 保留作為v1三分支歷史紀錄，不能用來代表目前架構。舊版說明見[ST² v1](history/ST2_V1.md)。

2026-10-28已驗證預設尺寸四圖匯出，以及64×96的示意／meta實際追蹤；驗證產物在tmp清理，以上SVG為正式文件資產。
