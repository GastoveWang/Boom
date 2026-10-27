# ST² torchview 圖片快照

這四張SVG由`python src/st2 visualize`使用torchview 0.2.7產生，對應2026-10-27的ST²預設架構：history=8、768×576（W×H）、batch1、depth2、meta裝置、無訓練checkpoint。圖上張量維度為PyTorch順序，標題尺寸為H×W。

- [完整模型](ST2_MODEL.svg)
- [正常畫面預測](ST2_PREDICTOR.svg)
- [背景運動](ST2_MOTION.svg)
- [目標關注](ST2_GATE.svg)

這是可縮放的實際forward圖，並非手畫概念圖或訓練效果。重新匯出會在output/st2/visualization建立獨立目錄，不自動覆寫本文件快照。操作見 [ST2_VISUALIZATION.md](../ST2_VISUALIZATION.md)。
