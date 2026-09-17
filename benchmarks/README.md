# Boom 外部演算法對照與基準評估庫 (Benchmarks & Baselines)

本目錄專門用於收納與維護**非本專案原生核心**的第三方成果、學術界論文實作或外部基準演算法（Baselines）。

---

## 目錄設計宗旨

1. **完全隔離干擾**：外部研究成果不直接侵入 `src/` 核心系統，避免依賴地獄或污染主系統架構。
2. **統一公平比較**：未來所有新增的比較演算法，皆可在此共用 `data/` 下的相同測試空拍影片與航拍圖資進行客觀評估。
3. **支援多演算法擴充**：未來加入任何新論文或新模型時，只需在此新增獨立子目錄。

---

## 目前收錄之基準演算法

| 目錄 | 演算法/論文來源 | 核心技術特性 |
| :--- | :--- | :--- |
| **`mod_ir/`** | **MOD-IR 論文實作** (*Moving Objects Detection from UAV-captured Video Sequences based on Image Registration*) | 基於 ORB 特徵配準消除相機運動、Kapur 最大熵差分閾值、Quick-shift 色彩區域分割與煙霧低飽和/柔和邊緣過濾規則。 |

---

## 如何新增其他比較演算法？

若未來要引進新論文或第三方演算法（例如基於 YOLO、Swin-Transformer、或稠密光流的煙霧偵測）：

1. 在 `benchmarks/` 下建立以演算法/論文命名的獨立子資料夾（例如 `benchmarks/fire_smoke_yolo/`）。
2. 在該目錄內放置其專屬程式碼與 `README.md`。
3. 輸出結果統一導向 `outputs/benchmarks/<演算法名稱>/`，確保主輸出目錄整潔。
