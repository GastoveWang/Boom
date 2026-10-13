<!-- 檔案功能：上游歷史檔案分類、用途與執行限制。 -->

# ASTT 上游歷史參考

這裡保存作者的舊模型、消融實驗與工具。正式煙霧資料實驗使用父目錄的 `train.py`、`test.py`。

| 分組 | 檔案 | 用途 |
|---|---|---|
| 架構比較 | model_original.py、model_baseline_ca.py、model_ste_tte.py | 原模型、baseline＋CSA、STE＋TTE |
| 歷史實驗 | Train_and_eval.py、train_val_UIT_ADrone*.py、eval.py | 作者的訓練／評估參考 |
| 設定與工具 | config.py、utils.py、data_utils.py、data_loaders.py | 作者的設定、資料讀取與 logging |
| 權重工具 | checkpoint.py、check_jax.py | 舊 JAX／ViT 權重載入、轉換與檢查 |
| 作者文件 | README_UPSTREAM.md | 上游 README、論文與引用 |

保留原演算法及 sibling import；歷史入口可找到父目錄的正式 model 相容模組。這些腳本原本仍含作者路徑、CUDA 寫法與舊模型介面，**整理目錄不代表已完成本機化**。它們可能需要 torchvision、timm、sklearn、TensorFlow 或 JAX；正式流程不需要安裝這些歷史依賴。`check_jax.py` 的上游 `src.*` 匯入也仍是研究參考。

不要在此處的 `config.py` 調整正式實驗。正式參數在 [../src/config/cli.py](../src/config/cli.py)。
