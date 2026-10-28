# HSTforU benchmark 套件

本目錄用於和 Boom 的 ST² 比較，模型名稱保持 **HSTforU**。
官方來源：[vt-le/HSTforU](https://github.com/vt-le/HSTforU)，固定 commit
`9548473306bbb89895b077275e8503de79653029`（取得日 2026-10-28）。

## 結構

- `origin/`：官方完整工作樹，未修改；含原始 README、資料集 train/test 入口、模型及網站資產。
- `UPSTREAM.json`：官方版本與 321 個原始檔的 SHA-256，路徑相對於 origin。
- `src/hstforu/`：獨立模型套件，複製官方模型與必要 loss/helper，只修正成 hstforu 命名空間的匯入。
- `pyproject.toml`：本機套件、依賴與 `hstforu` 命令。
- `.upstream-git/`：下載時保留的 Git 中繼資料，不納入 Boom 版本，也不作為子模組。

模型載入 API 支援官方預設 Ped2（PVTv2-B1）和 Avenue（PVTv2-B2）。ShanghaiTech 及 Drone 的原始程式保留於 origin；尚未包裝或驗證其資料評估流程。沒有接入 ST² 資料、Boom runner、GPS、煙霧事件確認或長時間訓練。

## 安裝與執行

建議另建環境，避免與其他需要 MMCV 2.x 的專案混用。從 Boom 根目錄執行：

```powershell
conda create -n hstforu python=3.10 -y
conda activate hstforu
python -m pip install setuptools==69.5.1 wheel
python -m pip install --no-build-isolation -e benchmark/HSTforU
python -m hstforu doctor
python -m hstforu forward --variant ped2 --device cpu
python -m hstforu forward --variant avenue --device cpu
```

GPU 需先在該環境安裝適合設備的 PyTorch／torchvision，再使用 `--device cuda`。
這次只在專案 tmp 安裝依賴驗證，完成後清除；未修改既有 boom 環境。
官方 README 提及的 requirements.txt 並未提供。本套件依實際 import 補齊模型依賴，固定 `timm==0.6.13`、`mmcv==1.7.2`；新版 MMCV 沒有本模型使用的 `mmcv.runner`。
舊版 MMCV 建置需要 pkg_resources，因此先安裝含該介面的 setuptools，再使用 `--no-build-isolation`。

## Python API

```python
import torch
from hstforu import build_model, load_checkpoint

model = build_model("ped2").eval()  # 不自動下載、也不載入任何權重
# load_checkpoint(model, "你的官方模型.pth")  # 僅載入可信任的權重，strict=True
frames = [torch.rand(1, 3, 256, 256) * 2 - 1 for _ in range(4)]
with torch.inference_mode():
    predicted = model(frames)  # [1, 3, 256, 256]，預測下一幀
```

輸入必須是四張歷史 RGB，各 `[B,3,H,W]`、值域 [-1,1]。已驗證 256×256；API 直接沿用官方 forward，不增加任意尺寸或八幀的相容承諾。`forward --size` 僅允許至少 32 且 32 整除的正方形，預設 256。輸出為未經額外裁切的預測 RGB，不是煙霧機率或分割遮罩。

可在前向命令加上 `--checkpoint` 進行嚴格權重載入；目前上游 README 權重表沒有可用下載連結，本次未取得或驗證訓練權重。ST² 權重不能載入 HSTforU。

## 驗證（2026-10-28）

- Python 3.10、PyTorch 2.13.0+cu130、torchvision 0.28.0+cu130、NumPy 2.2.6、timm 0.6.13、MMCV 1.7.2、einops 0.8.2；CPU eval、batch=1。
- Ped2：四張 256×256 → `[1,3,256,256]`，輸出皆有限；137,615,680 個參數。
- Avenue：相同輸入／輸出尺寸且皆有限；148,969,536 個參數。
- Ped2 封裝前後使用相同隨機種子及輸入，官方與套件輸出逐值完全相同，state_dict 鍵一致。
- 驗證資料與依賴位於 Boom `tmp/hstforu_setup_20261028/`，完成後刪除，未保留臨時測試程式或權重。

這是模型可執行性驗證，沒有測量煙霧 AUC、誤報、漏報、端到端 FPS 或定位誤差。

完整差異：[HSTforU 與 ST²](../../doc/HSTFORU_COMPARISON.md)。原作者設定與引用見 [官方 README](origin/README.md)。
