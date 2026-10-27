# ST² 模型、資料與訓練套件

Boom名稱保留，`src/boom/`負責辨識執行、顯示、事件與定位；本套件獨立負責ST²模型與研究流程。

## 從哪裡開始

從專案根目錄執行，不需在tools查找ST²腳本：

訓練後自動推論的最短指令是`python train_st2.py`，預設訓練30輪並以best.pt推論0603影片資料夾。以下為套件內各功能的獨立命令：

```powershell
conda activate boom
python src/st2 prepare
python src/st2 train --epochs 30 --device cuda
python src/st2 infer --checkpoint "output/st2/training/你的run/best.pt" --source "data/video/1440-1080/0603" --device cuda
python src/st2 export-tensorboard --run "output/st2/training/你的run"
```

各命令加`--help`查看參數。安裝 `python -m pip install -e ".[st2]"` 後，也可用`python -m st2 train`。直接執行單一入口如`python src/st2/cli/train.py --help`也可。

## 目錄分工

```text
src/st2/
  __main__.py           # 統一命令入口：python src/st2 <command>
  cli/                  # train、infer、prepare、export_tensorboard
  models/
    network.py          # ST2Config、ST2Model、ST2Criterion
    transformer.py      # 純 Transformer 三分支、patch與attention
    motion.py           # 幾何配準、背景loss與相機增強
  data/
    dataset.py          # 8幀歷史＋當前影格與9幀ROI mask
    sequence.py         # 來源分組、連續窗口與fingerprint
    annotations.py      # 逐幀標註與資料集匯出
    editor.py           # 影片標註UI
  engine/
    train.py            # 訓練、驗證與續訓
    infer.py            # 影片熱圖、分數與輸出
    runtime.py          # 純模型逐幀介面，供未來Boom adapter
    postprocess.py      # 分數換算與可選運動guard
    view.py             # 研究推論畫面
    tensorboard.py      # 既有metrics匯入TensorBoard
  common/
    utils.py            # 前處理、checkpoint、JSON與抽樣
```

`ST2Runtime`位於`engine/runtime.py`，只回傳熱圖與分數，不確認煙霧事件；未來由Boom adapter接入原有顯示／事件流程。模型計算、state_dict與既有資料／輸出路徑不因本次整理改變。

完整說明見[模型文件](../../doc/ST2.md)、[資料集文件](../../doc/ST2_DATASET.md)、[專案結構](../../doc/PROJECT_STRUCTURE.md)。
