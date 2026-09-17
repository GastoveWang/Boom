"""
==============================================================================
Boom 資料與模型工具鏈 - YOLO 標註模型訓練與 ONNX 匯出 (train_yolo.py)
==============================================================================

【工具定位】
本腳本為 Boom 專案的資料工程與模型訓練工具。使用 Ultralytics YOLO 在空拍標註
資料集上進行訓練，並自動將訓練出的最優模型 (best.pt) 轉換匯出為 ONNX 格式。

【核心功能】
1. 權重檔智慧檢索：自動在 `models/` 與歷史 `label/model` 目錄中搜尋預訓練模型。
2. 自動訓練配置：動態依據 GPU VRAM 配置 batch size、加入 early stopping。
3. ONNX 格式安全匯出：在暫存目錄中完成 ONNX 轉換，不污染主要目錄，若重名自動遞增流水號。

【執行方式】
python tools/train_yolo.py
==============================================================================
"""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


def find_local_model(model_dirs, model_name):
    """在給定的模型目錄清單中遞迴尋找指定檔名的 .pt 權重檔案。"""
    if isinstance(model_dirs, (str, Path)):
        model_dirs = [model_dirs]
    filename = f"{model_name}.pt".casefold()
    for mdir in model_dirs:
        p = Path(mdir)
        if not p.exists():
            continue
        matches = sorted(
            path for path in p.rglob("*")
            if path.is_file() and path.name.casefold() == filename
        )
        if len(matches) > 1:
            paths = "\n".join(str(path) for path in matches)
            raise ValueError(
                f"找到多份同名模型 {model_name}.pt，請使用唯一檔名並更新 MODEL_NAME：\n{paths}"
            )
        if matches:
            return matches[0]
    return None


def main():
    from ultralytics import YOLO

    # =========================================================
    # 1. 專案路徑
    # =========================================================

    TOOLS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = TOOLS_DIR.parent
    LABEL_DIR = PROJECT_ROOT / "label"

    # =========================================================
    # 2. Dataset / Model 設定
    # =========================================================

    DATASET_NAME = "detect_v2"
    MODEL_NAME = "detect_v1_yolo26m"
    IMAGE_SIZE = 640

    DATASET_DIR = (
        PROJECT_ROOT
        / "data"
        / "yolo_datasets"
        / DATASET_NAME
    )

    DATA_YAML = DATASET_DIR / "data.yaml"

    # =========================================================
    # 3. Model 資料夾
    # =========================================================

    MODEL_DIRS = [
        PROJECT_ROOT / "models",
        LABEL_DIR / "model",
    ]

    PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"
    TRAINED_DIR = PROJECT_ROOT / "models" / "checkpoints"
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    # 搜尋 models 與 label/model 根目錄與所有子資料夾；找不到時沿用下載流程。
    PRETRAINED_MODEL = find_local_model(MODEL_DIRS, MODEL_NAME) or (
        PRETRAINED_DIR
        / f"{MODEL_NAME}.pt"
    )

    # =========================================================
    # 4. Runs 設定
    # =========================================================

    RUNS_DIR = PROJECT_ROOT / "outputs" / "training_runs"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    EXPERIMENT_NAME = f"{DATASET_NAME}_{MODEL_NAME}"

    # =========================================================
    # 5. 顯示設定
    # =========================================================

    print("=" * 70)
    print("YOLO Training Configuration")
    print("=" * 70)

    print(f"Project Root     : {PROJECT_ROOT}")
    print(f"Dataset Name     : {DATASET_NAME}")
    print(f"Dataset YAML     : {DATA_YAML}")
    print(f"Model Name       : {MODEL_NAME}")
    print(f"Training Model   : {PRETRAINED_MODEL}")
    print(f"Runs Directory   : {RUNS_DIR}")
    print(f"Experiment Name  : {EXPERIMENT_NAME}")

    print("=" * 70)

    # =========================================================
    # 6. 檢查 data.yaml
    # =========================================================

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"\n找不到 data.yaml：\n{DATA_YAML}"
        )

    # =========================================================
    # 7. 載入 / 下載預訓練模型
    # =========================================================

    if PRETRAINED_MODEL.exists():
        print(f"\n找到本地訓練權重：\n{PRETRAINED_MODEL}")
        model = YOLO(str(PRETRAINED_MODEL))
    else:
        print(f"\n找不到 {MODEL_NAME}.pt，開始下載 / 載入 {MODEL_NAME}.pt...")
        downloaded_model = YOLO(f"{MODEL_NAME}.pt")
        source_path = Path(downloaded_model.ckpt_path)
        if not source_path.exists():
            raise FileNotFoundError(f"\n模型下載後找不到權重檔：{source_path}")
        shutil.copy2(source_path, PRETRAINED_MODEL)
        print(f"\n預訓練模型已儲存到：\n{PRETRAINED_MODEL}")
        model = YOLO(str(PRETRAINED_MODEL))

    # =========================================================
    # 8. 開始訓練
    # =========================================================

    print("\n" + "=" * 70)
    print("開始訓練")
    print("=" * 70 + "\n")

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=IMAGE_SIZE,
        batch=-1,
        device=0,
        patience=20,
        optimizer="auto",
        workers=8,
        project=str(RUNS_DIR),
        name=EXPERIMENT_NAME,
        exist_ok=True,
    )

    # =========================================================
    # 9. 找到訓練產生的 best.pt
    # =========================================================

    BEST_WEIGHT = (
        RUNS_DIR
        / EXPERIMENT_NAME
        / "weights"
        / "best.pt"
    )

    if not BEST_WEIGHT.exists():
        raise FileNotFoundError(f"\n訓練完成，但找不到 best.pt：\n{BEST_WEIGHT}")

    # =========================================================
    # 10. 設定 ONNX 輸出名稱
    # =========================================================

    OUTPUT_ONNX = TRAINED_DIR / f"{DATASET_NAME}_{MODEL_NAME}.onnx"
    output_index = 2
    while OUTPUT_ONNX.exists():
        OUTPUT_ONNX = TRAINED_DIR / f"{EXPERIMENT_NAME}_{output_index}.onnx"
        output_index += 1

    # =========================================================
    # 11. 在暫存目錄匯出，checkpoints 資料夾只保存 ONNX
    # =========================================================

    print(f"\n開始匯出 ONNX：\n{OUTPUT_ONNX}")

    try:
        with TemporaryDirectory(prefix="boom_onnx_") as temp_dir:
            temp_weight = Path(temp_dir) / OUTPUT_ONNX.with_suffix(".pt").name
            shutil.copy2(BEST_WEIGHT, temp_weight)
            exported_onnx = Path(YOLO(str(temp_weight)).export(
                format="onnx",
                imgsz=IMAGE_SIZE,
                batch=1,
                device="cpu",
                dynamic=False,
                simplify=False,
            ))
            if not exported_onnx.is_file():
                raise FileNotFoundError(f"找不到匯出的 ONNX 模型：{exported_onnx}")
            shutil.copy2(exported_onnx, OUTPUT_ONNX)
    except Exception as exc:
        raise RuntimeError(
            f"ONNX 匯出失敗；原始訓練權重仍保留於：{BEST_WEIGHT}"
        ) from exc

    # =========================================================
    # 12. 完成
    # =========================================================

    print("\n" + "=" * 70)
    print("Training Finished")
    print("=" * 70)
    print(f"\n完整訓練紀錄：\n{RUNS_DIR / EXPERIMENT_NAME}")
    print(f"\n原始 best.pt：\n{BEST_WEIGHT}")
    print(f"\nONNX 模型：\n{OUTPUT_ONNX}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
