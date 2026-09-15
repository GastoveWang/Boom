from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


def find_local_model(model_dir, model_name):
    """Search all model subfolders for a uniquely named training checkpoint."""
    filename = f"{model_name}.pt".casefold()
    matches = sorted(
        path for path in Path(model_dir).rglob("*")
        if path.is_file() and path.name.casefold() == filename
    )
    if len(matches) > 1:
        paths = "\n".join(str(path) for path in matches)
        raise ValueError(
            f"找到多份同名模型 {model_name}.pt，請使用唯一檔名並更新 MODEL_NAME：\n{paths}"
        )
    return matches[0] if matches else None


def main():
    from ultralytics import YOLO

    # =========================================================
    # 1. 專案路徑
    # =========================================================

    # 目前這支程式的位置：
    # Boom/label/train_yolo_labeling.py
    LABEL_DIR = Path(__file__).resolve().parent

    # Boom/
    PROJECT_ROOT = LABEL_DIR.parent

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

    MODEL_DIR = LABEL_DIR / "model"

    PRETRAINED_DIR = MODEL_DIR / "pretrained"
    TRAINED_DIR = MODEL_DIR / "trained"

    # 搜尋 model 根目錄與所有子資料夾；找不到時沿用下載流程。
    PRETRAINED_MODEL = find_local_model(MODEL_DIR, MODEL_NAME) or (
        PRETRAINED_DIR
        / f"{MODEL_NAME}.pt"
    )

    # =========================================================
    # 4. Runs 設定
    # =========================================================

    RUNS_DIR = LABEL_DIR / "runs"

    EXPERIMENT_NAME = f"{DATASET_NAME}_{MODEL_NAME}"

    # =========================================================
    # 5. 建立必要資料夾
    # =========================================================

    PRETRAINED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TRAINED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================================================
    # 6. 顯示設定
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
    # 7. 檢查 data.yaml
    # =========================================================

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"\n找不到 data.yaml：\n{DATA_YAML}"
        )

    # =========================================================
    # 8. 載入 / 下載預訓練模型
    # =========================================================

    if PRETRAINED_MODEL.exists():

        print(
            f"\n找到本地訓練權重：\n"
            f"{PRETRAINED_MODEL}"
        )

        model = YOLO(
            str(PRETRAINED_MODEL)
        )

    else:

        print(
            f"\n找不到 {MODEL_NAME}.pt"
        )

        print(
            f"開始下載 / 載入 {MODEL_NAME}.pt..."
        )

        # Ultralytics 會下載模型
        downloaded_model = YOLO(
            f"{MODEL_NAME}.pt"
        )

        # 取得實際下載位置
        source_path = Path(
            downloaded_model.ckpt_path
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"\n模型下載後找不到權重檔：\n{source_path}"
            )

        # 複製到我們固定的 pretrained 資料夾
        shutil.copy2(
            source_path,
            PRETRAINED_MODEL
        )

        print(
            f"\n預訓練模型已儲存到：\n"
            f"{PRETRAINED_MODEL}"
        )

        # 重新從固定位置讀取
        model = YOLO(
            str(PRETRAINED_MODEL)
        )

    # =========================================================
    # 9. 開始訓練
    # =========================================================

    print("\n" + "=" * 70)
    print("開始訓練")
    print("=" * 70 + "\n")

    model.train(

        # Dataset
        data=str(DATA_YAML),

        # -------------------------
        # Training
        # -------------------------

        epochs=100,

        imgsz=IMAGE_SIZE,

        # 自動根據 GPU VRAM 決定 batch
        batch=-1,

        # 使用第一張 NVIDIA GPU
        device=0,

        # Early stopping
        patience=20,

        # 讓 Ultralytics 自動選 optimizer
        optimizer="auto",

        # DataLoader workers
        workers=8,

        # -------------------------
        # Output
        # -------------------------

        project=str(RUNS_DIR),

        name=EXPERIMENT_NAME,

        exist_ok=True,
    )

    # =========================================================
    # 10. 找到訓練產生的 best.pt
    # =========================================================

    BEST_WEIGHT = (
        RUNS_DIR
        / EXPERIMENT_NAME
        / "weights"
        / "best.pt"
    )

    if not BEST_WEIGHT.exists():

        raise FileNotFoundError(
            f"\n訓練完成，但找不到 best.pt：\n"
            f"{BEST_WEIGHT}"
        )

    # =========================================================
    # 11. 設定 ONNX 輸出名稱
    # =========================================================

    OUTPUT_ONNX = (
        TRAINED_DIR
        / f"{DATASET_NAME}_{MODEL_NAME}.onnx"
    )

    # 避免覆蓋先前保存的 ONNX 模型。
    output_index = 2
    while OUTPUT_ONNX.exists():
        OUTPUT_ONNX = TRAINED_DIR / f"{EXPERIMENT_NAME}_{output_index}.onnx"
        output_index += 1

    # =========================================================
    # 12. 在暫存目錄匯出，trained 資料夾只保存 ONNX
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
    # 13. 完成
    # =========================================================

    print("\n" + "=" * 70)
    print("Training Finished")
    print("=" * 70)

    print(
        f"\n完整訓練紀錄：\n"
        f"{RUNS_DIR / EXPERIMENT_NAME}"
    )

    print(
        f"\n原始 best.pt：\n"
        f"{BEST_WEIGHT}"
    )

    print(
        f"\nONNX 模型：\n"
        f"{OUTPUT_ONNX}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
