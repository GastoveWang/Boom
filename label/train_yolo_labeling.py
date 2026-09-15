from ultralytics import YOLO
from pathlib import Path
import shutil


def main():
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

    DATASET_NAME = "detect_v1"
    MODEL_NAME = "yolo26m"

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

    PRETRAINED_MODEL = (
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
    print(f"Pretrained Model : {PRETRAINED_MODEL}")
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
            f"\n找到本地預訓練模型：\n"
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

        imgsz=640,

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
    # 11. 將 best.pt 改成 dataset + model 名稱
    # =========================================================

    OUTPUT_MODEL = (
        TRAINED_DIR
        / f"{DATASET_NAME}_{MODEL_NAME}.pt"
    )

    shutil.copy2(
        BEST_WEIGHT,
        OUTPUT_MODEL
    )

    # =========================================================
    # 12. 完成
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
        f"\n最終模型：\n"
        f"{OUTPUT_MODEL}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
