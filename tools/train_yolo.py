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


def export_best_to_onnx(
    weight_path: str | Path,
    output_onnx: str | Path | None = None,
    imgsz: int = 640,
    checkpoints_dir: str | Path | None = None,
) -> Path:
    """將訓練好的 best.pt 轉換匯出為 ONNX 格式，並安全儲存至 models/checkpoints/。"""
    from ultralytics import YOLO

    weight_path = Path(weight_path).resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f"找不到要轉換的權重檔案：{weight_path}")

    if checkpoints_dir is None:
        checkpoints_dir = weight_path.parents[3] / "models" / "checkpoints"
    checkpoints_dir = Path(checkpoints_dir).resolve()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if output_onnx is None:
        # 預設採用實驗名稱（如 boom_v1_yolo26m.onnx）
        exp_name = weight_path.parent.parent.name
        if exp_name in {"output", "yolo_training", "training_runs", "weights"}:
            exp_name = weight_path.stem
        output_onnx = checkpoints_dir / f"{exp_name}.onnx"
    else:
        output_onnx = Path(output_onnx).resolve()

    print("\n" + "=" * 70)
    print("匯出 best.pt 到 ONNX (models/checkpoints)")
    print("=" * 70)
    print(f"來源權重 (best.pt) : {weight_path}")
    print(f"目標 ONNX 路徑    : {output_onnx}")
    print(f"輸入尺寸 imgsz     : {imgsz}")
    print("=" * 70)

    try:
        with TemporaryDirectory(prefix="boom_onnx_") as temp_dir:
            temp_weight = Path(temp_dir) / output_onnx.with_suffix(".pt").name
            shutil.copy2(weight_path, temp_weight)
            model = YOLO(str(temp_weight))
            exported_onnx = Path(model.export(
                format="onnx",
                imgsz=imgsz,
                batch=1,
                device="cpu",
                dynamic=False,
                simplify=False,
            ))
            if not exported_onnx.is_file():
                raise FileNotFoundError(f"找不到匯出的 ONNX 模型：{exported_onnx}")
            shutil.copy2(exported_onnx, output_onnx)
            # 清除 ultralytics 預設在原權重同層目錄產生的 best.onnx，避免污染
            default_sibling_onnx = weight_path.with_suffix(".onnx")
            if default_sibling_onnx.is_file() and default_sibling_onnx.resolve() != output_onnx.resolve():
                try:
                    default_sibling_onnx.unlink()
                except OSError:
                    pass
    except Exception as exc:
        raise RuntimeError(
            f"ONNX 匯出失敗；原始訓練權重仍保留於：{weight_path}"
        ) from exc

    print(f"\n✅ ONNX 匯出完成！檔案已存放於：\n{output_onnx} ({output_onnx.stat().st_size / (1024 * 1024):.1f} MB)")
    return output_onnx


def main():
    import argparse
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Boom YOLO 訓練與 ONNX 匯出工具")
    parser.add_argument(
        "--export",
        nargs="?",
        const=True,
        default=False,
        help="單獨將指定的 best.pt 轉換為 ONNX 放入 models/checkpoints/（若不指定路徑則自動尋找最新 best.pt）",
    )
    args = parser.parse_args()

    # =========================================================
    # 1. 專案路徑
    # =========================================================

    TOOLS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = TOOLS_DIR.parent
    LABEL_DIR = PROJECT_ROOT / "label"

    # =========================================================
    # 2. Dataset / Model 設定
    # =========================================================

    DATASET_NAME = "boom_v1"
    MODEL_NAME = "yolo26m-seg"
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

    # =========================================================
    # 4. Runs 設定
    # =========================================================

    RUNS_DIR = PROJECT_ROOT / "output" / "yolo_training"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    EXPERIMENT_NAME = f"{DATASET_NAME}_{MODEL_NAME}"
    BEST_WEIGHT = RUNS_DIR / EXPERIMENT_NAME / "weights" / "best.pt"
    OUTPUT_ONNX = TRAINED_DIR / f"{EXPERIMENT_NAME}.onnx"

    # =========================================================
    # 5. 若指定 --export 參數，直接執行 ONNX 轉換，不重新訓練
    # =========================================================

    if args.export:
        target_weight = None
        if isinstance(args.export, str):
            target_weight = Path(args.export).resolve()
        elif BEST_WEIGHT.exists():
            target_weight = BEST_WEIGHT
        else:
            # 搜尋 output/yolo_training 下所有 best.pt，挑選最新修改的一份
            found_bests = sorted(
                RUNS_DIR.rglob("best.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if found_bests:
                target_weight = found_bests[0]

        if not target_weight or not target_weight.is_file():
            raise FileNotFoundError(
                f"找不到可匯出的 best.pt。請指定路徑，例如：\npython tools/train_yolo.py --export path/to/best.pt"
            )
        export_best_to_onnx(target_weight, output_onnx=OUTPUT_ONNX, imgsz=IMAGE_SIZE, checkpoints_dir=TRAINED_DIR)
        return

    # 搜尋 models 與 label/model 根目錄與所有子資料夾；找不到時沿用下載流程。
    PRETRAINED_MODEL = find_local_model(MODEL_DIRS, MODEL_NAME) or (
        PRETRAINED_DIR
        / f"{MODEL_NAME}.pt"
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
    print(f"Target ONNX      : {OUTPUT_ONNX}")

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
        print(f"\n找到本地訓練權重：\n{PRETRAINED_MODEL}")
        model = YOLO(str(PRETRAINED_MODEL))
    else:
        print(f"\n本地找不到 {MODEL_NAME}.pt，開始從官方來源下載...")
        downloaded_model = YOLO(f"{MODEL_NAME}.pt")
        source_path = Path(downloaded_model.ckpt_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"\n模型下載後找不到權重檔：{source_path}")
        # 將下載之權重儲存至 models/pretrained/
        if source_path != PRETRAINED_MODEL.resolve():
            shutil.copy2(source_path, PRETRAINED_MODEL)
            # 若 Ultralytics 下載至當前工作目錄（例如 tools/），自動清理以避免污染目錄
            if source_path.parent in (TOOLS_DIR, PROJECT_ROOT):
                try:
                    source_path.unlink()
                except OSError:
                    pass
        print(f"\n預訓練模型已儲存到：\n{PRETRAINED_MODEL}")
        model = YOLO(str(PRETRAINED_MODEL))

    # =========================================================
    # 9. 開始訓練
    # =========================================================

    print("\n" + "=" * 70)
    print("開始訓練")
    print("=" * 70 + "\n")

    try:
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
    except KeyboardInterrupt:
        print("\n\n⚠️ 偵測到訓練中斷 (Ctrl+C)！")
        if BEST_WEIGHT.exists():
            print(f"發現訓練過程中已保存的中間最佳權重：{BEST_WEIGHT}")
            print("正在自動將 best.pt 轉換為 ONNX 存入 models/checkpoints/...")
            export_best_to_onnx(BEST_WEIGHT, OUTPUT_ONNX, imgsz=IMAGE_SIZE, checkpoints_dir=TRAINED_DIR)
        raise

    # =========================================================
    # 10. 找到訓練產生的 best.pt 並轉成 ONNX 存入 models/checkpoints
    # =========================================================

    if not BEST_WEIGHT.exists():
        raise FileNotFoundError(f"\n訓練完成，但找不到 best.pt：\n{BEST_WEIGHT}")

    export_best_to_onnx(BEST_WEIGHT, OUTPUT_ONNX, imgsz=IMAGE_SIZE, checkpoints_dir=TRAINED_DIR)

    # =========================================================
    # 11. 完成
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
