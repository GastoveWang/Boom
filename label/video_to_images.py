"""將影片擷取為原始尺寸圖片，供後續標註使用。"""

import argparse
import math
from pathlib import Path


def extract_frames(video_path, output_dir=None, interval=0.0, image_format="png"):
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("擷取間隔必須是大於或等於 0 的有限秒數。")
    if image_format not in ("jpg", "png"):
        raise ValueError("圖片格式必須是 jpg 或 png。")
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到影片：{video_path}")

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if interval > 0 and (not math.isfinite(fps) or fps <= 0):
            raise RuntimeError("無法取得有效 FPS；可使用 --interval 0 擷取每一幀。")
        success, frame = capture.read()
        if not success:
            raise RuntimeError("影片沒有可解碼的影格。")

        if output_dir is None:
            base = Path(__file__).resolve().parents[1] / "data" / "extracted_frames"
            base.mkdir(parents=True, exist_ok=True)
            output_dir = base / video_path.stem
            index = 2
            while True:
                try:
                    output_dir.mkdir()
                    break
                except FileExistsError:
                    output_dir = base / f"{video_path.stem}_{index}"
                    index += 1
        else:
            output_dir = Path(output_dir).resolve()
            # 明確指定的目錄必須不存在，以免混入或覆蓋舊資料。
            output_dir.mkdir(parents=True, exist_ok=False)

        print(f"影片：{video_path}\n輸出：{output_dir}\nFPS：{fps:g}")
        frame_index = 0
        saved = 0
        next_sample = 0.0
        while success:
            # 依原始 FPS 換算時間；逐幀讀取，避免跳轉到關鍵幀造成重複。
            if interval == 0 or frame_index / fps + 1e-9 >= next_sample:
                destination = output_dir / f"{video_path.stem}_{frame_index:08d}.{image_format}"
                params = [cv2.IMWRITE_JPEG_QUALITY, 95] if image_format == "jpg" else []
                encoded_ok, encoded = cv2.imencode(f".{image_format}", frame, params)
                if not encoded_ok:
                    raise RuntimeError(f"圖片編碼失敗：{destination}")
                # Python 寫入支援 Windows 中文路徑，xb 禁止覆蓋既有檔案。
                with destination.open("xb") as image_file:
                    image_file.write(encoded.tobytes())
                saved += 1
                if interval > 0:
                    next_sample = (math.floor((frame_index / fps + 1e-9) / interval) + 1) * interval
                if saved % 100 == 0:
                    print(f"已擷取 {saved} 張圖片")
            frame_index += 1
            success, frame = capture.read()

        print(f"完成：讀取 {frame_index} 幀，保存 {saved} 張圖片至 {output_dir}")
        return output_dir, saved
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="輸入影片路徑")
    parser.add_argument("--output", type=Path, help="輸出至指定的新資料夾")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="每隔幾秒擷取一張（預設 0，擷取每一幀）")
    parser.add_argument("--format", choices=("jpg", "png"), default="png",
                        help="圖片格式（預設 png）")
    args = parser.parse_args()
    try:
        extract_frames(args.video, args.output, args.interval, args.format)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"錯誤：{exc}\n")


if __name__ == "__main__":
    main()
