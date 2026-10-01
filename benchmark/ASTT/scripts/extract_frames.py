# 檔案功能：正式 MP4 抽幀入口，保存影格與來源幀號／時間 manifest。
# 執行設定：doc/RUN_SETTINGS.md；ASTT 正式參數見 runner.py parser()，歷史 config.py 不影響正式流程。
"""Extract sequentially decoded MP4 frames; writes an index/timestamp manifest."""
import argparse
import csv
import math
from pathlib import Path
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description='Extract MP4 frames into one ASTT video folder')
    p.add_argument('video', type=Path)
    p.add_argument('--fps', type=float, default=10.)
    p.add_argument('--name', help='Output video folder name (defaults to MP4 stem)')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    p.add_argument('--data-dir', type=Path, default=PROJECT_ROOT/'data')
    args = p.parse_args()
    name = args.name or args.video.stem
    if not math.isfinite(args.fps) or args.fps <= 0 or Path(name).name != name or name in {'', '.', '..'}:
        p.error('Provide a positive finite FPS and a single folder name')
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        p.error(f'Cannot open {args.video}')
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(source_fps) or source_fps <= 0:
        cap.release()
        p.error('Video has no usable FPS metadata')
    if args.fps > source_fps + 1e-6:
        cap.release()
        p.error('Requested FPS exceeds source FPS; duplicate frames are not generated')
    root = args.data_dir.resolve()/'astt'/args.split
    output = (root if args.split == 'train' else root/'frames')/name
    try:
        output.mkdir(parents=True, exist_ok=False)
        source_index, written, next_time = 0, 0, 0.
        with (output/'frames.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['filename', 'source_frame_index', 'time_seconds'])
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                timestamp = source_index/source_fps
                if timestamp + 1e-9 >= next_time:
                    filename = f'frame_{written:08d}.jpg'
                    if not cv2.imwrite(str(output/filename), image):
                        raise IOError(f'Cannot write {output/filename}')
                    writer.writerow([filename, source_index, timestamp])
                    written += 1
                    next_time = written/args.fps
                source_index += 1
        if written == 0:
            raise ValueError('Video produced no decoded frames')
        print(f'{written} frames -> {output}; create labels in this extracted-frame order')
    finally:
        cap.release()


if __name__ == '__main__':
    main()
