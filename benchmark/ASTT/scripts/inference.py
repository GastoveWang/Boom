# 檔案功能：載入 ASTT 權重，直接對一支新影片推論並輸出標記 MP4。
# 所屬分工：scripts 正式入口；影片串流與事件輸出在 src/video_inference.py。
# 輸入／輸出：影片與 best.pth → 30 FPS 標記影片、逐幀分數及候選事件紀錄。
# 執行方式：python benchmark/ASTT/scripts/inference.py --video <影片> --checkpoint <權重>。
# 設定與說明：不需標記／.npy；操作見 doc/VIDEO_INFERENCE.md。

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASTT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.ASTT.src.video_inference import infer_video


def main():
    p = argparse.ArgumentParser(description='Load ASTT weights and process a new video at 30 FPS')
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=ASTT_ROOT / 'output')
    p.add_argument('--experiment-name', default='video_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--registration', action='store_true', help='Must match training')
    p.add_argument('--threshold', type=float, default=.1)
    p.add_argument('--min-area', type=int, default=25, help='Area in model pixels')
    p.add_argument('--morph-kernel', type=int, default=3)
    p.add_argument('--min-event-frames', type=int, default=3)
    p.add_argument('--event-iou', type=float, default=.3)
    p.add_argument('--event-max-gap', type=int, default=2)
    p.add_argument('--display', action='store_true', help='Show preview; Q / Esc stops early')
    p.add_argument('--save-heatmaps', action='store_true', help='Also save raw error maps and PNGs')
    p.add_argument('--heatmap-max', type=float, default=.25, help='Fixed display color range upper bound; does not change detection')
    p.add_argument('--max-frames', type=int, help='Process only this many 30 FPS frames for a quick check')
    p.set_defaults(num_workers=0)
    args = p.parse_args()
    try:
        infer_video(args)
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
