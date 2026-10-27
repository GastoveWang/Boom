# 檔案功能：Boom Smoke Heatmap 逐幀標註 UI、分組預覽及資料集版本匯出入口。
"""Launch the manual BBox UI or prepare a dataset from a saved project."""
import argparse
import json
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from st2.data.annotations import DATASET_ROOT, export_dataset, load_project, plan_split


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='ST2：時間軌、區段同位套用及逐幀微調；不需標物體類別。')
    parser.add_argument('--project', type=Path, help='載入本工具保存的標註 JSON')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preview', action='store_true', help='預覽群組分配，不開 UI 或寫入輸出')
    modes.add_argument('--export', action='store_true', help='從已完成的標註專案匯出新版本')
    parser.add_argument('--output-parent', type=Path, default=DATASET_ROOT)
    parser.add_argument('--name', default='st2')
    parser.add_argument('--ratios', type=float, nargs=2, default=(7., 3.), metavar=('TRAIN', 'VAL'))
    parser.add_argument('--history-frames', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    try:
        if args.preview or args.export:
            if not args.project:
                parser.error('--preview／--export 需要 --project。')
            project = load_project(args.project)
            if args.export:
                report = export_dataset(project, args.output_parent, args.name, args.ratios,
                    args.seed, args.history_frames, progress=lambda text: print(text, file=sys.stderr))
            else:
                report = plan_split(project, args.ratios, args.seed, args.history_frames)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            from st2.data.editor import launch
            launch(args.project, args.output_parent, args.name, args.ratios, args.history_frames, args.seed)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
