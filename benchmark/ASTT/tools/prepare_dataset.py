# 檔案功能：啟動 ASTT 影片區段標記 GUI，或從已保存專案預覽／匯出資料集。
# 所屬分工：tools：使用者的資料標記與 train／val（7:3）準備入口。
# 輸入／輸出：原始影片／影格及 30 FPS 異常區段 → 標記 JSON、30 FPS 版本資料集與 .npy。
# 執行方式：python benchmark/ASTT/tools/prepare_dataset.py。
# 設定與說明：參數由 main() 定義，操作見 doc/DATASET_PREPARATION.md。

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASTT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.ASTT.src.runtime.annotations import export_dataset, load_project, plan_split


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='以固定 30 FPS 標記與匯出異常時間區段，其餘影格正常；依來源群組產生 ASTT 資料集與 .npy。')
    parser.add_argument('--project', type=Path, help='載入本工具的標記 JSON')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--export', action='store_true', help='從標記專案匯出，不開啟 GUI')
    modes.add_argument('--preview', action='store_true', help='只預覽資料分組，不寫入輸出')
    parser.add_argument('--output-parent', type=Path, default=ASTT_ROOT / 'data' / 'prepared')
    parser.add_argument('--name', default='artillery-shell', help='新版本名稱前綴，預設 artillery-shell')
    parser.add_argument('--ratios', type=float, nargs=2, default=(7., 3.), metavar=('TRAIN', 'VAL'), help='train／val 比例，預設 7:3')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.export or args.preview:
        if not args.project:
            parser.error('--export／--preview 需要 --project。')
        try:
            sources = load_project(args.project)
            if args.export:
                report = export_dataset(sources, args.output_parent, args.name, args.ratios, args.seed,
                                        progress=lambda text: print(text, file=sys.stderr))
            else:
                report = plan_split(sources, args.ratios, args.seed)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            parser.error(str(exc))
    else:
        from benchmark.ASTT.src.ui.annotation_app import launch
        launch(args.project)


if __name__ == '__main__':
    main()
