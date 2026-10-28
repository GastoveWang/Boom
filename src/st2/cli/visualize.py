"""Export ST2 forward computation graphs using torchview."""
from pathlib import Path
import argparse
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, help='Optional trained ST2 checkpoint; otherwise use the current default architecture')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[3]/'output/st2/visualization')
    p.add_argument('--height', type=int, help='Defaults to checkpoint height or 576')
    p.add_argument('--width', type=int, help='Defaults to checkpoint width or 768')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--depth', type=int, default=2, help='Module depth; 1 for overview, 3 or more for details')
    p.add_argument('--device', choices=['meta', 'cpu'], default='meta')
    p.add_argument('--part', choices=['all', 'model', 'predictor', 'registration'], default='all',
                   help='Model/registration are explicit pipeline schematics; predictor is a torchview trace')
    p.add_argument('--format', dest='formats', nargs='+', choices=['svg', 'dot'], default=['svg'])
    p.add_argument('--direction', choices=['TB', 'LR'], default='TB')
    p.add_argument('--no-expand-nested', dest='expand_nested', action='store_false')
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        from st2.engine.visualize import export_graphs
        run, summary = export_graphs(**vars(args))
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        p.exit(1, f'ST2 torchview: {exc}\n')
    print(f'ST2 torchview output: {run}')
    for name, record in summary['parts'].items():
        print(f'  {name}: {record["status"]} ({", ".join(record["files"])})')
        for error in record['errors']:
            print(f'    {error}', file=sys.stderr)
    return 0 if summary['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
