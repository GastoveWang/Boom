"""Run ST2 from the project root: python src/st2 <command> [options]."""
from pathlib import Path
import argparse
from importlib import import_module
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    commands = {'train': 'train', 'infer': 'infer', 'prepare': 'prepare',
                'export-tensorboard': 'export_tensorboard', 'visualize': 'visualize'}
    parser = argparse.ArgumentParser(description='ST2 training, dataset preparation and model inference')
    parser.add_argument('command', choices=commands, help='Use <command> --help for command options')
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [sys.argv[0] + ' ' + args.command, *sys.argv[2:]]
    return import_module('st2.cli.' + commands[args.command]).main()


if __name__ == '__main__':
    sys.exit(main())
