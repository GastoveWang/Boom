"""Train ST2, then infer with best.pt; run directly from the project root.

Defaults: data/st2/st2_v1, 30 epochs, automatic CPU/CUDA selection,
then inference on data/video/1440-1080/0603. Override with normal ST2 options.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from st2.engine.train import main as train_main

DEFAULT_INFER_SOURCE = ROOT / 'data/video/1440-1080/0603'


def main(argv=None):
    return train_main(argv, default_infer_source=DEFAULT_INFER_SOURCE)


if __name__ == '__main__':
    sys.exit(main())
