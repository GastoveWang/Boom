# 檔案功能：訓練／測試 CLI、模型預設參數與預設資料／輸出路徑。
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / 'data'
OUTPUT_ROOT = PROJECT_ROOT / 'output'
PREPARED_ROOT = DATA_ROOT / 'astt' / 'prepared'
ANNOTATIONS_ROOT = OUTPUT_ROOT / 'astt' / 'annotations'
MODEL_KEYS = ('image_size', 'patch_size', 'emb_dim', 'mlp_dim', 'num_heads',
              'spatial_layers', 'temporal_layers', 'num_frames', 'dropout_rate')


def parser(training=True):
    p = argparse.ArgumentParser(description='ASTT future-frame prediction and candidate localization')
    p.add_argument('--data-dir', type=Path, default=DATA_ROOT, help='Root containing astt/; prepared datasets use their version directory')
    p.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT, help='Root containing astt/training/ and astt/results/')
    p.add_argument('--experiment-name', default=datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=0, help='0 is safest on Windows')
    p.add_argument('--registration', action='store_true', help='ECC align past inputs to latest input')
    p.add_argument('--checkpoint', type=Path, default=None, help='Initialize model / required for inference')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--threshold', type=float, default=.1, help='Absolute channel-mean squared error in [-1,1] image space')
    p.add_argument('--min-area', type=int, default=25, help='Minimum component area at model resolution')
    p.add_argument('--morph-kernel', type=int, default=3)
    p.add_argument('--min-event-frames', type=int, default=3)
    p.add_argument('--event-iou', type=float, default=.3)
    p.add_argument('--event-max-gap', type=int, default=2)
    if training:
        p.add_argument('--preflight', action='store_true', help='Diagnostic only: 4 train / 2 val batches, requires --epochs 1; keeps model settings')
        p.add_argument('--resume', type=Path, default=None, help='Restore model, optimizer, scheduler and RNG')
        p.add_argument('--epochs', type=int, default=30, help='Total target epochs, including resumed epochs')
        p.add_argument('--lr', type=float, default=1e-4)
        p.add_argument('--weight-decay', type=float, default=1e-4)
        p.add_argument('--image-size', type=int, default=256)
        p.add_argument('--patch-size', type=int, default=32)
        p.add_argument('--emb-dim', type=int, default=768)
        p.add_argument('--mlp-dim', type=int, default=3072)
        p.add_argument('--num-heads', type=int, default=8)
        p.add_argument('--spatial-layers', type=int, default=12)
        p.add_argument('--temporal-layers', type=int, default=12)
        p.add_argument('--num-frames', type=int, default=4)
        p.add_argument('--dropout-rate', type=float, default=.1)
        p.add_argument('--selection', choices=['mse', 'auc'], default='mse', help='Best checkpoint criterion, val only')
    else:
        p.add_argument('--split', choices=['val', 'test'], default='test')
    return p
