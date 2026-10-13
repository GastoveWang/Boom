# 檔案功能：正式訓練／測試 CLI、預設路徑與模型設定欄位。
# 所屬分工：config：相當於 Boom 的 config 設定層。
# 輸入／輸出：命令列參數 → argparse Namespace；預設讀取 Boom/data、寫入 Boom/output。
# 執行方式：由入口與 runner 呼叫 parser()，不直接執行。
# 設定與說明：正式參數見 src/config/cli.py；說明見 doc/RUN_SETTINGS.md。

import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_KEYS = ('image_size', 'patch_size', 'emb_dim', 'mlp_dim', 'num_heads',
              'spatial_layers', 'temporal_layers', 'num_frames', 'dropout_rate')


def parser(training=True):
    p = argparse.ArgumentParser(description='ASTT future-frame prediction and candidate localization')
    p.add_argument('--data-dir', type=Path, default=PROJECT_ROOT/'data', help='Root containing astt/')
    p.add_argument('--output-dir', type=Path, default=PROJECT_ROOT/'output')
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
