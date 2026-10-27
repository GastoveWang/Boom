"""Convert existing metrics.jsonl to a new TensorBoard event directory."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid

from st2.engine.train import log_scalars


def export_metrics(run):
    from torch.utils.tensorboard import SummaryWriter
    run = Path(run).resolve(strict=True)
    # Validate before creating output; preserve the original JSONL and checkpoints.
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('No epoch metrics to export')
    epochs = [row['epoch'] for row in rows]
    if any(type(epoch) is not int or epoch < 1 for epoch in epochs) or epochs != sorted(set(epochs)):
        raise ValueError('Epoch numbers must be positive, unique and increasing')
    if any(not isinstance(row.get(phase), dict) for row in rows for phase in ('train', 'val')):
        raise ValueError('Each epoch requires train and val metrics')
    output = run/f'tensorboard_import_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
    output.mkdir(exist_ok=False)
    with SummaryWriter(log_dir=str(output)) as writer:
        for row in rows:
            for phase in ('train', 'val'):
                log_scalars(writer, phase, row[phase], row['epoch'])
        config = run/'config.json'
        if config.is_file():
            writer.add_text('configuration', config.read_text(encoding='utf-8'), 0)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Existing training run containing metrics.jsonl')
    args = parser.parse_args()
    try:
        print(f'TensorBoard logs: {export_metrics(args.run)}')
    except (ImportError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'TensorBoard export: {exc}\n')


if __name__ == '__main__':
    main()
