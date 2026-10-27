"""Box-supervised target-change heatmaps; optionally infer after full training."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
from st2.common.utils import write_json, VIDEO_SUFFIXES
from st2.common.utils import save_checkpoint, subset


def parser():
    p = argparse.ArgumentParser(description='ST2: ROI supervision + normal prediction MSE; train then infer')
    p.add_argument('--data', type=Path, default=ROOT / 'data/st2/prepared/st2_v1')
    p.add_argument('--output', type=Path, default=ROOT / 'output/st2/training')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--history-frames', type=int, default=8, help='Past inputs; current scoring frame is separate')
    p.add_argument('--height', type=int, default=576)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--embedding-dim', type=int, default=128)
    p.add_argument('--model-variant', choices=['transformer'], default='transformer')
    p.add_argument('--patch-size', type=int, default=16, help='Transformer-only RGB patch side')
    p.add_argument('--transformer-layers', type=int, default=2, help='Transformer-only space/time block count')
    p.add_argument('--window-size', type=int, default=6, help='Transformer-only spatial window side in patches')
    p.add_argument('--camera-augmentation', type=float, default=.5, help='Training-only coherent camera trajectory probability')
    p.add_argument('--motion-loss-weight', type=float, default=.2)
    p.add_argument('--gate-floor', type=float, default=.05)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--positive-fraction', type=float, default=.5)
    p.add_argument('--workers', type=int, default=4, help='Parallel image loaders; use 0 for serial loading')
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--max-train-samples', type=int, default=0)
    p.add_argument('--max-val-samples', type=int, default=0)
    p.add_argument('--resume', type=Path)
    p.add_argument('--infer-source', type=Path, help='Run full-folder heatmap inference after final epoch succeeds')
    p.add_argument('--inference-max-frames', type=int, default=0, help='Limited inference only for preflight')
    p.add_argument('--heatmap-max', type=float, default=.02)
    p.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard event logging')
    return p


def make_loader(dataset, args, epoch, training):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    generator = torch.Generator().manual_seed(args.seed + epoch)
    sampler = None
    positive = sum(dataset.positive)
    negative = len(dataset) - positive
    if training and args.positive_fraction and positive and negative:
        weights = [args.positive_fraction / positive if p else (1 - args.positive_fraction) / negative
                   for p in dataset.positive]
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training and sampler is None,
                      sampler=sampler, num_workers=args.workers, generator=generator,
                      pin_memory=args.device.startswith('cuda'))


def log_scalars(writer, prefix, values, step):
    if writer is not None:
        for name, value in values.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                writer.add_scalar(f'{prefix}/{name}', value, step)


def run_epoch(model, criterion, loader, device, progress_path, epoch, optimizer=None, camera_augmentation=0.,
              writer=None, logger=None, total_epochs=None):
    import numpy as np
    import torch
    from scipy.stats import rankdata
    from tqdm import tqdm
    training = optimizer is not None
    model.train(training)
    total, count = {}, 0
    labels, raw_scores, weighted_scores, attention_scores = [], [], [], []
    intersection = union = predicted_pixels = actual_pixels = 0
    reconstruction_samples = 0
    started = last_report = time.perf_counter()
    phase = 'train' if training else 'val'
    description = f'Epoch {epoch}/{total_epochs or epoch} {phase}'
    # Interactive terminals reuse a single line. Redirected logs keep only epoch
    # summaries; no carriage-return progress spam in captured output.
    with torch.set_grad_enabled(training), tqdm(total=len(loader), desc=description, leave=False,
                                              dynamic_ncols=True, ascii=True, mininterval=.5,
                                              disable=not sys.stderr.isatty()) as bar:
        for step, batch in enumerate(loader, 1):
            values = [value.to(device, non_blocking=True) for value in batch]
            history, current, mask, normal = values[:4]
            sequence_masks = values[4] if len(values) == 5 else None
            if training and camera_augmentation:
                from st2.models.motion import augment_camera_clip
                history, current, mask, sequence_masks = augment_camera_clip(
                    history, current, mask, sequence_masks, camera_augmentation)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(history, current)
            losses = criterion(outputs, mask, normal, sequence_masks)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise ValueError('Non-finite loss')
            if training:
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            size = len(current)
            count += size
            reconstruction_samples += int(normal.sum())
            for name, value in losses.items():
                total[name] = total.get(name, 0.) + float(value.detach()) * size
            if not training:
                truth = mask.bool()
                predicted = outputs['attention'] >= .5
                intersection += int((truth & predicted).sum())
                union += int((truth | predicted).sum())
                predicted_pixels += int(predicted.sum())
                actual_pixels += int(truth.sum())
                labels.extend(mask.flatten(1).any(1).int().cpu().tolist())
                for key, destination in (('raw_mse', raw_scores), ('weighted_mse', weighted_scores), ('attention', attention_scores)):
                    flat = outputs[key].detach().flatten(1)
                    k = max(1, math.ceil(flat.shape[1] * .001))
                    destination.extend(flat.topk(k, dim=1).values.mean(1).cpu().tolist())
            now = time.perf_counter()
            bar.set_postfix(loss=f"{total['loss'] / count:.4f}", refresh=False)
            bar.update(1)
            if step == 1 or now - last_report >= 5 or step == len(loader):
                elapsed = now - started
                status = {'stage': phase, 'epoch': epoch, 'batch': step, 'batches': len(loader),
                          'elapsed_seconds': elapsed, 'estimated_remaining_seconds': elapsed / step * (len(loader) - step),
                          'loss': total['loss'] / count}
                write_json(progress_path, status)
                if logger is not None:
                    logger.info('%s %s/%s batch %s/%s loss=%.6f ETA=%.1fs',
                                phase, epoch, total_epochs or epoch, step, len(loader),
                                status['loss'], status['estimated_remaining_seconds'])
                if training:
                    global_step = (epoch - 1) * len(loader) + step
                    log_scalars(writer, 'train_step', {name: float(value.detach()) for name, value in losses.items()}, global_step)
                last_report = now
    result = {name: value / count for name, value in total.items()}
    result.update(seconds=time.perf_counter() - started, windows=count,
                  normal_reconstruction_windows=reconstruction_samples)
    if not training:
        y = np.asarray(labels)
        npositive, nnegative = int(y.sum()), int(len(y) - y.sum())

        def auc(values):
            if not npositive or not nnegative:
                return None
            ranks = rankdata(values, method='average')
            return float((ranks[y == 1].sum() - npositive * (npositive + 1) / 2) / (npositive * nnegative))

        result.update(weighted_frame_auc=auc(weighted_scores), raw_frame_auc=auc(raw_scores),
                      attention_frame_auc=auc(attention_scores), positive_windows=npositive,
                      roi_pixel_iou=intersection / union if union else None,
                      roi_pixel_precision=intersection / predicted_pixels if predicted_pixels else 0.,
                      roi_pixel_recall=intersection / actual_pixels if actual_pixels else None)
    return result


def selection(val):
    return (val['weighted_frame_auc'] if val['weighted_frame_auc'] is not None else -val['gate_loss'],
            -val['gate_loss'])


def main(argv=None, *, default_infer_source=None):
    p = parser()
    if default_infer_source is not None:
        p.set_defaults(infer_source=Path(default_infer_source))
        p.description = 'ST2: train, save best.pt, then run video inference after successful training'
    args = p.parse_args(argv)
    run = None
    writer = None
    logger = None
    inference_started = False
    try:
        import torch
        SummaryWriter = None
        if not args.no_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError('TensorBoard is missing. Install tensorboard or pass --no-tensorboard.') from exc
        from st2.data.sequence import dataset_fingerprint
        from st2.data.dataset import ST2Dataset
        from st2.models.network import ST2Config, ST2Model, ST2Criterion
        if min(args.epochs, args.batch_size) < 1 or min(args.workers, args.max_train_samples, args.max_val_samples,
                                                     args.inference_max_frames) < 0:
            raise ValueError('Invalid epoch, batch or sample limit')
        if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.positive_fraction < 1:
            raise ValueError('Invalid learning rate or sampling fraction')
        if not math.isfinite(args.heatmap_max) or args.heatmap_max <= 0:
            raise ValueError('heatmap-max must be positive and finite')
        if not 0 <= args.camera_augmentation <= 1 or not math.isfinite(args.motion_loss_weight) or args.motion_loss_weight < 0:
            raise ValueError('Invalid camera augmentation probability or motion loss weight')
        checkpoint = None
        if args.resume:
            args.resume = args.resume.resolve()
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
            if checkpoint.get('format') not in ('st2.checkpoint.v1', 'boom.roi_heatmap.checkpoint.v1'):
                raise ValueError('Resume requires an ST2 checkpoint')
            for name in ('history_frames', 'height', 'width', 'batch_size', 'embedding_dim', 'gate_floor',
                         'lr', 'positive_fraction', 'seed', 'max_train_samples', 'max_val_samples'):
                setattr(args, name, checkpoint['args'][name])
            # Reject historical architectures before restoring their settings.
            ST2Config.from_checkpoint(checkpoint['model_config'])
            args.model_variant = 'transformer'
            args.camera_augmentation = checkpoint['args'].get('camera_augmentation', 0.)
            args.motion_loss_weight = checkpoint['args'].get('motion_loss_weight', 0.)
            for name, default in (('patch_size', 16), ('transformer_layers', 2), ('window_size', 6)):
                setattr(args, name, checkpoint['model_config'].get(name, default))
        if args.infer_source:
            args.infer_source = args.infer_source.resolve(strict=True)
            candidates = list(args.infer_source.iterdir()) if args.infer_source.is_dir() else [args.infer_source]
            if not any(v.is_file() and v.suffix.lower() in VIDEO_SUFFIXES for v in candidates):
                raise ValueError('Inference source contains no supported videos')
        args.data = args.data.resolve()
        args.device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        device = torch.device(args.device)
        if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Use an available CUDA device or cpu')
        if device.type == 'cpu':
            torch.set_num_threads(min(8, torch.get_num_threads()))
        config = ST2Config(history_frames=args.history_frames, embedding_dim=args.embedding_dim,
                                  gate_floor=args.gate_floor, variant=args.model_variant,
                                  patch_size=args.patch_size, transformer_layers=args.transformer_layers,
                                  window_size=args.window_size)
        config.validate_image_size(args.height, args.width)
        if checkpoint and asdict(ST2Config.from_checkpoint(checkpoint['model_config'])) != asdict(config):
            raise ValueError('Checkpoint architecture mismatch')
        fingerprint = dataset_fingerprint(args.data)
        if checkpoint and checkpoint['dataset_fingerprint'] != fingerprint:
            raise ValueError('Dataset changed; start a new experiment')
        history_masks = config.uses_motion
        train = ST2Dataset(args.data, 'train', (args.height, args.width), args.history_frames, history_masks)
        val = ST2Dataset(args.data, 'val', (args.height, args.width), args.history_frames, history_masks)
        if not any(train.positive) or not any(s['normal_history'] for s in train.samples):
            raise ValueError('Training requires both target ROI windows and wholly normal historical windows')
        summary = {'train_windows': len(train), 'train_positive': sum(train.positive),
                   'train_normal_prediction': sum(s['normal_history'] for s in train.samples),
                   'val_windows': len(val), 'val_positive': sum(val.positive),
                   'history_frames': args.history_frames, 'input_size_hw': [args.height, args.width], 'device': str(device)}
        train = subset(train, args.max_train_samples, args.seed)
        val = subset(val, args.max_val_samples, args.seed + 1)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = ST2Model(config).to(device)
        criterion = ST2Criterion(motion_weight=args.motion_loss_weight).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        start = checkpoint['epoch'] + 1 if checkpoint else 1
        if start > args.epochs:
            raise ValueError('--epochs must exceed completed checkpoint epoch')
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        args.output.mkdir(parents=True, exist_ok=True)
        run = args.output.resolve() / f'st2_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
        run.mkdir(exist_ok=False)
        logger = logging.Logger(f'st2.{run.name}', level=logging.INFO)
        handler = logging.FileHandler(run/'training.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        if SummaryWriter is not None:
            writer = SummaryWriter(log_dir=str(run/'tensorboard'), flush_secs=10)
        saved_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        metadata = {'args': saved_args, 'model_config': asdict(config), 'dataset_fingerprint': fingerprint,
                    'dataset_summary': summary, 'selected_train_windows': len(train), 'selected_val_windows': len(val),
                    'selection': 'highest weighted top-0.1%-pixel frame ROC AUC, tie lower gate loss',
                    'loss': 'ST2 normal-window MSE + balanced gate BCE + positive Dice + background photometry',
                    'model_version': config.model_version,
                    'model_name': config.model_name, 'model_display_name': config.model_display_name,
                    'camera_augmentation': {'probability': args.camera_augmentation, 'training_only': True,
                                            'coherent_clip_and_masks': True},
                    'motion_loss_weight': args.motion_loss_weight,
                    'supervision': 'target inside box union; outside is interference; ignore windows excluded',
                    'torch_version': str(torch.__version__)}
        write_json(run / 'config.json', metadata)
        write_json(run / 'pipeline.json', {'stage': 'training', 'total_epochs': args.epochs, 'infer_source': saved_args['infer_source']})
        print(f'Training output: {run}', flush=True)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        logger.info('Training output: %s; dataset: %s', run, json.dumps(summary))
        if writer is not None:
            writer.add_text('configuration', json.dumps(metadata, ensure_ascii=False, indent=2), 0)
            print(f'TensorBoard logs: {run / "tensorboard"}', flush=True)
        best = (-math.inf, -math.inf)
        if checkpoint:
            candidate = checkpoint
            prior_best = args.resume.resolve().parent / 'best.pt'
            if prior_best.is_file():
                prior = torch.load(prior_best, map_location='cpu', weights_only=True)
                # An inherited best may originate before this resume run. Compare
                # training settings rather than output/resume paths or epoch limits.
                training_defaults = {'camera_augmentation': 0., 'motion_loss_weight': 0.}
                training_names = ('history_frames', 'height', 'width', 'batch_size', 'embedding_dim', 'gate_floor',
                                  'lr', 'positive_fraction', 'seed', 'max_train_samples', 'max_val_samples',
                                  'camera_augmentation', 'motion_loss_weight')
                same_training = all(prior.get('args', {}).get(name, training_defaults.get(name)) ==
                                    checkpoint['args'].get(name, training_defaults.get(name)) for name in training_names)
                if (prior.get('format') == checkpoint['format']
                        and asdict(ST2Config.from_checkpoint(prior['model_config'])) == asdict(config)
                        and prior.get('dataset_fingerprint') == fingerprint and same_training
                        and prior['epoch'] <= checkpoint['epoch'] and selection(prior['metrics']['val']) > selection(candidate['metrics']['val'])):
                    candidate = prior
            best = selection(candidate['metrics']['val'])
            save_checkpoint(candidate, run / 'best.pt')
        for epoch in range(start, args.epochs + 1):
            random.seed(args.seed + epoch)
            torch.manual_seed(args.seed + epoch)
            a = run_epoch(model, criterion, make_loader(train, args, epoch, True), device, run/'progress.json',
                          epoch, optimizer, camera_augmentation=args.camera_augmentation,
                          writer=writer, logger=logger, total_epochs=args.epochs)
            b = run_epoch(model, criterion, make_loader(val, args, 0, False), device, run/'progress.json', epoch,
                          logger=logger, total_epochs=args.epochs)
            if a['normal_reconstruction_windows'] == 0:
                raise ValueError('No normal prediction windows sampled; increase samples or reduce positive fraction')
            row = {'epoch': epoch, 'train': a, 'val': b}
            with (run / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            log_scalars(writer, 'train', a, epoch)
            log_scalars(writer, 'val', b, epoch)
            log_scalars(writer, 'optimizer', {'learning_rate': optimizer.param_groups[0]['lr']}, epoch)
            if writer is not None:
                writer.flush()
            state = {'format': 'st2.checkpoint.v1', 'epoch': epoch, 'model_config': asdict(config),
                     'args': saved_args, 'dataset_fingerprint': fingerprint, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'metrics': row}
            save_checkpoint(state, run / 'last.pt')
            if selection(b) > best:
                best = selection(b)
                save_checkpoint(state, run / 'best.pt')
            write_json(run / 'pipeline.json', {'stage': 'training', 'completed_epochs': epoch,
                                               'total_epochs': args.epochs, 'infer_source': saved_args['infer_source']})
            message = (f"Epoch {epoch}/{args.epochs} | train loss={a['loss']:.4f} | val loss={b['loss']:.4f} | "
                       f"weighted AUC={b['weighted_frame_auc']} | raw AUC={b['raw_frame_auc']}")
            print(message, flush=True)
            logger.info(message)
        if writer is not None:
            writer.close()
            writer = None
        # Inference runs only after all requested epochs and complete checkpoint saves.
        del model, optimizer, criterion, state
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if args.infer_source:
            from st2.engine.infer import run_inference
            inference_started = True
            logger.info('Training complete; starting inference from %s', run/'best.pt')
            write_json(run / 'pipeline.json', {'stage': 'inference', 'completed_epochs': args.epochs,
                                               'infer_source': saved_args['infer_source'], 'checkpoint': str(run/'best.pt')})
            result, status = run_inference(run/'best.pt', args.infer_source, run/'inference',
                                            args.device, args.batch_size, args.heatmap_max, args.inference_max_frames)
            write_json(run / 'pipeline.json', {'stage': status, 'completed_epochs': args.epochs,
                                               'inference_output': str(result), 'checkpoint': str(run/'best.pt')})
            return 1 if status == 'failed' else 0
        write_json(run / 'pipeline.json', {'stage': 'training_complete', 'completed_epochs': args.epochs})
        logger.info('Training complete: %s epochs', args.epochs)
        return 0
    except KeyboardInterrupt:
        if logger is not None:
            logger.warning('Interrupted; inference_started=%s', inference_started)
        if run:
            write_json(run / 'pipeline.json', {'stage': 'interrupted', 'last_checkpoint': str(run/'last.pt'),
                                               'inference_started': inference_started})
        print('Interrupted; completed checkpoints and partial inference outputs retained.', flush=True)
        return 130
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        if logger is not None:
            logger.exception('Training pipeline failed')
        if run:
            write_json(run / 'pipeline.json', {'stage': 'failed', 'error': str(exc)})
        p.exit(1, f'ST2: {exc}\n')
    finally:
        if writer is not None:
            writer.close()
        if logger is not None:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


if __name__ == '__main__':
    sys.exit(main())
