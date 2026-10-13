# 檔案功能：訓練、validation、測試推論、checkpoint 與結果保存。
# 所屬分工：runtime：相當於 Boom 的執行流程層。
# 輸入／輸出：連續影格與設定 → 模型權重、scores、heatmaps、候選事件與 metrics。
# 執行方式：由 train.py／test.py 呼叫 train()／infer()，不直接執行。
# 設定與說明：正式參數見 src/config/cli.py；說明見 doc/RUN_SETTINGS.md。

"""Windows-friendly training / validation / held-out inference."""
import csv
import json
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import VideoDataset
from ..detection.model import VisionTransformer
from ..localization.candidates import EventMerger, candidate_boxes
from ..config.cli import MODEL_KEYS, PROJECT_ROOT, parser

EVENT_FIELDS = ['video', 'event_id', 'start_frame', 'end_frame', 'confirmed_frame',
                'hits', 'x1', 'y1', 'x2', 'y2', 'peak_score', 'status']


def device_for(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available() else 'cpu' if name == 'auto' else name)


def validate_args(args, training):
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('Invalid batch size / worker count')
    if args.threshold < 0 or args.min_area < 1 or args.morph_kernel < 1 or args.morph_kernel % 2 == 0:
        raise ValueError('Invalid threshold / minimum area / odd morphology kernel')
    EventMerger(args.min_event_frames, args.event_iou, args.event_max_gap)
    if Path(args.experiment_name).name != args.experiment_name or args.experiment_name in {'.', '..', ''}:
        raise ValueError('experiment-name must be a single directory name')
    if training and (args.epochs < 1 or args.lr <= 0 or args.resume and args.checkpoint):
        raise ValueError('Invalid epochs / lr, or conflicting resume and checkpoint')
    if training and args.preflight and (args.epochs != 1 or args.resume):
        raise ValueError('Preflight requires --epochs 1 and cannot resume a full training schedule')


def load_checkpoint(path, device):
    # Only this project's tensor / primitive checkpoints; no arbitrary pickle objects.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not {'model', 'model_config'} <= checkpoint.keys():
        raise ValueError('Expected local ASTT checkpoint; upstream weights are not automatically compatible')
    return checkpoint


def dataset(args, split, config):
    root = args.data_dir.resolve()/'astt'/split
    ds = VideoDataset(root if split == 'train' else root/'frames', config['image_size'],
        config['num_frames'], root/'test_frame_masks', args.registration, normal_only=split=='train')
    if not len(ds):
        raise ValueError(f'No valid {split} windows under {root}; need at least {config["num_frames"]+1} contiguous frames per video')
    if split == 'train':
        print('Training normal clips only: unlabelled clips must be manually verified normal.')
    return ds


def assert_disjoint(a, b):
    overlap = set(a.videos) & set(b.videos)
    if overlap:
        raise ValueError(f'Use distinct video names for train and val (potential leakage): {sorted(overlap)}')
    files_a = {p.resolve() for frames in a.videos.values() for p in frames}
    files_b = {p.resolve() for frames in b.videos.values() for p in frames}
    if files_a & files_b:
        raise ValueError('Train and val reference overlapping files')


def loader(ds, args, shuffle=False):
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=False)


def progress_batches(batches, stage, path=None):
    """Report completed batches after processing, including image reads and GPU work."""
    total = len(batches)
    started = last_report = time.perf_counter()
    def report(step):
        elapsed = time.perf_counter() - started
        eta = elapsed / step * (total - step) if step else None
        status = dict(stage=stage, completed_batches=step, total_batches=total,
                      elapsed_seconds=elapsed, estimated_remaining_seconds=eta)
        remaining = f'{eta / 60:.1f} min' if eta is not None else 'pending first batch'
        print(f'[{stage}] {step}/{total} ({step / max(1, total):.1%}) '
              f'elapsed={elapsed / 60:.1f} min ETA={remaining}', flush=True)
        if path:
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
            temporary.replace(path)
    report(0)
    for step, batch in enumerate(batches, 1):
        if step == 1:
            print(f'[{stage}] First batch loaded; processing.', flush=True)
        yield batch
        now = time.perf_counter()
        if step == 1 or step == total or now - last_report >= 5:
            report(step)
            last_report = now


def limit_preflight(ds, count, balanced=False):
    """Select windows in memory; never copy, remove or relabel source images."""
    if balanced:
        buckets = {}
        for index, (video, start) in enumerate(ds.samples):
            label = int(ds.labels[video][start + ds.num_frames]) if video in ds.labels else -1
            buckets.setdefault(label, []).append(index)
        chosen = []
        quota = max(1, count // len(buckets))
        for indices in buckets.values():
            chosen.extend(indices[i] for i in np.linspace(0, len(indices) - 1, min(quota, len(indices)), dtype=int))
    else:
        chosen = np.linspace(0, len(ds.samples) - 1, min(count, len(ds.samples)), dtype=int).tolist()
    ds.samples = [ds.samples[i] for i in sorted(set(chosen))]


def auc_score(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return None
    # Mann-Whitney AUROC, average ranks for ties; no sklearn dependency required.
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), float)
    start = 0
    while start < len(scores):
        end = start+1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start+1+end)/2
        start = end
    positives = labels == 1
    n1, n0 = int(positives.sum()), int((labels == 0).sum())
    return float((ranks[positives].sum()-n1*(n1+1)/2)/(n1*n0))


@torch.inference_mode()
def evaluate(model, ds, args, device, output=None, stage='Validation', progress_path=None):
    model.eval()
    scores = {v: np.full(len(f), np.nan, dtype=np.float32) for v, f in ds.videos.items()}
    trackers = {v: EventMerger(args.min_event_frames, args.event_iou, args.event_max_gap) for v in ds.videos}
    loss_sum, count, labels_all, scores_all = 0., 0, [], []
    start_time = time.perf_counter()
    for batch in progress_batches(loader(ds, args), stage, progress_path):
        prediction = model(batch['inputs'].to(device))
        error = (prediction - batch['target'].to(device)).square().mean(1).cpu().numpy()
        if not np.isfinite(error).all():
            raise ValueError('Non-finite prediction error; inspect model/checkpoint and inputs')
        for i, error_map in enumerate(error):
            video, frame = batch['video'][i], int(batch['frame_index'][i])
            score = float(error_map.mean())
            scores[video][frame] = score
            loss_sum += score
            count += 1
            label = int(batch['label'][i])
            if label >= 0:
                labels_all.append(label)
                scores_all.append(score)
            if output:
                boxes = candidate_boxes(error_map, args.threshold, args.min_area, args.morph_kernel)
                displayed = trackers[video].update(frame, boxes, score)
                heat_dir, vis_dir = output/'heatmaps'/video, output/'visualizations'/video
                heat_dir.mkdir(parents=True, exist_ok=True)
                vis_dir.mkdir(parents=True, exist_ok=True)
                np.save(heat_dir/f'{frame:08d}.npy', error_map)
                # Fixed absolute color scale [0,4]; never min-max normalize per frame.
                heat = cv2.applyColorMap(np.clip(error_map/4*255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
                if not cv2.imwrite(str(heat_dir/f'{frame:08d}.png'), heat):
                    raise IOError('Failed writing heatmap')
                image = np.clip((batch['target'][i].numpy().transpose(1,2,0)+1)*127.5, 0,255).astype(np.uint8)
                image = cv2.addWeighted(image, .65, heat, .35, 0)
                for key, box, confirmed in displayed:
                    cv2.rectangle(image, box[:2], box[2:], (0,255,255) if confirmed else (128,128,128), 1)
                    cv2.putText(image, f'{key}: candidate' if confirmed else f'{key}: pending', box[:2],
                                cv2.FONT_HERSHEY_SIMPLEX, .35, (0,255,255), 1)
                if not cv2.imwrite(str(vis_dir/f'{frame:08d}.jpg'), image):
                    raise IOError('Failed writing visualization')
    per_video = {}
    for video, values in scores.items():
        valid = np.isfinite(values)
        if output:
            np.save(output/'scores'/f'{video}.npy', values)
        if video in ds.labels:
            per_video[video] = auc_score(ds.labels[video][valid], values[valid])
    elapsed = time.perf_counter()-start_time
    metrics = dict(mse=loss_sum/count, auc=auc_score(labels_all, scores_all),
                   per_video_auc=per_video, evaluated_frames=count, labelled_frames=len(labels_all),
                   missing_label_videos=sorted(set(ds.videos)-set(ds.labels)),
                   elapsed_seconds=elapsed, processing_fps=count/elapsed)
    metrics['preflight'] = bool(getattr(args, 'preflight', False))
    if output:
        with (output/'events.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
            writer.writeheader()
            for video, tracker in trackers.items():
                for event in tracker.finish():
                    x1,y1,x2,y2 = event['extent']
                    writer.writerow(dict(video=video, event_id=event['event_id'], start_frame=event['start_frame'],
                        end_frame=event['end_frame'], confirmed_frame=event['confirmed_frame'], hits=event['hits'],
                        x1=x1,y1=y1,x2=x2,y2=y2,peak_score=event['peak_score'],status='candidate'))
        (output/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        mapping = {v: [p.name for p in frames] for v, frames in ds.videos.items()}
        (output/'frame_index.json').write_text(json.dumps(mapping, indent=2), encoding='utf-8')
    return metrics


def create_output(args):
    output = args.output_dir.resolve()/'astt'/args.experiment_name
    output.mkdir(parents=True, exist_ok=False)  # never silently overwrite experiments
    for folder in ('checkpoints','scores','heatmaps','visualizations'):
        (output/folder).mkdir()
    with (output/'events.csv').open('w', newline='', encoding='utf-8') as stream:
        csv.writer(stream).writerow(EVENT_FIELDS)
    return output


def save_config(args, config, output):
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items()}
    settings['model_config'] = config
    settings['project_root'] = str(PROJECT_ROOT)
    settings['method'] = 'ASTT STE+TTE+CSA; spatial localization and event merging are Boom extensions'
    (output/'config.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')


def train(args):
    validate_args(args, True)
    print(f'Initializing ASTT on {args.device}; loading model and datasets.', flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    checkpoint = load_checkpoint(args.resume or args.checkpoint, device) if args.resume or args.checkpoint else None
    config = checkpoint['model_config'] if checkpoint else {key: getattr(args, key) for key in MODEL_KEYS}
    if checkpoint and bool(checkpoint.get('registration', False)) != args.registration:
        raise ValueError('Registration must match training checkpoint')
    model = VisionTransformer(**config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
    print('Model ready; loading train windows.', flush=True)
    train_ds = dataset(args, 'train', config)
    print('Loading validation windows.', flush=True)
    val_ds = dataset(args, 'val', config)
    print('Checking train / val separation.', flush=True)
    assert_disjoint(train_ds, val_ds)
    print('Dataset checks completed.', flush=True)
    if args.preflight:
        limit_preflight(train_ds, args.batch_size * 4)
        limit_preflight(val_ds, args.batch_size * 2, balanced=True)
        print('PREFLIGHT: limited real-data windows; results check execution only.', flush=True)
    if args.selection == 'auc' and (set(val_ds.labels) != set(val_ds.videos) or
            len(set(int(val_ds.labels[v][s+config['num_frames']]) for v,s in val_ds.samples)) < 2):
        raise ValueError('AUC selection requires labels for every val video and both scored classes')
    training_loader = loader(train_ds, args, True)
    print(f'Train: {len(train_ds)} windows / {len(training_loader)} batches; '
          f'val: {len(val_ds)} windows / {len(loader(val_ds, args))} batches; '
          f'batch_size={args.batch_size}, workers={args.num_workers}.', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
        total_steps=args.epochs*len(training_loader))
    start_epoch, best = 0, float('-inf') if args.selection == 'auc' else float('inf')
    if args.resume:
        if checkpoint['epochs'] != args.epochs or checkpoint['steps_per_epoch'] != len(training_loader) or checkpoint['selection'] != args.selection:
            raise ValueError('Resume requires original total epochs, steps per epoch and selection; use --checkpoint for a new schedule')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch, best = checkpoint['epoch'] + 1, checkpoint['best']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if torch.cuda.is_available() and checkpoint.get('cuda_rng'):
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        if start_epoch >= args.epochs:
            raise ValueError('Checkpoint already reached requested total epochs')
        prior_best_path = args.resume.resolve().parent/'best.pth'
        if not prior_best_path.is_file():
            raise ValueError('Resume requires sibling best.pth to preserve the previous best model')
        prior_best = load_checkpoint(prior_best_path, device)
        if prior_best['model_config'] != config or prior_best.get('best') != best:
            raise ValueError('Sibling best.pth does not match the resumed experiment')
    output = create_output(args)
    save_config(args, config, output)
    if args.resume:
        torch.save(prior_best, output/'checkpoints'/'best.pth')
    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total, count = 0., 0
        for batch in progress_batches(training_loader, f'Train epoch {epoch + 1}/{args.epochs}', output/'progress.json'):
            inputs, target = batch['inputs'].to(device), batch['target'].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs), target)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss; stopping before optimizer update')
            if args.preflight and count == 0:
                print('Preflight forward completed; starting backward.', flush=True)
            loss.backward()
            if args.preflight and count == 0:
                print('Preflight backward completed; updating optimizer.', flush=True)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(inputs)
            count += len(inputs)
        metrics = evaluate(model, val_ds, args, device, stage=f'Val epoch {epoch + 1}/{args.epochs}',
                           progress_path=output/'progress.json')
        value = metrics[args.selection]
        improved = value > best if args.selection == 'auc' else value < best
        if improved:
            best = value
        state = dict(model=model.state_dict(), model_config=config, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), epoch=epoch, best=best, epochs=args.epochs,
            steps_per_epoch=len(training_loader), selection=args.selection, registration=args.registration,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            preflight=args.preflight)
        print(f'Saving epoch {epoch + 1} checkpoints.', flush=True)
        torch.save(state, output/'checkpoints'/'last.pth')
        if improved:
            torch.save(state, output/'checkpoints'/'best.pth')
        row = dict(epoch=epoch+1, train_mse=total/count, val=metrics)
        history.append(row)
        (output/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    best_checkpoint = load_checkpoint(output/'checkpoints'/'best.pth', device)
    model.load_state_dict(best_checkpoint['model'])
    print('Reloading best checkpoint for validation outputs.', flush=True)
    evaluate(model, val_ds, args, device, output, stage='Best-model validation', progress_path=output/'progress.json')
    (output/'progress.json').write_text(json.dumps({'stage': 'Completed', 'preflight': args.preflight}), encoding='utf-8')
    print(f'Completed; best-model validation outputs: {output}', flush=True)


def infer(args):
    validate_args(args, False)
    if args.checkpoint is None:
        raise ValueError('Inference requires --checkpoint; random weights are not anomaly detection')
    device = device_for(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    if bool(checkpoint.get('registration', False)) != args.registration:
        raise ValueError('Registration must match checkpoint training configuration')
    config = checkpoint['model_config']
    model = VisionTransformer(**config).to(device)
    model.load_state_dict(checkpoint['model'])
    ds = dataset(args, args.split, config)
    output = create_output(args)
    save_config(args, config, output)
    print(json.dumps(evaluate(model, ds, args, device, output), indent=2))
    print(f'Outputs: {output}')
