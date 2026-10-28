"""Independent inference for normal-prediction MSE plus supervised ROI gate."""
import argparse
from collections import deque
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
from st2.common.utils import preprocess, write_json, VIDEO_SUFFIXES
from st2.engine.postprocess import ScoreNormalizer, CameraMotionGuard


def top_score(values):
    import numpy as np
    flat = values.reshape(-1)
    k = max(1, math.ceil(len(flat) * .001))
    return float(np.partition(flat, len(flat) - k)[-k:].mean())


def infer_one(source, directory, model, height, width, args, device, progress_path, progress):
    import cv2
    import numpy as np
    import torch
    from st2.engine.view import render_st2

    directory.mkdir(exist_ok=False)
    cap = cv2.VideoCapture(str(source))
    writer = None
    started = last_log = time.perf_counter()
    result = {'source': str(source), 'output': str(directory), 'status': 'running',
              'written_frames': 0, 'scored_frames': 0, 'max_raw_score': None, 'max_weighted_score': None}
    history = deque(maxlen=model.config.history_frames + 1)
    normalizer = ScoreNormalizer(args.score_max, args.score_knee)
    guard = CameraMotionGuard(model.config.history_frames, rotation_limit=args.motion_rotation_limit,
                              translation_limit=args.motion_translation_limit,
                              residual_floor=args.motion_residual_floor) if args.motion_mode == 'guard' else None
    curve = deque()
    pending = []

    def emit(frame, index, raw=None, weighted=None, attention=None, support=None, motion=None):
        motion = motion or {'motion_status': 'warming_up', 'motion_score_valid': False}
        row = {'frame_index': index, 'time_seconds': index / fps, 'status': 'warming_up' if raw is None else 'scored',
               'raw_mse_mean': None, 'weighted_mse_mean': None, 'raw_top_score': None,
               'weighted_top_score': None, 'attention_max': None, 'guarded_top_score': None,
               'raw_score_01': None, 'learned_weighted_score_01': None, 'target_score_01': None,
               'motion_status': motion['motion_status'], 'motion_score_valid': motion['motion_score_valid'],
               **{k: motion.get(k) for k in ('rotation_deg_per_second', 'translation_diagonal_per_second',
                                             'motion_inlier_ratio', 'motion_overlap', 'motion_residual_threshold',
                                             'motion_support_fraction')}}
        final_map = weighted
        if raw is not None:
            a, b = top_score(raw), top_score(weighted)
            if guard is not None and motion['motion_score_valid']:
                final_map = weighted * support
            final_score = top_score(final_map) if motion['motion_score_valid'] else None
            normalized_raw = normalizer(a)
            normalized_target = normalizer(final_score) if final_score is not None else None
            row.update(raw_mse_mean=float(raw.mean()), weighted_mse_mean=float(weighted.mean()),
                       raw_top_score=a, weighted_top_score=b, attention_max=float(attention.max()),
                       guarded_top_score=final_score, raw_score_01=normalized_raw,
                       learned_weighted_score_01=normalizer(b), target_score_01=normalized_target,
                       status='scored' if normalized_target is not None else 'camera_motion_uncertain')
            result['scored_frames'] += 1
            result['valid_target_score_frames'] += normalized_target is not None
            result['camera_uncertain_frames'] += normalized_target is None
            result['max_raw_score'] = max(result['max_raw_score'] or 0., a)
            result['max_weighted_score'] = max(result['max_weighted_score'] or 0., b)
            curve.append((index / fps, normalized_raw, normalized_target))
            while curve and curve[0][0] < index / fps - 10:
                curve.popleft()
            if args.save_heatmaps:
                np.savez_compressed(directory / 'heatmaps' / f'{index:08d}.npz',
                                    raw_mse=raw, weighted_mse=weighted, attention=attention,
                                    displayed_mse=final_map,
                                    target_score_valid=np.array(motion['motion_score_valid']))
        csv_writer.writerow(row)
        writer.write(render_st2(frame, raw, final_map, curve, index, fps, args.heatmap_max,
                                       motion_status=motion['motion_status'], normalized=True,
                                       raw_values=(row['raw_top_score'], row['weighted_top_score']),
                                       model_name=model.config.model_name))
        result['written_frames'] += 1

    def flush():
        if not pending:
            return
        past = torch.stack([p[0][:-1] for p in pending]).to(device)
        current = torch.stack([p[0][-1] for p in pending]).to(device)
        with torch.inference_mode():
            outputs = model(past, current)
        maps = {key: outputs[key].detach().float().cpu().numpy()[:, 0]
                for key in ('raw_mse', 'weighted_mse', 'attention')}
        if not all(np.isfinite(value).all() for value in maps.values()):
            raise ValueError('Non-finite heatmap')
        for i, (_, frame, index, support, motion) in enumerate(pending):
            emit(frame, index, maps['raw_mse'][i], maps['weighted_mse'][i], maps['attention'][i], support, motion)
        pending.clear()

    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open {source}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        source_width, source_height = int(cap.get(3)), int(cap.get(4))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or min(source_width, source_height) <= 0:
            raise ValueError('Invalid video FPS or dimensions')
        result.update(source_width=source_width, source_height=source_height, fps=fps, reported_frames=count)
        result.update(valid_target_score_frames=0, camera_uncertain_frames=0)
        writer = cv2.VideoWriter(str(directory / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 900))
        if not writer.isOpened():
            raise ValueError('Cannot open MP4 writer')
        if args.save_heatmaps:
            (directory / 'heatmaps').mkdir()
        with (directory / 'frame_scores.csv').open('x', newline='', encoding='utf-8') as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=['frame_index', 'time_seconds', 'status', 'raw_mse_mean',
                'weighted_mse_mean', 'raw_top_score', 'weighted_top_score', 'attention_max', 'guarded_top_score',
                'raw_score_01', 'learned_weighted_score_01', 'target_score_01', 'motion_status', 'motion_score_valid',
                'rotation_deg_per_second', 'translation_diagonal_per_second', 'motion_inlier_ratio',
                'motion_overlap', 'motion_residual_threshold', 'motion_support_fraction'])
            csv_writer.writeheader()
            index = 0
            while not args.max_frames or index < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (source_height, source_width):
                    raise ValueError('Source resolution changed mid-video')
                if guard is not None:
                    support, motion = guard.update(frame, fps, (height, width))
                else:
                    support, motion = None, {'motion_status': 'off', 'motion_score_valid': True}
                history.append(preprocess(frame, height, width))
                if len(history) <= model.config.history_frames:
                    emit(frame, index, motion=motion)
                else:
                    pending.append((torch.stack(list(history)), frame, index, support, motion))
                    if len(pending) >= args.batch_size:
                        flush()
                index += 1
                if time.perf_counter() - last_log >= 5:
                    progress.update(video=str(source), frames=index, reported_frames=count)
                    write_json(progress_path, progress)
                    print(f'  {source.name}: {index}/{count} frames', flush=True)
                    last_log = time.perf_counter()
            flush()
        if not index or (not args.max_frames and count > 0 and index < count):
            raise ValueError(f'Incomplete decode: {index}/{count}')
        result['status'] = 'limited' if args.max_frames and index >= args.max_frames else 'complete'
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        result.update(status='failed', error=str(exc))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    result['seconds'] = time.perf_counter() - started
    write_json(directory / 'summary.json', result)
    return result


def run_inference(checkpoint, source, output, device='auto', batch_size=4, heatmap_max=.02,
                  max_frames=0, save_heatmaps=False, score_max=.02, score_knee=.0001, motion_mode='off',
                  motion_rotation_limit=60., motion_translation_limit=.75, motion_residual_floor=.015,
                  *, direct_output=False):
    import torch
    from st2.models.network import ST2Config, ST2Model
    if batch_size < 1 or max_frames < 0 or not math.isfinite(heatmap_max) or heatmap_max <= 0:
        raise ValueError('Invalid inference parameters')
    normalizer = ScoreNormalizer(score_max, score_knee)
    if motion_mode not in ('off', 'guard'):
        raise ValueError('motion-mode must be off or guard')
    CameraMotionGuard(rotation_limit=motion_rotation_limit, translation_limit=motion_translation_limit,
                      residual_floor=motion_residual_floor)
    source = Path(source).resolve(strict=True)
    items = sorted(source.iterdir()) if source.is_dir() else [source]
    videos = [p for p in items if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    if not videos:
        raise ValueError('No supported videos at inference source')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if direct_output:
        if any(output.iterdir()):
            raise FileExistsError(f'Inference output directory is not empty: {output}')
        run = output
    else:
        run = output / f'st2_inference_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
        run.mkdir(exist_ok=False)
    snapshot = run / 'checkpoint.pt'
    checkpoint = Path(checkpoint).resolve()
    shutil.copyfile(checkpoint, snapshot)
    ck = torch.load(snapshot, map_location='cpu', weights_only=True)
    if ck.get('format') not in ('st2.checkpoint.v1', 'boom.roi_heatmap.checkpoint.v1'):
        raise ValueError('Requires an ST2 checkpoint')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Use an available CUDA device or cpu')
    if device.type == 'cpu':
        torch.set_num_threads(min(8, torch.get_num_threads()))
    config = ST2Config.from_checkpoint(ck['model_config'])
    config.validate_image_size(ck['args']['height'], ck['args']['width'])
    model = ST2Model(config).to(device).eval()
    model.load_state_dict(ck['model'])
    height, width = ck['args']['height'], ck['args']['width']
    args = argparse.Namespace(batch_size=batch_size, heatmap_max=heatmap_max,
                              max_frames=max_frames, save_heatmaps=save_heatmaps,
                              score_max=score_max, score_knee=score_knee, motion_mode=motion_mode,
                              motion_rotation_limit=motion_rotation_limit, motion_translation_limit=motion_translation_limit,
                              motion_residual_floor=motion_residual_floor)
    manifest = {'status': 'running', 'checkpoint_source': str(Path(checkpoint).resolve()),
                'checkpoint_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(), 'epoch': ck['epoch'],
                'model_config': asdict(config), 'input_size_hw': [height, width], 'source': str(source),
                'model_version': config.model_version,
                'model_name': config.model_name, 'model_display_name': config.model_display_name,
                'learned_background_motion': config.uses_motion,
                'heatmap_max': heatmap_max, 'batch_size': batch_size, 'max_frames': max_frames,
                'save_heatmaps': save_heatmaps, 'score_top_fraction': .001, 'frame_index_base': 0,
                'raw_mse_definition': 'mean RGB squared error in [0,1], predicted from past frames only',
                'weighted_definition': 'raw_mse * (gate_floor + (1-gate_floor)*attention); not raw MSE',
                'postprocess_version': 'roi_heatmap.v2.1',
                'normalization': {'formula': 'clip(log1p(score/knee)/log1p(maximum/knee),0,1)',
                                  'maximum': normalizer.maximum, 'knee': normalizer.knee,
                                  'scope': 'fixed and shared across videos and score types; not a probability'},
                'motion_guard': {'mode': motion_mode, 'rotation_limit_deg_per_second': motion_rotation_limit,
                                 'translation_limit_diagonal_per_second': motion_translation_limit,
                                 'residual_floor': motion_residual_floor, 'recovery_frames': config.history_frames,
                                 'invalid_score': 'blank/null; diagnostic heatmap retained; not zero'},
                'skipped_files': [str(p) for p in items if p.is_file() and p not in videos], 'videos': []}
    write_json(run / 'summary.json', manifest)
    print(f'Heatmap inference output: {run}', flush=True)
    progress = {'status': 'running', 'videos_total': len(videos), 'videos_finished': 0}
    write_json(run / 'progress.json', progress)
    for i, video in enumerate(videos, 1):
        print(f'Heatmap video {i}/{len(videos)}: {video.name}', flush=True)
        result = infer_one(video, run / f'{i:03d}_{video.stem}', model, height, width, args,
                           device, run / 'progress.json', progress)
        manifest['videos'].append(result)
        progress['videos_finished'] = i
        write_json(run / 'summary.json', manifest)
    manifest['status'] = ('failed' if any(v['status'] == 'failed' for v in manifest['videos']) else
                          'limited' if max_frames else 'complete')
    progress['status'] = manifest['status']
    write_json(run / 'summary.json', manifest)
    write_json(run / 'progress.json', progress)
    return run, manifest['status']


def main():
    p = argparse.ArgumentParser(description='8 past frames -> normal prediction -> raw and ROI-weighted MSE heatmaps')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'output/st2/inference')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--heatmap-max', type=float, default=.02)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--save-heatmaps', action='store_true')
    p.add_argument('--score-max', type=float, default=.02, help='Fixed score mapped to 1.00; shared across videos')
    p.add_argument('--score-knee', type=float, default=.0001, help='Fixed log normalization sensitivity; not per-video min/max')
    p.add_argument('--motion-mode', choices=['off', 'guard'], default='off',
                   help='Optional inference-only comparison; default off. Does not train camera-motion understanding.')
    p.add_argument('--motion-rotation-limit', type=float, default=60., help='Degrees/second; faster is uncertain')
    p.add_argument('--motion-translation-limit', type=float, default=.75, help='Image diagonals/second')
    p.add_argument('--motion-residual-floor', type=float, default=.015, help='Local change check in grayscale [0,1]')
    args = p.parse_args()
    try:
        _, status = run_inference(**vars(args))
        return 1 if status == 'failed' else 0
    except (ImportError, ValueError, OSError, RuntimeError, KeyError) as exc:
        p.exit(1, f'ST2 inference: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
