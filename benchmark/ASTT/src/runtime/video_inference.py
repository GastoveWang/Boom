# 檔案功能：單支影片以固定 30 FPS 串流推論，保存原圖／熱圖／疊圖／分數的成果影片。
# 所屬分工：runtime：執行環境；模型與候選框／事件邏輯沿用 detection、localization。
# 輸入／輸出：新影片、ASTT checkpoint → annotated.mp4、CSV、scores.npy、metrics.json。
# 執行方式：由 benchmark/ASTT/inference.py 呼叫 infer_video()。
# 設定與說明：30 FPS 與標記匯出共用對應表；不輸出無標籤影片的 AUC。

from collections import deque
import csv
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .annotations import Source, TARGET_FPS, sample_indices
from .dataset import align_inputs
from .runner import create_output, device_for, load_checkpoint, save_config, validate_args
from ..detection.model import VisionTransformer
from ..localization.candidates import candidate_boxes, EventMerger
from ..ui.inference_view import VIEW_WIDTH, VIEW_HEIGHT, heatmap_image, render_results


def video_frames(source, count):
    """Decode sequentially; repeat/drop exactly the same source frames as annotation export."""
    cap = cv2.VideoCapture(source.path)
    decoded = -1
    try:
        for index, source_index in enumerate(sample_indices(source)[:count]):
            source_index = int(source_index)
            while decoded < source_index:
                ok, image = cap.read()
                decoded += 1
                if not ok:
                    raise ValueError(f'Video decode failed at source frame {decoded}; output is incomplete')
            yield index, source_index, image
    finally:
        cap.release()


def pixel_box(box, width, height, size):
    return (max(0, math.floor(box[0] * width / size)), max(0, math.floor(box[1] * height / size)),
            min(width, math.ceil(box[2] * width / size)), min(height, math.ceil(box[3] * height / size)))


@torch.inference_mode()
def infer_video(args):
    validate_args(args, False)
    heatmap_max = getattr(args, 'heatmap_max', .25)
    if not math.isfinite(heatmap_max) or heatmap_max <= 0:
        raise ValueError('--heatmap-max must be a positive finite number')
    source = Source.inspect(args.video)
    if source.kind != 'video':
        raise ValueError('--video must be a video file, not an image folder')
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError('--max-frames must be positive')
    count = min(source.frame_count, args.max_frames or source.frame_count)
    device = device_for(args.device)
    print(f'Loading checkpoint on {device}.', flush=True)
    # Optimizer tensors in training checkpoints stay on CPU and are discarded after loading weights.
    checkpoint = load_checkpoint(args.checkpoint, torch.device('cpu'))
    if bool(checkpoint.get('registration', False)) != args.registration:
        raise ValueError('Registration must match checkpoint training configuration')
    config = checkpoint['model_config']
    size, history_count = config['image_size'], config['num_frames']
    if count <= history_count:
        raise ValueError(f'Need at least {history_count + 1} frames after conversion to 30 FPS')
    model = VisionTransformer(**config)
    model.load_state_dict(checkpoint['model'])
    diagnostic_weights = bool(checkpoint.get('preflight', False))
    del checkpoint
    model = model.to(device).eval()
    output = create_output(args)
    save_config(args, config, output)
    history = deque(maxlen=history_count)
    tracker = EventMerger(args.min_event_frames, args.event_iou, args.event_max_gap)
    pending, scores = [], np.full(count, np.nan, dtype=np.float32)
    writer = None
    width = height = written = 0
    stopped = preview_open = False
    started = last_report = time.perf_counter()
    window_title = 'ASTT candidates - Q / Esc to stop'
    frames = video_frames(source, count)

    def progress(force=False):
        nonlocal last_report
        now = time.perf_counter()
        if force or written == history_count + 1 or now - last_report >= 5:
            elapsed = now - started
            remaining = elapsed / written * (count - written) if written else None
            state = dict(stage='Video inference', completed_frames=written, total_frames=count,
                         elapsed_seconds=elapsed, estimated_remaining_seconds=remaining)
            temporary = output / 'progress.tmp'
            temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
            temporary.replace(output / 'progress.json')
            eta = f'{remaining / 60:.1f} min' if remaining is not None else 'pending first frame'
            print(f'[Video] {written}/{count} ({written / count:.1%}) ETA={eta}', flush=True)
            last_report = now

    def write_frame(index, source_index, image, error, rows):
        nonlocal writer, width, height, written, stopped, preview_open
        if writer is None:
            h, w = image.shape[:2]
            width, height = w, h
            writer = cv2.VideoWriter(str(output / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                     TARGET_FPS, (VIEW_WIDTH, VIEW_HEIGHT))
            if not writer.isOpened():
                raise OSError('Cannot open MP4 writer')
        displayed = []
        score = None
        if error is None:
            state = 'warming up'
        else:
            score = float(error.mean())
            scores[index] = score
            boxes = candidate_boxes(error, args.threshold, args.min_area, args.morph_kernel)
            displayed = tracker.update(index, boxes, score)
            state = 'candidate' if any(item[2] for item in displayed) else 'pending' if displayed else 'no candidate'
            displayed = [(key, pixel_box(box, width, height, size), confirmed) for key, box, confirmed in displayed]
            if args.save_heatmaps:
                np.save(output / 'heatmaps' / f'{index:08d}.npy', error)
                heat = heatmap_image(error, heatmap_max)
                if not cv2.imwrite(str(output / 'heatmaps' / f'{index:08d}.png'), heat):
                    raise OSError('Cannot write heatmap')
        rendered = render_results(image, error, displayed, scores, index, TARGET_FPS, state, args.threshold, heatmap_max)
        writer.write(rendered)
        rows.writerow([index, source_index, index / TARGET_FPS, source_index / source.fps,
                       '' if score is None else score, state])
        written += 1
        if args.display:
            factor = min(1., 1280 / VIEW_WIDTH, 720 / VIEW_HEIGHT)
            cv2.imshow(window_title, cv2.resize(rendered, (round(VIEW_WIDTH * factor), round(VIEW_HEIGHT * factor))))
            preview_open = True
            if cv2.waitKey(1) & 0xff in (27, ord('q'), ord('Q')):
                stopped = True
        progress()

    def flush_batch(rows):
        if not pending:
            return
        inputs = torch.stack([item[3] for item in pending]).to(device)
        targets = torch.stack([item[4] for item in pending]).to(device)
        errors = (model(inputs) - targets).square().mean(1).cpu().numpy()
        if not np.isfinite(errors).all():
            raise ValueError('Non-finite prediction error')
        for item, error in zip(pending, errors):
            if stopped:
                break
            write_frame(item[0], item[1], item[2], error, rows)
        pending.clear()

    try:
        progress(True)
        with (output / 'frame_scores.csv').open('w', newline='', encoding='utf-8') as stream:
            rows = csv.writer(stream)
            rows.writerow(['frame_index', 'source_frame_index', 'time_seconds', 'source_time_seconds', 'score', 'status'])
            for index, source_index, image in frames:
                resized = cv2.resize(image, (size, size))
                if len(history) < history_count:
                    write_frame(index, source_index, image, None, rows)
                else:
                    past = align_inputs(list(history)) if args.registration else list(history)
                    tensor = torch.from_numpy(np.stack([*past, resized]).transpose(0, 3, 1, 2).copy()).float() / 127.5 - 1
                    pending.append((index, source_index, image, tensor[:-1], tensor[-1]))
                    if len(pending) == args.batch_size:
                        flush_batch(rows)
                history.append(resized)
                if stopped:
                    break
            if not stopped:
                flush_batch(rows)
        np.save(output / 'scores.npy', scores[:written], allow_pickle=False)
        fields = ['video', 'event_id', 'start_frame', 'end_frame', 'confirmed_frame', 'start_seconds',
                  'end_seconds', 'confirmed_seconds', 'hits', 'x1', 'y1', 'x2', 'y2', 'peak_score', 'status']
        with (output / 'events.csv').open('w', newline='', encoding='utf-8') as stream:
            rows = csv.DictWriter(stream, fieldnames=fields)
            rows.writeheader()
            for event in tracker.finish():
                x1, y1, x2, y2 = pixel_box(event['extent'], width, height, size)
                rows.writerow(dict(video=source.name, event_id=event['event_id'], start_frame=event['start_frame'],
                                   end_frame=event['end_frame'], confirmed_frame=event['confirmed_frame'],
                                   start_seconds=event['start_frame'] / TARGET_FPS, end_seconds=event['end_frame'] / TARGET_FPS,
                                   confirmed_seconds=event['confirmed_frame'] / TARGET_FPS, hits=event['hits'],
                                   x1=x1, y1=y1, x2=x2, y2=y2, peak_score=event['peak_score'], status='candidate'))
        elapsed = time.perf_counter() - started
        valid = scores[:written][np.isfinite(scores[:written])]
        metrics = dict(video=source.path, source_fps=source.fps, output_fps=TARGET_FPS,
                       processed_frames=written, evaluated_frames=len(valid), warmup_frames=min(written, history_count),
                       processed_full_video=written == source.frame_count, stopped_early=stopped,
                       mse=float(valid.mean()) if len(valid) else None, auc=None, labelled_frames=0,
                       elapsed_seconds=elapsed, processing_fps=written / elapsed,
                       bbox_coordinate_space='source_video_pixels', source_width=width, source_height=height,
                       output_width=VIEW_WIDTH, output_height=VIEW_HEIGHT,
                       video_layout='original_heatmap_visualization_scores', heatmap_display_max=heatmap_max,
                       diagnostic_checkpoint=diagnostic_weights)
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        progress(True)
        (output / 'progress.json').write_text(json.dumps({'stage': 'Stopped' if stopped else 'Completed',
                                                         'completed_frames': written}), encoding='utf-8')
        print(f'{"Stopped early" if stopped else "Completed"}: {output / "annotated.mp4"}', flush=True)
        return metrics
    except BaseException as exc:
        (output / 'progress.json').write_text(json.dumps({'stage': 'Failed', 'completed_frames': written,
                                                         'error': str(exc)}), encoding='utf-8')
        raise
    finally:
        frames.close()
        if writer is not None:
            writer.release()
        if preview_open:
            cv2.destroyWindow(window_title)
