"""Original/MSE/target-weighted overlay panels and causal score curves."""
import cv2
import numpy as np


def render_st2(frame, raw, weighted, scores, index, fps, maximum=.02, *,
                       motion_status=None, normalized=False, raw_values=None, model_name='ST2'):
    canvas = np.full((900, 1920, 3), (25, 22, 19), dtype=np.uint8)

    def text(value, xy, color=(235, 235, 235), scale=.65):
        cv2.putText(canvas, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    state = 'warming up' if raw is None else 'scored: learned ROI weighting'
    if raw is not None and scores and scores[-1][2] is None:
        state = 'CAMERA MOTION: target score unavailable'
    text(f'{model_name} | Target-change heatmap | Frame {index + 1} | {index / fps:.2f}s | {state}', (20, 30), scale=.85)
    text(f'Fixed MSE color range: 0 - {maximum:g} | Input RGB [0,1] | Weighted map is not raw MSE', (20, 60))
    h, w = frame.shape[:2]
    factor = min(608 / w, 420 / h)
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    original = cv2.resize(frame, size)

    def colored(values):
        return cv2.resize(cv2.applyColorMap(np.clip(values / maximum * 255, 0, 255).astype(np.uint8),
                                           cv2.COLORMAP_JET), size)

    panels = [original, colored(raw) if raw is not None else None,
              cv2.addWeighted(original, .55, colored(weighted), .45, 0) if weighted is not None else None]
    target_title = 'Target-weighted MSE overlay' if motion_status in (None, 'off') else 'Target MSE + motion check'
    for column, (title, panel) in enumerate(zip(('Original video', 'Raw prediction MSE', target_title), panels)):
        x = 640 * column
        cv2.rectangle(canvas, (x + 8, 76), (x + 632, 582), (55, 52, 47), 1)
        text(title, (x + 18, 106), scale=.8)
        if panel is None:
            text('Waiting for 8 past frames', (x + 90, 330))
        else:
            left, top = x + 16 + (608 - size[0]) // 2, 124 + (420 - size[1]) // 2
            canvas[top:top + size[1], left:left + size[0]] = panel
        if column and panel is not None:
            ramp = np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (12, 1))
            canvas[560:572, x + 150:x + 470] = cv2.applyColorMap(ramp, cv2.COLORMAP_JET)
            text('0', (x + 120, 572))
            text(f'{maximum:g}+', (x + 480, 572))
    title = ('Normalized score 0.00-1.00 | raw (gray), target (yellow) | gaps = unavailable, not zero'
             if normalized else 'Top 0.1% pixel mean | last 10 seconds | raw MSE (gray), weighted MSE (yellow)')
    text(title, (20, 620))
    valid = [v for _, a, b in scores for v in (a, b) if v is not None]
    ymax = 1. if normalized else max(.001, max(valid, default=0.) * 1.15)
    x0, x1, y0, y1 = 100, 1874, 650, 820
    for fraction in (0., .5, 1.):
        y = round(y1 - fraction * (y1 - y0))
        cv2.line(canvas, (x0, y), (x1, y), (65, 62, 57), 1)
        text(f'{fraction * ymax:.2f}' if normalized else f'{fraction * ymax:.4f}', (14, y + 5), scale=.5)
    start = max(0., index / fps - 10)
    for channel, color in ((1, (175, 175, 175)), (2, (0, 220, 255))):
        previous = None
        for row in scores:
            t, value = row[0], row[channel]
            if not start <= t <= index / fps or value is None:
                previous = None
                continue
            point = (round(x0 + (t - start) / 10 * (x1 - x0)), round(y1 - value / ymax * (y1 - y0)))
            if previous is not None:
                cv2.line(canvas, previous, point, color, 2, cv2.LINE_AA)
            previous = point
    text(f'{start:.1f}s', (x0, 850))
    text(f'{start + 10:.1f}s', (x1 - 80, 850))
    if scores:
        target = scores[-1][2]
        target_text = 'N/A' if target is None else (f'{target:.2f}' if normalized else f'{target:.6f}')
        if normalized:
            text(f'Current score: raw={scores[-1][1]:.2f}, target={target_text} | {motion_status} | Not a probability', (20, 875))
            if raw_values and raw_values[0] is not None:
                text(f'Original MSE top scores: raw={raw_values[0]:.6f}, learned-weighted={raw_values[1]:.6f}', (20, 895), scale=.55)
        else:
            text(f'Current raw={scores[-1][1]:.6f}, target-weighted={target_text}', (20, 882))
    return canvas
