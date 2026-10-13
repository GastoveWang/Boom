# 檔案功能：將原畫面、誤差熱圖、候選框疊圖與逐幀分數曲線組成成果影片。
# 所屬分工：ui：只負責呈現；不改變模型分數、候選框或事件判斷。
# 輸入／輸出：原畫面、像素誤差、來源座標候選框、分數序列 → 1920×900 BGR 畫面。
# 執行方式：由 runtime/video_inference.py 呼叫 render_results()。
# 設定與說明：熱圖固定色階，曲線顯示最近十秒原始 MSE；見 doc/VIDEO_INFERENCE.md。

import cv2
import numpy as np

VIEW_WIDTH, VIEW_HEIGHT = 1920, 900
BACKGROUND = (25, 22, 19)
TEXT = (235, 235, 235)
MUTED = (175, 175, 175)
ACCENT = (240, 205, 80)


def heatmap_image(error, maximum):
    return cv2.applyColorMap(np.clip(error / maximum * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)


def label(canvas, text, point, scale=.7, color=TEXT, thickness=1):
    cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_results(image, error, displayed, scores, index, fps, state, threshold, heatmap_max):
    """Read-only presentation; displayed boxes use source-image pixels (exclusive x2/y2)."""
    canvas = np.full((VIEW_HEIGHT, VIEW_WIDTH, 3), BACKGROUND, np.uint8)
    score = float(scores[index]) if np.isfinite(scores[index]) else None
    score_text = f'{score:.5f}' if score is not None else 'N/A (warming up)'
    label(canvas, f'ASTT | {index / fps:.2f}s | Frame {index + 1} | {state}', (18, 30), .85)
    label(canvas, f'Frame score (mean MSE): {score_text}', (1040, 30), .85, ACCENT)
    label(canvas, f'Pixel candidate cutoff: {threshold:g} | Heatmap display range: 0 - {heatmap_max:g} (fixed)',
          (18, 60), .65, MUTED)
    h, w = image.shape[:2]
    factor = min(608 / w, 420 / h)
    fitted_w, fitted_h = max(1, round(w * factor)), max(1, round(h * factor))
    original = cv2.resize(image, (fitted_w, fitted_h))
    heat = cv2.resize(heatmap_image(error, heatmap_max), (fitted_w, fitted_h)) if error is not None else None
    overlay = cv2.addWeighted(original, .6, heat, .4, 0) if heat is not None else original.copy()
    for event_id, box, confirmed in displayed:
        x1, y1, x2, y2 = box
        a = (round(x1 * fitted_w / w), round(y1 * fitted_h / h))
        b = (min(fitted_w - 1, max(a[0], round(x2 * fitted_w / w) - 1)),
             min(fitted_h - 1, max(a[1], round(y2 * fitted_h / h) - 1)))
        color = (0, 255, 255) if confirmed else (155, 155, 155)
        cv2.rectangle(overlay, a, b, color, 2)
        text = f'#{event_id}'
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)
        left = max(2, min(a[0], fitted_w - text_width - 4))
        bottom = min(fitted_h - baseline - 2, max(text_height + 3, a[1]))
        cv2.rectangle(overlay, (left - 2, bottom - text_height - 2),
                      (left + text_width + 2, bottom + baseline), BACKGROUND, -1)
        label(overlay, text, (left, bottom), .55, color, 2)
    titles = ('Original video', 'Heatmap (pixel MSE)', 'Visualization (overlay + events)')
    for column, (title, panel) in enumerate(zip(titles, (original, heat, overlay))):
        x = column * 640
        cv2.rectangle(canvas, (x + 8, 76), (x + 632, 582), (45, 42, 37), 1)
        label(canvas, title, (x + 18, 106), .8, TEXT, 2)
        left = x + 16 + (608 - fitted_w) // 2
        top = 124 + (420 - fitted_h) // 2
        if panel is not None:
            canvas[top:top + fitted_h, left:left + fitted_w] = panel
        else:
            label(canvas, 'Waiting for enough history frames', (x + 80, 336), .65, MUTED)
    if heat is not None:
        ramp = np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (14, 1))
        canvas[560:574, 766:1086] = cv2.applyColorMap(ramp, cv2.COLORMAP_JET)
        label(canvas, '0', (718, 573), .6, MUTED)
        label(canvas, f'{heatmap_max:g}+', (1100, 573), .6, MUTED)

    # Curve contains only past/current frames, never scores from a future batch.
    start = max(0, index - round(fps * 10))
    values = np.asarray(scores[start:index + 1])
    valid = values[np.isfinite(values)]
    maximum = max(.01, float(valid.max()) * 1.15) if len(valid) else .01
    label(canvas, 'Scores | mean MSE per frame | last 10 seconds | Y-axis adapts; values remain raw MSE',
          (18, 615), .7)
    x0, x1, y0, y1 = 100, 1874, 638, 820
    for fraction in (0., .5, 1.):
        y = round(y1 - fraction * (y1 - y0))
        cv2.line(canvas, (x0, y), (x1, y), (65, 60, 52), 1)
        label(canvas, f'{maximum * fraction:.4f}', (18, y + 5), .55, MUTED)
    # Before ten seconds, use the available duration; afterward show a rolling ten-second window.
    span = max(1 / fps, min(10., index / fps))
    def point(k, value):
        ago = (index - (start + k)) / fps
        return (round(x1 - ago / span * (x1 - x0)), round(y1 - float(value) / maximum * (y1 - y0)))
    previous = None
    for k, value in enumerate(values):
        if not np.isfinite(value):
            previous = None
            continue
        current = point(k, value)
        if previous is not None:
            cv2.line(canvas, previous, current, ACCENT, 2, cv2.LINE_AA)
        previous = current
    if len(values) and np.isfinite(values[-1]):
        cv2.circle(canvas, point(len(values) - 1, values[-1]), 5, (0, 255, 255), -1)
    for ago in (span, span / 2, 0):
        x = round(x1 - ago / span * (x1 - x0))
        label(canvas, f'{index / fps - ago:.1f}s' if index / fps >= ago else 'before start',
              (min(x1 - 55, max(x0, x - 30)), 846), .55, MUTED)
    label(canvas, 'MSE is prediction error, not smoke probability. Gray: pending | Yellow: candidate.',
          (18, 883), .65, MUTED)
    return canvas
