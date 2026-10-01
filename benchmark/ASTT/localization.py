# 檔案功能：誤差候選 bbox 與事件去抖合併，非 GPS 定位。
# 執行設定：doc/RUN_SETTINGS.md；ASTT 正式參數見 runner.py parser()，歷史 config.py 不影響正式流程。
"""Project extension: spatial prediction error candidates, NOT smoke classes."""
import cv2
import numpy as np


def candidate_boxes(error, threshold=.1, min_area=25, kernel=3):
    mask = (error > threshold).astype(np.uint8)
    if kernel > 1:
        structure = np.ones((kernel, kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, structure)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, structure)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return [(int(x), int(y), int(x+w), int(y+h)) for x, y, w, h, area in stats[1:count] if area >= min_area]


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.


class EventMerger:
    """Greedy one-to-one IoU tracking with consecutive-hit debounce and gap grace."""
    def __init__(self, min_frames=3, iou_threshold=.3, max_gap=2):
        if min_frames < 1 or max_gap < 0 or not 0 < iou_threshold <= 1:
            raise ValueError('Invalid event merge parameters')
        self.min_frames, self.threshold, self.max_gap = min_frames, iou_threshold, max_gap
        self.active, self.finished, self.next_id = {}, [], 1

    def update(self, frame, boxes, score):
        for key, event in list(self.active.items()):
            if frame - event['end_frame'] > self.max_gap + 1:
                self._finish(key)
        available = set(self.active)
        assignments = {}
        pairs = sorted(((iou(box, self.active[key]['bbox']), idx, key)
                        for idx, box in enumerate(boxes) for key in available), reverse=True)
        used_boxes = set()
        for overlap, idx, key in pairs:
            if overlap < self.threshold:
                break
            if idx not in used_boxes and key in available:
                assignments[idx] = key
                used_boxes.add(idx)
                available.remove(key)
        display = []
        for idx, box in enumerate(boxes):
            key = assignments.get(idx)
            if key is None:
                key = self.next_id
                self.next_id += 1
                self.active[key] = dict(event_id=key, start_frame=frame, end_frame=frame-1,
                    bbox=box, hits=0, streak=0, max_streak=0, peak_score=score,
                    confirmed_frame=None, extent=box)
            event = self.active[key]
            event['streak'] = event['streak'] + 1 if frame == event['end_frame'] + 1 else 1
            event['max_streak'] = max(event['max_streak'], event['streak'])
            if event['confirmed_frame'] is None and event['streak'] >= self.min_frames:
                event['confirmed_frame'] = frame
            event['end_frame'], event['bbox'] = frame, box
            event['hits'] += 1
            event['peak_score'] = max(event['peak_score'], score)
            a = event['extent']
            event['extent'] = (min(a[0],box[0]),min(a[1],box[1]),max(a[2],box[2]),max(a[3],box[3]))
            display.append((key, box, event['confirmed_frame'] is not None))
        return display

    def _finish(self, key):
        event = self.active.pop(key)
        if event['confirmed_frame'] is not None:
            self.finished.append(event)

    def finish(self):
        for key in list(self.active):
            self._finish(key)
        return self.finished
