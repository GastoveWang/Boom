"""Nine-frame records: eight past inputs plus a labelled current scoring frame."""
import math
import json

import torch

from st2.data.sequence import ST2SequenceDataset


def boxes_to_mask(boxes, height, width):
    mask = torch.zeros(1, height, width)
    for cx, cy, w, h in boxes:
        left = max(0, min(width, math.floor((cx - w / 2) * width)))
        right = max(0, min(width, math.ceil((cx + w / 2) * width)))
        top = max(0, min(height, math.floor((cy - h / 2) * height)))
        bottom = max(0, min(height, math.ceil((cy + h / 2) * height)))
        mask[:, top:bottom, left:right] = 1
    return mask


class ST2Dataset(ST2SequenceDataset):
    def __init__(self, root, split, image_size=(576, 768), history_frames=8, return_history_masks=False):
        self.history_frames = history_frames
        self.return_history_masks = return_history_masks
        # Base loader certifies overlapping windows, ignore states and group split.
        super().__init__(root, split, image_size, num_queries=1_000_000,
                         history_frames=history_frames + 1)
        self.frame_boxes = {}
        if return_history_masks:
            for folder in {s['folder'] for s in self.samples}:
                for line in (folder / 'frames.jsonl').read_text(encoding='utf-8').splitlines():
                    frame = json.loads(line)
                    if frame['state'] == 'ignore':
                        continue
                    boxes = frame.get('roi_xyxy', [])
                    if frame['state'] == 'roi' and not boxes:
                        raise ValueError(f'Missing historical ROI boxes: {folder}/{frame["image"]}')
                    normalized = []
                    for box in boxes:
                        if (len(box) != 4 or any(not math.isfinite(v) for v in box)
                                or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
                            raise ValueError(f'Invalid historical ROI box: {folder}/{frame["image"]}')
                        x1, y1, x2, y2 = box
                        normalized.append([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
                    self.frame_boxes[(folder, frame['image'])] = normalized

    def __getitem__(self, index):
        frames, _ = super().__getitem__(index)
        sample = self.samples[index]
        mask = boxes_to_mask(sample['boxes'], self.height, self.width)
        result = (frames[:-1], frames[-1], mask, torch.tensor(sample['normal_history']))
        if self.return_history_masks:
            sequence_masks = torch.stack([boxes_to_mask(self.frame_boxes[(sample['folder'], image)],
                                                        self.height, self.width) for image in sample['images']])
            # The existing validated sample target remains authoritative, including
            # any floating-point rasterization differences in xyxy -> cxcywh.
            sequence_masks[-1] = mask
            return (*result, sequence_masks)
        return result
