"""Read exported smoke ROI historical clips without depending on benchmark ASTT."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def dataset_fingerprint(root, include_targets=True):
    digest = hashlib.sha256()
    root = Path(root).resolve()
    paths = [root / 'project.json', root / 'split_report.json']
    for split in ('train', 'val'):
        for name in ('source.json', 'frames.jsonl', 'samples.jsonl'):
            paths += sorted(root.glob(f'{split}/*/{name}'))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        if include_targets:
            digest.update(path.read_bytes())
        else:
            value = path.read_text(encoding='utf-8')
            if path.name == 'frames.jsonl':
                value = [{k: frame[k] for k in ('image', 'frame_index', 'state')}
                         for frame in map(json.loads, value.splitlines())]
            elif path.name == 'samples.jsonl':
                value = [{k: sample[k] for k in ('input_images', 'target_frame')}
                         for sample in map(json.loads, value.splitlines())]
            elif path.name == 'project.json':
                value = [{'id': source['id'], 'origin': source['origin']} for source in json.loads(value)['sources']]
            elif path.name == 'split_report.json':
                report = json.loads(value)
                value = {k: report[k] for k in ('history_frames', 'assignments')}
            else:
                value = {'id': json.loads(value)['id']}
            digest.update(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()


class ST2SequenceDataset(Dataset):
    """Direct resize preserves normalized cxcywh; images are never padded/cropped.

    Validate exported frame states and history before training. The exported
    negative labels mean normal outside annotated intervals, not a smoke class.
    """

    def __init__(self, root, split, image_size=(576, 768), num_queries=16, history_frames=8, read_targets=True):
        self.root = Path(root).resolve()
        if split not in ('train', 'val'):
            raise ValueError('split must be train or val')
        self.split = split
        self.read_targets = read_targets
        self.height, self.width = image_size
        if any(v < 16 or v % 8 for v in image_size):
            raise ValueError('Image height/width must be >=16 and divisible by 8')
        report = json.loads((self.root / 'split_report.json').read_text(encoding='utf-8'))
        project = json.loads((self.root / 'project.json').read_text(encoding='utf-8'))
        exported_frames = report['history_frames']
        if type(exported_frames) is not int or exported_frames < 1:
            raise ValueError('Invalid history_frames')
        self.num_frames = exported_frames if history_frames is None else history_frames
        if type(self.num_frames) is not int or self.num_frames < exported_frames:
            raise ValueError('Requested history_frames must be an integer >= exported history_frames')
        # Check actual directories against original group assignments, not just summary counts.
        sources = {source['id']: source for source in project['sources']}
        groups = {'train': set(), 'val': set()}
        for partition in groups:
            for folder in sorted((self.root / partition).iterdir()):
                if not folder.is_dir():
                    continue
                meta = json.loads((folder / 'source.json').read_text(encoding='utf-8'))
                sid = meta['id']
                if sid not in sources or report['assignments'].get(sid) != partition:
                    raise ValueError(f'Source assignment mismatch: {folder}')
                groups[partition].add(sources[sid]['origin'].strip().casefold())
        if groups['train'] & groups['val']:
            raise ValueError('Source groups leak between train and val')
        self.samples = []
        for folder in sorted((self.root / split).iterdir()):
            if not folder.is_dir():
                continue
            frames = {}
            folder_samples = {}
            for line in (folder / 'frames.jsonl').read_text(encoding='utf-8').splitlines():
                frame = json.loads(line)
                name = frame['image']
                if name in frames:
                    raise ValueError(f'Duplicate exported image record: {folder / name}')
                path = (folder / name).resolve()
                if not path.is_relative_to(folder.resolve()) or not path.is_file():
                    raise ValueError(f'Missing or unsafe image: {path}')
                frames[name] = frame
            for line in (folder / 'samples.jsonl').read_text(encoding='utf-8').splitlines():
                sample = json.loads(line)
                names = sample['input_images']
                if len(names) != exported_frames or any(name not in frames for name in names):
                    raise ValueError(f'Invalid historical window in {folder}')
                history = [frames[name] for name in names]
                indices = [frame['frame_index'] for frame in history]
                if indices != list(range(sample['target_frame'] - exported_frames + 1,
                                         sample['target_frame'] + 1)):
                    raise ValueError(f'Non-consecutive historical window in {folder}')
                if any(frame['state'] not in ('normal', 'roi') for frame in history):
                    raise ValueError(f'Ignored/unknown frame in historical window: {folder}')
                target = {'labels': [], 'boxes': []}
                if read_targets:
                    target = sample['target']
                    if any(type(label) is not int for label in target['labels']):
                        raise ValueError(f'Class labels must be integers: {folder}')
                    labels = torch.tensor(target['labels'], dtype=torch.int64)
                    boxes = torch.tensor(target['boxes'], dtype=torch.float32).reshape(-1, 4)
                    if len(labels) != len(boxes) or (labels != 0).any() or len(boxes) > num_queries:
                        raise ValueError(f'Invalid single-class targets/query capacity: {folder}')
                    corners = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                                         boxes[:, :2] + boxes[:, 2:] / 2), dim=1)
                    if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any() or (corners < -1e-6).any() or (corners > 1 + 1e-6).any():
                        raise ValueError(f'Invalid target box coordinates: {folder}')
                    if bool(len(boxes)) != (history[-1]['state'] == 'roi'):
                        raise ValueError(f'Target/frame state mismatch: {folder}')
                # Store plain metadata: Windows workers must not pickle thousands
                # of small tensors/shared-memory handles before their first batch.
                end = sample['target_frame']
                if end in folder_samples:
                    raise ValueError(f'Duplicate target frame in {folder}')
                folder_samples[end] = {'folder': folder, 'images': names, 'labels': target['labels'],
                                       'boxes': target['boxes'], 'target_frame': end,
                                       'normal_history': all(frame['state'] == 'normal' for frame in history)}
            extra_frames = self.num_frames - exported_frames
            for end, sample in folder_samples.items():
                # Extend through certified overlapping windows only. A missing
                # window breaks the chain even if file indices look consecutive.
                starts = range(end - extra_frames, end + 1)
                if any(index not in folder_samples for index in starts):
                    continue
                images = list(folder_samples[end - extra_frames]['images'])
                images.extend(folder_samples[index]['images'][-1]
                              for index in range(end - extra_frames + 1, end + 1))
                if any(folder_samples[index]['images'] != images[offset:offset + exported_frames]
                       for offset, index in enumerate(starts)):
                    raise ValueError(f'Inconsistent overlapping historical windows in {folder}')
                self.samples.append(dict(sample, images=images,
                                         normal_history=all(folder_samples[index]['normal_history']
                                                            for index in starts)))
        if not self.samples:
            raise ValueError(f'No samples in {self.root / split}')
        self.positive = [bool(len(sample['boxes'])) for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        images = []
        for name in sample['images']:
            with Image.open(sample['folder'] / name) as image:
                image = image.convert('RGB').resize((self.width, self.height), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32).copy() / 255.
                images.append(torch.from_numpy(array).permute(2, 0, 1))
        if not self.read_targets:
            return torch.stack(images)
        return torch.stack(images), {'labels': torch.tensor(sample['labels'], dtype=torch.int64),
                                     'boxes': torch.tensor(sample['boxes'], dtype=torch.float32).reshape(-1, 4)}
