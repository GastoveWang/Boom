# 檔案功能：每影片連續影格窗口、缺幀／標籤校驗與可選 ECC。
# 執行設定：doc/RUN_SETTINGS.md；ASTT 正式參數見 runner.py parser()，歷史 config.py 不影響正式流程。
"""Per-video future-frame windows. No concatenation across videos or gaps."""
from pathlib import Path
import re
import warnings
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', path.name)]


def frame_number(path):
    match = re.search(r'(\d+)$', path.stem)
    return int(match.group(1)) if match else None


def align_inputs(images):
    """ECC-align historical inputs to latest INPUT, never to future target."""
    reference = cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    result = []
    for image in images[:-1]:
        moving = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        try:
            _, warp = cv2.findTransformECC(reference, moving, np.eye(2, 3, dtype=np.float32),
                cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-5))
            image = cv2.warpAffine(image, warp, (image.shape[1], image.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        except cv2.error:
            warnings.warn('ECC did not converge; using unregistered input', RuntimeWarning)
        result.append(image)
    return result + [images[-1]]


class VideoDataset(Dataset):
    def __init__(self, frames_root, image_size=256, num_frames=4, labels_root=None,
                 registration=False, normal_only=False):
        self.root = Path(frames_root)
        if not self.root.is_dir():
            raise FileNotFoundError(f'Missing frames directory: {self.root}')
        self.size, self.num_frames = image_size, num_frames
        self.registration = registration
        self.videos, self.labels, self.samples = {}, {}, []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            frames = sorted((p for p in folder.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}), key=natural_key)
            if not frames:
                continue
            self.videos[folder.name] = frames
            label_path = Path(labels_root) / (folder.name + '.npy') if labels_root else None
            if label_path and label_path.exists():
                labels = np.load(label_path, allow_pickle=False)
                if labels.shape != (len(frames),) or not np.isin(labels, [0, 1]).all():
                    raise ValueError(f'{label_path}: expected {len(frames)} binary frame labels')
                self.labels[folder.name] = labels
            for start in range(len(frames) - num_frames):
                window = frames[start:start + num_frames + 1]
                numbers = [frame_number(p) for p in window]
                if all(n is not None for n in numbers) and any(b != a+1 for a, b in zip(numbers, numbers[1:])):
                    continue
                if normal_only and folder.name in self.labels and self.labels[folder.name][start:start+num_frames+1].any():
                    continue
                self.samples.append((folder.name, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, start = self.samples[index]
        paths = self.videos[video][start:start+self.num_frames+1]
        images = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'Cannot decode {path}')
            original_h, original_w = image.shape[:2]
            images.append(cv2.resize(image, (self.size, self.size)))
        if self.registration:
            images = align_inputs(images[:-1]) + [images[-1]]
        tensor = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy()).float() / 127.5 - 1
        target_index = start + self.num_frames
        return dict(inputs=tensor[:-1], target=tensor[-1], video=video,
            frame_index=target_index, filename=paths[-1].name,
            label=int(self.labels[video][target_index]) if video in self.labels else -1,
            original_height=original_h, original_width=original_w)
