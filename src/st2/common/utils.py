"""Shared I/O for smoke heatmap training and inference."""
import random
import json

VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}

def write_json(path, value):
    staging = path.with_suffix('.pending.json')
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    staging.replace(path)

def preprocess(frame, height, width):
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    # Same RGB, Pillow bilinear direct resize and [0,1] mapping as training.
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).copy() / 255.
    return torch.from_numpy(array).permute(2, 0, 1)

def subset(dataset, maximum, seed):
    """Seeded subset; stratify only when the dataset explicitly provides labels."""
    if not maximum or maximum >= len(dataset):
        return dataset
    from torch.utils.data import Subset
    rng = random.Random(seed)
    if not hasattr(dataset, 'positive'):
        return Subset(dataset, rng.sample(range(len(dataset)), maximum))
    positives = [i for i, positive in enumerate(dataset.positive) if positive]
    negatives = [i for i, positive in enumerate(dataset.positive) if not positive]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positive_count = min(len(positives), max(1, round(maximum * sum(dataset.positive) / len(dataset))))
    if negatives and maximum > 1:
        positive_count = min(positive_count, maximum - 1)
    indices = positives[:positive_count] + negatives[:maximum - positive_count]
    if len(indices) < maximum:
        indices += positives[positive_count:positive_count + maximum - len(indices)]
    rng.shuffle(indices)
    selection = Subset(dataset, indices)
    selection.positive = [dataset.positive[i] for i in indices]
    return selection

def save_checkpoint(state, destination):
    """Publish a complete checkpoint atomically inside this run's owned directory."""
    import torch
    staging = destination.with_suffix('.pending.pt')
    try:
        torch.save(state, staging)
        staging.replace(destination)
    finally:
        if staging.exists():
            staging.unlink()
