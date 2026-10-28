"""Model-only streaming inference for a future Boom detector adapter.

No renderer, event confirmation, camera reader or geographic assumptions live
here. A runner supplies BGR frames; a detector adapter owns event decisions.
"""
from pathlib import Path
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from st2.models.network import ST2Config, ST2Model
from st2.engine.postprocess import ScoreNormalizer
from st2.common.utils import preprocess


@dataclass(frozen=True)
class FramePrediction:
    frame_index: int
    ready: bool
    raw_mse: object = None
    weighted_mse: object = None
    attention: object = None
    score_01: object = None
    registration_status: str = 'warming_up'
    registration_quality: object = None
    registration_matrix: object = None


class ST2Runtime:
    """Causal, one-stream checkpoint inference; scores are not probabilities.

    Discontinuous frame indices or changed source dimensions restart warmup.
    The optional inference-only camera guard is not part of this model API.
    """
    def __init__(self, checkpoint, device='auto'):
        checkpoint = Path(checkpoint).resolve()
        ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if ck.get('format') not in ('st2.checkpoint.v1', 'boom.roi_heatmap.checkpoint.v1'):
            raise ValueError('Expected ST2 or compatible heatmap checkpoint')
        self.config = ST2Config.from_checkpoint(ck['model_config'])
        self.height, self.width = ck['args']['height'], ck['args']['width']
        self.config.validate_image_size(self.height, self.width)
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        if self.device.type not in ('cpu', 'cuda') or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Use an available CUDA device or cpu')
        self.model = ST2Model(self.config).to(self.device).eval()
        self.model.load_state_dict(ck['model'], strict=True)
        self.normalizer = ScoreNormalizer()
        self.history = deque(maxlen=self.config.history_frames + 1)
        self.reset()

    def reset(self):
        self.history.clear()
        self.last_index = None
        self.source_shape = None

    def process_frame(self, frame_bgr, frame_index):
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3
                or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8
                or min(frame_bgr.shape[:2]) < 1):
            raise ValueError('Expected nonempty uint8 BGR image')
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError('Frame index must be a nonnegative integer')
        if ((self.last_index is not None and frame_index != self.last_index + 1)
                or (self.source_shape is not None and self.source_shape != frame_bgr.shape)):
            self.reset()
        self.last_index, self.source_shape = frame_index, frame_bgr.shape
        self.history.append(preprocess(frame_bgr, self.height, self.width))
        if len(self.history) <= self.config.history_frames:
            return FramePrediction(frame_index, False)
        clip = torch.stack(list(self.history)).to(self.device)
        with torch.inference_mode():
            outputs = self.model(clip[:-1][None], clip[-1][None])
        maps = {key: outputs[key][0, 0].float().cpu().numpy()
                for key in ('raw_mse', 'weighted_mse', 'attention')}
        if not all(np.isfinite(value).all() for value in maps.values()):
            raise ValueError('Non-finite model output')
        flat = maps['weighted_mse'].reshape(-1)
        k = max(1, int(np.ceil(len(flat) * .001)))
        score = float(np.partition(flat, len(flat) - k)[-k:].mean())
        return FramePrediction(frame_index, True, **maps,
                               score_01=self.normalizer(score),
                               registration_status='aligned' if bool(outputs['registration_success'][0, -1]) else 'uncertain',
                               registration_quality=outputs['registration_quality'][0, -1].cpu().numpy(),
                               registration_matrix=outputs['registration_matrix'][0, -1].cpu().numpy())
