"""Current ST² configuration, Transformer model and training criterion."""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class ST2Config:
    history_frames: int = 8
    embedding_dim: int = 128
    num_heads: int = 4
    dropout: float = .1
    variant: str = 'transformer'
    patch_size: int = 16
    transformer_layers: int = 2
    window_size: int = 6
    supervision: str = 'normal-frames-only'
    registration_backend: str = 'loftr-ransac'
    registration_max_side: int = 320
    registration_confidence: float = .2
    registration_ransac_threshold: float = 3.
    registration_min_inliers: int = 12
    registration_min_ratio: float = .25
    registration_min_coverage: float = .05
    registration_min_overlap: float = .3
    registration_min_std: float = .01

    @classmethod
    def from_checkpoint(cls, values):
        if values.get('supervision') != 'normal-frames-only':
            raise ValueError('Requires box-free ST2 v3 weights; earlier Gate checkpoints require retraining')
        if values.get('registration_backend') != 'loftr-ransac':
            raise ValueError('Requires LoFTR+RANSAC registration; legacy ST2Motion checkpoints require retraining')
        if values.get('variant') != 'transformer':
            raise ValueError('Historical baseline/camera-aware models were removed; use an ST2 Transformer checkpoint')
        return cls(**values)

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.history_frames, self.embedding_dim, self.num_heads)):
            raise ValueError('History, embedding and head counts must be positive integers')
        if self.embedding_dim % 4 or self.embedding_dim % self.num_heads:
            raise ValueError('Embedding must be divisible by four and num_heads')
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        if self.supervision != 'normal-frames-only':
            raise ValueError('Only frame-level normal training without boxes is supported')
        if self.variant != 'transformer':
            raise ValueError('Only the current ST2 Transformer architecture is supported')
        if self.registration_backend != 'loftr-ransac':
            raise ValueError('Only LoFTR+RANSAC registration is supported')
        if type(self.registration_max_side) is not int or self.registration_max_side < 64:
            raise ValueError('registration_max_side must be an integer >= 64')
        if type(self.registration_min_inliers) is not int or self.registration_min_inliers < 3:
            raise ValueError('registration_min_inliers must be >= 3')
        if not math.isfinite(self.registration_ransac_threshold) or self.registration_ransac_threshold <= 0:
            raise ValueError('registration_ransac_threshold must be positive and finite')
        for name in ('registration_confidence', 'registration_min_ratio', 'registration_min_coverage', 'registration_min_overlap', 'registration_min_std'):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in (0,1]')
        if any(type(v) is not int or v < 1 for v in
               (self.patch_size, self.transformer_layers, self.window_size)):
            raise ValueError('Patch, Transformer layer and window sizes must be positive integers')

    @property
    def uses_registration(self):
        return True

    @property
    def model_version(self):
        return 'st2.v3.0-box-free'

    @property
    def model_name(self):
        return 'ST2'

    @property
    def model_display_name(self):
        return 'Spatiotemporal Predictor + LoFTR/RANSAC (ST² v3, box-free)'

    def validate_image_size(self, height, width):
        divisor = self.patch_size
        if min(height, width) < max(16, divisor) or height % divisor or width % divisor:
            raise ValueError(f'Image dimensions must be at least {max(16, divisor)} and divisible by {divisor}')


class ST2Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        from st2.models.transformer import ST2Predictor
        from st2.models.registration import LoFTRRegistration
        self.config = config or ST2Config()
        self.predictor = ST2Predictor(self.config)
        self.registration = LoFTRRegistration(self.config)

    def forward(self, history, current):
        if history.ndim != 5 or current.shape != (history.shape[0], 3, *history.shape[-2:]):
            raise ValueError('Current frame must match the historical RGB batch and image size')
        self.config.validate_image_size(*history.shape[-2:])
        predicted = self.predictor(history)
        raw_mse = (predicted - current).square().mean(1, keepdim=True)
        sequence = torch.stack((history[:, -1], current), 1).detach()
        registration = self.registration(sequence)
        return {'predicted': predicted, 'raw_mse': raw_mse,
                'compensated_mse': registration['registration_mse'][:, -1:],
                'compensated_valid': registration['registration_valid'][:, -1:], **registration}


class ST2Criterion(nn.Module):
    """Normal-frame prediction MSE only; no boxes, masks or Gate supervision."""
    def forward(self, outputs):
        reconstruction = outputs['raw_mse'].mean()
        return {'loss': reconstruction, 'reconstruction': reconstruction}
