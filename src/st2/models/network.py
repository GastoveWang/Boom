"""Current ST² configuration, Transformer model and training criterion."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ST2Config:
    history_frames: int = 8
    embedding_dim: int = 128
    num_heads: int = 4
    dropout: float = .1
    gate_floor: float = .05
    variant: str = 'transformer'
    patch_size: int = 16
    transformer_layers: int = 2
    window_size: int = 6

    @classmethod
    def from_checkpoint(cls, values):
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
        if not math.isfinite(self.gate_floor) or not 0 <= self.gate_floor < 1:
            raise ValueError('gate_floor must be in [0,1)')
        if self.variant != 'transformer':
            raise ValueError('Only the current ST2 Transformer architecture is supported')
        if any(type(v) is not int or v < 1 for v in
               (self.patch_size, self.transformer_layers, self.window_size)):
            raise ValueError('Patch, Transformer layer and window sizes must be positive integers')

    @property
    def uses_motion(self):
        return True

    @property
    def model_version(self):
        return 'st2.v1.0'

    @property
    def model_name(self):
        return 'ST2'

    @property
    def model_display_name(self):
        return 'Smoke-Targeted Spatiotemporal Transformer (ST² v1.0)'

    def validate_image_size(self, height, width):
        divisor = self.patch_size
        if min(height, width) < max(16, divisor) or height % divisor or width % divisor:
            raise ValueError(f'Image dimensions must be at least {max(16, divisor)} and divisible by {divisor}')


class ST2Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        from st2.models.transformer import ST2Predictor, ST2Motion, ST2Gate
        self.config = config or ST2Config()
        self.predictor = ST2Predictor(self.config)
        self.motion = ST2Motion(self.config)
        self.gate = ST2Gate(self.config)

    def forward(self, history, current):
        if history.ndim != 5 or current.shape != (history.shape[0], 3, *history.shape[-2:]):
            raise ValueError('Current frame must match the historical RGB batch and image size')
        self.config.validate_image_size(*history.shape[-2:])
        predicted = self.predictor(history)
        raw_mse = (predicted - current).square().mean(1, keepdim=True)
        sequence = torch.cat((history, current[:, None]), 1).detach()
        motion = self.motion(sequence)
        parameters = motion['motion_parameters'].detach()
        statistics = torch.cat((parameters.mean(1), parameters.square().mean(1).sqrt()), 1)
        logits = self.gate(sequence, raw_mse.detach(), motion['motion_mse'].detach(), statistics)
        attention = logits.sigmoid()
        weighted_mse = raw_mse.detach() * (self.config.gate_floor + (1-self.config.gate_floor)*attention)
        return {'predicted': predicted, 'raw_mse': raw_mse, 'gate_logits': logits,
                'attention': attention, 'weighted_mse': weighted_mse, **motion}


class ST2Criterion(nn.Module):
    def __init__(self, motion_weight=.2):
        super().__init__()
        if not math.isfinite(motion_weight) or motion_weight < 0:
            raise ValueError('motion_weight must be nonnegative and finite')
        self.motion_weight = motion_weight

    def forward(self, outputs, mask, normal_eligible, sequence_masks=None):
        logits = outputs['gate_logits']
        if mask.shape != logits.shape or normal_eligible.shape != (len(mask),):
            raise ValueError('Mask and normal eligibility shapes must match outputs')
        if not torch.isfinite(mask).all() or ((mask != 0) & (mask != 1)).any():
            raise ValueError('ROI masks must be binary and finite')
        if normal_eligible.dtype != torch.bool:
            raise ValueError('Normal eligibility must be boolean')
        # Normalize positive and negative pixels separately: tiny ROI supervision
        # must not disappear into hundreds of thousands of background pixels.
        pixel_bce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
        positive_count, negative_count = mask.sum(), (1 - mask).sum()
        positive_bce = (pixel_bce * mask).sum() / positive_count.clamp(min=1)
        negative_bce = (pixel_bce * (1 - mask)).sum() / negative_count.clamp(min=1)
        active = (positive_count > 0).to(logits.dtype) + (negative_count > 0).to(logits.dtype)
        balanced_bce = (positive_bce + negative_bce) / active.clamp(min=1)
        positive_images = mask.flatten(1).sum(1) > 0
        if (normal_eligible & positive_images).any():
            raise ValueError('A target ROI frame cannot be eligible for normal-only prediction training')
        if positive_images.any():
            p = outputs['attention'][positive_images].flatten(1)
            y = mask[positive_images].flatten(1)
            dice = (1 - (2 * (p * y).sum(1) + 1e-6) / (p.sum(1) + y.sum(1) + 1e-6)).mean()
        else:
            dice = logits.sum() * 0
        if normal_eligible.any():
            reconstruction = outputs['raw_mse'][normal_eligible].mean()
        else:
            reconstruction = outputs['raw_mse'].sum() * 0
        gate_loss = balanced_bce + dice
        motion_loss = logits.sum() * 0
        motion_regularization = logits.sum() * 0
        if 'motion_error' in outputs:
            from st2.models.motion import background_photometric_loss
            if sequence_masks is None or not torch.equal(sequence_masks[:, -1], mask):
                raise ValueError('Current ROI mask must agree with final sequence mask')
            motion_loss = background_photometric_loss(outputs, sequence_masks)
            motion_regularization = outputs['motion_regularization']
        return {'loss': reconstruction + gate_loss + self.motion_weight * (motion_loss + .001 * motion_regularization),
                'reconstruction': reconstruction, 'gate_loss': gate_loss, 'balanced_bce': balanced_bce,
                'dice_loss': dice, 'motion_photometric': motion_loss, 'motion_regularization': motion_regularization}
