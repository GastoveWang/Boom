"""Learned global inter-frame motion and coherent clip augmentation.

All matrices map current/output normalized coordinates to previous/input
coordinates, as required by affine_grid/grid_sample with align_corners=False.
The branch learns from background photometry, not from inference-time RANSAC.
"""
import torch
from torch.nn import functional as F


def similarity_matrix(parameters, height, width):
    """tx/ty normalized, rotation radians, log scale; rotation is in pixel geometry."""
    tx, ty, angle, log_scale = parameters.unbind(-1)
    scale = log_scale.exp()
    cosine, sine = scale * angle.cos(), scale * angle.sin()
    return torch.stack((cosine, -sine * height / width, tx,
                        sine * width / height, cosine, ty), -1).reshape(-1, 2, 3)


def warp_images(images, theta, mode='bilinear'):
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode=mode, padding_mode='zeros', align_corners=False), grid




def background_photometric_loss(outputs, sequence_masks):
    """Exclude target ROIs in BOTH frames of every adjacent pair, including overlaps."""
    b, steps, _, h, w = outputs['motion_error'].shape
    if sequence_masks is None or sequence_masks.shape != (b, steps + 1, 1, h, w):
        raise ValueError('ST2 motion loss requires ROI masks for all historical and current frames')
    if not torch.isfinite(sequence_masks).all() or ((sequence_masks != 0) & (sequence_masks != 1)).any():
        raise ValueError('Sequence masks must be binary and finite')
    source = sequence_masks[:, :-1].reshape(-1, 1, h, w)
    theta = outputs['motion_theta'].detach().reshape(-1, 2, 3)
    source_roi, _ = warp_images(source, theta, mode='nearest')
    target_roi = sequence_masks[:, 1:].reshape(-1, 1, h, w)
    # Exclude a small margin around target boundaries and bilinear mixing.
    excluded = F.max_pool2d(torch.maximum(source_roi, target_roi), 3, stride=1, padding=1)
    background = ((excluded < .01).reshape(b, steps, 1, h, w)
                  & outputs['motion_valid']).to(outputs['motion_error'].dtype)
    error = outputs['motion_error']
    per_pair = (error * background).sum((-3, -2, -1)) / background.sum((-3, -2, -1)).clamp(min=1)
    available = background.sum((-3, -2, -1)) > 0
    return (per_pair * available).sum() / available.sum().clamp(min=1)


def augment_camera_clip(history, current, mask, sequence_masks=None, probability=.5):
    """Continuous camera trajectory; every frame and ROI uses its matching grid.

    Crop/rotation/translation stay inside the input (no synthetic black borders).
    One smooth trajectory per sample, never independent random transforms/frame.
    """
    if not 0 <= probability <= 1:
        raise ValueError('Camera augmentation probability must be in [0,1]')
    if probability == 0:
        return history, current, mask, sequence_masks
    sequence = torch.cat((history, current[:, None]), 1)
    b, length, _, h, w = sequence.shape
    selected = (torch.rand(b, 1, device=history.device) < probability).to(history.dtype)
    phase = torch.linspace(-1, 1, length, device=history.device, dtype=history.dtype)[None]
    angle = (torch.rand(b, 1, device=history.device) * 2 - 1) * .12 * phase * selected
    dx = (torch.rand(b, 1, device=history.device) * 2 - 1) * .04 * phase * selected
    dy = (torch.rand(b, 1, device=history.device) * 2 - 1) * .04 * phase * selected
    # Account for rectangular pixel rotation; .75 crop provides a safe margin.
    scale = (1 - .25 * selected).expand(-1, length)
    parameters = torch.stack((dx, dy, angle, scale.log()), -1).reshape(-1, 4)
    theta = similarity_matrix(parameters, h, w)
    grid = F.affine_grid(theta, (b * length, 3, h, w), align_corners=False)
    # Extreme caller aspect ratios may need a tighter crop to avoid padding.
    extent = grid.abs().flatten(1).amax(1).clamp(min=1)
    grid = grid / extent[:, None, None, None]
    frames = F.grid_sample(sequence.reshape(-1, 3, h, w), grid, padding_mode='border', align_corners=False)
    frames = frames.reshape(b, length, 3, h, w)
    if sequence_masks is not None:
        if sequence_masks.shape != (b, length, 1, h, w):
            raise ValueError('Camera augmentation sequence mask shape mismatch')
        masks = F.grid_sample(sequence_masks.reshape(-1, 1, h, w), grid, mode='nearest',
                              padding_mode='zeros', align_corners=False).reshape(b, length, 1, h, w)
        return frames[:, :-1], frames[:, -1], masks[:, -1], masks
    target_grid = grid.reshape(b, length, h, w, 2)[:, -1]
    mask = F.grid_sample(mask, target_grid, mode='nearest', align_corners=False)
    return frames[:, :-1], frames[:, -1], mask, None
