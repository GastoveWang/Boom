"""ST2 v2: Transformer predictor with external LoFTR/RANSAC registration.

Predictor uses linear patch projections and attention; registration is external.
Warping, patch assembly and the final MSE weighting remain explicit operations.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def spatial_positions(height, width, dim, reference):
    """Dynamic 2-D sinusoidal positions; no fixed training-resolution table."""
    frequency = torch.exp(torch.arange(dim // 4, device=reference.device,
                                      dtype=torch.float32) * (-math.log(10000.) / (dim // 4)))
    y = torch.arange(height, device=reference.device)[:, None] * frequency
    x = torch.arange(width, device=reference.device)[:, None] * frequency
    yy = torch.cat((y.sin(), y.cos()), -1)[:, None].expand(-1, width, -1)
    xx = torch.cat((x.sin(), x.cos()), -1)[None].expand(height, -1, -1)
    return torch.cat((yy, xx), -1).to(reference.dtype)


class PatchEmbedding(nn.Module):
    def __init__(self, channels, dim, patch):
        super().__init__()
        self.patch = patch
        self.projection = nn.Linear(channels * patch * patch, dim)

    def forward(self, images):
        b, c, h, w = images.shape
        p = self.patch
        if h % p or w % p:
            raise ValueError(f'Image dimensions must be divisible by patch size {p}')
        patches = images.reshape(b, c, h // p, p, w // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(b, h // p, w // p, c * p * p)
        return self.projection(patches)


def assemble_patches(values, patch, channels):
    b, gh, gw, _ = values.shape
    return values.reshape(b, gh, gw, channels, patch, patch).permute(
        0, 3, 1, 4, 2, 5).reshape(b, channels, gh * patch, gw * patch)


def encoder_layer(config):
    return nn.TransformerEncoderLayer(config.embedding_dim, config.num_heads,
                                     config.embedding_dim * 4, config.dropout,
                                     activation='gelu', batch_first=True, norm_first=True)


class WindowAttention(nn.Module):
    """Shifted non-wrapping windows with masked padding, including small inputs."""
    def __init__(self, config, shifted=False):
        super().__init__()
        self.window = config.window_size
        self.offset = self.window // 2 if shifted else 0
        self.layer = encoder_layer(config)

    def forward(self, tokens):
        b, h, w, d = tokens.shape
        s, offset = self.window, self.offset
        ph, pw = (-(h + offset)) % s, (-(w + offset)) % s
        padded = F.pad(tokens.permute(0, 3, 1, 2), (offset, pw, offset, ph)).permute(0, 2, 3, 1)
        hh, ww = padded.shape[1:3]
        valid = F.pad(torch.ones((h, w), device=tokens.device, dtype=torch.bool),
                      (offset, pw, offset, ph), value=False)
        windows = padded.reshape(b, hh // s, s, ww // s, s, d).permute(
            0, 1, 3, 2, 4, 5).reshape(-1, s * s, d)
        mask = ~valid.reshape(hh // s, s, ww // s, s).permute(0, 2, 1, 3).reshape(-1, s * s)
        mask = mask.repeat(b, 1)
        windows = self.layer(windows, src_key_padding_mask=mask)
        result = windows.reshape(b, hh // s, ww // s, s, s, d).permute(
            0, 1, 3, 2, 4, 5).reshape(b, hh, ww, d)
        return result[:, offset:offset+h, offset:offset+w]


class SpaceTimeEncoder(nn.Module):
    def __init__(self, config, length):
        super().__init__()
        self.length = length
        self.embedding = PatchEmbedding(3, config.embedding_dim, config.patch_size)
        self.time_positions = nn.Parameter(torch.randn(1, length, 1, 1, config.embedding_dim) * .02)
        self.spatial = nn.ModuleList(WindowAttention(config, i % 2 == 1)
                                     for i in range(config.transformer_layers))
        self.temporal = nn.ModuleList(encoder_layer(config) for _ in self.spatial)
        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, sequence):
        b, t, c, h, w = sequence.shape
        if t != self.length or c != 3:
            raise ValueError('Unexpected RGB sequence length')
        tokens = self.embedding(sequence.reshape(b * t, c, h, w))
        gh, gw, d = tokens.shape[1:]
        tokens = tokens.reshape(b, t, gh, gw, d)
        tokens = tokens + self.time_positions + spatial_positions(gh, gw, d, tokens)[None, None]
        for spatial, temporal in zip(self.spatial, self.temporal):
            tokens = spatial(tokens.reshape(b * t, gh, gw, d)).reshape(b, t, gh, gw, d)
            time = tokens.permute(0, 2, 3, 1, 4).reshape(b * gh * gw, t, d)
            tokens = temporal(time).reshape(b, gh, gw, t, d).permute(0, 3, 1, 2, 4)
        return self.norm(tokens)


class PatchDecoder(nn.Module):
    """One query at each grid point attends its temporal memory, then local space."""
    def __init__(self, config, channels):
        super().__init__()
        self.patch, self.channels = config.patch_size, channels
        self.query = nn.Parameter(torch.randn(1, 1, config.embedding_dim) * .02)
        self.cross = nn.TransformerDecoderLayer(config.embedding_dim, config.num_heads,
                                               config.embedding_dim * 4, config.dropout,
                                               activation='gelu', batch_first=True, norm_first=True)
        self.spatial = WindowAttention(config, shifted=True)
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, channels * self.patch ** 2)

    def forward(self, memory, extra=None):
        b, t, gh, gw, d = memory.shape
        timeline = memory.permute(0, 2, 3, 1, 4).reshape(b * gh * gw, t, d)
        query = timeline[:, -1:] + self.query
        if extra is not None:
            query = query + extra.reshape(b * gh * gw, 1, d)
        decoded = self.cross(query, timeline).reshape(b, gh, gw, d)
        decoded = self.norm(self.spatial(decoded))
        return assemble_patches(self.output(decoded), self.patch, self.channels)


class ST2Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SpaceTimeEncoder(config, config.history_frames)
        self.decoder = PatchDecoder(config, 3)

    def forward(self, history):
        delta = self.decoder(self.encoder(history))
        return (history[:, -1] + .25 * delta.tanh()).clamp(0, 1)
