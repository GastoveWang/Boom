# 檔案功能：本機 STE/TTE/CSA 模型與下一幀預測 decoder。
# 執行設定：doc/RUN_SETTINGS.md；ASTT 正式參數見 runner.py parser()，歷史 config.py 不影響正式流程。
"""Final ASTT: shared STE -> chronological TTE -> temporal CSA -> decoder.

See README_LOCAL.md for the dimensionally explicit interpretation of Eq. 11.
Encoder and decoder blocks are adapted from upstream model_ste_tte.py.
"""
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, tokens, dim, mlp_dim, layers, heads, dropout):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, tokens, dim) * .02)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.dropout(x + self.position)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class CrossAttention(nn.Module):
    """Aligned temporal CLS query; all tokens (including aligned CLS) as K/V.

    att @ V reduces N+1 tokens to one, avoiding Eq.11's ambiguous broadcast.
    align2(aligned_cls + attention) + original_cls implements the two skips.
    """
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        if dim % num_heads:
            raise ValueError('emb_dim must be divisible by num_heads')
        self.heads = num_heads
        self.scale = (dim // num_heads) ** -.5
        self.align1 = nn.Linear(dim, dim, bias=False)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.align2 = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        original = tokens[:, :1]
        aligned = self.align1(original)
        context = torch.cat((aligned, tokens[:, 1:]), dim=1)
        b, n, d = context.shape
        def heads(x):
            return x.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = heads(self.wq(aligned)), heads(self.wk(context)), heads(self.wv(context))
        weights = self.dropout((q @ k.transpose(-2, -1) * self.scale).softmax(-1))
        attended = (weights @ v).transpose(1, 2).reshape(b, 1, d)
        return original + self.align2(aligned + attended)


class Decoder(nn.Module):
    def __init__(self, dim, image_size):
        super().__init__()
        if image_size % 16:
            raise ValueError('image_size must be divisible by 16')
        self.side = image_size // 16
        self.dense = nn.Sequential(nn.Linear(dim, 256 * self.side ** 2), nn.ELU())
        def basic(a, b):
            return [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                    nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()]
        def up(a, b, factor):
            return [nn.ConvTranspose2d(a, b, 3 if factor == 2 else 4,
                                      stride=factor, padding=1,
                                      output_padding=1 if factor == 2 else 2),
                    nn.BatchNorm2d(b), nn.ReLU()]
        self.network = nn.Sequential(*basic(256, 128), *up(128, 128, 2),
                                     *basic(128, 64), *up(64, 64, 2), *up(64, 64, 4),
                                     *basic(64, 32), nn.Conv2d(32, 3, 3, padding=1), nn.Tanh())

    def forward(self, x):
        return self.network(self.dense(x).reshape(x.shape[0], 256, self.side, self.side))


class VisionTransformer(nn.Module):
    def __init__(self, image_size=256, patch_size=32, emb_dim=768, mlp_dim=3072,
                 num_heads=8, spatial_layers=12, temporal_layers=12,
                 num_frames=4, dropout_rate=.1):
        super().__init__()
        if image_size % patch_size or min(spatial_layers, temporal_layers, num_frames) < 1:
            raise ValueError('Invalid image/patch size, layers, or frame count')
        self.image_size, self.num_frames = image_size, num_frames
        self.embedding = nn.Conv2d(3, emb_dim, patch_size, stride=patch_size)
        self.spatial_cls = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.spatial_transformer = Encoder((image_size // patch_size) ** 2 + 1,
            emb_dim, mlp_dim, spatial_layers, num_heads, dropout_rate)
        self.temporal_transformer = Encoder(num_frames + 1, emb_dim, mlp_dim,
            temporal_layers, num_heads, dropout_rate)
        self.cross_att = CrossAttention(emb_dim, num_heads, dropout_rate)
        self.decoder = Decoder(emb_dim, image_size)

    def forward(self, x):
        b, t, c, h, w = x.shape
        if (t, c, h, w) != (self.num_frames, 3, self.image_size, self.image_size):
            raise ValueError('Expected B,T,3,H,W matching model configuration')
        patches = self.embedding(x.reshape(b*t, c, h, w)).flatten(2).transpose(1, 2)
        spatial = self.spatial_transformer(torch.cat((self.spatial_cls.expand(b*t, -1, -1), patches), 1))
        chronological = spatial[:, 0].reshape(b, t, -1)
        temporal = self.temporal_transformer(torch.cat((self.temporal_cls.expand(b, -1, -1), chronological), 1))
        return self.decoder(self.cross_att(temporal)[:, 0])
