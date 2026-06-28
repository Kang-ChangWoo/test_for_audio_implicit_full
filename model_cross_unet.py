"""Deep pix2pix (8-down) U-Net ENCODER as the token producer for the ray cross model.

The ray queries cross-attend the *high-level* embedding of the strong 8-down pix2pix
encoder (the architecture that wins at 256x512), not a shallow feature -- so cross
inherits the globally-integrated representation (better front/back) while keeping the
ray-conditioned implicit decode.

Drop-in for model.AudioEncoder: forward(spec, want_tokens) -> (z, tok)
  encoder: 8 downs (256x512 -> 1x2 global bottleneck), like the best U-Net
  tok : e4 high-level embedding (16x32 = 512 tokens, ngf*8) -> dim
  z   : pooled bottleneck (e8) -> audio_dim
"""
import torch
import torch.nn as nn

from model_unet_raymod import Down


class UNetTokenEncoder(nn.Module):
    def __init__(self, width=48, audio_dim=256, dim=192, in_ch=2, ngf=64):
        super().__init__()
        self.e1 = Down(in_ch,   ngf,     norm=False)   # /2   128x256
        self.e2 = Down(ngf,     ngf * 2)               # /4   64x128
        self.e3 = Down(ngf * 2, ngf * 4)               # /8   32x64
        self.e4 = Down(ngf * 4, ngf * 8)               # /16  16x32  (high-level token stage)
        self.e5 = Down(ngf * 8, ngf * 8)               # /32  8x16
        self.e6 = Down(ngf * 8, ngf * 8)               # /64  4x8
        self.e7 = Down(ngf * 8, ngf * 8)               # /128 2x4
        self.e8 = Down(ngf * 8, ngf * 8, norm=False)   # /256 1x2  (global bottleneck)
        self.gpool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Linear(ngf * 8, audio_dim), nn.GELU())
        self.tok = nn.Linear(ngf * 8, dim)             # e4 (high-level) tokens -> dim
        self.tok_dim = dim

    def forward(self, spec, want_tokens=False):
        e4 = self.e4(self.e3(self.e2(self.e1(spec))))   # (B, ngf*8, H/16, W/16) high-level
        deep = self.e8(self.e7(self.e6(self.e5(e4))))   # (B, ngf*8, H/256, W/256) bottleneck
        z = self.gpool(deep)                            # (B, audio_dim)
        if not want_tokens:
            return z, None
        tok = self.tok(e4.flatten(2).transpose(1, 2))   # (B, 16*32, dim) = 512 high-level tokens
        return z, tok
