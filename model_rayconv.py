"""Ray-feature extraction + CONV decoder (hybrid).

Finding so far: cross's per-ray attention decode lacks spatial integration -> weak
front/back; full U-Net wins via its conv decoder (skip) spatial integration.
This model keeps the ray directional inductive bias for FEATURE extraction but
decodes with a conv up-decoder for spatial coherence:

  spec -> audio tokens
  coarse ray grid (hc x wc) -> ray queries -> cross-attend tokens -> (dim, hc, wc) feature map
  conv up-decoder (x2 per stage) -> sigmoid depth (H x W)

Shares the {"D","D0","extras"} interface (train_fullmap / eval_fullmap).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import conv_bn, Refine, AudioEncoder, CrossBlock


class RayConvNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        self.hc, self.wc = cfg.coarse_h, cfg.coarse_w
        self.bb = AudioEncoder(cfg.width, cfg.audio_dim, cfg.dim,
                               in_ch=getattr(cfg, "in_ch", 2),
                               hi_tokens=getattr(cfg, "hi_tokens", False))
        self.ray_mlp = nn.Sequential(nn.Linear(cfg.coarse_feat_dim, cfg.dim), nn.GELU(),
                                     nn.Linear(cfg.dim, cfg.dim), nn.GELU())
        self.cross = nn.ModuleList([CrossBlock(cfg.dim, cfg.n_heads) for _ in range(cfg.n_cross)])
        nup = int(round(math.log2(self.H / self.hc)))      # x2 upsamples to reach H
        ch, outc = cfg.dim, cfg.width
        ups = []
        for _ in range(nup):
            ups += [nn.Upsample(scale_factor=2, mode="nearest"), conv_bn(ch, outc), Refine(outc)]
            ch = outc
        self.up = nn.Sequential(*ups)
        self.head = nn.Sequential(conv_bn(ch, ch), nn.Conv2d(ch, 1, 3, 1, 1))

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        assert coarse_feat is not None, "RayConvNet needs coarse_feat (RayBank.feat)"
        B = spec.size(0)
        _, tok = self.bb(spec, want_tokens=True)
        q = self.ray_mlp(coarse_feat[None].expand(B, -1, -1))      # (B, hc*wc, dim)
        for blk in self.cross:
            q = blk(q, tok)
        x = q.transpose(1, 2).reshape(B, -1, self.hc, self.wc)     # (B, dim, hc, wc)
        D = torch.sigmoid(self.head(self.up(x)))
        if D.shape[-2:] != (self.H, self.W):
            D = F.interpolate(D, (self.H, self.W), mode="bilinear", align_corners=False)
        return {"D": D, "D0": D, "extras": {}}
