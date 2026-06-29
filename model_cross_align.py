"""Feature-aligned ray decoder: high-res audio feature + ray cross-attn + conv smoothing.

Lesson from experiments: pure per-ray decode is discrete because each ray only sees
GLOBAL audio tokens and lacks its own LOCAL high-res feature + neighbour coupling.
This model fixes all three:
  (1) audio feature kept at decent res (e2 = /4 = 64x128, NOT collapsed to /8)
  (2) feature-ALIGNED: each ray on the 64x128 grid is fused with the e2 feature at its
      own (theta,phi) (= the e2 cell), i.e. an implicit "skip"
  (3) ray cross-attends GLOBAL context tokens (e4 16x32), then a small CONV smooths the
      64x128 ray-feature grid (local coupling) before x4 bilinear upsample to full ERP.

Returns {"D","D0","extras"} for train_fullmap / eval_fullmap.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_unet_coarse import UNet8Encoder
from model import CrossBlock, conv_bn, Refine
from ray_features import RayBank
import copy


class CrossAlign(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        self.hc, self.wc = cfg.img_h // 4, cfg.img_w // 4      # 64x128 = e2 grid
        ngf = getattr(cfg, "ngf", 64); dim = cfg.dim
        self.enc = UNet8Encoder(getattr(cfg, "in_ch", 2), ngf)
        # ray PE on the 64x128 grid (RayBank handles theta/phi/seam/mic-PE conventions)
        pcfg = copy.copy(cfg); pcfg.img_h, pcfg.img_w = self.hc, self.wc
        bank = RayBank(pcfg, device="cpu")
        self.register_buffer("ray_feat", bank.feat)            # (hc*wc, F)
        self.ray_proj = nn.Sequential(nn.Linear(bank.feat_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.align_proj = nn.Linear(ngf * 2, dim)              # e2 (64x128) aligned local feature
        self.kv_proj = nn.Linear(ngf * 8, dim)                 # e4 (16x32) global context tokens
        self.cross = nn.ModuleList([CrossBlock(dim, cfg.n_heads)
                                    for _ in range(getattr(cfg, "ray_cross_layers", 2))])
        # conv smoothing (local coupling) on the 64x128 ray-feature grid
        self.smooth = nn.Sequential(conv_bn(dim, ngf * 2), Refine(ngf * 2), conv_bn(ngf * 2, ngf))
        self.head = nn.Sequential(conv_bn(ngf, ngf), nn.Conv2d(ngf, 1, 3, 1, 1))

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        B = spec.size(0)
        e2 = self.enc.e2(self.enc.e1(spec))                    # (B, ngf*2, 64, 128) high-res
        e4 = self.enc.e4(self.enc.e3(e2))                      # (B, ngf*8, 16, 32) global
        kv = self.kv_proj(e4.flatten(2).transpose(1, 2))      # (B, 512, dim) global context
        q = self.ray_proj(self.ray_feat)[None].expand(B, -1, -1)        # (B, hc*wc, dim) ray PE
        q = q + self.align_proj(e2.flatten(2).transpose(1, 2))         # + aligned local audio feature
        for blk in self.cross:
            q = blk(q, kv)
        x = q.transpose(1, 2).reshape(B, -1, self.hc, self.wc)         # (B, dim, 64, 128)
        x = self.smooth(x)                                             # conv local smoothing
        D = torch.sigmoid(self.head(x))
        D = F.interpolate(D, (self.H, self.W), mode="bilinear", align_corners=False)
        return {"D": D, "D0": D, "extras": {}}
