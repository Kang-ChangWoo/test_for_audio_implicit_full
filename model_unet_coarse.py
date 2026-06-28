"""Strong U-Net8 encoder + COARSE spherical-layout heads (not a dense full-res decoder).

Goal: keep the winning U-Net8 (+5ch+flip) encoder as the feature extractor, but
replace the dense per-pixel decoder with a band-limited coarse head, and revive the
ray/theta-phi implicit idea as a COARSE ray-token field (16x32), not full-res per-pixel.

All models return the shared interface {"D": full(B,1,H,W), "D0": coarse-upsampled,
"extras": {...}} so train_fullmap.py / eval_fullmap.py work unchanged.

Heads:
  UNetCoarse          : pooled bottleneck z -> MLP -> 16x32 ERP depth -> bilinear up
  UNetSH              : z -> low-order SH coeffs -> synthesize band-limited depth
  UNetRayCoarse       : e4 audio tokens (K/V) + coarse RayBank ray tokens (Q) cross-attn -> 16x32
  UNetCoarseResidual  : UNetCoarse + small low-pass residual (constrained)
"""
import copy
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_unet_raymod import Down
from model import CrossBlock, SelfBlock, conv_bn, Refine
from ray_features import RayBank
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "test_for_audio_clip"))
from sh import erp_grid, sh_basis_matrix  # noqa: E402


class UNet8Encoder(nn.Module):
    """Explicit pix2pix-style 8-down encoder (256x512 -> 1x2). Exposes the e4 mid
    feature (16x32, ngf*8 = strong globally-integrated tokens) and the e8 bottleneck."""
    def __init__(self, in_ch, ngf=64):
        super().__init__()
        self.e1 = Down(in_ch,   ngf,     norm=False)   # 128x256
        self.e2 = Down(ngf,     ngf * 2)               # 64x128
        self.e3 = Down(ngf * 2, ngf * 4)               # 32x64
        self.e4 = Down(ngf * 4, ngf * 8)               # 16x32   <- token / coarse stage
        self.e5 = Down(ngf * 8, ngf * 8)               # 8x16
        self.e6 = Down(ngf * 8, ngf * 8)               # 4x8
        self.e7 = Down(ngf * 8, ngf * 8)               # 2x4
        self.e8 = Down(ngf * 8, ngf * 8, norm=False)   # 1x2     <- global bottleneck
        self.cmid, self.cdeep = ngf * 4, ngf * 8

    def forward(self, spec):
        e1 = self.e1(spec); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        e8 = self.e8(self.e7(self.e6(self.e5(e4))))
        return {"e3": e3, "e4": e4, "bottleneck": e8}


def _upsample(d_c, H, W):
    return F.interpolate(d_c, size=(H, W), mode="bilinear", align_corners=False)


class UNetCoarse(nn.Module):
    """U-Net8 encoder -> pooled z -> MLP -> Hc x Wc band-limited ERP depth -> upsample."""
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        self.hc, self.wc = cfg.coarse_head_h, cfg.coarse_head_w
        ngf = getattr(cfg, "ngf", 64)
        self.enc = UNet8Encoder(getattr(cfg, "in_ch", 2), ngf)
        self.gpool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(ngf * 8, cfg.embed_dim), nn.GELU(),
                                  nn.Linear(cfg.embed_dim, self.hc * self.wc))

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        z = self.gpool(self.enc(spec)["bottleneck"])             # (B, ngf*8)
        d_c = torch.sigmoid(self.head(z)).view(-1, 1, self.hc, self.wc)
        D = _upsample(d_c, self.H, self.W)
        return {"D": D, "D0": D, "extras": {"D_coarse": d_c}}


class UNetSH(nn.Module):
    """U-Net8 encoder -> low-order SH coefficients -> synthesize band-limited depth."""
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        ngf = getattr(cfg, "ngf", 64)
        order = getattr(cfg, "coarse_sh_order", 4)
        self.enc = UNet8Encoder(getattr(cfg, "in_ch", 2), ngf)
        self.gpool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        el, az = erp_grid(self.H, self.W)
        B = sh_basis_matrix(order, el, az).astype("float32")     # (N, (order+1)^2)
        self.register_buffer("sh", torch.from_numpy(B))
        self.head = nn.Sequential(nn.Linear(ngf * 8, cfg.embed_dim), nn.GELU(),
                                  nn.Linear(cfg.embed_dim, B.shape[1]))

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        z = self.gpool(self.enc(spec)["bottleneck"])             # (B, ngf*8)
        coef = self.head(z)                                       # (B, Kc)
        d = torch.einsum("bk,nk->bn", coef, self.sh)             # (B, N) band-limited
        D = torch.sigmoid(d).view(-1, 1, self.H, self.W)
        return {"D": D, "D0": D, "extras": {"sh_coef": coef}}


class UNetRayCoarse(nn.Module):
    """U-Net8 e4 audio tokens (K/V) + coarse RayBank ray tokens (Q) -> cross-attn -> 16x32."""
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        self.hc, self.wc = cfg.ray_coarse_h, cfg.ray_coarse_w
        ngf = getattr(cfg, "ngf", 64); dim = cfg.dim
        self.enc = UNet8Encoder(getattr(cfg, "in_ch", 2), ngf)
        self.kv_proj = nn.Linear(ngf * 8, dim)                   # e4 tokens (16x32) -> dim
        pcfg = copy.copy(cfg); pcfg.img_h, pcfg.img_w = self.hc, self.wc
        bank = RayBank(pcfg, device="cpu")
        self.register_buffer("ray_feat", bank.feat)              # (Hc*Wc, F) fixed
        self.ray_proj = nn.Sequential(nn.Linear(bank.feat_dim, dim), nn.GELU(),
                                      nn.Linear(dim, dim))
        self.cross = nn.ModuleList([CrossBlock(dim, cfg.n_heads)
                                    for _ in range(getattr(cfg, "ray_cross_layers", 2))])
        self.selfb = nn.ModuleList([SelfBlock(dim, cfg.n_heads)
                                    for _ in range(getattr(cfg, "ray_self_layers", 1))])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        B = spec.size(0)
        e4 = self.enc(spec)["e4"]                                # (B, ngf*8, 16, 32)
        kv = self.kv_proj(e4.flatten(2).transpose(1, 2))         # (B, 512, dim)
        q = self.ray_proj(self.ray_feat)[None].expand(B, -1, -1)  # (B, Hc*Wc, dim)
        for blk in self.cross:
            q = blk(q, kv)
        for blk in self.selfb:
            q = blk(q)
        d_c = torch.sigmoid(self.head(q)).transpose(1, 2).reshape(B, 1, self.hc, self.wc)
        D = _upsample(d_c, self.H, self.W)
        return {"D": D, "D0": D, "extras": {"D_coarse": d_c, "ray_tokens": q}}


class UNetCoarseResidual(nn.Module):
    """Coarse band-limited layout + small constrained low-pass residual (residual_h x w)."""
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        self.hc, self.wc = cfg.coarse_head_h, cfg.coarse_head_w
        self.rh, self.rw = cfg.residual_h, cfg.residual_w
        self.rscale = getattr(cfg, "residual_scale", 0.1)
        ngf = getattr(cfg, "ngf", 64)
        self.enc = UNet8Encoder(getattr(cfg, "in_ch", 2), ngf)
        self.gpool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(ngf * 8, cfg.embed_dim), nn.GELU(),
                                  nn.Linear(cfg.embed_dim, self.hc * self.wc))
        # residual from the e3 (32x64) feature -> 1ch at residual res, zero-init (starts off)
        self.res = nn.Conv2d(ngf * 4, 1, 3, 1, 1)
        nn.init.zeros_(self.res.weight); nn.init.zeros_(self.res.bias)

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        f = self.enc(spec)
        z = self.gpool(f["bottleneck"])
        d_c = torch.sigmoid(self.head(z)).view(-1, 1, self.hc, self.wc)
        D0 = _upsample(d_c, self.H, self.W)
        r = self.res(f["e3"])                                    # (B,1,32,64)
        r = F.adaptive_avg_pool2d(r, (self.rh, self.rw))         # band-limit
        r = _upsample(self.rscale * torch.tanh(r), self.H, self.W)
        D = (D0 + r).clamp(1e-3, 1.0)
        return {"D": D, "D0": D0, "extras": {"D_coarse": d_c, "residual": r}}
