"""RayDPT: ray-conditioned multi-scale Dense-Prediction-Transformer decoder.

Interpretation (not a copy of DPT): keep the U-Net8 *encoder* (strong audio
encoding), but decode on the ERP RAY GRID with DPT-style multi-scale fusion +
local SPHERICAL attention. The implicit directional bias lives in the whole
ray-conditioned feature pyramid, not in a final scalar ray-MLP.

  spec -> U-Net8 encoder -> {e2 64x128, e3 32x64, e4 16x32}
  ray query pyramid {Q16, Q32, Q64} from spherical RayBank features
  audio<->ray = GLOBAL cross-attn at COARSE tokens (e4=512, e3=2048); the 64x128
      e2 tokens (8192) would make cross-attn O(N^2)=8192^2, so fine detail enters
      as a DPT SKIP (1x1 conv of e2) instead — this is exactly how a U-Net injects
      encoder detail, which is the spatial-integration win we want to absorb.
  ray<->ray = LOCAL spherical window attention at 32x64 and 64x128 (circular
      azimuth wrap + spherical relative-position bias).
  DPT fusion coarse->fine -> head -> ERP depth.

Returns {"D","D0","extras":{"D_coarse"}} so the COARSE composite loss in
train_fullmap (dense + coarse-16x32 + low-pass) applies.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CrossBlock, conv_bn, Refine
from model_unet_coarse import UNet8Encoder
from ray_features import RayBank


# ---- local spherical window attention (ray <-> ray) -------------------------
def _window_kv(t, win):
    """(B,C,H,W) -> (B,C,win*win,H,W): neighbours via circular-W / replicate-H pad."""
    pad = win // 2
    t = torch.cat([t[..., -pad:], t, t[..., :pad]], dim=-1)        # circular azimuth wrap
    t = F.pad(t, (0, 0, pad, pad), mode="replicate")               # replicate elevation (poles)
    B, C, Hp, Wp = t.shape
    H, W = Hp - 2 * pad, Wp - 2 * pad
    cols = F.unfold(t, kernel_size=win)                            # (B, C*win*win, H*W)
    return cols.view(B, C, win * win, H, W)


def _geom_bias_feats(H, W, win):
    """(H, win*win, 3): [wrapped dtheta, dphi, cos angular distance] per row/offset."""
    pad = win // 2
    el = (math.pi / 2 - (torch.arange(H).float() + 0.5) / H * math.pi)     # (H,)
    offs = [(dr, dc) for dr in range(-pad, pad + 1) for dc in range(-pad, pad + 1)]
    out = torch.zeros(H, len(offs), 3)
    dphi_u, dth_u = math.pi / H, 2 * math.pi / W
    for h in range(H):
        ei = el[h]
        for k, (dr, dc) in enumerate(offs):
            ej = el[min(max(h + dr, 0), H - 1)]
            dth = dc * dth_u
            cosang = (torch.sin(ei) * torch.sin(ej)
                      + torch.cos(ei) * torch.cos(ej) * math.cos(dth))
            out[h, k] = torch.tensor([dth, dr * dphi_u, float(cosang)])
    return out


class LocalSphericalAttention(nn.Module):
    def __init__(self, dim, heads, H, W, win=5):
        super().__init__()
        self.h, self.dh, self.win = heads, dim // heads, win
        self.scale = self.dh ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.register_buffer("geom", _geom_bias_feats(H, W, win))          # (H,K,3)
        self.bias_mlp = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, heads))

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.to_qkv(x).chunk(3, 1)
        kw = _window_kv(k, self.win).view(B, self.h, self.dh, self.win * self.win, H, W)
        vw = _window_kv(v, self.win).view(B, self.h, self.dh, self.win * self.win, H, W)
        q = q.view(B, self.h, self.dh, H, W)
        attn = torch.einsum("bndhw,bndkhw->bnkhw", q, kw) * self.scale     # (B,nh,K,H,W)
        bias = self.bias_mlp(self.geom).permute(2, 1, 0)                   # (nh,K,H)
        attn = attn + bias[None, :, :, :, None]
        attn = attn.softmax(dim=2)
        out = torch.einsum("bnkhw,bndkhw->bndhw", attn, vw).reshape(B, C, H, W)
        return x + self.proj(out)


# ---- RayDPT ------------------------------------------------------------------
class RayDPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        ngf = getattr(cfg, "ngf", 64); dim = cfg.dim; heads = cfg.n_heads
        nL = getattr(cfg, "ray_cross_layers", 2)
        self.enc = UNet8Encoder(getattr(cfg, "in_ch", 2), ngf)

        def bank(h, w):
            pc = copy.copy(cfg); pc.img_h, pc.img_w = h, w
            b = RayBank(pc, device="cpu"); return b.feat, b.feat_dim
        f16, fd = bank(16, 32); f32, _ = bank(32, 64); f64, _ = bank(64, 128)
        self.register_buffer("rf16", f16); self.register_buffer("rf32", f32)
        self.register_buffer("rf64", f64)
        # SHARED ray-query MLP across scales: ray feats are direction functions
        # (resolution-independent, same feat_dim), so one projection keeps the same
        # direction -> same embedding at every scale (scale-consistent, fewer params).
        self.ray_proj = nn.Sequential(nn.Linear(fd, dim), nn.GELU(), nn.Linear(dim, dim))
        # audio kv: e4 (512 tok), e3 (2048 tok). 64-scale reuses e4 (cheap global cue).
        self.kv_e4 = nn.Linear(ngf * 8, dim)
        self.kv_e3 = nn.Linear(ngf * 4, dim)
        mk_cr = lambda: nn.ModuleList([CrossBlock(dim, heads) for _ in range(nL)])
        self.cr16, self.cr32, self.cr64 = mk_cr(), mk_cr(), mk_cr()
        # DPT encoder skips (U-Net detail injection)
        self.se4 = nn.Conv2d(ngf * 8, dim, 1)
        self.se3 = nn.Conv2d(ngf * 4, dim, 1)
        self.se2 = nn.Conv2d(ngf * 2, dim, 1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.refine32 = Refine(dim); self.refine64 = Refine(dim)
        self.lsa32 = LocalSphericalAttention(dim, heads, 32, 64, getattr(cfg, "raydpt_win32", 5))
        self.lsa64 = LocalSphericalAttention(dim, heads, 64, 128, getattr(cfg, "raydpt_win64", 3))
        self.coarse_head = nn.Conv2d(dim, 1, 1)
        self.head = nn.Sequential(conv_bn(dim, ngf), conv_bn(ngf, ngf), nn.Conv2d(ngf, 1, 3, 1, 1))
        self.lite = getattr(cfg, "raydpt_lite", False)        # 2-scale (32,64) lite variant

    def _cross(self, rp, rf, blocks, kv, B, h, w):
        q = rp(rf)[None].expand(B, -1, -1)
        for blk in blocks:
            q = blk(q, kv)
        return q.transpose(1, 2).reshape(B, -1, h, w)

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        B = spec.size(0)
        e1 = self.enc.e1(spec); e2 = self.enc.e2(e1); e3 = self.enc.e3(e2); e4 = self.enc.e4(e3)
        kv4 = self.kv_e4(e4.flatten(2).transpose(1, 2))        # (B,512,dim)
        if self.lite:
            # 2-scale lite: ONE ray cross-attn at 32x64 (Q32 <- e4), e3/e2 projection
            # skips + local spherical attn. Isolates the DPT-fusion / ray-grid gain.
            F32 = self._cross(self.ray_proj, self.rf32, self.cr32, kv4, B, 32, 64)
            m = F32 + self.se3(e3)                              # 32x64
            d_c = torch.sigmoid(self.coarse_head(F.adaptive_avg_pool2d(m, (16, 32))))
            x = self.lsa32(self.refine32(m))                   # 32x64
            x = self.lsa64(self.refine64(self.up(x) + self.se2(e2)))   # 64x128
        else:
            kv3 = self.kv_e3(e3.flatten(2).transpose(1, 2))    # (B,2048,dim)
            F16 = self._cross(self.ray_proj, self.rf16, self.cr16, kv4, B, 16, 32)
            F32 = self._cross(self.ray_proj, self.rf32, self.cr32, kv3, B, 32, 64)
            F64 = self._cross(self.ray_proj, self.rf64, self.cr64, kv4, B, 64, 128)
            m16 = F16 + self.se4(e4)                            # 16x32
            d_c = torch.sigmoid(self.coarse_head(m16))         # (B,1,16,32) coarse layout
            x = self.lsa32(self.refine32(self.up(m16) + F32 + self.se3(e3)))   # 32x64
            x = self.lsa64(self.refine64(self.up(x) + F64 + self.se2(e2)))     # 64x128
        D = torch.sigmoid(self.head(x))
        D = F.interpolate(D, (self.H, self.W), mode="bilinear", align_corners=False)
        return {"D": D, "D0": D, "extras": {"D_coarse": d_c}}
