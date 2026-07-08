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

from model import CrossBlock, SelfBlock, conv_bn, Refine, FFN
from model_unet_coarse import UNet8Encoder
from ray_features import RayBank


# ---- local spherical window attention (ray <-> ray) -------------------------
def _window_kv(t, win, wrap=True):
    """(B,C,H,W) -> (B,C,win*win,H,W): neighbours. wrap=True: circular-W + replicate-H
    (spherical). wrap=False: zero-pad both (planar image local attention)."""
    pad = win // 2
    if wrap:
        t = torch.cat([t[..., -pad:], t, t[..., :pad]], dim=-1)    # circular azimuth wrap
        t = F.pad(t, (0, 0, pad, pad), mode="replicate")           # replicate elevation (poles)
    else:
        t = F.pad(t, (pad, pad, pad, pad), mode="constant", value=0)   # planar zero-pad
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
    """mode: spherical (wrap+replicate + great-circle bias) | nobias (wrap, no bias)
    | planar (zero-pad, no bias) | off (identity). Ablation of the spherical design."""
    def __init__(self, dim, heads, H, W, win=5, mode="spherical"):
        super().__init__()
        self.mode = mode
        if mode == "off":
            return
        self.h, self.dh, self.win = heads, dim // heads, win
        self.scale = self.dh ** -0.5
        self.wrap = (mode != "planar")
        self.use_bias = (mode == "spherical")
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        if self.use_bias:
            self.register_buffer("geom", _geom_bias_feats(H, W, win))      # (H,K,3)
            self.bias_mlp = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, heads))

    def forward(self, x):
        if self.mode == "off":
            return x
        B, C, H, W = x.shape
        q, k, v = self.to_qkv(x).chunk(3, 1)
        kw = _window_kv(k, self.win, self.wrap).view(B, self.h, self.dh, self.win * self.win, H, W)
        vw = _window_kv(v, self.win, self.wrap).view(B, self.h, self.dh, self.win * self.win, H, W)
        q = q.view(B, self.h, self.dh, H, W)
        attn = torch.einsum("bndhw,bndkhw->bnkhw", q, kw) * self.scale     # (B,nh,K,H,W)
        if self.use_bias:
            bias = self.bias_mlp(self.geom).permute(2, 1, 0)              # (nh,K,H)
            attn = attn + bias[None, :, :, :, None]
        attn = attn.softmax(dim=2)
        out = torch.einsum("bnkhw,bndkhw->bndhw", attn, vw).reshape(B, C, H, W)
        return x + self.proj(out)


# ---- E22/E27: global coarse ray<->ray self-attn + cos-angular-distance bias --
def _coarse_cosang(H, W):
    """(N,N) cos angular distance between all ERP cell-centre directions (N=H*W)."""
    el = (math.pi / 2 - (torch.arange(H).float() + 0.5) / H * math.pi)     # (H,)
    az = (-math.pi + (torch.arange(W).float() + 0.5) / W * 2 * math.pi)    # (W,)
    E = el[:, None].expand(H, W).reshape(-1); A = az[None, :].expand(H, W).reshape(-1)
    dirs = torch.stack([torch.cos(E) * torch.cos(A), torch.cos(E) * torch.sin(A), torch.sin(E)], 1)
    return (dirs @ dirs.t()).clamp(-1, 1)                                  # (N,N)


class CoarseGeoSelfAttn(nn.Module):
    """Global ray<->ray self-attention at the coarse layout scale (16x32=512 tok),
    with a learned per-head bias driven by the true cos-angular-distance (E22 self-attn
    + E27 geometry-aware bias). Residual, pre-norm."""
    def __init__(self, dim, heads, H, W, geo=True):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.scale = self.dh ** -0.5
        self.geo = geo
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        if geo:                                                          # ray-grounding: cos-ang bias
            self.register_buffer("cosang", _coarse_cosang(H, W))          # (N,N)
            self.bias_mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, heads))

    def forward(self, x):                                                # (B,C,H,W)
        B, C, H, W = x.shape; N = H * W
        t = x.flatten(2).transpose(1, 2)                                 # (B,N,C)
        q, k, v = self.to_qkv(self.norm(t)).chunk(3, -1)
        q = q.view(B, N, self.h, self.dh).transpose(1, 2)
        k = k.view(B, N, self.h, self.dh).transpose(1, 2)
        v = v.view(B, N, self.h, self.dh).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale                    # (B,h,N,N)
        if self.geo:
            bias = self.bias_mlp(self.cosang[..., None]).permute(2, 0, 1)    # (h,N,N)
            attn = attn + bias[None]
        attn = attn.softmax(-1)
        o = self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))
        return (t + o).transpose(1, 2).reshape(B, C, H, W)


# ---- Cue-factorized branches + separate-K/V cross-attention ------------------
class CueEncoder(nn.Module):
    """Lightweight cue-specific encoder: 256x512 -> 16x32 (=512 tokens) x dim.
    4 stride-2 conv blocks. Cheap branch producing coarse cue tokens aligned to the
    ray coarse scale (so K/V routing is not confounded by token resolution)."""
    def __init__(self, in_ch, dim):
        super().__init__()
        chs = [in_ch, 32, 64, 128, dim]
        self.net = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(chs[i], chs[i + 1], 4, 2, 1), nn.BatchNorm2d(chs[i + 1]), nn.GELU())
            for i in range(4)])

    def forward(self, x):
        z = self.net(x)                                  # (B,dim,16,32)
        return z.flatten(2).transpose(1, 2)              # (B,512,dim)


class CueCrossBlock(nn.Module):
    """Cross-attention with SEPARATE key/value sources (per-branch LayerNorm on K,V)."""
    def __init__(self, dim, heads):
        super().__init__()
        self.nq = nn.LayerNorm(dim); self.nk = nn.LayerNorm(dim); self.nv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = FFN(dim)

    def forward(self, q, k, v):
        a, _ = self.attn(self.nq(q), self.nk(k), self.nv(v))
        q = q + a
        return q + self.ffn(self.nq(q))


# ---- RayDPT ------------------------------------------------------------------
class RayDPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        ngf = getattr(cfg, "ngf", 64); dim = cfg.dim; heads = cfg.n_heads
        nL = getattr(cfg, "ray_cross_layers", 2)
        self.enc = UNet8Encoder(getattr(cfg, "in_ch", 2), ngf)
        # --- Cue factorization (Group A input stems + Group B K/V routing); RayDPT intact when off ---
        self.cue_stems = getattr(cfg, "cue_stems", False)
        if self.cue_stems:                                # two-stem input -> MAIN encoder
            cm, cs = getattr(cfg, "cue_cmag", 32), getattr(cfg, "cue_cspatial", 32)
            mk_stem = lambda ic, oc: nn.Sequential(nn.Conv2d(ic, oc, 3, 1, 1), nn.GELU(),
                                                   nn.Conv2d(oc, oc, 3, 1, 1), nn.GELU())
            self.stem_mag = mk_stem(2, cm); self.stem_spatial = mk_stem(3, cs)
            self.enc = UNet8Encoder(cm + cs, ngf)         # encoder consumes fused stem output
        self.cue_route = getattr(cfg, "cue_route", False)
        if self.cue_route:                                # cue coarse branches + routed F16 (Group B)
            self.cue_dup = getattr(cfg, "cue_dup_input", False)       # F2 control: both branches see all ch
            self.cue_rand = getattr(cfg, "cue_random_split", False)   # F1 control: non-semantic split
            self.cue_adapter = getattr(cfg, "cue_adapter", False)     # F4/A3: adapters on shared e4 (no cue enc)
            self.cue_fused_mode = getattr(cfg, "cue_fused_mode", "kv4")   # C: kv4|concat|add|gate
            self.cue_dual = getattr(cfg, "cue_dual", False)           # D1: parallel spatial+mag attention
            self.mag_idx = [0, 3] if self.cue_rand else [0, 1]
            self.spa_idx = [1, 2, 4] if self.cue_rand else [2, 3, 4]
            mch = 5 if self.cue_dup else len(self.mag_idx)
            sch = 5 if self.cue_dup else len(self.spa_idx)
            if self.cue_adapter:
                self.ad_mag = nn.Linear(dim, dim); self.ad_spatial = nn.Linear(dim, dim)
            else:
                self.cue_mag_enc = CueEncoder(mch, dim); self.cue_spatial_enc = CueEncoder(sch, dim)
            if self.cue_fused_mode == "concat":
                self.fuse_proj = nn.Linear(dim * 2, dim)
            elif self.cue_fused_mode == "gate":
                self.fuse_gate = nn.Parameter(torch.zeros(1))
            self.cr16_route = nn.ModuleList([CueCrossBlock(dim, heads) for _ in range(nL)])
            if self.cue_dual:
                self.cr16_mag = nn.ModuleList([CueCrossBlock(dim, heads) for _ in range(nL)])
                self.dual_proj = nn.Linear(dim * 2, dim)
            self.key_src = getattr(cfg, "kv_key_source", "fused")
            self.val_src = getattr(cfg, "kv_value_source", "fused")

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
        self.lsa32 = LocalSphericalAttention(dim, heads, 32, 64, getattr(cfg, "raydpt_win32", 5), mode=getattr(cfg, "lsa_mode", "spherical"))
        self.lsa64 = LocalSphericalAttention(dim, heads, 64, 128, getattr(cfg, "raydpt_win64", 3), mode=getattr(cfg, "lsa_mode", "spherical"))
        self.coarse_head = nn.Conv2d(dim, 1, 1)
        self.head = nn.Sequential(conv_bn(dim, ngf), conv_bn(ngf, ngf), nn.Conv2d(ngf, 1, 3, 1, 1))
        # multi-scale-KV fusion: ray queries cross-attend COMPACT acoustic memory from
        # several encoder scales (e4 + pooled e3 + pooled e2), instead of raw e2/e3
        # DPT skip-addition (coordinate mismatch: spec-axes != ERP-axes).
        self.msf = getattr(cfg, "raydpt_msf", False)
        if self.msf:
            self.e3_pool = nn.AdaptiveAvgPool2d((16, 32))     # 32x64 -> 16x32 (512 tok)
            self.e2_pool = nn.AdaptiveAvgPool2d((16, 32))     # 64x128 -> 16x32 (512 tok)
            self.kv_e3s = nn.Linear(ngf * 4, dim)
            self.kv_e2 = nn.Linear(ngf * 2, dim)
        # ablation: no ray conditioning -> LEARNED direction-agnostic queries (same
        # decoder capacity, but queries carry NO spherical direction info).
        self.noray = getattr(cfg, "raydpt_noray", False)
        if self.noray:
            self.q16 = nn.Parameter(torch.randn(16 * 32, dim) * 0.02)
            self.q32 = nn.Parameter(torch.randn(32 * 64, dim) * 0.02)
            self.q64 = nn.Parameter(torch.randn(64 * 128, dim) * 0.02)
        # Task2 ablation: ray<->audio via a single GLOBAL mean-pooled audio code (no per-ray
        # cross-attn). ray bank features condition on that code (concat + MLP), then CSA.
        self.cross_mode = getattr(cfg, "cross_mode", "cross")
        if self.cross_mode == "global":
            self.gcond = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        # E22/E27: coarse global geo self-attn; E29: gated DPT skips
        self.coarse_sa = getattr(cfg, "raydpt_coarse_sa", False)
        if self.coarse_sa:
            _nb = getattr(cfg, "coarse_sa_blocks", 1)
            self.csa = nn.ModuleList([CoarseGeoSelfAttn(dim, heads, 16, 32, geo=getattr(cfg, "coarse_sa_geo", True)) for _ in range(_nb)])
        self.gated_skip = getattr(cfg, "raydpt_gated_skip", False)
        if self.gated_skip:
            self.g4 = nn.Conv2d(dim * 2, dim, 1)
            self.g3 = nn.Conv2d(dim * 2, dim, 1)
            self.g2 = nn.Conv2d(dim * 2, dim, 1)
        self.lite = getattr(cfg, "raydpt_lite", False)        # 2-scale (32,64) lite variant
        # full-decode: LEARNED upsample 64x128 -> 256x512 (+e1 skip), like U-Net's decoder
        # tail, instead of parameter-free bilinear x4. Closes the fairness gap vs pix2pix.
        self.full_decode = getattr(cfg, "raydpt_full_decode", False)
        if self.full_decode:
            self.proj_fd = conv_bn(dim, ngf)                  # dim->ngf at 64x128 (cheap before up)
            self.se1 = nn.Conv2d(ngf, ngf, 1)                 # e1 (128x256) DPT skip
            self.dec1 = Refine(ngf)                           # at 128x256
            self.dec2 = nn.Sequential(conv_bn(ngf, ngf), Refine(ngf))   # at 256x512
            self.head_fd = nn.Conv2d(ngf, 1, 3, 1, 1)

    def _cross(self, rp, rf, blocks, kv, B, h, w):
        q = rp(rf)[None].expand(B, -1, -1)
        for blk in blocks:
            q = blk(q, kv)
        return q.transpose(1, 2).reshape(B, -1, h, w)

    def _cross_q(self, q0, blocks, kv, B, h, w):          # learned query (no ray conditioning)
        q = q0[None].expand(B, -1, -1)
        for blk in blocks:
            q = blk(q, kv)
        return q.transpose(1, 2).reshape(B, -1, h, w)

    def _global(self, rp, rf, g, B, h, w):                # Task2: condition ray bank on ONE global audio code
        q = rp(rf)[None].expand(B, -1, -1)                # (B,N,dim) ray features (no audio retrieval)
        gg = g[:, None, :].expand(-1, q.size(1), -1)      # broadcast global code
        x = self.gcond(torch.cat([q, gg], -1))            # concat + MLP conditioning
        return x.transpose(1, 2).reshape(B, -1, h, w)

    def _skip(self, base, s, gate):                       # E29: gated (vs raw-add) DPT skip
        if gate is None:
            return base + s
        return base + torch.sigmoid(gate(torch.cat([base, s], 1))) * s

    def _csa(self, m):                                    # E51: stacked post-fusion geo self-attn
        for blk in self.csa:
            m = blk(m)
        return m

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        B = spec.size(0)
        enc_in = spec
        if self.cue_stems:                                # Group A: two-stem input fusion
            enc_in = torch.cat([self.stem_mag(spec[:, :2]), self.stem_spatial(spec[:, 2:5])], 1)
        e1 = self.enc.e1(enc_in); e2 = self.enc.e2(e1); e3 = self.enc.e3(e2); e4 = self.enc.e4(e3)
        kv4 = self.kv_e4(e4.flatten(2).transpose(1, 2))        # (B,512,dim)
        if self.cue_route:                                # Group B: cue-specific coarse K/V at F16
            if self.cue_adapter:                          # F4: adapters on shared e4 (no cue encoders)
                zm, zs = self.ad_mag(kv4), self.ad_spatial(kv4)
            else:
                mx = spec if self.cue_dup else spec[:, self.mag_idx]
                sx = spec if self.cue_dup else spec[:, self.spa_idx]
                zm, zs = self.cue_mag_enc(mx), self.cue_spatial_enc(sx)
            if self.cue_fused_mode == "concat":
                zf = self.fuse_proj(torch.cat([zm, zs], -1))
            elif self.cue_fused_mode == "add":
                zf = zm + zs
            elif self.cue_fused_mode == "gate":
                g = torch.sigmoid(self.fuse_gate); zf = g * zm + (1 - g) * zs
            else:
                zf = kv4
            feat = {"fused": zf, "spatial": zs, "magnitude": zm}
            self._cue_zm, self._cue_zs = zm, zs
            self._cue_K = feat[self.key_src]; self._cue_V = feat[self.val_src]
        if self.lite:
            # 2-scale lite: ONE ray cross-attn at 32x64 (Q32 <- e4), e3/e2 projection
            # skips + local spherical attn. Isolates the DPT-fusion / ray-grid gain.
            F32 = self._cross(self.ray_proj, self.rf32, self.cr32, kv4, B, 32, 64)
            m = F32 + self.se3(e3)                              # 32x64
            d_c = torch.sigmoid(self.coarse_head(F.adaptive_avg_pool2d(m, (16, 32))))
            x = self.lsa32(self.refine32(m))                   # 32x64
            x = self.lsa64(self.refine64(self.up(x) + self.se2(e2)))   # 64x128
        elif self.noray:
            # ablation: same decoder but LEARNED (direction-agnostic) queries.
            kv3 = self.kv_e3(e3.flatten(2).transpose(1, 2))
            F16 = self._cross_q(self.q16, self.cr16, kv4, B, 16, 32)
            F32 = self._cross_q(self.q32, self.cr32, kv3, B, 32, 64)
            F64 = self._cross_q(self.q64, self.cr64, kv4, B, 64, 128)
            m16 = F16 + self.se4(e4)
            d_c = torch.sigmoid(self.coarse_head(m16))
            x = self.lsa32(self.refine32(self.up(m16) + F32 + self.se3(e3)))
            x = self.lsa64(self.refine64(self.up(x) + F64 + self.se2(e2)))
        elif self.msf:
            # multi-scale compact KV: each ray scale queries the right acoustic memory.
            kv3 = self.kv_e3(e3.flatten(2).transpose(1, 2))                          # 2048 (32x64)
            kv3s = self.kv_e3s(self.e3_pool(e3).flatten(2).transpose(1, 2))          # 512  (pooled 16x32)
            kv2 = self.kv_e2(self.e2_pool(e2).flatten(2).transpose(1, 2))            # 512  (pooled 16x32)
            kv32 = torch.cat([kv4, kv3], 1)                                          # 2560
            kv64 = torch.cat([kv4, kv3s, kv2], 1)                                    # 1536 (compact)
            F16 = self._cross(self.ray_proj, self.rf16, self.cr16, kv4, B, 16, 32)
            F32 = self._cross(self.ray_proj, self.rf32, self.cr32, kv32, B, 32, 64)
            F64 = self._cross(self.ray_proj, self.rf64, self.cr64, kv64, B, 64, 128)
            m16 = F16 + self.se4(e4)                            # e4 same coord at coarsest -> keep
            d_c = torch.sigmoid(self.coarse_head(m16))
            x = self.lsa32(self.refine32(self.up(m16) + F32))  # e3 enters via kv32 (no raw skip)
            x = self.lsa64(self.refine64(self.up(x) + F64))    # e2 enters via kv2  (no raw skip)
        elif self.cross_mode == "global":
            # single global audio code (mean-pool of coarse tokens) — NO per-ray retrieval.
            g = kv4.mean(dim=1)                                # (B,dim) global acoustic code
            F16 = self._global(self.ray_proj, self.rf16, g, B, 16, 32)
            F32 = self._global(self.ray_proj, self.rf32, g, B, 32, 64)
            F64 = self._global(self.ray_proj, self.rf64, g, B, 64, 128)
            g4 = self.g4 if self.gated_skip else None
            g3 = self.g3 if self.gated_skip else None
            g2 = self.g2 if self.gated_skip else None
            m16 = self._skip(F16, self.se4(e4), g4)
            if self.coarse_sa:
                m16 = self._csa(m16)
            d_c = torch.sigmoid(self.coarse_head(m16))
            x = self.lsa32(self.refine32(self._skip(self.up(m16) + F32, self.se3(e3), g3)))
            x = self.lsa64(self.refine64(self._skip(self.up(x) + F64, self.se2(e2), g2)))
        else:
            kv3 = self.kv_e3(e3.flatten(2).transpose(1, 2))    # (B,2048,dim)
            if self.cue_route:                             # Group B: routed K/V at coarse F16
                q0 = self.ray_proj(self.rf16)[None].expand(B, -1, -1)
                if self.cue_dual:                          # D1: parallel spatial + magnitude attention
                    qs, qm = q0, q0
                    for blk in self.cr16_route: qs = blk(qs, self._cue_zs, self._cue_zs)
                    for blk in self.cr16_mag: qm = blk(qm, self._cue_zm, self._cue_zm)
                    q = self.dual_proj(torch.cat([qs, qm], -1))
                else:
                    q = q0
                    for blk in self.cr16_route:
                        q = blk(q, self._cue_K, self._cue_V)
                F16 = q.transpose(1, 2).reshape(B, -1, 16, 32)
            else:
                F16 = self._cross(self.ray_proj, self.rf16, self.cr16, kv4, B, 16, 32)
            F32 = self._cross(self.ray_proj, self.rf32, self.cr32, kv3, B, 32, 64)
            F64 = self._cross(self.ray_proj, self.rf64, self.cr64, kv4, B, 64, 128)
            g4 = self.g4 if self.gated_skip else None
            g3 = self.g3 if self.gated_skip else None
            g2 = self.g2 if self.gated_skip else None
            m16 = self._skip(F16, self.se4(e4), g4)            # 16x32
            if self.coarse_sa:
                m16 = self._csa(m16)                           # E22/E27/E51 geo self-attn
            d_c = torch.sigmoid(self.coarse_head(m16))         # (B,1,16,32) coarse layout
            x = self.lsa32(self.refine32(self._skip(self.up(m16) + F32, self.se3(e3), g3)))   # 32x64
            x = self.lsa64(self.refine64(self._skip(self.up(x) + F64, self.se2(e2), g2)))     # 64x128
        if self.full_decode:                                   # LEARNED upsample 64x128 -> 256x512
            xf = self.up(self.proj_fd(x))                       # 128x256, ngf
            xf = self.dec1(xf + self.se1(e1))                  # + e1 skip
            xf = self.dec2(self.up(xf))                        # 256x512, ngf
            D = torch.sigmoid(self.head_fd(xf))
        else:
            D = torch.sigmoid(self.head(x))                    # 64x128 head
            D = F.interpolate(D, (self.H, self.W), mode="bilinear", align_corners=False)  # parameter-free x4
        return {"D": D, "D0": D, "extras": {"D_coarse": d_c}}


# ---- RayDPT + Acoustic Perceiver Resampler --------------------------------- #
class RayDPTResampler(nn.Module):
    """U-Net8 audio encoder -> multi-scale acoustic tokens -> Perceiver/Q-Former-style
    resampler (learned latents compress them into a compact scene memory) -> physical
    ERP ray queries cross-attend the compact memory -> DPT fusion + spherical attn.

    Rationale: spectrogram grid != ERP ray grid, so don't raw-skip e2/e3 (coordinate
    mismatch). learned latents = "scene acoustic summary"; ray queries f(theta,phi) =
    "depth in this direction?". Ray cross-attn cost drops to Q x N_latents.
    """
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        ngf = getattr(cfg, "ngf", 64); dim = cfg.dim; heads = cfg.n_heads
        nL = getattr(cfg, "ray_cross_layers", 2)
        nR = getattr(cfg, "resampler_layers", 3)
        N = getattr(cfg, "resampler_latents", 64)
        self.enc = UNet8Encoder(getattr(cfg, "in_ch", 2), ngf)

        def bank(h, w):
            pc = copy.copy(cfg); pc.img_h, pc.img_w = h, w
            b = RayBank(pc, device="cpu"); return b.feat, b.feat_dim
        f16, fd = bank(16, 32); f32, _ = bank(32, 64); f64, _ = bank(64, 128)
        self.register_buffer("rf16", f16); self.register_buffer("rf32", f32); self.register_buffer("rf64", f64)
        self.ray_proj = nn.Sequential(nn.Linear(fd, dim), nn.GELU(), nn.Linear(dim, dim))

        # multi-scale acoustic tokenizers (+ per-scale marker embedding)
        self.kv_e4 = nn.Linear(ngf * 8, dim)
        self.kv_e3 = nn.Linear(ngf * 4, dim)
        self.kv_e2 = nn.Linear(ngf * 2, dim)
        self.e2_pool = nn.AdaptiveAvgPool2d((32, 64))         # 64x128 -> 32x64 (keep cost sane)
        self.scale_emb = nn.Parameter(torch.zeros(3, dim))    # e4,e3,e2 markers

        # acoustic Perceiver resampler: learned latents <- acoustic tokens
        self.latents = nn.Parameter(torch.randn(N, dim) * 0.02)
        self.rs_cross = nn.ModuleList([CrossBlock(dim, heads) for _ in range(nR)])
        self.rs_self = nn.ModuleList([SelfBlock(dim, heads) for _ in range(nR)])

        # ray-query decoder reads the compact memory L
        mk = lambda: nn.ModuleList([CrossBlock(dim, heads) for _ in range(nL)])
        self.cr16, self.cr32, self.cr64 = mk(), mk(), mk()
        self.se4 = nn.Conv2d(ngf * 8, dim, 1)                 # coarse aligned skip (16x32) only
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.refine32 = Refine(dim); self.refine64 = Refine(dim)
        self.lsa32 = LocalSphericalAttention(dim, heads, 32, 64, getattr(cfg, "raydpt_win32", 5), mode=getattr(cfg, "lsa_mode", "spherical"))
        self.lsa64 = LocalSphericalAttention(dim, heads, 64, 128, getattr(cfg, "raydpt_win64", 3), mode=getattr(cfg, "lsa_mode", "spherical"))
        self.coarse_head = nn.Conv2d(dim, 1, 1)
        # learned full-decode 64x128 -> 256x512 (+e1 skip)
        self.proj_fd = conv_bn(dim, ngf); self.se1 = nn.Conv2d(ngf, ngf, 1)
        self.dec1 = Refine(ngf); self.dec2 = nn.Sequential(conv_bn(ngf, ngf), Refine(ngf))
        self.head_fd = nn.Conv2d(ngf, 1, 3, 1, 1)

    def _cross(self, rf, blocks, kv, B, h, w):
        q = self.ray_proj(rf)[None].expand(B, -1, -1)
        for blk in blocks:
            q = blk(q, kv)
        return q.transpose(1, 2).reshape(B, -1, h, w)

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        B = spec.size(0)
        e1 = self.enc.e1(spec); e2 = self.enc.e2(e1); e3 = self.enc.e3(e2); e4 = self.enc.e4(e3)
        A4 = self.kv_e4(e4.flatten(2).transpose(1, 2)) + self.scale_emb[0]      # 512
        A3 = self.kv_e3(e3.flatten(2).transpose(1, 2)) + self.scale_emb[1]      # 2048
        A2 = self.kv_e2(self.e2_pool(e2).flatten(2).transpose(1, 2)) + self.scale_emb[2]  # 2048
        A = torch.cat([A4, A3, A2], dim=1)                                       # (B, ~4608, dim)
        L = self.latents[None].expand(B, -1, -1)                                 # (B, N, dim)
        for cb, sb in zip(self.rs_cross, self.rs_self):
            L = cb(L, A); L = sb(L)                                              # compact scene memory
        F16 = self._cross(self.rf16, self.cr16, L, B, 16, 32)
        F32 = self._cross(self.rf32, self.cr32, L, B, 32, 64)
        F64 = self._cross(self.rf64, self.cr64, L, B, 64, 128)
        m16 = F16 + self.se4(e4)
        d_c = torch.sigmoid(self.coarse_head(m16))
        x = self.lsa32(self.refine32(self.up(m16) + F32))
        x = self.lsa64(self.refine64(self.up(x) + F64))
        xf = self.up(self.proj_fd(x)); xf = self.dec1(xf + self.se1(e1)); xf = self.dec2(self.up(xf))
        D = torch.sigmoid(self.head_fd(xf))
        return {"D": D, "D0": D, "extras": {"D_coarse": d_c}}
