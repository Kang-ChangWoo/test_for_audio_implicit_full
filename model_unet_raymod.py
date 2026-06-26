"""Ray-conditioned sparse modulation on a strong pix2pix-style U-Net (A15/A16).

Framing (important): this is NOT "pure RayMLP beats U-Net". It is a STRONG U-Net
whose decoder skip feature is *modulated* by sparse spherical ray queries that
cross-attend the U-Net's own audio-encoder tokens. The control is the SAME big
U-Net with `ray_mod_scale=0` (or `--arch unet`), so any win is attributable to
the directional inductive bias the ray queries add, not to extra capacity.

Pipeline:
    spec -> U-Net encoder (e1..e6, explicit so we can tap mid features)
    ray_feat(coarse_h x coarse_w) -> RayMLP -> ray tokens q
    q  cross-attend  e_stage tokens (audio-conditioned)  -> q'
    q' -> ray map -> Conv1x1 (ZERO-init) -> (gamma, beta)
    e_stage_mod = e_stage * (1 + s*tanh(gamma)) + s*beta     # FiLM, s small
    U-Net decoder uses e_stage_mod in the skip path -> sigmoid depth

Zero-init FiLM + small scale `s` => the net starts as a plain U-Net and only
learns to deviate where ray modulation helps, so the baseline is never harmed.

Returns the shared dict interface: {"D", "D0", "extras"}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CrossBlock


# Stage the ray grid attaches to. 8x16 ray grid -> e3 (8x16); 16x32 -> e2 (16x32).
# Each entry: (encoder feature index, channel multiplier of ngf at that stage).
_STAGE = {"e3": (3, 4), "e2": (2, 2)}


class Down(nn.Module):
    """pix2pix encoder block: Conv(4,2,1) (/2) + optional BN + LeakyReLU."""

    def __init__(self, ci, co, norm=True):
        super().__init__()
        layers = [nn.Conv2d(ci, co, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(co))
        layers.append(nn.LeakyReLU(0.2))            # non-inplace: e6 act is reused by d5
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """pix2pix decoder block: ReLU + ConvT(4,2,1) (x2) + optional BN."""

    def __init__(self, ci, co, norm=True):
        super().__init__()
        layers = [nn.ReLU(), nn.ConvTranspose2d(ci, co, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(co))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNetRayMod(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_ch = getattr(cfg, "in_ch", 2)
        ngf = getattr(cfg, "ngf", 96)
        dim = getattr(cfg, "dim", 256)
        self.dim = dim
        self.ngf = ngf
        self.ch, self.cw = cfg.coarse_h, cfg.coarse_w
        self.ray_scale = getattr(cfg, "ray_mod_scale", 0.1)
        # ray_mod_stage may name several encoder stages joined by '+', e.g. "e2+e3".
        # The ray queries live on the 8x16 grid (anchored to e3 tokens); each target
        # stage gets its own zero-init FiLM head (ray_map is upsampled to e2 if needed).
        self.stages = [s for s in getattr(cfg, "ray_mod_stage", "e3").split("+") if s]
        assert all(s in _STAGE for s in self.stages), \
            f"ray_mod_stage parts must be in {list(_STAGE)}, got {self.stages}"

        # --- explicit encoder: 64x128 -> 32x64 -> 16x32 -> 8x16 -> 4x8 -> 2x4 -> 1x2
        self.e1 = Down(in_ch,   ngf,     norm=False)   # 32x64
        self.e2 = Down(ngf,     ngf * 2)               # 16x32
        self.e3 = Down(ngf * 2, ngf * 4)               # 8x16
        self.e4 = Down(ngf * 4, ngf * 8)               # 4x8
        self.e5 = Down(ngf * 8, ngf * 8)               # 2x4
        self.e6 = Down(ngf * 8, ngf * 8, norm=False)   # 1x2 (innermost: no norm)

        # --- sparse ray branch
        self.ray_mlp = nn.Sequential(
            nn.Linear(cfg.coarse_feat_dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU())
        self.kv_proj = nn.Linear(ngf * 4, dim)         # anchor: e3 tokens (8x16) -> dim
        self.cross = nn.ModuleList([CrossBlock(dim, cfg.n_heads)
                                    for _ in range(cfg.n_cross)])
        # one FiLM head per modulated stage; ZERO-init -> starts as a plain U-Net
        self.film = nn.ModuleDict()
        for s in self.stages:
            conv = nn.Conv2d(dim, 2 * ngf * _STAGE[s][1], 1)
            nn.init.zeros_(conv.weight); nn.init.zeros_(conv.bias)
            self.film[s] = conv

        # --- decoder (mirror, with skips); cat doubles the channel count
        self.d5 = Up(ngf * 8,  ngf * 8)                # 1x2  -> 2x4
        self.d4 = Up(ngf * 16, ngf * 8)                # 2x4  -> 4x8   (cat e5)
        self.d3 = Up(ngf * 16, ngf * 4)                # 4x8  -> 8x16  (cat e4)
        self.d2 = Up(ngf * 8,  ngf * 2)                # 8x16 -> 16x32 (cat e3_mod)
        self.d1 = Up(ngf * 4,  ngf)                    # 16x32-> 32x64 (cat e2)
        self.out = nn.Sequential(
            nn.ReLU(), nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1), nn.Sigmoid())

    def _ray_field(self, e3, coarse_feat):
        """Ray queries cross-attend the e3 (8x16) tokens -> audio-conditioned ray map."""
        B = e3.size(0)
        rf = coarse_feat[None].expand(B, -1, -1)               # (B, N, F)
        q = self.ray_mlp(rf)                                   # (B, N, dim)
        tok = self.kv_proj(e3.flatten(2).transpose(1, 2))      # (B, 8*16, dim)
        for blk in self.cross:
            q = blk(q, tok)
        return q.transpose(1, 2).reshape(B, self.dim, self.ch, self.cw)   # (B,dim,8,16)

    def _film(self, feat, ray_map, stage):
        rm = ray_map if ray_map.shape[-2:] == feat.shape[-2:] else \
            F.interpolate(ray_map, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = self.film[stage](rm).chunk(2, dim=1)
        return feat * (1 + self.ray_scale * torch.tanh(gamma)) + self.ray_scale * beta, gamma, beta

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        assert coarse_feat is not None, "UNetRayMod needs coarse_feat (RayBank.feat)"
        e1 = self.e1(spec)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        ray_map = self._ray_field(e3, coarse_feat)
        extras = {"ray_map": ray_map}
        feats = {"e2": e2, "e3": e3}
        for s in self.stages:                                  # modulate each named stage
            feats[s], g, b = self._film(feats[s], ray_map, s)
            extras[f"gamma_{s}"], extras[f"beta_{s}"] = g, b
        e2, e3 = feats["e2"], feats["e3"]
        # diag back-compat: expose a canonical gamma/beta (prefer e3, else first stage)
        ks = "e3" if "e3" in self.stages else (self.stages[0] if self.stages else None)
        if ks is not None:
            extras["gamma"], extras["beta"] = extras[f"gamma_{ks}"], extras[f"beta_{ks}"]

        x = self.d5(e6)
        x = self.d4(torch.cat([x, e5], 1))
        x = self.d3(torch.cat([x, e4], 1))
        x = self.d2(torch.cat([x, e3], 1))
        x = self.d1(torch.cat([x, e2], 1))
        D = self.out(torch.cat([x, e1], 1))
        return {"D": D, "D0": D, "extras": extras}
