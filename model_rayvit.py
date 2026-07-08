"""RayViT: pretrained ViT-B/16 encoder (fine-tuned) + ray-conditioned cross-attention decoder.
Ray-bank queries cross-attend the ViT patch tokens, then coarse geo self-attn + DPT upsample.

modes (cfg.rayvit_mode):
  single      — ViT last-layer 16x32 tokens only (pure proposal); no fine multi-scale.
  multiscale  — ViT intermediate blocks {3,6,9,12} as pseudo multi-scale KV.
  hybrid      — ViT coarse tokens + U-Net8 e2/e3 fine skips (fixes ViT's single-scale detail loss).
Anticipated issues handled by flags: vit_freeze (overfit), vit_pretrained (ImageNet prior control).
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models import ViT_B_16_Weights

from model import CrossBlock, conv_bn, Refine
from model_raydpt import CoarseGeoSelfAttn, LocalSphericalAttention
from model_unet_coarse import UNet8Encoder
from ray_features import RayBank


def _interp_pos(pos, old, new):
    cls, patch = pos[:, :1], pos[:, 1:]
    D = patch.shape[-1]; oh, ow = old; nh, nw = new
    patch = patch.reshape(1, oh, ow, D).permute(0, 3, 1, 2)
    patch = F.interpolate(patch, (nh, nw), mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, nh * nw, D)
    return torch.cat([cls, patch], 1)


class ViTEncoder(nn.Module):
    """ViT-B/16 with in_ch->3 adapter, pos-embed interpolated to (gh,gw), optional intermediates."""
    def __init__(self, in_ch, gh, gw, pretrained=True, freeze=False, multiscale=False):
        super().__init__()
        vit = tvm.vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        self.adapter = nn.Conv2d(in_ch, 3, 1)
        self.patch_embed = vit.conv_proj            # Conv2d(3,768,16,16) stride16
        self.cls = vit.class_token
        self.encoder = vit.encoder
        self.encoder.pos_embedding = nn.Parameter(
            _interp_pos(vit.encoder.pos_embedding.data, (14, 14), (gh, gw)))
        self.gh, self.gw, self.multiscale = gh, gw, multiscale
        if freeze:
            for p in self.patch_embed.parameters(): p.requires_grad = False
            for p in self.encoder.parameters(): p.requires_grad = False
            self.cls.requires_grad = False

    def forward(self, x):
        x = self.adapter(x)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)     # (B, gh*gw, 768)
        B = x.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        x = x + self.encoder.pos_embedding
        x = self.encoder.dropout(x)
        feats = []
        for i, blk in enumerate(self.encoder.layers):
            x = blk(x)
            if self.multiscale and i in (2, 5, 8, 11):
                feats.append(x[:, 1:, :])
        x = self.encoder.ln(x)
        return x[:, 1:, :], feats                              # (B, gh*gw, 768), [intermediates]


class RayViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        dim = cfg.dim; heads = cfg.n_heads; nL = getattr(cfg, "ray_cross_layers", 2)
        ngf = getattr(cfg, "ngf", 64)
        self.mode = getattr(cfg, "rayvit_mode", "single")
        self.vit = ViTEncoder(getattr(cfg, "in_ch", 5), 16, 32,
                              getattr(cfg, "vit_pretrained", True),
                              getattr(cfg, "vit_freeze", False),
                              self.mode == "multiscale")
        self.vit_proj = nn.Linear(768, dim)
        if self.mode == "multiscale":
            self.ms_proj = nn.ModuleList([nn.Linear(768, dim) for _ in range(4)])

        def bank(h, w):
            pc = copy.copy(cfg); pc.img_h, pc.img_w = h, w
            b = RayBank(pc, device="cpu"); return b.feat, b.feat_dim
        f16, fd = bank(16, 32); f32, _ = bank(32, 64); f64, _ = bank(64, 128)
        self.register_buffer("rf16", f16); self.register_buffer("rf32", f32); self.register_buffer("rf64", f64)
        self.ray_proj = nn.Sequential(nn.Linear(fd, dim), nn.GELU(), nn.Linear(dim, dim))
        mk = lambda: nn.ModuleList([CrossBlock(dim, heads) for _ in range(nL)])
        self.cr16, self.cr32, self.cr64 = mk(), mk(), mk()

        self.coarse_sa = getattr(cfg, "raydpt_coarse_sa", True)
        if self.coarse_sa:
            nb = getattr(cfg, "coarse_sa_blocks", 1)
            self.csa = nn.ModuleList([CoarseGeoSelfAttn(dim, heads, 16, 32,
                                     geo=getattr(cfg, "coarse_sa_geo", True)) for _ in range(nb)])
        if self.mode == "hybrid":                              # U-Net8 fine skips
            self.enc = UNet8Encoder(getattr(cfg, "in_ch", 5), ngf)
            self.se3 = nn.Conv2d(ngf * 4, dim, 1); self.se2 = nn.Conv2d(ngf * 2, dim, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.refine32 = Refine(dim); self.refine64 = Refine(dim)
        self.lsa32 = LocalSphericalAttention(dim, heads, 32, 64, 5)
        self.lsa64 = LocalSphericalAttention(dim, heads, 64, 128, 3)
        self.coarse_head = nn.Conv2d(dim, 1, 1)
        self.proj_fd = conv_bn(dim, ngf); self.dec1 = Refine(ngf)
        self.dec2 = nn.Sequential(conv_bn(ngf, ngf), Refine(ngf)); self.head_fd = nn.Conv2d(ngf, 1, 3, 1, 1)

    def _cross(self, rf, blocks, kv, B, h, w):
        q = self.ray_proj(rf)[None].expand(B, -1, -1)
        for blk in blocks:
            q = blk(q, kv)
        return q.transpose(1, 2).reshape(B, -1, h, w)

    def forward(self, spec, *a, **k):
        B = spec.size(0)
        tok, feats = self.vit(spec)
        kv = self.vit_proj(tok)                                # (B,512,dim)
        if self.mode == "multiscale" and feats:
            ms = [p(f) for p, f in zip(self.ms_proj, feats)]
            kv32 = torch.cat([kv, ms[0], ms[1]], 1); kv64 = torch.cat([kv, ms[2], ms[3]], 1)
        else:
            kv32 = kv64 = kv
        F16 = self._cross(self.rf16, self.cr16, kv, B, 16, 32)
        F32 = self._cross(self.rf32, self.cr32, kv32, B, 32, 64)
        F64 = self._cross(self.rf64, self.cr64, kv64, B, 64, 128)
        e2 = e3 = None
        if self.mode == "hybrid":
            e1 = self.enc.e1(spec); e2 = self.enc.e2(e1); e3 = self.enc.e3(e2)
        m16 = F16
        if self.coarse_sa:
            for blk in self.csa:
                m16 = blk(m16)
        d_c = torch.sigmoid(self.coarse_head(m16))
        x = self.up(m16) + F32
        if e3 is not None: x = x + self.se3(e3)
        x = self.lsa32(self.refine32(x))
        x = self.up(x) + F64
        if e2 is not None: x = x + self.se2(e2)
        x = self.lsa64(self.refine64(x))
        xf = self.up(self.proj_fd(x)); xf = self.dec1(xf)
        xf = self.dec2(self.up(xf))
        D = torch.sigmoid(self.head_fd(xf))
        return {"D": D, "D0": D, "extras": {"D_coarse": d_c}}
