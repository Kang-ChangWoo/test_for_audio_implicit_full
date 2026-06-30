"""EchoDiffusion "success-type" transfer, faithfully ported (no 5ch, no flip).

The point of these two models is to bring over the load-bearing pieces of
EchoDiffusion that our earlier `wave_unet8` threw away:
  (1) a PRETRAINED audio encoder (wav2vec2, frozen) over the RAW binaural
      waveform -> rich acoustic representation (vs a random 4-layer 1D-CNN),
  (2) the CIDE class-embedding bottleneck: wav2vec2 -> 100-way scene-class
      distribution -> learned scene embedding (a CLIP-token analog),
  (3) CROSS-ATTENTION conditioning (vs broadcast channel-concat): spatial /
      ray tokens attend to the scene tokens, exactly like EchoDiffusion feeds
      its scene embedding into the SD-UNet via `c_crossattn`.

EchoUNet : winning U-Net8 backbone + scene cross-attn injected at the e4 (16x32)
           mid stage, ControlNet/zero-init style (identity at init).
EchoRay  : RayDPT redone with the scene prior fused into EVERY cross-attn scale,
           keeping the multi-scale ray pyramid + local spherical ray<->ray attn.

Both consume (spec, wave) and return the shared {"D","D0","extras"} interface.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CrossBlock, SelfBlock, conv_bn, Refine
from model_unet_coarse import UNet8Encoder
from model_raydpt import LocalSphericalAttention
from ray_features import RayBank


# --------------------------------------------------------------------------- #
# (1)+(2) pretrained wav2vec2 scene encoder + CIDE class-embedding  (frozen w2v)
# --------------------------------------------------------------------------- #
class EchoSceneEncoder(nn.Module):
    """raw binaural wave (B,2,T) -> scene context tokens (B, 1+Tf, dim).

    Token 0 is the CIDE scene-class embedding (global room/scene prior); the
    remaining tokens are per-frame wav2vec2 features (fine acoustic context).
    """

    def __init__(self, dim, n_class=100):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.w2v.eval()
        for p in self.w2v.parameters():
            p.requires_grad_(False)
        wd = self.w2v.config.hidden_size                       # 768
        self.to_mono = nn.Conv1d(2, 1, 1)                      # binaural -> mono (learned mix)
        self.frame_proj = nn.Linear(wd, dim)                   # per-frame context tokens
        # CIDE: pooled w2v -> 100-class soft distribution -> learned scene embedding
        self.fc = nn.Sequential(nn.Linear(wd, 400), nn.GELU(), nn.Linear(400, n_class))
        self.embeddings = nn.Parameter(torch.randn(n_class, dim))
        self.adapter = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-4)

    def train(self, mode=True):                                # keep frozen w2v in eval (no dropout/BN drift)
        super().train(mode)
        self.w2v.eval()
        return self

    def forward(self, wave):
        x = self.to_mono(wave).squeeze(1)                      # (B,T)
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
        with torch.no_grad():
            feat = self.w2v(x).last_hidden_state               # (B,Tf,768)
        frames = self.frame_proj(feat)                         # (B,Tf,dim)
        pooled = feat.mean(1)                                  # (B,768)
        probs = self.fc(pooled).softmax(-1)                    # (B,n_class)
        cls = probs @ self.embeddings                          # (B,dim)
        cls = cls + self.gamma * self.adapter(cls)             # CIDE adapter (residual)
        return torch.cat([cls[:, None, :], frames], dim=1)     # (B,1+Tf,dim)


# --------------------------------------------------------------------------- #
# (3) scene cross-attention injected into a spatial feature map (ControlNet-ish)
# --------------------------------------------------------------------------- #
class SceneCrossAttn(nn.Module):
    def __init__(self, ch, dim, heads, n_layers=2, self_attn=True):
        super().__init__()
        self.proj_in = nn.Conv2d(ch, dim, 1)
        self.cross = nn.ModuleList([CrossBlock(dim, heads) for _ in range(n_layers)])
        self.selfb = SelfBlock(dim, heads) if self_attn else None
        self.proj_out = nn.Conv2d(dim, ch, 1)
        nn.init.zeros_(self.proj_out.weight); nn.init.zeros_(self.proj_out.bias)   # identity at init

    def forward(self, x, scene):
        B, C, H, W = x.shape
        t = self.proj_in(x).flatten(2).transpose(1, 2)         # (B,HW,dim)
        for blk in self.cross:
            t = blk(t, scene)
        if self.selfb is not None:
            t = self.selfb(t)
        t = t.transpose(1, 2).reshape(B, -1, H, W)
        return x + self.proj_out(t)                            # zero-init residual


def _up(ci, co, outer=False):
    if outer:
        return nn.Sequential(nn.ConvTranspose2d(ci, 1, 4, 2, 1), nn.Sigmoid())
    return nn.Sequential(nn.ConvTranspose2d(ci, co, 4, 2, 1, bias=False),
                         nn.BatchNorm2d(co), nn.ReLU(True))


# --------------------------------------------------------------------------- #
# EchoUNet: U-Net8 backbone + scene cross-attn at the e4 (16x32) mid stage
# --------------------------------------------------------------------------- #
class EchoUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ngf = getattr(cfg, "ngf", 64); dim = cfg.dim; heads = cfg.n_heads
        self.enc = UNet8Encoder(getattr(cfg, "in_ch", 2), ngf)
        self.scene = EchoSceneEncoder(dim)
        self.cond4 = SceneCrossAttn(ngf * 8, dim, heads,
                                    n_layers=getattr(cfg, "ray_cross_layers", 2))
        # pix2pix-style decoder with skips (e7..e1)
        self.u8 = _up(ngf * 8, ngf * 8)        # 1x2  -> 2x4
        self.u7 = _up(ngf * 16, ngf * 8)       # 2x4  -> 4x8   (cat e7)
        self.u6 = _up(ngf * 16, ngf * 8)       # 4x8  -> 8x16  (cat e6)
        self.u5 = _up(ngf * 16, ngf * 8)       # 8x16 -> 16x32 (cat e5)
        self.u4 = _up(ngf * 16, ngf * 4)       # 16x32-> 32x64 (cat e4)
        self.u3 = _up(ngf * 8, ngf * 2)        # 32x64-> 64x128(cat e3)
        self.u2 = _up(ngf * 4, ngf)            # 64x128->128x256(cat e2)
        self.u1 = _up(ngf * 2, 1, outer=True)  # 128x256->256x512(cat e1)

    def forward(self, spec, wave=None, coarse_feat=None, sh_basis=None):
        assert wave is not None, "EchoUNet requires raw waveform (audio_src=wave)"
        e1 = self.enc.e1(spec); e2 = self.enc.e2(e1); e3 = self.enc.e3(e2); e4 = self.enc.e4(e3)
        e4 = self.cond4(e4, self.scene(wave))                  # inject scene prior at mid stage
        e5 = self.enc.e5(e4); e6 = self.enc.e6(e5); e7 = self.enc.e7(e6); e8 = self.enc.e8(e7)
        d = self.u8(e8)
        d = self.u7(torch.cat([d, e7], 1))
        d = self.u6(torch.cat([d, e6], 1))
        d = self.u5(torch.cat([d, e5], 1))
        d = self.u4(torch.cat([d, e4], 1))
        d = self.u3(torch.cat([d, e3], 1))
        d = self.u2(torch.cat([d, e2], 1))
        D = self.u1(torch.cat([d, e1], 1))
        return {"D": D, "D0": D, "extras": {}}


# --------------------------------------------------------------------------- #
# EchoRay: RayDPT multi-scale ray pyramid, with the scene prior fused into
# every cross-attn scale + local spherical ray<->ray attention.
# --------------------------------------------------------------------------- #
class EchoRay(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        ngf = getattr(cfg, "ngf", 64); dim = cfg.dim; heads = cfg.n_heads
        nL = getattr(cfg, "ray_cross_layers", 2)
        self.enc = UNet8Encoder(getattr(cfg, "in_ch", 2), ngf)
        self.scene = EchoSceneEncoder(dim)

        def bank(h, w):
            pc = copy.copy(cfg); pc.img_h, pc.img_w = h, w
            b = RayBank(pc, device="cpu"); return b.feat, b.feat_dim
        f16, fd = bank(16, 32); f32, _ = bank(32, 64); f64, _ = bank(64, 128)
        self.register_buffer("rf16", f16); self.register_buffer("rf32", f32); self.register_buffer("rf64", f64)
        self.ray_proj = nn.Sequential(nn.Linear(fd, dim), nn.GELU(), nn.Linear(dim, dim))
        self.kv_e4 = nn.Linear(ngf * 8, dim)
        self.kv_e3 = nn.Linear(ngf * 4, dim)
        mk = lambda: nn.ModuleList([CrossBlock(dim, heads) for _ in range(nL)])
        self.cr16, self.cr32, self.cr64 = mk(), mk(), mk()
        self.se4 = nn.Conv2d(ngf * 8, dim, 1)
        self.se3 = nn.Conv2d(ngf * 4, dim, 1)
        self.se2 = nn.Conv2d(ngf * 2, dim, 1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.refine32 = Refine(dim); self.refine64 = Refine(dim)
        self.lsa32 = LocalSphericalAttention(dim, heads, 32, 64, getattr(cfg, "raydpt_win32", 5))
        self.lsa64 = LocalSphericalAttention(dim, heads, 64, 128, getattr(cfg, "raydpt_win64", 3))
        self.coarse_head = nn.Conv2d(dim, 1, 1)
        # learned full-decode 64x128 -> 256x512 (+ e1 skip), like RayDPT full_decode
        self.proj_fd = conv_bn(dim, ngf)
        self.se1 = nn.Conv2d(ngf, ngf, 1)
        self.dec1 = Refine(ngf)
        self.dec2 = nn.Sequential(conv_bn(ngf, ngf), Refine(ngf))
        self.head_fd = nn.Conv2d(ngf, 1, 3, 1, 1)

    def _cross(self, rf, blocks, kv, B, h, w):
        q = self.ray_proj(rf)[None].expand(B, -1, -1)
        for blk in blocks:
            q = blk(q, kv)
        return q.transpose(1, 2).reshape(B, -1, h, w)

    def forward(self, spec, wave=None, coarse_feat=None, sh_basis=None):
        assert wave is not None, "EchoRay requires raw waveform (audio_src=wave)"
        B = spec.size(0)
        e1 = self.enc.e1(spec); e2 = self.enc.e2(e1); e3 = self.enc.e3(e2); e4 = self.enc.e4(e3)
        scene = self.scene(wave)                                       # (B,S,dim) global acoustic prior
        # fuse scene tokens into EVERY scale's K/V (spatial audio tokens + scene)
        kv4 = torch.cat([self.kv_e4(e4.flatten(2).transpose(1, 2)), scene], dim=1)
        kv3 = torch.cat([self.kv_e3(e3.flatten(2).transpose(1, 2)), scene], dim=1)
        F16 = self._cross(self.rf16, self.cr16, kv4, B, 16, 32)
        F32 = self._cross(self.rf32, self.cr32, kv3, B, 32, 64)
        F64 = self._cross(self.rf64, self.cr64, kv4, B, 64, 128)
        m16 = F16 + self.se4(e4)                                       # 16x32
        d_c = torch.sigmoid(self.coarse_head(m16))                     # coarse layout (B,1,16,32)
        x = self.lsa32(self.refine32(self.up(m16) + F32 + self.se3(e3)))   # 32x64
        x = self.lsa64(self.refine64(self.up(x) + F64 + self.se2(e2)))     # 64x128
        xf = self.up(self.proj_fd(x))                                  # 128x256
        xf = self.dec1(xf + self.se1(e1))
        xf = self.dec2(self.up(xf))                                    # 256x512
        D = torch.sigmoid(self.head_fd(xf))
        return {"D": D, "D0": D, "extras": {"D_coarse": d_c}}
