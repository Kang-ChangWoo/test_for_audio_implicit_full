"""Pretrained ViT-B/16 audio->ERP-depth, adapted from
baseline/models/pretrain/pretrained_vit.py to THIS repo's flat-cfg + dict interface.

Pipeline: 2ch spec -> Conv1x1 pseudo-RGB -> ViT-B/16 (ImageNet pretrained, pos-embed
interpolated to our 4x8 patch grid) -> progressive ConvTranspose decoder -> sigmoid depth.

Our input is 64x128 (patch16 -> grid 4x8 = 32 tokens), not the baseline's 256x512.
Returns the shared {"D","D0","extras"} interface so train_fullmap/eval_fullmap can use it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import ViT_B_16_Weights


def _interp_pos_embed(pos_embed, old_grid, new_grid):
    cls = pos_embed[:, :1, :]
    patch = pos_embed[:, 1:, :]
    D = patch.shape[-1]; oH, oW = old_grid; nH, nW = new_grid
    patch = patch.reshape(1, oH, oW, D).permute(0, 3, 1, 2)
    patch = F.interpolate(patch, size=(nH, nW), mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, nH * nW, D)
    return torch.cat([cls, patch], dim=1)


class _Decoder(nn.Module):
    """patch-token grid -> dense depth, ConvTranspose x2 per stage."""
    def __init__(self, embed_dim, gh, gw):
        super().__init__()
        self.gh, self.gw = gh, gw
        ch = embed_dim; ups = []
        for out in (256, 128, 64, 32):
            ups.append(nn.Sequential(
                nn.ConvTranspose2d(ch, out, 4, 2, 1), nn.BatchNorm2d(out), nn.ReLU(True)))
            ch = out
        self.up = nn.ModuleList(ups)
        self.head = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(16, 1, 1))

    def forward(self, tokens, th, tw):
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, self.gh, self.gw)
        for layer in self.up:
            x = layer(x)
        x = torch.sigmoid(self.head(x))
        if x.shape[2] != th or x.shape[3] != tw:
            x = F.interpolate(x, (th, tw), mode="bilinear", align_corners=False)
        return x


class ViTDepth(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.th, self.tw = cfg.img_h, cfg.img_w
        in_ch = getattr(cfg, "in_ch", 2)
        pretrained = getattr(cfg, "vit_pretrained", True)
        freeze = getattr(cfg, "vit_freeze", False)
        patch, embed = 16, 768

        self.input_adapter = nn.Conv2d(in_ch, 3, 1)              # spec -> pseudo-RGB
        vit = tv_models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        self.patch_embed = vit.conv_proj
        self.cls_token = vit.class_token
        self.encoder = vit.encoder

        self.gh, self.gw = self.th // patch, self.tw // patch        # 4 x 8
        new_pos = _interp_pos_embed(vit.encoder.pos_embedding.data, (14, 14), (self.gh, self.gw))
        self.encoder.pos_embedding = nn.Parameter(new_pos)

        # ERP-valid positional encoding (vs the interpolated PLANAR ImageNet PE):
        # encode each patch-centre's sphere direction via RayBank (Fourier-xyz / SH),
        # which is azimuth-periodic and pole-aware. Disable the built-in planar PE.
        self.vit_pe = getattr(cfg, "vit_pe", "planar")
        if self.vit_pe != "planar":
            import copy as _copy
            from ray_features import RayBank
            pcfg = _copy.copy(cfg); pcfg.img_h, pcfg.img_w = self.gh, self.gw
            pcfg.use_mic_pe = False
            pcfg.use_xyz = True
            pcfg.use_fourier_pe = self.vit_pe in ("fourier", "both")
            pcfg.use_sh_pe = self.vit_pe in ("sh", "both")
            bank = RayBank(pcfg, device="cpu")
            self.register_buffer("geom_feat", bank.feat)             # (N, F) fixed
            self.pe_proj = nn.Linear(bank.feat_dim, embed)
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed))
            self.encoder.pos_embedding = nn.Parameter(torch.zeros_like(new_pos),
                                                      requires_grad=False)  # built-in PE off

        if freeze:
            for p in self.patch_embed.parameters(): p.requires_grad = False
            for p in self.encoder.parameters(): p.requires_grad = False
            self.cls_token.requires_grad = False

        self.decoder = _Decoder(embed, self.gh, self.gw)

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        x = self.input_adapter(spec)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)          # (B, gh*gw, 768)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        if self.vit_pe != "planar":                                 # add ERP geometric PE
            x = x + self.pe_proj(self.geom_feat).unsqueeze(0)       # (1,N,768) broadcast
            cls = cls + self.cls_pos
        x = self.encoder(torch.cat([cls, x], dim=1))                # built-in PE is planar or zero
        D = self.decoder(x[:, 1:, :], self.th, self.tw)
        return {"D": D, "D0": D, "extras": {}}
