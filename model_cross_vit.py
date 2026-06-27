"""Pretrained ViT-B/16 spectrogram encoder, drop-in for AudioEncoder so the
ray-conditioned cross-attention model can use ViT patch tokens as keys/values.

Interface matches model.AudioEncoder.forward(spec, want_tokens) -> (z, tok):
  z   : (B, audio_dim) global latent (from CLS token)
  tok : (B, n_patch, dim) patch tokens for ray cross-attention
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ViT_B_16_Weights

from model_vit import _interp_pos_embed


class ViTTokenEncoder(nn.Module):
    def __init__(self, width=48, audio_dim=256, dim=192, in_ch=2,
                 img_h=256, img_w=512, pretrained=True, freeze=False):
        super().__init__()
        patch, embed = 16, 768
        self.input_adapter = nn.Conv2d(in_ch, 3, 1)              # spec -> pseudo-RGB
        vit = tv_models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        self.patch_embed = vit.conv_proj
        self.cls_token = vit.class_token
        self.encoder = vit.encoder
        gh, gw = img_h // patch, img_w // patch                  # 16 x 32 at 256x512
        self.encoder.pos_embedding = nn.Parameter(
            _interp_pos_embed(vit.encoder.pos_embedding.data, (14, 14), (gh, gw)))
        if freeze:
            for p in self.patch_embed.parameters(): p.requires_grad = False
            for p in self.encoder.parameters(): p.requires_grad = False
            self.cls_token.requires_grad = False
        self.z_proj = nn.Sequential(nn.Linear(embed, audio_dim), nn.GELU())
        self.tok_proj = nn.Linear(embed, dim)
        self.tok_dim = dim

    def forward(self, spec, want_tokens=False):
        x = self.input_adapter(spec)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)       # (B, gh*gw, 768)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = self.encoder(torch.cat([cls, x], dim=1))             # (B, 1+gh*gw, 768)
        z = self.z_proj(x[:, 0])                                 # (B, audio_dim)
        if not want_tokens:
            return z, None
        tok = self.tok_proj(x[:, 1:])                            # (B, gh*gw, dim)
        return z, tok
