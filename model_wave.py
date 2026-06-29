"""WaveUNet: proven 5ch-spectrogram U-Net + raw-waveform 1D-CNN conditioning.

EchoDiffusion (wjzhang) feeds the RAW waveform through a learned encoder (wav2vec2)
to a GLOBAL scene embedding injected via cross-attention. We adapt that idea to our
best spatial backbone: the binaural waveform is encoded by a trainable 1D-CNN to a
compact global vector, broadcast and concatenated to the U-Net input (FiLM-at-input).
The spectrogram U-Net (skip-connection, the winning arch) carries the spatial layout;
the waveform branch supplies the fine-timing / room-acoustic global prior that the
log-mag spectrogram (+ nearest resize) discards.

Returns the shared {"D","D0","extras"} interface.
"""
import copy
import torch
import torch.nn as nn

from model_unet import UNet


class WaveEncoder(nn.Module):
    """Raw binaural waveform (B,2,T) -> global embedding (B,out_ch)."""
    def __init__(self, out_ch=8, ngf=32):
        super().__init__()
        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.BatchNorm1d(o), nn.GELU())
        self.net = nn.Sequential(
            blk(2, ngf, 15, 4),
            blk(ngf, ngf * 2, 11, 4),
            blk(ngf * 2, ngf * 4, 9, 4),
            blk(ngf * 4, ngf * 4, 7, 4),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(ngf * 4, ngf * 4), nn.GELU(), nn.Linear(ngf * 4, out_ch))

    def forward(self, wave):
        return self.head(self.net(wave))                          # (B, out_ch)


class WaveUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cg = getattr(cfg, "wave_ch", 8)
        self.wave_enc = WaveEncoder(out_ch=self.cg, ngf=getattr(cfg, "wave_ngf", 32))
        icfg = copy.copy(cfg)
        icfg.in_ch = getattr(cfg, "in_ch", 5) + self.cg          # spec channels + wave embedding
        self.unet = UNet(icfg)

    def forward(self, spec, wave=None, coarse_feat=None, sh_basis=None):
        assert wave is not None, "WaveUNet requires raw waveform (audio_src=wave)"
        g = self.wave_enc(wave)                                   # (B, cg)
        gmap = g[:, :, None, None].expand(-1, -1, spec.size(2), spec.size(3))
        return self.unet(torch.cat([spec, gmap], 1))
