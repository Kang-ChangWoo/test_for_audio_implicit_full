"""Wrappers for the published-baseline pretrained backbones (channel-adapted):
baseline/models/pretrain/{pretrained_vit,pretrained_resnet}.py. A 1x1 conv adapts
in_ch -> 3ch pseudo-RGB, then a pretrained ViT-B/16 / ResNet-50 encoder + decoder.
These are the "change only the input channels" deploy baselines for the paper."""
import importlib.util
import torch
import torch.nn as nn
from types import SimpleNamespace

_BASE = "/root/storage/implementation/shared_audio/baseline/models/pretrain"


def _load(name):
    """Load a baseline module by file path, bypassing the package __init__ (which
    pulls in unet_foa -> data.sh_basis and clashes with our local data.py)."""
    spec = importlib.util.spec_from_file_location(f"_bl_{name}", f"{_BASE}/{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


PretrainedViT = _load("pretrained_vit").PretrainedViT
PretrainedResNet = _load("pretrained_resnet").PretrainedResNet

# comparison methods. BatVision = self-contained -> file-path load.
_spec = importlib.util.spec_from_file_location(
    "_bl_batvision", "/root/storage/implementation/shared_audio/baseline/comparison_methods/batvision/batvision.py")
_bv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_bv)
BatVisionUNet = _bv.BatVisionUNet

# EchoDiffusion uses relative imports; build a SYNTHETIC package with only its clean deps
# (aspp_asff, diffusion_unet) so we skip the real __init__ (which pulls ambi_sh -> data.sh_basis).
import sys  # noqa: E402
import types  # noqa: E402
_ED = "/root/storage/implementation/shared_audio/baseline/comparison_methods/echodiffusion"
_pkg = types.ModuleType("edpkg"); _pkg.__path__ = [_ED]; sys.modules["edpkg"] = _pkg


def _edsub(name):
    sp = importlib.util.spec_from_file_location(f"edpkg.{name}", f"{_ED}/{name}.py")
    m = importlib.util.module_from_spec(sp); sys.modules[f"edpkg.{name}"] = m
    sp.loader.exec_module(m); return m


_edsub("aspp_asff"); _edsub("diffusion_unet")
_EchoDiffusion = _edsub("echodiffusion").EchoDiffusion


def _shim(cfg):
    """Provide the cfg.dataset.images_size / depth_norm the baseline models expect."""
    return SimpleNamespace(dataset=SimpleNamespace(
        images_size=[cfg.img_h, cfg.img_w], depth_norm=True))


class PViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.m = PretrainedViT(_shim(cfg), input_nc=getattr(cfg, "in_ch", 5),
                               pretrained=getattr(cfg, "vit_pretrained", True),
                               freeze_encoder=getattr(cfg, "vit_freeze", False))

    def forward(self, spec, *a, **k):
        return {"D": self.m(spec), "D0": None, "extras": {}}


class PResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.m = PretrainedResNet(_shim(cfg), input_nc=getattr(cfg, "in_ch", 5),
                                  pretrained=getattr(cfg, "vit_pretrained", True),
                                  freeze_encoder=getattr(cfg, "vit_freeze", False))

    def forward(self, spec, *a, **k):
        return {"D": self.m(spec), "D0": None, "extras": {}}


class BatVis(nn.Module):
    """BatVision comparison method (Christensen et al.) — pix2pix U-Net, depth-normalised."""
    def __init__(self, cfg):
        super().__init__()
        self.m = BatVisionUNet(_shim(cfg), input_nc=getattr(cfg, "in_ch", 5))

    def forward(self, spec, *a, **k):
        return {"D": self.m(spec), "D0": None, "extras": {}}


class EchoDiff(nn.Module):
    """Faithful EchoDiffusion (wjzhang-ai): 2ch spec + raw wave -> CIDE(wav2vec2)+ASPP+
    diffusion-UNet. Returns depth in METERS -> divide by max_depth so the pipeline's
    *max_depth restores it. Takes (spec, wave); uses first 2 spec channels (log-mag L/R)."""
    def __init__(self, cfg):
        super().__init__()
        self.md = float(getattr(cfg, "max_depth", 10.0))
        self.m = _EchoDiffusion(max_depth=self.md)

    def forward(self, spec, wave=None, *a, **k):
        d = self.m(spec[:, :2], wave)                 # meters
        return {"D": d / self.md, "D0": None, "extras": {}}
