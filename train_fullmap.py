"""Train the A0-style full-map decoder + optional audio correction (A9-A12).

  A9   python train_fullmap.py --run-name A9_fullmap  --correction none
  A10  python train_fullmap.py --run-name A10_cross   --correction cross
  A11  python train_fullmap.py --run-name A11_shaux    --correction sh
  A12  python train_fullmap.py --run-name A12_film     --correction film

Whole-map prediction + masked MAE (matches the A0 det baseline recipe). The
correction branches add a ZERO-initialised coarse term on top of D0.
"""

import os
import sys
import json
import math
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from config import get_cfg
from data import make_loader, apply_audio_mode, shuffle_audio_batch, swap_audio_lr, chan_stats_raw
from ray_features import RayBank
from model_fullmap import FullMapNet
from metrics import MetricBank, cos_lat, gaussian_blur_erp
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "test_for_audio_clip"))
from sh import SHGrid  # noqa: E402

N_VAL = None   # None => evaluate the FULL val split each epoch (was 1500-subset)


def prep_audio(spec, cfg, norm=None):
    if spec.shape[1] > cfg.in_ch:            # channel ablation: keep first in_ch
        spec = spec[:, :cfg.in_ch]
    zc = getattr(cfg, "zero_chan", -1)       # per-channel ablation: zero out one input channel
    if 0 <= zc < spec.shape[1]:
        spec = spec.clone(); spec[:, zc] = 0.0
    spec = apply_audio_mode(spec, cfg.audio_mode)
    if cfg.shuffle_audio:
        spec = shuffle_audio_batch(spec)
    if norm is not None:                     # per-channel train-set normalisation
        spec = (spec - norm[0]) / norm[1]
    return spec


def masked_mae(D, gt, mask):
    return ((D - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)


def dense_loss(D, gt, mask, cfg):
    """Main term with alternative WEIGHTING schemes:
      w_depth_gamma: per-pixel weight gt**gamma (gamma>0 upweights FAR->RMSE; <0 NEAR->AbsRel).
      berhu: reverse-Huber (L1 small err, L2 large err) - error-magnitude weighting."""
    e = (D - gt).abs()
    if getattr(cfg, "berhu", False):
        c = 0.2 * (e * mask).max().clamp(min=1e-6)
        e = torch.where(e <= c, e, (e * e + c * c) / (2 * c))
    g = getattr(cfg, "w_depth_gamma", 0.0)
    if abs(g) > 1e-9:
        w = (gt.clamp(min=1e-3) ** g) * mask
        return (e * w).sum() / w.sum().clamp(min=1e-6)
    return (e * mask).sum() / mask.sum().clamp(min=1e-6)


def tv(x):
    return (x[..., :, 1:] - x[..., :, :-1]).abs().mean() + (x[..., 1:, :] - x[..., :-1, :]).abs().mean()


def erp_dirs(H, W, device):
    """ERP unit ray directions (3,H,W): el top->bottom, az 0..2pi (matches geometry)."""
    i = (torch.arange(H, device=device).float() + 0.5) / H
    j = (torch.arange(W, device=device).float() + 0.5) / W
    el = (math.pi / 2 - i * math.pi).view(H, 1)
    az = (j * 2 * math.pi).view(1, W)
    x = torch.cos(el) * torch.cos(az)
    y = torch.cos(el) * torch.sin(az)
    z = torch.sin(el) * torch.ones_like(az)
    return torch.stack([x, y, z], 0)                       # (3,H,W)


def normal_loss(D, gt, mask, dirs):
    """type-3: 3D surface-normal cosine loss on the ERP point cloud (edge-aware)."""
    def normals(P):                                        # P (B,3,H,W)
        du = P[:, :, 1:, :-1] - P[:, :, :-1, :-1]          # elevation grad
        dv = P[:, :, :-1, 1:] - P[:, :, :-1, :-1]          # azimuth grad
        return F.normalize(torch.cross(dv, du, dim=1), dim=1, eps=1e-6)
    npred = normals(D * dirs[None]); ngt = normals(gt * dirs[None])
    m = mask[:, :, :-1, :-1]
    cos = (npred * ngt).sum(1, keepdim=True)
    return ((1 - cos) * m).sum() / m.sum().clamp(min=1e-6)


def chamfer_loss(D, gt, mask, dirs, k=1024):
    """type-2: symmetric Chamfer distance between subsampled 3D point clouds."""
    B, _, H, W = D.shape
    df = dirs.view(3, -1)                                  # (3,HW)
    Pp = (D.view(B, 1, -1) * df).permute(0, 2, 1)          # (B,HW,3)
    Pg = (gt.view(B, 1, -1) * df).permute(0, 2, 1)
    idx = torch.multinomial(mask.view(B, -1).clamp(min=1e-6), k, replacement=True)   # (B,k)
    bg = torch.arange(B, device=D.device)[:, None]
    Sp, Sg = Pp[bg, idx], Pg[bg, idx]                      # (B,k,3)
    d = torch.cdist(Sp, Sg)                                # (B,k,k)
    return 0.5 * (d.min(2).values.mean() + d.min(1).values.mean())


def warm_start(model, run, freeze, device):
    """Load decoder weights (bb/to_z/fc/up/head) from a trained run; optional freeze."""
    ck = torch.load(os.path.join("out", run, "best.pth"), map_location="cpu", weights_only=False)
    msd = model.state_dict()
    keep = {k: v for k, v in ck["state_dict"].items() if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    print(f"[warm] loaded {len(keep)} tensors from {run}", flush=True)
    if freeze:
        dec = ("bb.", "to_z.", "fc.", "up.", "head.")
        nf = 0
        for n, p in model.named_parameters():
            if n.startswith(dec):
                p.requires_grad_(False); nf += 1
        print(f"[warm] froze {nf} decoder params", flush=True)


def chan_stats(cfg, device):
    """Per-channel mean/std over a RAW train sample (for chan_norm)."""
    return chan_stats_raw(cfg, device)


@torch.no_grad()
def quick_val(model, loader, cfg, device, extra, wlat, norm=None):
    model.eval(); tot = 0.0; wn = 0.0; seen = 0
    wave_arch = getattr(cfg, "arch", "fullmap") in ("wave", "echo_unet", "echo_ray", "echo_bin")
    for b in loader:
        spec = prep_audio(b["spec"].to(device), cfg, norm)
        gt = b["depth"].to(device); mask = b["mask"].to(device)
        wave = b["wave"].to(device) if "wave" in b else None
        D = (model(spec, wave) if wave_arch else
             model(spec, extra.get("coarse_feat"), extra.get("sh_basis")))["D"] * cfg.max_depth
        w = wlat * mask
        tot += ((D - gt * cfg.max_depth).abs() * w).sum().item(); wn += w.sum().item()
        seen += spec.size(0)
        if N_VAL and seen >= N_VAL:
            break
    return tot / max(wn, 1e-6)


def main():
    cfg = get_cfg()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(cfg.out_dir, cfg.run_name); os.makedirs(run_dir, exist_ok=True)

    # correction-branch precompute (also build the ray bank for unet_raymod modulation)
    extra = {}
    if cfg.correction in ("cross", "cross_sup") or getattr(cfg, "arch", "fullmap") in ("unet_raymod", "rayconv"):
        ccfg = copy.copy(cfg); ccfg.img_h, ccfg.img_w = cfg.coarse_h, cfg.coarse_w
        cbank = RayBank(ccfg, device=device)
        cfg.coarse_feat_dim = cbank.feat_dim
        extra["coarse_feat"] = cbank.feat
    elif cfg.correction == "sh":
        shg = SHGrid(cfg.img_h, cfg.img_w, order=cfg.corr_sh_order)
        extra["sh_basis"] = torch.from_numpy(shg.B).to(device)          # (N,Kc)
        extra["sh_pinv"] = torch.from_numpy(shg.B_pinv).to(device)       # (Kc,N)
    norm = chan_stats(cfg, device) if cfg.chan_norm else None
    print(f"[cfg] correction={cfg.correction} {vars(cfg)}", flush=True)

    if getattr(cfg, "arch", "fullmap") == "unet":
        from model_unet import UNet
        model = UNet(cfg).to(device)
    elif getattr(cfg, "arch", "fullmap") == "unet_raymod":
        from model_unet_raymod import UNetRayMod
        model = UNetRayMod(cfg).to(device)
    elif getattr(cfg, "arch", "fullmap") == "vit":
        from model_vit import ViTDepth
        model = ViTDepth(cfg).to(device)
    elif getattr(cfg, "arch", "fullmap") == "rayconv":
        from model_rayconv import RayConvNet
        model = RayConvNet(cfg).to(device)
    elif getattr(cfg, "arch", "fullmap") == "cross_align":
        from model_cross_align import CrossAlign
        model = CrossAlign(cfg).to(device)
    elif getattr(cfg, "arch", "fullmap") == "wave":
        from model_wave import WaveUNet
        model = WaveUNet(cfg).to(device)
    elif getattr(cfg, "arch", "fullmap") == "raydpt":
        if getattr(cfg, "raydpt_resampler", False):
            from model_raydpt import RayDPTResampler
            model = RayDPTResampler(cfg).to(device)
        else:
            from model_raydpt import RayDPT
            model = RayDPT(cfg).to(device)
    elif getattr(cfg, "arch", "fullmap") in ("echo_unet", "echo_ray", "echo_bin"):
        from model_echo import EchoUNet, EchoRay, EchoBin
        model = {"echo_unet": EchoUNet, "echo_ray": EchoRay, "echo_bin": EchoBin}[cfg.arch](cfg).to(device)
    elif getattr(cfg, "arch", "fullmap") in ("unet_coarse", "unet_sh", "unet_raycoarse", "unet_coarse_res"):
        from model_unet_coarse import UNetCoarse, UNetSH, UNetRayCoarse, UNetCoarseResidual
        model = {"unet_coarse": UNetCoarse, "unet_sh": UNetSH,
                 "unet_raycoarse": UNetRayCoarse, "unet_coarse_res": UNetCoarseResidual}[cfg.arch](cfg).to(device)
    else:
        model = FullMapNet(cfg).to(device)
    COARSE_ARCH = getattr(cfg, "arch", "fullmap") in ("unet_coarse", "unet_sh", "unet_raycoarse", "unet_coarse_res", "raydpt", "echo_ray")
    WAVE_ARCH = getattr(cfg, "arch", "fullmap") in ("wave", "echo_unet", "echo_ray", "echo_bin")
    if cfg.init_decoder:
        warm_start(model, cfg.init_decoder, cfg.freeze_decoder, device)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    tr = make_loader(cfg, "train", shuffle=True)
    va = make_loader(cfg, "val", shuffle=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(tr); warm = max(1, len(tr))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wlat = cos_lat(cfg.img_h, device).view(1, 1, cfg.img_h, 1)
    dirs3d = erp_dirs(cfg.img_h, cfg.img_w, device) if (cfg.w_normal > 0 or cfg.w_chamfer > 0) else None

    best = 1e9; hist = []
    ema = ({k: v.detach().clone() for k, v in model.state_dict().items()}   # E14: weight EMA
           if cfg.w_ema > 0 else None)
    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); run = {}
        for b in tr:
            spec = prep_audio(b["spec"].to(device, non_blocking=True), cfg, norm)
            gt = b["depth"].to(device); mask = b["mask"].to(device)
            wave = b["wave"].to(device, non_blocking=True) if "wave" in b else None
            if getattr(cfg, "flip_aug", False):              # correct L/R mirror aug (no spec-time-flip)
                fm = torch.rand(spec.size(0), device=device) < 0.5
                if fm.any():
                    spec = spec.clone(); gt = gt.clone(); mask = mask.clone()
                    spec[fm] = swap_audio_lr(spec[fm])
                    gt[fm] = torch.flip(gt[fm], dims=[-1])
                    mask[fm] = torch.flip(mask[fm], dims=[-1])
                    if wave is not None:
                        wave = wave.clone(); wave[fm] = wave[fm][:, [1, 0]]   # swap ears
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=getattr(cfg, "amp", False)):
                out = model(spec, wave) if WAVE_ARCH else model(spec, extra.get("coarse_feat"), extra.get("sh_basis"))
            main = dense_loss(out["D"], gt, mask, cfg)
            logs = {"mae": float(main.detach())}
            if COARSE_ARCH:
                # band-limited objective: dense + coarse-layout + low-pass (+ residual TV)
                loss = cfg.w_dense * main
                gt_c = F.adaptive_avg_pool2d(gt, (cfg.coarse_head_h, cfg.coarse_head_w))
                m_c = F.adaptive_avg_pool2d(mask, (cfg.coarse_head_h, cfg.coarse_head_w))
                if "D_coarse" in out["extras"] and out["extras"]["D_coarse"].shape[-2:] == gt_c.shape[-2:]:
                    lc = masked_mae(out["extras"]["D_coarse"], gt_c, m_c)
                else:
                    lc = masked_mae(F.adaptive_avg_pool2d(out["D"], (cfg.coarse_head_h, cfg.coarse_head_w)), gt_c, m_c)
                ll = masked_mae(gaussian_blur_erp(out["D"], 3.0), gaussian_blur_erp(gt, 3.0), mask)
                loss = loss + cfg.w_coarse_layout * lc + cfg.w_low * ll
                logs["lc"] = float(lc.detach()); logs["llow"] = float(ll.detach())
                if "residual" in out["extras"]:
                    tvr = tv(out["extras"]["residual"]); loss = loss + cfg.w_tv_res * tvr
                    logs["tvr"] = float(tvr.detach())
            else:
                loss = main
            if cfg.correction == "sh":
                gt_coef = (gt.view(gt.size(0), -1) @ extra["sh_pinv"].T)     # (B,Kc)
                aux = (out["extras"]["coef"] - gt_coef).abs().mean()
                loss = loss + cfg.w_sh_aux * aux; logs["shaux"] = float(aux.detach())
            if cfg.correction == "cross_sup":
                dc = out["extras"]["Dcorr"]                                  # (B,1,H,W)
                rtgt = gaussian_blur_erp((gt - out["D0"].detach()), 3.0).clamp(-cfg.res_scale, cfg.res_scale)
                rsup = masked_mae(dc, rtgt, mask); tvl = tv(dc)
                loss = loss + cfg.w_res_sup * rsup + cfg.w_tv * tvl
                logs["rsup"] = float(rsup.detach()); logs["tv"] = float(tvl.detach())
            if cfg.w_normal > 0:                                            # type-3: 3D surface-normal cosine
                ln = normal_loss(out["D"], gt, mask, dirs3d)
                loss = loss + cfg.w_normal * ln; logs["lnorm"] = float(ln.detach())
            if cfg.w_chamfer > 0:                                           # type-2: 3D Chamfer
                lch = chamfer_loss(out["D"], gt, mask, dirs3d, cfg.chamfer_k)
                loss = loss + cfg.w_chamfer * lch; logs["lch"] = float(lch.detach())
            if cfg.w_rel > 0:                                               # E2 recipe: relative-depth (AbsRel) loss
                lrel = (((out["D"] - gt).abs() / gt.clamp(min=1e-3)) * mask).sum() / mask.sum().clamp(min=1e-6)
                loss = loss + cfg.w_rel * lrel; logs["lrel"] = float(lrel.detach())
            if cfg.w_scale > 0:                                             # per-scene scale guide: pred mean -> gt mean
                msum = mask.sum(dim=(1, 2, 3)).clamp(min=1e-6)
                pm = (out["D"] * mask).sum(dim=(1, 2, 3)) / msum
                gm = (gt * mask).sum(dim=(1, 2, 3)) / msum
                lscl = (pm - gm).abs().mean()
                loss = loss + cfg.w_scale * lscl; logs["lscl"] = float(lscl.detach())
            if cfg.w_silog > 0:                                             # E4: scale-invariant log loss
                dlog = (torch.log(out["D"].clamp(min=1e-3)) - torch.log(gt.clamp(min=1e-3))) * mask
                ms = mask.sum().clamp(min=1e-6)
                m1 = dlog.sum() / ms; m2 = (dlog * dlog).sum() / ms
                lsi = m2 - 0.85 * m1 * m1
                loss = loss + cfg.w_silog * lsi; logs["lsi"] = float(lsi.detach())
            if cfg.w_grad > 0:                                              # E34: edge-aware gradient matching
                D = out["D"]
                dxp = D[..., :, 1:] - D[..., :, :-1]; dxg = gt[..., :, 1:] - gt[..., :, :-1]
                mx = mask[..., :, 1:] * mask[..., :, :-1]
                dyp = D[..., 1:, :] - D[..., :-1, :]; dyg = gt[..., 1:, :] - gt[..., :-1, :]
                my = mask[..., 1:, :] * mask[..., :-1, :]
                lgr = (((dxp - dxg).abs() * mx).sum() / mx.sum().clamp(min=1e-6)
                       + ((dyp - dyg).abs() * my).sum() / my.sum().clamp(min=1e-6))
                loss = loss + cfg.w_grad * lgr; logs["lgr"] = float(lgr.detach())
            if cfg.w_swap_eq > 0:                                            # tip8: swap-equivariance
                out_sw = model(swap_audio_lr(spec), extra.get("coarse_feat"), extra.get("sh_basis"))
                eq = masked_mae(out_sw["D"], torch.flip(out["D"].detach(), dims=[-1]), mask)
                loss = loss + cfg.w_swap_eq * eq; logs["eq"] = float(eq.detach())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if ema is not None:                                            # E14: track weight EMA
                d = cfg.w_ema; msd = model.state_dict()
                for k in ema:
                    if ema[k].dtype.is_floating_point:
                        ema[k].mul_(d).add_(msd[k].detach(), alpha=1 - d)
                    else:
                        ema[k].copy_(msd[k])
            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + v
        run = {k: v / len(tr) for k, v in run.items()}
        raw = None
        if ema is not None:                                                # eval/checkpoint the EMA weights
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema, strict=True)
        vmae = quick_val(model, va, cfg, device, extra, wlat, norm)
        a = out["extras"].get("alpha")
        hist.append({"epoch": ep, "val_mae_m": vmae, "alpha": a, **run})
        print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s  {run}  val_MAE={vmae:.4f}m"
              + (f"  alpha={a:+.3f}" if a is not None else ""), flush=True)
        if vmae < best:
            best = vmae
            torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg),
                        "norm": (norm[0].cpu(), norm[1].cpu()) if norm is not None else None},
                       os.path.join(run_dir, "best.pth"))
        if raw is not None:                                                # restore true iterate for next epoch
            model.load_state_dict(raw, strict=True)
    json.dump({"best_val_mae_m": best, "hist": hist, "cfg": vars(cfg)},
              open(os.path.join(run_dir, "train_done.json"), "w"), indent=2)
    print(f"[done] best val MAE = {best:.4f} m -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
