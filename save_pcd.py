"""Export contender predictions as colored point clouds (PLY, MeshLab-native).

Scenes: same deterministic pick as viz_contenders, 2 each from train/val/test.
Models: U-Net8, RayDPT-full, RayDPT-lite, cross-flip (+ GT reference).
ERP radial depth -> xyz on the unit-direction sphere * depth(m); color = turbo(depth).
"""
import os
import numpy as np
import matplotlib.cm as cm

import eval_fullmap as ev_fm
from viz_contenders import pick_scenes, preds_for, best_seed, mae_of, DEV

OUT = "out/pcd"; os.makedirs(OUT, exist_ok=True)

MODELS = [
    ("UNet8",       "Bnode2_unet8_5chflip", "fm"),
    ("RayDPTfull",  "C_raydpt_5chflip",     "fm"),
    ("RayDPTlite",  "C_raydptlite_5chflip", "fm"),
    ("crossflip",   "Bnode2_cross_flip",    "impl"),
]
TURBO = cm.get_cmap("turbo")


def erp_dirs(H, W):
    """ERP unit directions matching the model geometry (el top->bottom, az 0..2pi)."""
    i = (np.arange(H) + 0.5) / H
    j = (np.arange(W) + 0.5) / W
    el = (np.pi / 2 - i * np.pi)[:, None]              # (H,1)
    az = (j * 2 * np.pi)[None, :]                       # (1,W)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el) * np.ones_like(az)
    return np.stack([x, y, z], -1)                     # (H,W,3)


def save_ply(path, depth, dirs, vmax):
    """depth (H,W) metres -> colored point cloud, dropping holes/far-clamp."""
    r = depth.astype(np.float32)
    valid = (r > 0.05) & (r < 9.9)
    pts = (r[..., None] * dirs)[valid]                 # (N,3)
    col = (TURBO(np.clip(r[valid] / max(vmax, 1e-6), 0, 1))[:, :3] * 255).astype(np.uint8)
    N = pts.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (rr, gg, bb) in zip(pts, col):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {rr} {gg} {bb}\n")
    return N


def main():
    base_cfg = ev_fm.load(f"out/{best_seed(MODELS[0][1])}", DEV)[1]
    alls = pick_scenes(base_cfg)
    # 2 each from train/val/test, in the contenders pool order
    scenes = []
    for sp in ("train", "val", "test"):
        scenes += [s for s in alls if s["split"] == sp][:2]
    H, W = base_cfg.img_h, base_cfg.img_w
    dirs = erp_dirs(H, W)
    print(f"[scenes] {[(s['split'], s['key']) for s in scenes]}", flush=True)

    # GT clouds
    for s in scenes:
        tag = f"{s['split']}_{s['key'].replace('/','-')}"
        n = save_ply(f"{OUT}/{tag}__GT.ply", s["gt"], dirs, s["vmax"])
        print(f"[GT]  {tag}  N={n}", flush=True)

    # model clouds
    for lab, base, typ in MODELS:
        run = best_seed(base)
        if run is None:
            print(f"[skip] {base}"); continue
        preds = preds_for(run, typ, scenes)
        for s in scenes:
            tag = f"{s['split']}_{s['key'].replace('/','-')}"
            n = save_ply(f"{OUT}/{tag}__{lab}.ply", preds[s["key"]], dirs, s["vmax"])
            print(f"[{lab}] {tag} ({run}, MAE={mae_of(run):.3f})  N={n}", flush=True)
    print(f"[done] PLY clouds in {OUT}/", flush=True)


if __name__ == "__main__":
    main()
