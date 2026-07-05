"""Selected-model fig grid + PCD export to a SEPARATE folder (out/pcd_sel).
Models: RayDPT+E2, cross_flip, GCC U-Net8, U-Net8 5ch+flip, U-Net8 w20, crossself_flip.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import eval_fullmap as ev_fm
import viz_contenders as vc
from viz_contenders import preds_for, best_seed, mae_of, DEV
from save_pcd import erp_dirs, save_ply

vc.WANT = {"train": 2, "val": 2, "test": 2}          # 6 scenes (same as pcd set)
FIG = "out/figs"; PCD = "out/pcd_sel"
os.makedirs(FIG, exist_ok=True); os.makedirs(PCD, exist_ok=True)

MODELS = [
    ("RayDPT+E2",      "R_raydpt_e2",          "fm"),
    ("cross_flip",     "Bnode2_cross_flip",    "impl"),
    ("GCC U-Net8",     "Bnode2_gcc_unet8",     "fm"),
    ("U-Net8 5ch+flip", "Bnode2_unet8_5chflip", "fm"),
    ("U-Net8 w20",     "Bnode2_unet8_5chflip_w20", "fm"),
]


def main():
    cfg = ev_fm.load(f"out/{best_seed('Bnode2_unet8_5chflip')}", DEV)[1]
    scenes = vc.pick_scenes(cfg)
    resolved = []
    for lab, base, typ in MODELS:
        r = best_seed(base)
        if r is None:
            print(f"[skip] {base}"); continue
        resolved.append((lab, r, typ))
    cache = {}
    for lab, run, typ in resolved:
        print(f"[infer] {run} ({typ})", flush=True)
        cache[run] = preds_for(run, typ, scenes)

    # --- figure: GT + 6 models, rows = scenes ---
    cols = [("GT depth", None, None)] + [(f"{l}\n{mae_of(r):.3f}", r, t) for l, r, t in resolved]
    ncol, nrow = len(cols), len(scenes)
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 1.9 * nrow))
    for i, s in enumerate(scenes):
        for j, (lab, run, typ) in enumerate(cols):
            a = ax[i, j]
            img = s["gt"] if run is None else cache[run][s["key"]]
            im = a.imshow(img, cmap="turbo", vmin=0, vmax=s["vmax"])
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(lab, fontsize=9)
            if j == 0:
                a.set_ylabel(f"{s['split']}\n{s['key']}", fontsize=7)
        plt.colorbar(im, ax=ax[i, -1], fraction=0.046, pad=0.04)
    fig.suptitle("selected models: GT + RayDPT+E2 / cross_flip / GCC / U-Net8(5ch+flip,w20) / crossself_flip. "
                 "title = test MAE[m].", y=1.002, fontsize=11)
    fig.tight_layout()
    out = f"{FIG}/fig_sel.png"; fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}", flush=True)

    # --- PCD to separate folder ---
    H, W = cfg.img_h, cfg.img_w
    dirs = erp_dirs(H, W)
    for s in scenes:                                      # GT reference clouds
        tag = f"{s['split']}_{s['key'].replace('/','-')}"
        save_ply(f"{PCD}/{tag}__GT.ply", s["gt"], dirs, s["vmax"])
    for lab, run, typ in resolved:
        clean = lab.replace("+", "").replace(" ", "").replace("-", "")
        for s in scenes:
            tag = f"{s['split']}_{s['key'].replace('/','-')}"
            n = save_ply(f"{PCD}/{tag}__{clean}.ply", cache[run][s["key"]], dirs, s["vmax"])
        print(f"[pcd] {lab} -> {clean} ({run})", flush=True)
    print(f"[done] figs/fig_sel.png + {PCD}/*.ply", flush=True)


if __name__ == "__main__":
    main()
