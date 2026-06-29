"""Grid visualisation using each family's BEST seed (min test MAE).

Same layout as viz_grid: every figure carries GT | Best U-Net | Best cross as
references + up to 4 other methods (7 cols). Rows = 10 scenes (2 train/2 val/6 test).
"""
import os, re, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from viz_grid import pick_scenes, preds_for, mae, FIG

os.makedirs(FIG, exist_ok=True)


def best_seed(base):
    """Return the run-dir (base_sN) with the lowest test MAE, or None."""
    cands = []
    for d in os.listdir("out"):
        if re.sub(r"_s\d+$", "", d) != base:
            continue
        p = f"out/{d}/metrics_test.json"
        if not os.path.exists(p):
            continue
        try:
            cands.append((json.load(open(p))["test"]["MAE"], d))
        except Exception:
            pass
    return min(cands)[1] if cands else None


# (label, base-family, type)
REF_UNET = ("Best U-Net", "Bnode2_unet8_5chflip", "fm")
REF_CROSS = ("Best cross", "Bnode2_cross_flip", "impl")
FAMILIES = [
    ("U-Net w20",         "Bnode2_unet8_5chflip_w20", "fm"),
    ("U-Net no-log",      "Bnode2_unet8nolog",        "fm"),
    ("cross unet-enc 5ch","Bnode2_cross_unetenc5",    "impl"),
    ("cross 5ch+flip",    "Bnode2_cross_5chflip",     "impl"),
    ("cross no-log",      "Bnode2_cross_nolog",       "impl"),
    ("U-Net SH4 head",    "C_unet8_sh4_5chflip",      "fm"),
    ("cross unet-enc 2ch","Bnode2_cross_unetenc",     "impl"),
    ("U-Net SH6 head",    "C_unet8_sh6_5chflip",      "fm"),
    ("U-Net raycoarse16", "C_unet8_raycoarse16_5chflip", "fm"),
    ("cross 5ch",         "Bnode2_cross5ch",          "impl"),
    ("hybrid 5ch",        "Bnode2_hybrid5ch",         "impl"),
    ("U-Net coarse-res",  "C_unet8_coarseres_5chflip","fm"),
    ("U-Net coarse16",    "C_unet8_coarse16_5chflip", "fm"),
    ("U-Net coarse32",    "C_unet8_coarse32_5chflip", "fm"),
    ("rayconv dense 5ch", "Bnode2_rayconv5d",         "fm"),
]


def resolve(items):
    out = []
    for lab, base, typ in items:
        run = best_seed(base)
        if run is None:
            print(f"[skip] {base}: no completed seed"); continue
        seed = run.split("_s")[-1]
        out.append((f"{lab}\n(s{seed})", run, typ))
    return out


def main():
    ru = resolve([REF_UNET])[0]; rc = resolve([REF_CROSS])[0]
    others = resolve(FAMILIES)
    print(f"[refs] {ru[1]} / {rc[1]}")
    scenes = pick_scenes()
    cache = {}
    for lab, run, typ in [ru, rc] + others:
        print(f"[infer] {run}", flush=True)
        cache[run] = preds_for(run, typ, scenes)

    BATCH = 4
    groups = [others[i:i+BATCH] for i in range(0, len(others), BATCH)]
    for gi, grp in enumerate(groups, 1):
        cols = [("GT depth", None, None), ru, rc] + grp
        ncol, nrow = len(cols), len(scenes)
        fig, ax = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 1.85 * nrow))
        for i, s in enumerate(scenes):
            vmax = s["vmax"]
            for j, (lab, run, typ) in enumerate(cols):
                a = ax[i, j]
                img = s["gt"] if run is None else cache[run][s["key"]]
                im = a.imshow(img, cmap="turbo", vmin=0, vmax=vmax)
                a.set_xticks([]); a.set_yticks([])
                if i == 0:
                    extra = "" if run is None else f"\n{mae(run):.3f}"
                    a.set_title(lab + extra, fontsize=8)
                if j == 0:
                    a.set_ylabel(f"{s['split']}\n{s['key']}", fontsize=7)
            plt.colorbar(im, ax=ax[i, -1], fraction=0.046, pad=0.04)
        fig.suptitle(f"b{gi}: BEST-seed per family. refs (GT | Best U-Net | Best cross) "
                     f"+ {len(grp)} methods. title=test MAE[m]. rows: 2 train/2 val/6 test.",
                     y=1.002, fontsize=10)
        fig.tight_layout()
        out = f"{FIG}/fig_grid_best_b{gi}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
