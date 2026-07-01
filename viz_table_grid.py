"""Grid viz of the comparison-table models (fig_grid_best style).
Each page: GT | U-Net8(5ch+flip) ref + 4 methods. Rows = 10 scenes.
Handles fm / impl / wave forward signatures (reuses viz_contenders.preds_for).
"""
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import eval_fullmap as ev_fm
import viz_contenders as vc
from viz_contenders import preds_for, best_seed, mae_of, DEV

vc.WANT = {"train": 2, "val": 2, "test": 6}      # 10 scenes (fig_grid_best layout)
FIG = "out/figs"; os.makedirs(FIG, exist_ok=True)

REF = ("U-Net8\n(5ch+flip)", "Bnode2_unet8_5chflip", "fm")
METHODS = [
    ("U-Net8 w20", "Bnode2_unet8_5chflip_w20", "fm"),
    ("GCC U-Net8", "Bnode2_gcc_unet8", "fm"),
    ("U-Net8 noflip", "B_unet8_5ch", "fm"),
    ("U-Net8 +normal", "U_unet8_normal", "fm"),
    ("U-Net8 +chamfer", "U_unet8_chamfer", "fm"),
    ("U-Net8 +scale1", "U_unet8_scale1", "fm"),
    ("U-Net8 nolog", "Bnode2_unet8nolog", "fm"),
    ("cross flip", "Bnode2_cross_flip", "impl"),
    ("echo_unet", "E_echo_unet", "wave"),
    ("echo_bin", "E_echo_bin", "wave"),
    ("RayDPT orig", "C_raydpt_5chflip", "fm"),
    ("RayDPT-lite", "C_raydptlite_5chflip", "fm"),
    ("RayDPT+E2", "R_raydpt_e2", "fm"),
    ("RayDPT-msf", "C_raydpt_msf", "fm"),
    ("RayDPT-rsmp", "C_raydpt_rsmp", "fm"),
]


def resolve(items):
    out = []
    for lab, base, typ in items:
        r = best_seed(base)
        if r is None:
            print(f"[skip] {base}"); continue
        out.append((f"{lab}\n{mae_of(r):.3f}", r, typ))
    return out


def main():
    cfg = ev_fm.load(f"out/{best_seed(REF[1])}", DEV)[1]
    scenes = vc.pick_scenes(cfg)
    ref = resolve([REF])[0]
    methods = resolve(METHODS)
    cache = {}
    for lab, run, typ in [ref] + methods:
        print(f"[infer] {run} ({typ})", flush=True)
        cache[run] = preds_for(run, typ, scenes)

    BATCH = 4
    groups = [methods[i:i + BATCH] for i in range(0, len(methods), BATCH)]
    for gi, grp in enumerate(groups, 1):
        cols = [("GT depth", None, None), ref] + grp
        ncol, nrow = len(cols), len(scenes)
        fig, ax = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 1.85 * nrow))
        for i, s in enumerate(scenes):
            for j, (lab, run, typ) in enumerate(cols):
                a = ax[i, j]
                img = s["gt"] if run is None else cache[run][s["key"]]
                im = a.imshow(img, cmap="turbo", vmin=0, vmax=s["vmax"])
                a.set_xticks([]); a.set_yticks([])
                if i == 0:
                    a.set_title(lab, fontsize=8)
                if j == 0:
                    a.set_ylabel(f"{s['split']}\n{s['key']}", fontsize=6)
            plt.colorbar(im, ax=ax[i, -1], fraction=0.046, pad=0.04)
        fig.suptitle(f"table b{gi}/{len(groups)}: GT | U-Net8(5ch+flip) + {len(grp)} methods. "
                     f"title = test MAE[m]. rows: 2 train / 2 val / 6 test.", y=1.001, fontsize=10)
        fig.tight_layout()
        out = f"{FIG}/fig_table_b{gi}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
