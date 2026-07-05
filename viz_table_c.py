"""Depth-map comparison grid: best RayDPT group (3) + best U-Net8 group (3) + GT.
Paginated as fig_table_c1/c2/c3.png (scenes split across pages)."""
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import eval_fullmap as ev_fm
import viz_contenders as vc
from viz_contenders import preds_for, best_seed, mae_of, DEV

vc.WANT = {"train": 20, "val": 20, "test": 20}   # 60 scenes (4x)
FIG = "out/figs"; os.makedirs(FIG, exist_ok=True)
PER_PAGE = 10

# group -> (label, family-base). all fm (train_fullmap / eval_fullmap).
MODELS = [
    ("RayDPT: E22",       "Q5_e22_coarsesa"),        # best honest comp
    ("RayDPT: coarse-sa", "Q6_csaonly"),             # best d1
    ("RayDPT: +E2",       "R_raydpt_e2"),            # best AbsRel (ray)
    ("U-Net8 5ch+flip",   "Bnode2_unet8_5chflip"),   # champion baseline
    ("U-Net8 GCC",        "Bnode2_gcc_unet8"),       # best MAE
    ("U-Net8 +normal",    "U_unet8_normal"),         # best RMSE
]


def main():
    cfg = ev_fm.load(f"out/{best_seed('Bnode2_unet8_5chflip')}", DEV)[1]
    scenes = vc.pick_scenes(cfg)
    resolved = []
    for lab, base in MODELS:
        r = best_seed(base)
        if r is None:
            print(f"[skip] {base}"); continue
        try:
            preds = preds_for(r, "fm", scenes)
        except Exception as e:
            print(f"[skip] {base}: {e}"); continue
        resolved.append((f"{lab}\nMAE {mae_of(r):.3f}", preds))
        print(f"[ok] {lab} <- {r}", flush=True)

    cols = [("GT depth", None)] + resolved
    pages = [scenes[i:i + PER_PAGE] for i in range(0, len(scenes), PER_PAGE)]
    for pi, page in enumerate(pages, 1):
        ncol, nrow = len(cols), len(page)
        fig, ax = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 1.95 * nrow), squeeze=False)
        for i, s in enumerate(page):
            for j, (lab, preds) in enumerate(cols):
                a = ax[i][j]
                img = s["gt"] if preds is None else preds[s["key"]]
                im = a.imshow(img, cmap="turbo", vmin=0, vmax=s["vmax"])
                a.set_xticks([]); a.set_yticks([])
                if i == 0:
                    a.set_title(lab, fontsize=9)
                if j == 0:
                    a.set_ylabel(f"{s['split']}\n{s['key']}", fontsize=7)
            plt.colorbar(im, ax=ax[i][-1], fraction=0.046, pad=0.04)
        fig.suptitle(f"[c{pi}] GT + best RayDPT (E22/coarse-sa/+E2) + best U-Net8 "
                     f"(5ch+flip/GCC/+normal).  title = test MAE[m].", y=1.003, fontsize=11)
        fig.tight_layout()
        out = f"{FIG}/fig_table_c{pi}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
