"""Focused comparison: GT | Best U-Net | Best cross | cross-vitenc s1, over N test samples."""
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from data import make_loader
from viz_grid import preds_for, mae, FIG
import eval_fullmap as ev_fm

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 12

COLS = [("Best U-Net\nunet8_5chflip", "Bnode2_unet8_5chflip_s0", "fm"),
        ("Best cross\ncross_flip",     "Bnode2_cross_flip_s0",    "impl"),
        ("cross-vitenc s1\n(eval중 미정)", "Bnode2_cross_vitenc_s1", "impl")]


def pick_test(n):
    base = ev_fm.load(f"out/{COLS[0][1]}", DEV)[1]; md = base.max_depth
    loader = make_loader(base, "test", shuffle=False)
    keys, means, gts = [], [], []
    seen = 0
    for b in loader:
        for j in range(b["depth"].size(0)):
            g = b["depth"][j, 0].numpy() * md; m = b["mask"][j, 0].numpy()
            keys.append(b["key"][j]); means.append(float((g*m).sum()/max(m.sum(),1))); gts.append(g)
        seen += b["depth"].size(0)
        if seen >= 400: break
    order = np.argsort(means)
    pick = np.linspace(0, len(order)-1, n).round().astype(int)
    return [dict(split="test", key=keys[order[p]], gt=gts[order[p]],
                 vmax=float(max(1.0, np.percentile(gts[order[p]], 99)))) for p in pick]


def main():
    scenes = pick_test(N)
    cache = {run: preds_for(run, typ, scenes) for _, run, typ in COLS}
    titles = ["GT depth"] + [c[0] for c in COLS]
    fig, ax = plt.subplots(N, 4, figsize=(2.2*4, 1.85*N))
    for i, s in enumerate(scenes):
        vmax = s["vmax"]
        imgs = [s["gt"]] + [cache[c[1]][s["key"]] for c in COLS]
        for j, img in enumerate(imgs):
            a = ax[i, j]; im = a.imshow(img, cmap="turbo", vmin=0, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                t = titles[j]
                if j > 0:
                    run = COLS[j-1][1]; m = mae(run); t += f"\n{m:.3f}" if m else "\n(eval중)"
                a.set_title(t, fontsize=9)
            if j == 0:
                a.set_ylabel(s["key"], fontsize=7)
        plt.colorbar(im, ax=ax[i, -1], fraction=0.046, pad=0.04)
    fig.suptitle("GT vs Best U-Net (0.886) vs Best cross (0.903) vs cross-vitenc s1 — 12 test samples "
                 "(GT-mean-depth spread)", y=1.003, fontsize=11)
    fig.tight_layout()
    out = f"{FIG}/fig_vitenc_cmp.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); print(f"[saved] {out}")


if __name__ == "__main__":
    main()
