"""Plot predicted ERP radial-depth maps: GT vs best implicit (cross) vs U-Net.

Same test samples fed to both models; saves out/figs/compare_cross_vs_unet.png.
Usage: python viz_compare.py --imp A4_cross_s0 --unet Aunet_s0 --n 5
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as D
from eval import load_model as load_imp, predict_full
from eval_fullmap import load as load_fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imp", default="A4_cross_s0")
    ap.add_argument("--unet", default="Aunet_s0")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    dev = "cuda"

    mi, ci, bank, shf = load_imp(os.path.join("out", args.imp), dev)
    mu, cu, extra = load_fm(os.path.join("out", args.unet), dev)
    md = ci.max_depth

    # spread samples across the whole test set (diverse scenes), same for both
    ds = D.make_dataset(ci, "test")
    N = len(ds)
    idxs = [int(i * N / args.n) for i in range(args.n)]
    items = [ds[i] for i in idxs]
    spec = torch.stack([it["spec"] for it in items]).to(dev)
    gt = (torch.stack([it["depth"] for it in items]) * md).numpy()[:, 0]
    mask = torch.stack([it["mask"] for it in items]).numpy()[:, 0]
    keys = [it["key"] for it in items]

    with torch.no_grad():
        pi = (predict_full(mi, spec, bank, ci, shf) * md).cpu().numpy()[:, 0]
        pu = (mu(spec, extra.get("coarse_feat"), extra.get("sh_basis"))["D"] * md).cpu().numpy()[:, 0]

    # blank invalid GT pixels
    gtm = np.where(mask > 0, gt, np.nan)
    ei = np.abs(pi - gt) * (mask > 0)
    eu = np.abs(pu - gt) * (mask > 0)

    cols = ["GT (radial depth)", f"cross implicit\n{args.imp}", f"U-Net\n{args.unet}",
            "|err| cross", "|err| U-Net"]
    n = args.n
    fig, ax = plt.subplots(n, 5, figsize=(18, 3.0 * n))
    if n == 1:
        ax = ax[None, :]
    for r in range(n):
        maed_i = ei[r][mask[r] > 0].mean()
        maed_u = eu[r][mask[r] > 0].mean()
        ims = [gtm[r], pi[r], pu[r], ei[r], eu[r]]
        cmaps = ["turbo", "turbo", "turbo", "magma", "magma"]
        vmaxs = [md, md, md, 4.0, 4.0]
        for c in range(5):
            a = ax[r][c]
            im = a.imshow(ims[c], cmap=cmaps[c], vmin=0, vmax=vmaxs[c], aspect="auto")
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=11)
            fig.colorbar(im, ax=a, fraction=0.025, pad=0.01)
        ax[r][0].set_ylabel(f"{keys[r]}", fontsize=8)
        ax[r][3].set_xlabel(f"MAE={maed_i:.3f}m", fontsize=9)
        ax[r][4].set_xlabel(f"MAE={maed_u:.3f}m", fontsize=9)

    fig.suptitle(f"ERP radial depth @256x512 — GT vs cross-implicit vs U-Net "
                 f"(cross is smooth/band-limited; U-Net chases unpredictable detail)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs("out/figs", exist_ok=True)
    out = "out/figs/compare_cross_vs_unet.png"
    fig.savefig(out, dpi=110)
    print(f"[viz] saved {out}  (samples: {keys})", flush=True)


if __name__ == "__main__":
    main()
