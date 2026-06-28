"""Montage: GT vs one representative model per paradigm, on the same test samples."""
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import data as D
from eval import load_model as load_imp, predict_full
from eval_fullmap import load as load_fm

dev = "cuda"
# (label, run, kind)  kind: imp=train.py model, fm=train_fullmap model
MODELS = [
    ("cross_flip\n(implicit 2ch)", "Bnode2_cross_flip_s0", "imp"),
    ("B_unet8_5ch\n(U-Net 8d +5ch)", "B_unet8_5ch_s0", "fm"),
    ("ViT-B/16\n(A22)", "A22_vit_aug_s0", "fm"),
    ("A9 fullmap\n(global bottleneck)", "A9_fullmap_s0", "fm"),
    ("Aunet\n(6d U-Net)", "Aunet_s0", "fm"),
]
IDX = [0, 700, 1400, 2100, 2800]   # fixed test indices (same key order across caches)


def predict(run, kind):
    rd = os.path.join("out", run)
    if kind == "imp":
        m, cfg, bank, shf = load_imp(rd, dev)
        ds = D.make_dataset(cfg, "test")
        keys = [ds[i]["key"] for i in IDX]
        spec = torch.stack([ds[i]["spec"] for i in IDX]).to(dev)
        with torch.no_grad():
            pr = (predict_full(m, spec, bank, cfg, shf) * cfg.max_depth).cpu().numpy()[:, 0]
        gt = (torch.stack([ds[i]["depth"] for i in IDX]) * cfg.max_depth).numpy()[:, 0]
        mask = torch.stack([ds[i]["mask"] for i in IDX]).numpy()[:, 0]
        return pr, gt, mask, keys
    else:
        m, cfg, extra = load_fm(rd, dev)
        ds = D.make_dataset(cfg, "test")
        keys = [ds[i]["key"] for i in IDX]
        spec = torch.stack([ds[i]["spec"] for i in IDX]).to(dev)
        if "norm" in extra:
            spec = (spec - extra["norm"][0]) / extra["norm"][1]
        with torch.no_grad():
            pr = (m(spec, extra.get("coarse_feat"), extra.get("sh_basis"))["D"] * cfg.max_depth).cpu().numpy()[:, 0]
        gt = (torch.stack([ds[i]["depth"] for i in IDX]) * cfg.max_depth).numpy()[:, 0]
        mask = torch.stack([ds[i]["mask"] for i in IDX]).numpy()[:, 0]
        return pr, gt, mask, keys


preds = {}
gt = mask = keys = None
for lab, run, kind in MODELS:
    try:
        pr, g, mk, keys = predict(run, kind)
        preds[lab] = pr; gt = g; mask = mk
        print(f"ok {run}", flush=True)
    except Exception as e:
        print(f"FAIL {run}: {e}", flush=True)

ncol = 1 + len(preds)
nrow = len(IDX)
fig, ax = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
cols = ["GT"] + list(preds.keys())
md = 10.0
for r in range(nrow):
    gtm = np.where(mask[r] > 0, gt[r], np.nan)
    ims = [gtm] + [preds[c][r] for c in preds]
    for c in range(ncol):
        a = ax[r][c]
        a.imshow(ims[c], cmap="turbo", vmin=0, vmax=md, aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        if r == 0:
            a.set_title(cols[c], fontsize=10)
        if c > 0:
            e = np.abs(preds[cols[c]][r] - gt[r])[mask[r] > 0].mean()
            a.set_xlabel(f"MAE={e:.2f}", fontsize=8)
fig.suptitle("ERP radial depth @256x512 — GT vs paradigm representatives", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs("out/figs", exist_ok=True)
fig.savefig("out/figs/all_models_montage.png", dpi=105)
print("saved out/figs/all_models_montage.png", flush=True)
