"""Top-5 models (Bnode2_ + B_ families) montage, 10 test samples -> out/figs/top5_montage.png."""
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import data as D
from eval import load_model as load_imp, predict_full
from eval_fullmap import load as load_fm

dev = "cuda"
MODELS = [  # (label, run, kind) — ranked by MAE_plain
    ("B_unet8_5ch\n0.753", "B_unet8_5ch_s0", "fm"),
    ("B_unet8nolog_aug\n0.758", "B_unet8nolog_aug_s0", "fm"),
    ("Bnode2_cross_flip\n0.761", "Bnode2_cross_flip_s0", "imp"),
    ("Bnode2_unet8nolog\n0.763", "Bnode2_unet8nolog_s0", "fm"),
    ("B_unet8nolog\n0.764", "B_unet8nolog_s0", "fm"),
]
IDX = [int(i * 3600 / 10) for i in range(10)]   # 10 spread test indices
MD = 10.0


def predict(run, kind):
    rd = os.path.join("out", run)
    if kind == "imp":
        m, cfg, bank, shf = load_imp(rd, dev); ds = D.make_dataset(cfg, "test")
        spec = torch.stack([ds[i]["spec"] for i in IDX]).to(dev)
        with torch.no_grad():
            pr = (predict_full(m, spec, bank, cfg, shf) * cfg.max_depth).cpu().numpy()[:, 0]
    else:
        m, cfg, extra = load_fm(rd, dev); ds = D.make_dataset(cfg, "test")
        spec = torch.stack([ds[i]["spec"] for i in IDX]).to(dev)
        if "norm" in extra: spec = (spec - extra["norm"][0]) / extra["norm"][1]
        with torch.no_grad():
            pr = (m(spec, extra.get("coarse_feat"), extra.get("sh_basis"))["D"] * cfg.max_depth).cpu().numpy()[:, 0]
    gt = (torch.stack([ds[i]["depth"] for i in IDX]) * cfg.max_depth).numpy()[:, 0]
    mask = torch.stack([ds[i]["mask"] for i in IDX]).numpy()[:, 0]
    return pr, gt, mask


preds = {}; gt = mask = None
for lab, run, kind in MODELS:
    try:
        pr, g, mk = predict(run, kind); preds[lab] = pr; gt = g; mask = mk; print("ok", run, flush=True)
    except Exception as e:
        print("FAIL", run, e, flush=True)

ncol = 1 + len(preds); nrow = len(IDX)
fig, ax = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.3 * nrow))
cols = ["GT"] + list(preds.keys())
for r in range(nrow):
    ims = [np.where(mask[r] > 0, gt[r], np.nan)] + [preds[c][r] for c in preds]
    for c in range(ncol):
        a = ax[r][c]; a.imshow(ims[c], cmap="turbo", vmin=0, vmax=MD, aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        if r == 0: a.set_title(cols[c], fontsize=9)
        if c > 0:
            e = np.abs(preds[cols[c]][r] - gt[r])[mask[r] > 0].mean()
            a.set_xlabel(f"{e:.2f}", fontsize=7)
fig.suptitle("Top-5 models (Bnode2_+B_) — ERP radial depth @256x512, 10 test samples", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
os.makedirs("out/figs", exist_ok=True)
fig.savefig("out/figs/top5_montage.png", dpi=105)
print("saved out/figs/top5_montage.png", flush=True)
