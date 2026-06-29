"""Grid visualisation of representative C_/Bnode2_ experiments (seed s0).

Every figure carries 3 references — GT | Best U-Net | Best cross — plus up to 4
other methods (7 columns). Figures numbered a1, a2, ... Rows = 10 scenes:
2 train, 2 val, 6 test (the last 4 test added for diversity), picked by GT
mean-depth spread.
"""
import os, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from data import make_loader
import eval as ev_impl
import eval_fullmap as ev_fm
from train import predict_full

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG = "out/figs"; os.makedirs(FIG, exist_ok=True)

BEST_UNET = ("Best U-Net\n(unet8 5ch+flip)", "Bnode2_unet8_5chflip_s0", "fm")
BEST_CROSS = ("Best cross\n(cross flip)", "Bnode2_cross_flip_s0", "impl")

# (label, run, type) — type: 'impl'=RayDepthModel(eval), 'fm'=FullMapNet(eval_fullmap)
OTHERS = [
    ("U-Net w20",            "Bnode2_unet8_5chflip_w20_s0", "fm"),
    ("U-Net no-log",         "Bnode2_unet8nolog_s0",        "fm"),
    ("cross unet-enc 5ch",   "Bnode2_cross_unetenc5_s0",    "impl"),
    ("cross 5ch+flip",       "Bnode2_cross_5chflip_s0",     "impl"),
    ("cross no-log",         "Bnode2_cross_nolog_s0",       "impl"),
    ("U-Net SH4 head",       "C_unet8_sh4_5chflip_s0",      "fm"),
    ("cross unet-enc 2ch",   "Bnode2_cross_unetenc_s0",     "impl"),
    ("U-Net SH6 head",       "C_unet8_sh6_5chflip_s0",      "fm"),
    ("U-Net raycoarse16",    "C_unet8_raycoarse16_5chflip_s0", "fm"),
    ("cross 5ch",            "Bnode2_cross5ch_s0",          "impl"),
    ("hybrid 5ch",           "Bnode2_hybrid5ch_s0",         "impl"),
    ("U-Net coarse-res",     "C_unet8_coarseres_5chflip_s0","fm"),
    ("U-Net coarse16",       "C_unet8_coarse16_5chflip_s0", "fm"),
    ("U-Net coarse32",       "C_unet8_coarse32_5chflip_s0", "fm"),
    ("rayconv dense 5ch",    "Bnode2_rayconv5d_s0",         "fm"),
]

# all runs we must run inference for (refs first)
ALLRUNS = [BEST_UNET, BEST_CROSS] + OTHERS


def mae(run):
    p = f"out/{run}/metrics_test.json"
    try: return json.load(open(p))["test"]["MAE"]
    except: return None


# ---------- 1. pick 10 diverse scenes ----------------------------------------
def pick_scenes():
    """Return list of dicts: {split, key, gt(np HxW), vmax}. Diversity by GT mean depth."""
    base = ev_fm.load(f"out/{BEST_UNET[1]}", DEV)[1]   # any cfg => GT is cache-independent
    md = base.max_depth                                # GT in cache is normalised [0,1]
    want = {"train": 2, "val": 2, "test": 6}
    chosen = []
    for split, k in want.items():
        loader = make_loader(base, split, shuffle=False)
        keys, means, gts = [], [], []
        seen = 0
        for b in loader:
            for j in range(b["depth"].size(0)):
                g = b["depth"][j, 0].numpy() * md       # -> metres, matches preds
                m = b["mask"][j, 0].numpy()
                mu = float((g * m).sum() / max(m.sum(), 1))
                keys.append(b["key"][j]); means.append(mu); gts.append(g)
            seen += b["depth"].size(0)
            if seen >= 240:    # sample a decent pool then pick spread
                break
        order = np.argsort(means)
        # evenly spaced over the mean-depth-sorted pool -> max diversity
        pick = np.linspace(0, len(order) - 1, k).round().astype(int)
        for pi in pick:
            i = order[pi]
            chosen.append(dict(split=split, key=keys[i], gt=gts[i],
                               vmax=float(max(1.0, np.percentile(gts[i], 99)))))
    return chosen


# ---------- 2. run every model on the chosen scenes --------------------------
@torch.no_grad()
def preds_for(run, typ, scenes):
    if typ == "impl":
        m, cfg, bank, shf = ev_impl.load_model(f"out/{run}", DEV)
    else:
        m, cfg, extra = ev_fm.load(f"out/{run}", DEV)
    want_keys = {s["key"] for s in scenes}
    spec_by_key = {}
    for split in ("train", "val", "test"):
        need = {s["key"] for s in scenes if s["split"] == split}
        if not need: continue
        loader = make_loader(cfg, split, shuffle=False)
        got = 0
        for b in loader:
            for j in range(b["spec"].size(0)):
                kk = b["key"][j]
                if kk in need:
                    spec_by_key[kk] = b["spec"][j:j+1].clone()
                    got += 1
            if got >= len(need): break
    out = {}
    for s in scenes:
        spec = spec_by_key[s["key"]].to(DEV)
        if spec.shape[1] > getattr(cfg, "in_ch", 2):
            spec = spec[:, :cfg.in_ch]
        if typ == "impl":
            P = predict_full(m, spec, bank, cfg, shf).cpu()[0, 0].numpy() * cfg.max_depth
        else:
            if "norm" in extra:
                spec = (spec - extra["norm"][0]) / extra["norm"][1]
            P = m(spec, extra.get("coarse_feat"), extra.get("sh_basis"))["D"].cpu()[0, 0].numpy() * cfg.max_depth
        out[s["key"]] = P
    del m; torch.cuda.empty_cache()
    return out


def main():
    scenes = pick_scenes()
    print(f"[scenes] {len(scenes)}:")
    for s in scenes:
        print(f"   {s['split']:5s} {s['key']:14s} vmax={s['vmax']:.1f}")

    cache = {}
    for lab, run, typ in ALLRUNS:
        print(f"[infer] {run}", flush=True)
        cache[run] = preds_for(run, typ, scenes)

    # figure batches: refs + 4 others each
    BATCH = 4
    groups = [OTHERS[i:i+BATCH] for i in range(0, len(OTHERS), BATCH)]
    for gi, grp in enumerate(groups, 1):
        cols = [("GT depth", None, None), BEST_UNET, BEST_CROSS] + grp
        ncol = len(cols); nrow = len(scenes)
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
        fig.suptitle(f"a{gi}: predicted ERP radial depth — refs (GT | Best U-Net | Best cross) "
                     f"+ {len(grp)} methods. title=test MAE[m]. rows: 2 train / 2 val / 6 test "
                     f"(GT-mean-depth spread).", y=1.002, fontsize=10)
        fig.tight_layout()
        out = f"{FIG}/fig_grid_a{gi}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
