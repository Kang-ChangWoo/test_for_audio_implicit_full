"""Output visualisation of the CONTENDER models over MANY scenes.

Best-seed per family: GT | U-Net8 | GCC | Wave | cross-flip | RayDPT-full | RayDPT-lite.
Rows = scenes (train/val/test, spread by GT mean depth). Multi-page PNGs.
Handles fm / impl / wave forward signatures.
"""
import os, re, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from data import make_loader
import eval as ev_impl
import eval_fullmap as ev_fm
from train import predict_full

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG = "out/figs"; os.makedirs(FIG, exist_ok=True)

# (label, family-base, type)  type: 'fm' | 'impl' | 'wave'
FAMILIES = [
    ("U-Net8",       "Bnode2_unet8_5chflip", "fm"),
    ("GCC U-Net8",   "Bnode2_gcc_unet8",     "fm"),
    ("Wave U-Net8",  "Bnode2_wave_unet8",    "wave"),
    ("cross flip",   "Bnode2_cross_flip",    "impl"),
    ("RayDPT-full",  "C_raydpt_5chflip",     "fm"),
    ("RayDPT-lite",  "C_raydptlite_5chflip", "fm"),
]
WANT = {"train": 6, "val": 6, "test": 24}     # 36 scenes
PER_PAGE = 12


def best_seed(base):
    cands = []
    for d in os.listdir("out"):
        if re.sub(r"_s\d+$", "", d) != base:
            continue
        p = f"out/{d}/metrics_test.json"
        if os.path.exists(p):
            try:
                cands.append((json.load(open(p))["test"]["MAE"], d))
            except Exception:
                pass
    return min(cands)[1] if cands else None


def mae_of(run):
    try:
        return json.load(open(f"out/{run}/metrics_test.json"))["test"]["MAE"]
    except Exception:
        return None


def pick_scenes(cfg):
    md = cfg.max_depth
    chosen = []
    for split, k in WANT.items():
        loader = make_loader(cfg, split, shuffle=False)
        keys, means, gts = [], [], []
        seen = 0
        for b in loader:
            for j in range(b["depth"].size(0)):
                g = b["depth"][j, 0].numpy() * md
                m = b["mask"][j, 0].numpy()
                keys.append(b["key"][j]); gts.append(g)
                means.append(float((g * m).sum() / max(m.sum(), 1)))
            seen += b["depth"].size(0)
            if seen >= 600:
                break
        order = np.argsort(means)
        pick = np.linspace(0, len(order) - 1, k).round().astype(int)
        for pi in pick:
            i = order[pi]
            chosen.append(dict(split=split, key=keys[i], gt=gts[i],
                               vmax=float(max(1.0, np.percentile(gts[i], 99)))))
    return chosen


@torch.no_grad()
def preds_for(run, typ, scenes):
    if typ == "impl":
        m, cfg, bank, shf = ev_impl.load_model(f"out/{run}", DEV)
    else:
        m, cfg, extra = ev_fm.load(f"out/{run}", DEV)
    # gather spec (+wave) by key
    spec_by, wave_by = {}, {}
    for split in ("train", "val", "test"):
        need = {s["key"] for s in scenes if s["split"] == split}
        if not need:
            continue
        loader = make_loader(cfg, split, shuffle=False); got = 0
        for b in loader:
            for j in range(b["spec"].size(0)):
                kk = b["key"][j]
                if kk in need and kk not in spec_by:
                    spec_by[kk] = b["spec"][j:j+1].clone()
                    if "wave" in b:
                        wave_by[kk] = b["wave"][j:j+1].clone()
                    got += 1
            if got >= len(need):
                break
    out = {}
    for s in scenes:
        spec = spec_by[s["key"]].to(DEV)
        if spec.shape[1] > getattr(cfg, "in_ch", 2):
            spec = spec[:, :cfg.in_ch]
        if typ == "impl":
            P = predict_full(m, spec, bank, cfg, shf).cpu()[0, 0].numpy() * cfg.max_depth
        elif typ == "wave":
            wave = wave_by[s["key"]].to(DEV)
            P = m(spec, wave)["D"].cpu()[0, 0].numpy() * cfg.max_depth
        else:
            if "norm" in extra:
                spec = (spec - extra["norm"][0]) / extra["norm"][1]
            P = m(spec, extra.get("coarse_feat"), extra.get("sh_basis"))["D"].cpu()[0, 0].numpy() * cfg.max_depth
        out[s["key"]] = P
    del m; torch.cuda.empty_cache()
    return out


def main():
    runs = []
    for lab, base, typ in FAMILIES:
        r = best_seed(base)
        if r is None:
            print(f"[skip] {base}: no completed seed"); continue
        runs.append((f"{lab}\n(s{r.split('_s')[-1]} {mae_of(r):.3f})", r, typ))
    # cfg for scene picking (any fm model)
    base_cfg = ev_fm.load(f"out/{runs[0][1]}", DEV)[1]
    scenes = pick_scenes(base_cfg)
    print(f"[scenes] {len(scenes)} (train/val/test = {WANT})", flush=True)

    cache = {}
    for lab, run, typ in runs:
        print(f"[infer] {run} ({typ})", flush=True)
        cache[run] = preds_for(run, typ, scenes)

    pages = [scenes[i:i+PER_PAGE] for i in range(0, len(scenes), PER_PAGE)]
    for pi, pg in enumerate(pages, 1):
        cols = [("GT depth", None, None)] + runs
        ncol, nrow = len(cols), len(pg)
        fig, ax = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 1.85 * nrow))
        for i, s in enumerate(pg):
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
        fig.suptitle(f"contenders p{pi}/{len(pages)}: ERP radial depth. "
                     f"title=family (seed, test MAE[m]).", y=1.001, fontsize=10)
        fig.tight_layout()
        out = f"{FIG}/fig_contenders_p{pi}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
