"""Metric bar-chart comparison: E22 champion vs best U-Net8 vs best RayDPT.
Averages seeds per family; 5 panels (MAE, RMSE, AbsRel, d1, honest comp)."""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# (label, family-base, group)
MODELS = [
    ("E22 champion\n(coarse-sa+EMA)", "Q5_e22_coarsesa",       "E22"),
    ("U-Net8 5ch+flip",               "Bnode2_unet8_5chflip",  "UNet"),
    ("GCC U-Net8",                     "Bnode2_gcc_unet8",      "UNet"),
    ("U-Net8 +normal",                 "U_unet8_normal",        "UNet"),
    ("RayDPT+E2",                      "R_raydpt_e2",           "RayDPT"),
    ("RayDPT orig",                    "C_raydpt_5chflip",      "RayDPT"),
]
COL = {"E22": "#d62728", "UNet": "#1f77b4", "RayDPT": "#2ca02c"}


def comp(d):
    return d["RMSE"] / 1.6 + (1 - d["delta1"]) / 0.46 + 0.3 * d["AbsRel"] / 0.4


def fam(base):
    ds = [json.load(open(f))["test"] for f in glob.glob(f"out/{base}_s*/metrics_test.json")]
    if not ds:
        return None
    n = len(ds)
    m = {k: sum(d[k] for d in ds) / n for k in ["MAE", "RMSE", "AbsRel", "delta1"]}
    m["comp"] = comp(m); m["n"] = n
    return m


def main():
    rows = [(lab, fam(base), grp) for lab, base, grp in MODELS]
    rows = [(lab, m, grp) for lab, m, grp in rows if m]
    labs = [f"{lab}  (n={m['n']})" for lab, m, grp in rows]
    cols = [COL[grp] for _, _, grp in rows]

    panels = [("MAE [m] ↓", "MAE"), ("RMSE [m] ↓", "RMSE"), ("AbsRel ↓", "AbsRel"),
              ("δ1 ↑", "delta1"), ("honest comp ↓\n(RMSE+d1 weighted)", "comp")]
    fig, ax = plt.subplots(1, 5, figsize=(19, 4.6))
    y = np.arange(len(rows))[::-1]
    for a, (title, key) in zip(ax, panels):
        vals = [m[key] for _, m, _ in rows]
        best = max(vals) if key == "delta1" else min(vals)
        a.barh(y, vals, color=cols, edgecolor="black", linewidth=0.6)
        for yi, v in zip(y, vals):
            a.text(v, yi, f" {v:.4f}" + ("  ★" if v == best else ""),
                   va="center", ha="left", fontsize=8,
                   fontweight="bold" if v == best else "normal")
        a.set_yticks(y); a.set_yticklabels(labs if a is ax[0] else [], fontsize=8)
        a.set_title(title, fontsize=11)
        lo, hi = min(vals), max(vals); pad = (hi - lo) * 0.35 + 1e-6
        a.set_xlim(lo - pad * 0.3, hi + pad * 1.6)
        a.grid(axis="x", alpha=0.3)
    fig.suptitle("Metric comparison: E22 champion vs best U-Net8 vs best RayDPT  "
                 "(★ = best per metric; red=E22, blue=U-Net, green=RayDPT)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("out/figs", exist_ok=True)
    out = "out/figs/fig_metrics_e22.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")
    for lab, m, _ in rows:
        print(f"  {lab:32} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} AbsRel={m['AbsRel']:.4f} "
              f"d1={m['delta1']:.3f} comp={m['comp']:.3f} (n={m['n']})")


if __name__ == "__main__":
    main()
