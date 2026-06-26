"""Aggregate ALL full-res (256x512) results into a grouped markdown table.

Groups runs by base name (strips the trailing _s<seed>), reports mean +/- std over
seeds for the key metrics + the audio-use control (shuffle) and L/R-mirror
consistency. Writes RESULTS_full.md. Safe to run anytime (uses whatever evals exist).
"""
import json
import os
import re
import glob
import numpy as np

OUT = "out"


def base(name):
    m = re.match(r"^(.*)_s\d+$", name)
    return m.group(1) if m else name


def load(run):
    p = os.path.join(OUT, run, "metrics_test.json")
    return json.load(open(p)) if os.path.exists(p) else None


# 64x128 reference (sibling experiments) for the same run names, MAE_plain
REF_DIRS = ["../test_for_audio_implicit/out", "../test_for_audio_implicit_n1/out"]


def ref_maeplain(group, seed_runs):
    """mean MAE_plain at 64x128 for the same base run names, if those runs exist."""
    vals = []
    for r in seed_runs:
        for rd in REF_DIRS:
            p = os.path.join(rd, r, "metrics_test.json")
            if os.path.exists(p):
                try:
                    vals.append(json.load(open(p))["test"]["MAE_plain"]); break
                except Exception:
                    pass
    return float(np.mean(vals)) if vals else None


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), float(np.std(xs)), len(xs)


def main():
    runs = sorted(d for d in os.listdir(OUT)
                  if os.path.isdir(os.path.join(OUT, d)))
    trained = [r for r in runs if os.path.exists(os.path.join(OUT, r, "train_done.json"))]
    evaled = [r for r in trained if load(r) is not None]

    # group metrics_test runs
    groups = {}
    for r in evaled:
        groups.setdefault(base(r), []).append(r)

    rows = []
    for g, rs in groups.items():
        ds = [load(r) for r in rs]
        mae = mean_std([d["test"].get("MAE") for d in ds])
        maep = mean_std([d["test"].get("MAE_plain") for d in ds])
        low = mean_std([d["test"].get("MAE_low") for d in ds])
        d1 = mean_std([d["test"].get("delta1") for d in ds])
        shuf = mean_std([d.get("shuffle", {}).get("MAE") for d in ds])
        swapc = mean_std([d["test"].get("mirror_better_rate") for d in ds])
        ref = ref_maeplain(g, rs)
        rows.append(dict(g=g, n=mae[2], mae=mae, maep=maep, low=low, d1=d1, shuf=shuf, mb=swapc, ref=ref))
    rows.sort(key=lambda x: (x["mae"][0] if x["mae"][0] is not None else 9))

    L = []
    L.append("# test_for_audio_implicit_full — FULL-RES (256x512) results\n")
    L.append(f"Radial depth, scene_split (72/9/9), loaded from actual files at 256x512 "
             f"(no 64x128 downsampling). cos-lat weighted metrics. seed = mean ± std.\n")
    L.append(f"**Progress: {len(evaled)}/{len(trained)} trained runs evaluated "
             f"(of 93 total jobs).**\n")
    L.append("| model (seed-grouped) | n | test MAE ↓ | MAE_plain | (64×128 plain) | MAE_low | δ<1.25 ↑ | shuffle MAE | mirror_better |")
    L.append("|---|---|---|---|---|---|---|---|---|")

    def f(ms, d=4):
        m, s, n = ms
        return "—" if m is None else (f"{m:.{d}f} ± {s:.{d}f}" if n > 1 else f"{m:.{d}f}")

    for r in rows:
        refc = "—" if r["ref"] is None else f"{r['ref']:.4f}"
        L.append(f"| {r['g']} | {r['n']} | {f(r['mae'])} | {f(r['maep'])} | {refc} | {f(r['low'])} "
                 f"| {f(r['d1'],3)} | {f(r['shuf'])} | {f(r['mb'],3)} |")

    # prob runs (separate file)
    probs = sorted(glob.glob(os.path.join(OUT, "*", "prob_eval.json")))
    if probs:
        L.append("\n## Probabilistic coarse head (prob_eval.json)\n")
        L.append("| run | keys |")
        L.append("|---|---|")
        for p in probs:
            name = os.path.basename(os.path.dirname(p))
            d = json.load(open(p))
            kv = ", ".join(f"{k}={round(v,4) if isinstance(v,(int,float)) else v}"
                           for k, v in d.items() if isinstance(v, (int, float, str)))
            L.append(f"| {name} | {kv[:200]} |")

    # not-yet-evaluated
    missing = [r for r in trained if r not in evaled]
    if missing:
        L.append(f"\n_Pending eval ({len(missing)}): {', '.join(missing)}_")
    untrained = 93 - len(trained)
    if untrained > 0:
        L.append(f"\n_Still training: ~{untrained} runs not yet finished._")

    txt = "\n".join(L) + "\n"
    open("RESULTS_full.md", "w").write(txt)
    print(txt)
    print(f"[agg] wrote RESULTS_full.md ({len(evaled)} runs, {len(rows)} groups)")


if __name__ == "__main__":
    main()
