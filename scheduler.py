"""Parallel job scheduler: run EVERY full-res (256x512) experiment across all GPUs.

One job per GPU at a time; as a GPU frees, the next ready job is dispatched, so all
8 GPUs stay busy. Dependencies: A14 warm-starts from A9_* (must finish first); RIR
jobs (in_ch 3/5) wait for their local cache. in_ch=3/5 caches are built in the
background (CPU/NFS) while in_ch=2 GPU jobs run.

Run:  nohup python scheduler.py --gpus 0,1,2,3,4,5,6,7 > logs/_sched.log 2>&1 &
"""
import argparse
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
EP = "--epochs 25 --num-workers 6"


def imp(name, model, seed, lr="2e-3", extra="", in_ch=2):
    return dict(name=name, in_ch=in_ch, deps=[], kind="imp",
                cmd=f"python train.py --model {model} --run-name {name} --seed {seed} "
                    f"{EP} --batch-size 24 --n-rays 2048 --lr {lr} --in-ch {in_ch} {extra}")


def fm(name, seed, arch="fullmap", lr="2e-3", bs=32, extra="", in_ch=2, deps=None):
    return dict(name=name, in_ch=in_ch, deps=deps or [], kind="fm",
                cmd=f"python train_fullmap.py --arch {arch} --run-name {name} --seed {seed} "
                    f"{EP} --batch-size {bs} --lr {lr} --in-ch {in_ch} {extra}")


def prob(name, k):
    return dict(name=name, in_ch=2, deps=[], kind="prob",
                cmd=f"python train_prob.py --run-name {name} --seed 0 {EP} "
                    f"--batch-size 32 --lr 2e-3 --prob-k {k}")


# pix2pix-U-Net common knobs
U96 = "--ngf 96 --dim 256 --n-heads 8 --n-cross 2 --coarse-h 8 --coarse-w 16 --weight-decay 1e-4 --ray-mod-scale 0.1"
U64 = "--ngf 64 --dim 256 --n-heads 8 --n-cross 2 --coarse-h 8 --coarse-w 16 --weight-decay 5e-4"
STRONG = "--ray-mod-scale 0.4 --ray-mod-stage e2+e3"

JOBS = [
    # ---------- implicit ray models (train.py, in_ch=2, bs24) ----------
    imp("A1_rayonly_s0", "rayonly", 0, extra="--audio-mode none"),
    imp("A1_rayonly_s1", "rayonly", 1, extra="--audio-mode none"),
    imp("A2_raymlp_s0", "raymlp", 0), imp("A2_raymlp_s1", "raymlp", 1), imp("A2_raymlp_s2", "raymlp", 2),
    imp("A2_shuf_s0", "raymlp", 0, extra="--shuffle-audio True"),
    imp("A2_shuf_s1", "raymlp", 1, extra="--shuffle-audio True"),
    imp("A4_cross_s0", "cross", 0, lr="3e-4"), imp("A4_cross_s1", "cross", 1, lr="3e-4"),
    imp("A4_cross_s2", "cross", 2, lr="3e-4"),
    imp("A4_cross_shuf_s0", "cross", 0, lr="3e-4", extra="--shuffle-audio True"),
    imp("A3_crossSH_s0", "cross", 0, lr="3e-4", extra="--use-sh-pe True"),
    imp("A5_crossMic_s0", "cross", 0, lr="3e-4", extra="--use-mic-pe True"),
    imp("A4_ffmask_s0", "cross", 0, lr="3e-4", extra="--mask-farfield True"),
    imp("A8_hybrid_s0", "hybrid", 0, lr="3e-4"),
    imp("A6_crossself_s0", "crossself", 0, lr="3e-4"),
    imp("A6_crossself_s1", "crossself", 1, lr="3e-4"),
    imp("A6_crossself_s2", "crossself", 2, lr="3e-4"),
    imp("A6sec_s0", "crossself", 0, lr="3e-4", extra="--sector-sample True"),
    imp("A6sec_s1", "crossself", 1, lr="3e-4", extra="--sector-sample True"),
    imp("A6sec_s2", "crossself", 2, lr="3e-4", extra="--sector-sample True"),
    imp("T_mlpskip", "cross", 0, lr="3e-4", extra="--ray-mlp-skip True"),
    imp("T_film", "cross", 0, lr="3e-4", extra="--ray-film True"),
    imp("T_progpe", "cross", 0, lr="3e-4", extra="--prog-pe True"),
    imp("T_sector", "cross", 0, lr="3e-4", extra="--sector-sample True"),
    imp("T_all", "cross", 0, lr="3e-4",
        extra="--ray-mlp-skip True --ray-film True --prog-pe True --sector-sample True"),

    # ---------- A9-A12 full-map decoder (train_fullmap fullmap, in_ch=2, bs32) ----------
    fm("A9_fullmap_s0", 0, extra="--correction none"),
    fm("A9_fullmap_s1", 1, extra="--correction none"),
    fm("A9_fullmap_s2", 2, extra="--correction none"),
    fm("A10_cross_s0", 0, extra="--correction cross"),
    fm("A11_shaux_s0", 0, extra="--correction sh"),
    fm("A11_shaux_s1", 1, extra="--correction sh"),
    fm("A11_shaux_s2", 2, extra="--correction sh"),
    fm("A12_film_s0", 0, extra="--correction film"),

    # ---------- pix2pix U-Net baseline (Aunet, ngf64, bs64) ----------
    fm("Aunet_s0", 0, arch="unet", bs=64), fm("Aunet_s1", 1, arch="unet", bs=64),
    fm("Aunet_s2", 2, arch="unet", bs=64),

    # ---------- A15/A16 capacity vs ray-mod (ngf96, bs48, lr1e-3) ----------
    fm("A15_bigunet_s0", 0, arch="unet", lr="1e-3", bs=48, extra=U96),
    fm("A15_bigunet_s1", 1, arch="unet", lr="1e-3", bs=48, extra=U96),
    fm("A15_bigunet_s2", 2, arch="unet", lr="1e-3", bs=48, extra=U96),
    fm("A16_raymod8x16_s0", 0, arch="unet_raymod", lr="1e-3", bs=48, extra=U96),
    fm("A16_raymod8x16_s1", 1, arch="unet_raymod", lr="1e-3", bs=48, extra=U96),
    fm("A16_raymod8x16_s2", 2, arch="unet_raymod", lr="1e-3", bs=48, extra=U96),

    # ---------- A18 regularized ngf64 (bs64, lr2e-3, wd5e-4, scale0.1) ----------
    fm("A18_unet64reg_s0", 0, arch="unet", bs=64, extra=U64),
    fm("A18_unet64reg_s1", 1, arch="unet", bs=64, extra=U64),
    fm("A18_unet64reg_s2", 2, arch="unet", bs=64, extra=U64),
    fm("A18_raymod64reg_s0", 0, arch="unet_raymod", bs=64, extra=U64 + " --ray-mod-scale 0.1"),
    fm("A18_raymod64reg_s1", 1, arch="unet_raymod", bs=64, extra=U64 + " --ray-mod-scale 0.1"),

    # ---------- A19 strong ray (ngf64, bs64) ----------
    fm("A19_raymodStrong_s0", 0, arch="unet_raymod", bs=64, extra=U64 + " " + STRONG),
    fm("A19_raymodStrong_s1", 1, arch="unet_raymod", bs=64, extra=U64 + " " + STRONG),
    fm("A19_raymodStrong_s2", 2, arch="unet_raymod", bs=64, extra=U64 + " " + STRONG),

    # ---------- full-val re-runs (_fv) ----------
    fm("A18_unet64reg_fv_s0", 0, arch="unet", bs=64, extra=U64),
    fm("A18_unet64reg_fv_s1", 1, arch="unet", bs=64, extra=U64),
    fm("A18_unet64reg_fv_s2", 2, arch="unet", bs=64, extra=U64),
    fm("A19_raymodStrong_fv_s0", 0, arch="unet_raymod", bs=64, extra=U64 + " " + STRONG),
    fm("A19_raymodStrong_fv_s1", 1, arch="unet_raymod", bs=64, extra=U64 + " " + STRONG),
    fm("A19_raymodStrong_fv_s2", 2, arch="unet_raymod", bs=64, extra=U64 + " " + STRONG),
    fm("A15_bigunet_fv_s0", 0, arch="unet", lr="1e-3", bs=48, extra=U96),
    fm("A15_bigunet_fv_s1", 1, arch="unet", lr="1e-3", bs=48, extra=U96),
    fm("A15_bigunet_fv_s2", 2, arch="unet", lr="1e-3", bs=48, extra=U96),
    fm("A16_raymod_fv_s0", 0, arch="unet_raymod", lr="1e-3", bs=48, extra=U96),
    fm("A16_raymod_fv_s1", 1, arch="unet_raymod", lr="1e-3", bs=48, extra=U96),
    fm("A16_raymod_fv_s2", 2, arch="unet_raymod", lr="1e-3", bs=48, extra=U96),

    # ---------- flip-aug (A20/A21, ngf64, bs64) ----------
    fm("A20_unet64_aug_s0", 0, arch="unet", bs=64, extra=U64 + " --flip-aug True"),
    fm("A20_unet64_aug_s1", 1, arch="unet", bs=64, extra=U64 + " --flip-aug True"),
    fm("A20_unet64_aug_s2", 2, arch="unet", bs=64, extra=U64 + " --flip-aug True"),
    fm("A21_raymodStrong_aug_s0", 0, arch="unet_raymod", bs=64, extra=U64 + " " + STRONG + " --flip-aug True"),
    fm("A21_raymodStrong_aug_s1", 1, arch="unet_raymod", bs=64, extra=U64 + " " + STRONG + " --flip-aug True"),
    fm("A21_raymodStrong_aug_s2", 2, arch="unet_raymod", bs=64, extra=U64 + " " + STRONG + " --flip-aug True"),

    # ---------- ViT (A22 planar, A23 ERP-PE), bs32, lr3e-4, flip-aug ----------
    fm("A22_vit_aug_s0", 0, arch="vit", lr="3e-4", bs=32, extra="--weight-decay 5e-4 --flip-aug True"),
    fm("A22_vit_aug_s1", 1, arch="vit", lr="3e-4", bs=32, extra="--weight-decay 5e-4 --flip-aug True"),
    fm("A22_vit_aug_s2", 2, arch="vit", lr="3e-4", bs=32, extra="--weight-decay 5e-4 --flip-aug True"),
]
# A23 ViT ERP-PE variants
for pe in ("fourier", "sh", "both"):
    for s in (0, 1, 2):
        JOBS.append(fm(f"A23_vit_{pe}_s{s}", s, arch="vit", lr="3e-4", bs=32,
                       extra=f"--weight-decay 5e-4 --flip-aug True --vit-pe {pe} --sh-order 6 --fourier-bands 6"))
# prob heads
JOBS += [prob("P_k1", 1), prob("P_k5", 5), prob("P_k10", 10)]

# ---------- RIR feature runs (in_ch 3/5) + A14 (depends on A9) ----------
JOBS += [
    fm("A13_mag2_s0", 0, extra="--correction none", in_ch=2),
    fm("A13_ild3_s0", 0, extra="--correction none", in_ch=3),
    fm("A13_ipd5_s0", 0, extra="--correction none", in_ch=5),
    fm("A13_ipd5_s1", 0, extra="--correction none", in_ch=5),
    fm("A13_ipd5_s2", 2, extra="--correction none", in_ch=5),
    fm("A14_logmag_s0", 0, lr="1e-3", extra="--correction cross_sup --init-decoder A9_fullmap_s0",
       deps=["A9_fullmap_s0"]),
    fm("A14_logmag_s1", 1, lr="1e-3", extra="--correction cross_sup --init-decoder A9_fullmap_s1",
       deps=["A9_fullmap_s1"]),
    fm("A14_frozen_s0", 0, lr="1e-3",
       extra="--correction cross_sup --init-decoder A9_fullmap_s0 --freeze-decoder True",
       deps=["A9_fullmap_s0"]),
    fm("A14_rir5_s0", 0, lr="2e-3", in_ch=5,
       extra="--correction cross_sup --chan-norm True"),
]

# ---- controlled comparison: baseline-faithful U-Net vs our best (cross),
#      both radial + NO input log + masked L1; U-Net is 8-down (1x2 global bottleneck) ----
for s in (0, 1, 2):
    JOBS.append(fm(f"B_unet8nolog_s{s}", s, arch="unet", bs=64,
                   extra="--unet-downs 8 --log-spec False"))
    JOBS.append(imp(f"B_cross_nolog_s{s}", "cross", s, lr="3e-4", extra="--log-spec False"))

# ---- node-A PRIORITY set (distinct 'Bnode2_' names so they don't clash with node B's B_*) ----
# Run the comparison FIRST: baseline-faithful U-Net (8-down, no-log) vs our best (cross)
# under matched settings — cross_nolog (matched to U-Net) and cross_flip (matched to ViT:
# log + flip-aug, isolates whether ViT's edge is just augmentation).
PRIORITY = []
for s in (0, 1, 2):
    PRIORITY.append(fm(f"Bnode2_unet8nolog_s{s}", s, arch="unet", bs=64,
                       extra="--unet-downs 8 --log-spec False"))
    PRIORITY.append(imp(f"Bnode2_cross_nolog_s{s}", "cross", s, lr="3e-4", extra="--log-spec False"))
    PRIORITY.append(imp(f"Bnode2_cross_flip_s{s}", "cross", s, lr="3e-4", extra="--flip-aug True"))
# follow-up: give cross the winning ingredients — richer 5ch (phase/IPD) input, and
# band-limited hybrid (SH-coarse + implicit residual) to stop detail-chasing.
for s in (0, 1, 2):
    PRIORITY.append(imp(f"Bnode2_cross5ch_s{s}", "cross", s, lr="3e-4", in_ch=5))
    PRIORITY.append(imp(f"Bnode2_hybrid5ch_s{s}", "hybrid", s, lr="3e-4", in_ch=5))
# PRIORITY first, then the remaining main runs; drop the B_* (those belong to node B).
JOBS = PRIORITY + [j for j in JOBS if not j["name"].startswith("B_")]


def done(name):
    return os.path.exists(os.path.join("out", name, "train_done.json"))


def cache_ready(in_ch):
    d = f"/root/implicit_full_cache/ic{in_ch}_256x512"
    return all(os.path.exists(os.path.join(d, f"train_{k}.npy")) for k in ("spec", "depth", "mask"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    # kick off in_ch 3 & 5 cache builds in the background (CPU/NFS), GPUs run in_ch2 meanwhile
    cache_procs = []
    for ic in (3, 5):
        if not cache_ready(ic):
            lf = open(f"logs/_cache_ic{ic}.log", "w")
            cache_procs.append(subprocess.Popen(
                f"python build_fullcache.py --in-ch {ic} --num-workers 16 --splits val test train",
                shell=True, stdout=lf, stderr=subprocess.STDOUT))
    print(f"[sched] {len(JOBS)} jobs, gpus={gpus}, cache builds={len(cache_procs)}", flush=True)

    pending = {j["name"]: j for j in JOBS if not done(j["name"])}
    skipped = [j["name"] for j in JOBS if done(j["name"])]
    if skipped:
        print(f"[sched] {len(skipped)} already done, skipping", flush=True)
    running = {}   # gpu -> (proc, name, t0)
    free = list(gpus)

    while pending or running:
        # reap finished
        for g in list(running):
            proc, name, t0 = running[g]
            if proc.poll() is not None:
                ok = done(name)
                print(f"[sched] {'DONE' if ok else 'EXIT(rc=%d)' % proc.returncode} {name} "
                      f"({time.time()-t0:.0f}s) -> gpu{g} free", flush=True)
                running.pop(g); free.append(g)
        # dispatch
        for g in list(free):
            cand = None
            for name, j in pending.items():
                if all(done(d) for d in j["deps"]) and cache_ready(j["in_ch"]):
                    cand = j; break
            if cand is None:
                break
            pending.pop(cand["name"]); free.remove(g)
            lf = open(os.path.join("logs", cand["name"] + ".log"), "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            proc = subprocess.Popen(cand["cmd"], shell=True, stdout=lf,
                                    stderr=subprocess.STDOUT, env=env)
            running[g] = (proc, cand["name"], time.time())
            print(f"[sched] LAUNCH {cand['name']} on gpu{g}  ({len(pending)} pending, "
                  f"{len(running)} running)", flush=True)
        time.sleep(10)

    for p in cache_procs:
        p.wait()
    print("[sched] ALL JOBS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
