#!/usr/bin/env python3
"""Node-B controlled-comparison launcher (4-GPU, queue-scheduled).

Runs the B_ comparison (baseline-faithful U-Net vs our best cross-attn implicit,
SAME no-log setting) PLUS node-1-finding add-ons layered on the SAME B_ base, to
verify whether they improve/worsen each architecture at full resolution:

  base      B_unet8nolog / B_cross_nolog            (no-log 2ch)
  +flip-aug B_unet8nolog_aug / B_cross_nolog_aug    (correct L/R mirror aug)
  +5ch/IPD  B_unet8_5ch / B_cross_5ch               (RIR ILD+IPD, in_ch=5)

3 seeds each -> 18 runs (9 unet ~45min, 9 cross ~6h). This node has 4 GPUs, so a
greedy queue keeps all 4 busy, cross(long)-first; the cheap U-Net jobs fill gaps.
Each job = train then eval (inline). 5ch jobs are gated on the ic5 no-log cache,
which is built in the background once the ic2 no-log cache is ready. When all 18
finish, agg_full.py rescans out/ and refreshes RESULTS_full.md.
"""
import os, sys, time, threading, subprocess, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
NGPU = 4
ML = open("logs/_extra_nodeB.master.log", "a", buffering=1)
CACHE = "/root/implicit_full_cache"
IC2 = f"{CACHE}/ic2_256x512_nolog/train_spec.npy"
IC5 = f"{CACHE}/ic5_256x512_nolog/train_spec.npy"


def say(*a):
    msg = f"[{datetime.datetime.now():%H:%M:%S}] " + " ".join(str(x) for x in a)
    print(msg, flush=True); ML.write(msg + "\n")


def unet_cmds(name, seed, in_ch, flip):
    tr = (f"python train_fullmap.py --arch unet --run-name {name} --seed {seed} "
          f"--epochs 25 --batch-size 64 --num-workers 6 --lr 2e-3 --in-ch {in_ch} "
          f"--unet-downs 8 --log-spec False"
          + (" --flip-aug True" if flip else ""))
    ev = f"python eval_fullmap.py --run-name {name} --controls True"
    return tr + " && " + ev


def cross_cmds(name, seed, in_ch, flip):
    tr = (f"python train.py --model cross --run-name {name} --seed {seed} "
          f"--epochs 25 --batch-size 24 --n-rays 2048 --num-workers 6 --lr 3e-4 "
          f"--in-ch {in_ch} --log-spec False"
          + (" --flip-aug True" if flip else ""))
    ev = f"python eval.py --run-name {name} --controls True"
    return tr + " && " + ev


# job = (priority, name, command, needs_ic5).  lower priority dispatched first;
# cross(0) before unet(1) so the 6h bottleneck starts as early as possible.
JOBS = []
for s in (0, 1, 2):
    JOBS.append((0, f"B_cross_nolog_s{s}",     cross_cmds(f"B_cross_nolog_s{s}",     s, 2, False), False))
    JOBS.append((0, f"B_cross_nolog_aug_s{s}", cross_cmds(f"B_cross_nolog_aug_s{s}", s, 2, True),  False))
    JOBS.append((0, f"B_cross_5ch_s{s}",       cross_cmds(f"B_cross_5ch_s{s}",       s, 5, False), True))
    JOBS.append((1, f"B_unet8nolog_s{s}",      unet_cmds(f"B_unet8nolog_s{s}",       s, 2, False), False))
    JOBS.append((1, f"B_unet8nolog_aug_s{s}",  unet_cmds(f"B_unet8nolog_aug_s{s}",   s, 2, True),  False))
    JOBS.append((1, f"B_unet8_5ch_s{s}",       unet_cmds(f"B_unet8_5ch_s{s}",        s, 5, False), True))
JOBS.sort(key=lambda j: j[0])

lock = threading.Lock()
pending = list(JOBS)          # not yet claimed
done = []
running_cross = 0            # live count of cross jobs on GPUs
MAXCROSS = 3                 # soft cap: keep >=1 GPU for cheap U-Net WHILE unet work remains


def ic5_ready():
    return os.path.exists(IC5)


def claim():
    """Return (job, reason). Cross is the 6h bottleneck; while any U-Net job is
    still runnable we cap concurrent cross at MAXCROSS so a GPU stays free for the
    cheap U-Net runs (early complete results, no makespan cost). Once U-Net work
    is exhausted the cap lifts and all GPUs pour into the remaining cross jobs."""
    global running_cross
    with lock:
        if not pending:
            return None, "empty"
        ready5 = ic5_ready()
        runnable = [(i, j) for i, j in enumerate(pending) if not (j[3] and not ready5)]
        if not runnable:
            return None, "blocked"        # only 5ch jobs left, ic5 cache not ready
        cross_ok = running_cross < MAXCROSS
        pick = None
        for i, j in runnable:             # pending is cross-first; honor cap
            if j[0] == 0 and not cross_ok:   # priority 0 == cross
                continue
            pick = (i, j); break
        if pick is None:                  # only cross runnable & cap hit -> lift cap
            pick = runnable[0]
        i, j = pick
        if j[0] == 0:
            running_cross += 1
        return pending.pop(i), "ok"


def worker(gpu):
    global running_cross
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        job, why = claim()
        if job is None:
            if why == "empty":
                return
            time.sleep(30)               # blocked on ic5 cache; retry
            continue
        prio, name, cmd, _ = job
        say(f"gpu{gpu} START {name}")
        t0 = time.time()
        with open(f"logs/{name}.log", "w") as lf:
            rc = subprocess.call(["bash", "-c", cmd], env=env, stdout=lf, stderr=subprocess.STDOUT)
        dt = (time.time() - t0) / 60.0
        with lock:
            done.append((name, rc))
            if prio == 0:
                running_cross -= 1
        say(f"gpu{gpu} DONE  {name} rc={rc} ({dt:.0f} min)")


def main():
    say("=== node-B extra launcher start (4 GPU, 18 runs) ===")
    # 1) wait for the ic2 no-log cache (gates everything)
    while not os.path.exists(IC2):
        say("waiting for ic2 no-log cache ..."); time.sleep(60)
    say("ic2 no-log cache ready")
    # 2) kick off ic5 no-log cache build in the background (for the 5ch add-on)
    if not ic5_ready():
        say("building ic5 no-log cache in background ...")
        subprocess.Popen(
            ["bash", "-c",
             "python build_fullcache.py --in-ch 5 --log-spec False --num-workers 24 "
             "--splits val test train > logs/_cache_ic5_nolog_nodeB.log 2>&1"])
    else:
        say("ic5 no-log cache already present")
    # 3) run the queue across 4 GPU workers
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in range(NGPU)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 4) aggregate
    say(f"all {len(done)} jobs finished; aggregating -> RESULTS_full.md")
    subprocess.call(["bash", "-c", "python agg_full.py > logs/_agg_extra_nodeB.log 2>&1"])
    fails = [n for n, rc in done if rc != 0]
    say(f"EXTRA DONE. failures={fails if fails else 'none'}")


if __name__ == "__main__":
    main()
