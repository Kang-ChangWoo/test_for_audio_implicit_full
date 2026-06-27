#!/usr/bin/env python3
"""Node-B controlled comparison + n1 add-ons, SAFE (low-resource) launcher.

Why this file instead of run_extra_nodeB.py: the box reset under the original
launcher because it drove all 4 RTX-4090s (3 heavy cross trains + 1 U-Net) at
once *while* a 24-worker NFS cache build ran -> ~1.8 kW GPU + heavy CPU/IO, host
brown-out. We can't cap GPU power (no permission in this container), so the only
safety lever is concurrency. This launcher:

  * uses ONLY 2 GPUs concurrently (gpu0, gpu1); gpu2/gpu3 stay idle as headroom
    -> ~900 W sustained, roughly half the load that crashed the box.
  * num-workers=4 per job (was 6), no concurrent cache build (both ic2+ic5 nolog
    caches are already materialised).
  * staggers job starts by 20 s so two cold-start power spikes never coincide.
  * runs each job at nice -n 10 to keep the host responsive.
  * idempotent: skips any run whose out/<name>/metrics_test.json already exists,
    so it is safe to re-launch after an interruption.

Scope (user-confirmed): base + flip-aug + 5ch-IPD, both archs, 3 seeds = 18 runs.
(The n1 'time-window 2x' add-on has no flag in this codebase -> excluded.)

Each job = train then eval (inline). When all finish, agg_full.py rescans out/
and refreshes RESULTS_full.md.
"""
import os, time, threading, subprocess, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

NGPU = 2                      # only 2 of 4 GPUs concurrently (safety)
GPUS = [0, 1]
NWORK = 4                    # dataloader workers per job (was 6)
STAGGER = 20                 # seconds between worker cold-starts
CACHE = "/root/implicit_full_cache"
IC2 = f"{CACHE}/ic2_256x512_nolog/train_spec.npy"
IC5 = f"{CACHE}/ic5_256x512_nolog/train_spec.npy"
ML = open("logs/_extra_nodeB_safe.master.log", "a", buffering=1)


def say(*a):
    msg = f"[{datetime.datetime.now():%H:%M:%S}] " + " ".join(str(x) for x in a)
    print(msg, flush=True); ML.write(msg + "\n")


def done_already(name):
    return os.path.exists(os.path.join("out", name, "metrics_test.json"))


def unet_cmds(name, seed, in_ch, flip):
    tr = (f"nice -n 10 python train_fullmap.py --arch unet --run-name {name} --seed {seed} "
          f"--epochs 25 --batch-size 64 --num-workers {NWORK} --lr 2e-3 --in-ch {in_ch} "
          f"--unet-downs 8 --log-spec False" + (" --flip-aug True" if flip else ""))
    ev = f"nice -n 10 python eval_fullmap.py --run-name {name} --controls True"
    return tr + " && " + ev


def cross_cmds(name, seed, in_ch, flip):
    tr = (f"nice -n 10 python train.py --model cross --run-name {name} --seed {seed} "
          f"--epochs 25 --batch-size 24 --n-rays 2048 --num-workers {NWORK} --lr 3e-4 "
          f"--in-ch {in_ch} --log-spec False" + (" --flip-aug True" if flip else ""))
    ev = f"nice -n 10 python eval.py --run-name {name} --controls True"
    return tr + " && " + ev


# job = (is_cross, name, command, needs_ic5).  Cross is the ~5-6h bottleneck.
# Ordering: cross FIRST so the long-pole comparison (baseline U-Net vs best cross)
# starts at t=0 and its results stream in throughout, not only at the back half.
JOBS = []
for s in (0, 1, 2):
    JOBS.append((1, f"B_cross_nolog_s{s}",     cross_cmds(f"B_cross_nolog_s{s}",     s, 2, False), False))
    JOBS.append((1, f"B_cross_nolog_aug_s{s}", cross_cmds(f"B_cross_nolog_aug_s{s}", s, 2, True),  False))
    JOBS.append((1, f"B_cross_5ch_s{s}",       cross_cmds(f"B_cross_5ch_s{s}",       s, 5, False), True))
    JOBS.append((0, f"B_unet8nolog_s{s}",      unet_cmds(f"B_unet8nolog_s{s}",       s, 2, False), False))
    JOBS.append((0, f"B_unet8nolog_aug_s{s}",  unet_cmds(f"B_unet8nolog_aug_s{s}",   s, 2, True),  False))
    JOBS.append((0, f"B_unet8_5ch_s{s}",       unet_cmds(f"B_unet8_5ch_s{s}",        s, 5, False), True))
JOBS.sort(key=lambda j: -j[0])   # cross (1) first

lock = threading.Lock()
pending = list(JOBS)
done = []
skipped = []
running_cross = 0                # live count of heavy cross trains on GPUs
MAXCROSS = 1                     # while ANY U-Net job is still runnable, cap concurrent
                                 # cross at 1 -> the two GPUs run (1 heavy cross + 1 light
                                 # U-Net) instead of (2 heavy cross). Lower/asymmetric power
                                 # AND cross still starts immediately. Once U-Net work is
                                 # exhausted the cap lifts and both GPUs pour into cross.


def claim():
    """Pick the next job. Cross is dispatched first, but while a runnable U-Net job
    remains we keep concurrent cross <= MAXCROSS so we never run two heavy trains at
    once needlessly (safest power profile). Returns (job, is_cross) or (None, why)."""
    global running_cross
    with lock:
        if not pending:
            return None, "empty"
        ready5 = os.path.exists(IC5)
        runnable = [(i, j) for i, j in enumerate(pending) if not (j[3] and not ready5)]
        if not runnable:
            return None, "blocked"            # only ic5-gated jobs left, cache missing
        unet_left = any(j[0] == 0 for _, j in runnable)
        cross_ok = running_cross < MAXCROSS
        pick = None
        for i, j in runnable:                 # runnable is cross-first
            if j[0] == 1 and not cross_ok and unet_left:
                continue                      # hold this cross; a U-Net can pair instead
            pick = (i, j); break
        if pick is None:                      # only cross runnable & cap hit -> lift cap
            pick = runnable[0]
        i, j = pick
        if j[0] == 1:
            running_cross += 1
        return pending.pop(i), j[0]


def worker(gpu, delay):
    global running_cross
    time.sleep(delay)                         # stagger cold-starts
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        job, is_cross = claim()
        if job is None:
            if is_cross == "empty":
                return
            time.sleep(30); continue          # blocked on ic5 cache; retry
        _, name, cmd, _ = job
        try:
            if done_already(name):
                say(f"gpu{gpu} SKIP {name} (metrics_test.json exists)")
                with lock:
                    skipped.append(name)
                continue
            say(f"gpu{gpu} START {name}" + (" [cross]" if is_cross else " [unet]"))
            t0 = time.time()
            with open(f"logs/{name}.log", "w") as lf:
                rc = subprocess.call(["bash", "-c", cmd], env=env, stdout=lf, stderr=subprocess.STDOUT)
            dt = (time.time() - t0) / 60.0
            with lock:
                done.append((name, rc))
            say(f"gpu{gpu} DONE  {name} rc={rc} ({dt:.0f} min)")
        finally:
            if is_cross:                      # always release the cross slot
                with lock:
                    running_cross -= 1


def main():
    say(f"=== node-B SAFE launcher start ({NGPU} GPU {GPUS}, workers={NWORK}, 18 runs) ===")
    for f, tag in ((IC2, "ic2"), (IC5, "ic5")):
        say(f"{tag} no-log cache: {'present' if os.path.exists(f) else 'MISSING'}")
    if not os.path.exists(IC2):
        say("FATAL: ic2 no-log cache missing; aborting."); return
    threads = [threading.Thread(target=worker, args=(GPUS[i], i * STAGGER), daemon=True)
               for i in range(NGPU)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    say(f"all jobs settled: ran={len(done)} skipped={len(skipped)}; aggregating -> RESULTS_full.md")
    subprocess.call(["bash", "-c", "python agg_full.py > logs/_agg_extra_nodeB_safe.log 2>&1"])
    fails = [n for n, rc in done if rc != 0]
    say(f"SAFE EXTRA DONE. failures={fails if fails else 'none'} skipped={skipped}")


if __name__ == "__main__":
    main()
