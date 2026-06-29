"""Unified idle-GPU pool: runs ALL remaining experiments across every free GPU.
Replaces the serial run_extra*/front_pool chain. Done-check skips finished runs
(metrics_test.json / prob_eval.json); cache-check defers jobs whose local cache
isn't built yet. One job per idle GPU (mem<1500MiB). Each job = train then eval.
"""
import os, subprocess, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
CK = "/root/implicit_full_cache"
EP = "--epochs 25 --num-workers 6"

def imp(name, model, seed, lr, extra, bs, cache=None, ev="--controls True"):
    cmd = (f"python train.py --model {model} --run-name {name} --seed {seed} {EP} "
           f"--batch-size {bs} --n-rays 2048 --lr {lr} {extra} "
           f"&& python eval.py --run-name {name} {ev}")
    return dict(name=name, cmd=cmd, cache=cache, art="metrics_test.json")

def fm(name, seed, arch, lr, extra, bs, cache=None):
    cmd = (f"python train_fullmap.py --arch {arch} --run-name {name} --seed {seed} {EP} "
           f"--batch-size {bs} --lr {lr} {extra} "
           f"&& python eval_fullmap.py --run-name {name} --controls True")
    return dict(name=name, cmd=cmd, cache=cache, art="metrics_test.json")

def prob(name, seed, extra, bs, cache=None):
    cmd = (f"python train_prob.py --run-name {name} --seed {seed} {EP} --batch-size {bs} "
           f"--lr 2e-3 {extra} && python eval_prob.py --run-name {name}")
    return dict(name=name, cmd=cmd, cache=cache, art="prob_eval.json")

IC5 = f"{CK}/ic5_256x512"; IC5W = f"{CK}/ic5_256x512_w20"; FOA = f"{CK}/ic4_256x512_foa"
IC_GCC = f"{CK}/ic6_256x512_gcc"; IC_WAVE = f"{CK}/ic5_256x512_wave"
JOBS = []
for s in (0, 1, 2):
    # --- front-strengthening (anti-discreteness) ---  (front-weighted-loss removed: no effect)
    JOBS += [imp(f"Bnode2_cross_hitok_s{s}", "cross", s, "3e-4", "--in-ch 2 --hi-tokens True", 12)]
    JOBS += [imp(f"Bnode2_cross_5chflip_s{s}", "cross", s, "3e-4", "--in-ch 5 --flip-aug True", 24, IC5)]
    # --- ViT encoder for cross / pix2pix U-Net encoder for cross (front-strong tokens) ---
    JOBS += [imp(f"Bnode2_cross_vitenc_s{s}", "cross", s, "3e-4", "--in-ch 2 --cross-enc vit --flip-aug True", 16)]
    JOBS += [imp(f"Bnode2_cross_unetenc_s{s}", "cross", s, "3e-4", "--in-ch 2 --cross-enc unet --ngf 64 --flip-aug True", 16)]
    JOBS += [imp(f"Bnode2_cross_unetenc5_s{s}", "cross", s, "3e-4", "--in-ch 5 --cross-enc unet --ngf 64 --flip-aug True", 16, IC5)]
    # --- #1 combo + #2 richer window (U-Net) ---
    JOBS += [fm(f"Bnode2_unet8_5chflip_s{s}", s, "unet", "2e-3", "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True", 48, IC5)]
    JOBS += [fm(f"Bnode2_unet8_5chflip_w20_s{s}", s, "unet", "2e-3", "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --audio-window-m 20", 48, IC5W)]
    # --- #3a uncertainty ---
    JOBS += [prob(f"P_5ch_k5_s{s}", s, "--prob-k 5 --in-ch 5", 32, IC5)]
    # --- aggregation suite ---
    JOBS += [imp(f"Bnode2_crossself_flip_s{s}", "crossself", s, "3e-4", "--in-ch 2 --flip-aug True", 16)]
    JOBS += [imp(f"Bnode2_cross_hitokflip_s{s}", "cross", s, "3e-4", "--in-ch 2 --hi-tokens True --flip-aug True", 12)]
    JOBS += [imp(f"Bnode2_crossself_hitokflip_s{s}", "crossself", s, "3e-4", "--in-ch 2 --hi-tokens True --flip-aug True", 8)]
# --- NEW: ray-sampling sweep (cross+flip), 2 seeds each ---
for s in (0, 1):
    for nr in (1024, 4096, 8192):
        bs = {1024: 24, 4096: 16, 8192: 8}[nr]
        JOBS.append(imp(f"Bnode2_cross_flip_nr{nr}_s{s}", "cross", s, "3e-4",
                        f"--in-ch 2 --flip-aug True --n-rays {nr}", bs))
# (ray-sampling jobs carry a 2nd --n-rays in extra; argparse last-wins so nr applies.)
# --- NEW: FOA (ambisonic, 4ch) richer input -- QUEUED LAST (runs only after all else) ---
for s in (0, 1, 2):
    JOBS.append(fm(f"Bnode2_foa_unet8_s{s}", s, "unet", "2e-3", "--ngf 64 --unet-downs 8 --in-ch 4 --audio-src foa --flip-aug True", 48, FOA))
    JOBS.append(imp(f"Bnode2_foa_cross_s{s}", "cross", s, "3e-4", "--in-ch 4 --audio-src foa --flip-aug True", 16, FOA))


# --- coarse-layout heads on U-Net8 encoder (band-limited; ray as 16x32 coarse field) ---
for s in (0, 1, 2):
    CL = "--in-ch 5 --unet-downs 8 --flip-aug True"
    JOBS.append(fm(f"C_unet8_coarse16_5chflip_s{s}", s, "unet_coarse", "2e-3", CL + " --coarse-head-h 16 --coarse-head-w 32", 48, IC5))
    JOBS.append(fm(f"C_unet8_coarse32_5chflip_s{s}", s, "unet_coarse", "2e-3", CL + " --coarse-head-h 32 --coarse-head-w 64", 48, IC5))
    JOBS.append(fm(f"C_unet8_sh4_5chflip_s{s}", s, "unet_sh", "2e-3", CL + " --coarse-sh-order 4", 48, IC5))
    JOBS.append(fm(f"C_unet8_sh6_5chflip_s{s}", s, "unet_sh", "2e-3", CL + " --coarse-sh-order 6", 48, IC5))
    JOBS.append(fm(f"C_unet8_raycoarse16_5chflip_s{s}", s, "unet_raycoarse", "2e-3", CL + " --ray-coarse-h 16 --ray-coarse-w 32", 32, IC5))
    JOBS.append(fm(f"C_unet8_coarseres_5chflip_s{s}", s, "unet_coarse_res", "2e-3", CL, 48, IC5))
    JOBS.append(fm(f"Bnode2_rayconv5d_s{s}", s, "rayconv", "2e-3", "--in-ch 5 --coarse-h 64 --coarse-w 128 --flip-aug True", 8, IC5))

# --- cross_align: high-res audio feature (e2 64x128) + ray cross-attn + conv smoothing ---
# Fixes ray "discreteness": each ray gets its own aligned local feature + neighbour
# coupling via conv, instead of only global tokens. Judge vs cross_flip / U-Net.
for s in (0, 1, 2):
    JOBS.append(fm(f"C_cross_align_5chflip_s{s}", s, "cross_align",
                   "3e-4", "--in-ch 5 --flip-aug True --ray-cross-layers 2", 24, IC5))

# --- richer-input bets: GCC-PHAT (waveform-derived ITD, 6ch) + raw-waveform WaveUNet ---
# GCC-PHAT recovers the fine binaural timing log-mag throws away (handedness/range);
# WaveUNet feeds the RAW waveform through a 1D-CNN global prior (EchoDiffusion-style).
for s in (0, 1, 2):
    JOBS.append(fm(f"Bnode2_gcc_unet8_s{s}", s, "unet", "2e-3",
                   "--ngf 64 --unet-downs 8 --in-ch 6 --audio-src gcc --flip-aug True", 48, IC_GCC))
    JOBS.append(fm(f"Bnode2_wave_unet8_s{s}", s, "wave", "2e-3",
                   "--ngf 64 --unet-downs 8 --in-ch 5 --audio-src wave --flip-aug True", 40, IC_WAVE))

# --- RayDPT: ray-conditioned multi-scale DPT decoder (global audio cross-attn at
# coarse tokens + e2 DPT skip + local spherical window attention) ---
for s in (0, 1, 2):
    JOBS.append(fm(f"C_raydpt_5chflip_s{s}", s, "raydpt", "3e-4",
                   "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2", 16, IC5))

# explicit front-of-queue ordering: just-added RayDPT runs FIRST, then the other
# richer-input / research-focus jobs, then everything else (stable within a rank).
FRONT = ["C_raydpt", "Bnode2_gcc_", "Bnode2_wave_", "C_cross_align",
         "C_unet8", "rayconv5d", "cross_unetenc"]
def _rank(n):
    for i, p in enumerate(FRONT):
        if p in n:
            return i
    return len(FRONT)
JOBS = sorted(JOBS, key=lambda j: _rank(j["name"]))      # stable sort preserves order within a rank


# optional split: restrict to a GPU subset and/or skip a name substring (run elsewhere)
ALLOW = set(int(x) for x in os.environ.get("MEGA_GPUS", "0,1,2,3,4,5,6,7").split(","))
SKIP = os.environ.get("MEGA_SKIP", "")
if SKIP:
    JOBS = [j for j in JOBS if SKIP not in j["name"]]


def done(j): return os.path.exists(os.path.join("out", j["name"], j["art"]))
def cache_ready(j): return j["cache"] is None or os.path.exists(j["cache"] + "/train_spec.npy")
def idle_gpus():
    o = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"]).decode()
    return [int(l.split(",")[0]) for l in o.strip().splitlines()
            if int(l.split(",")[1]) < 1500 and int(l.split(",")[0]) in ALLOW]


def main():
    running = {}
    print(f"[mega] {len(JOBS)} jobs total", flush=True)
    while True:
        for g in list(running):
            p, n = running[g]
            if p.poll() is not None:
                print(f"[mega] finished {n} gpu{g}", flush=True); running.pop(g)
        pend = [j for j in JOBS if not done(j) and j["name"] not in {n for _, n in running.values()} and cache_ready(j)]
        nd = [j for j in JOBS if not done(j)]
        if not nd and not running:
            print("[mega] ALL DONE", flush=True); break
        if pend:
            for g in idle_gpus():
                if g in running or not pend:
                    continue
                j = pend.pop(0)
                lf = open(f"logs/{j['name']}.log", "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                running[g] = (subprocess.Popen(j["cmd"], shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env), j["name"])
                print(f"[mega] launch {j['name']} gpu{g} ({len([x for x in JOBS if not done(x)])} left)", flush=True)
        time.sleep(20)
    subprocess.run("python agg_full.py > logs/_agg_mega.log 2>&1; python update_readme.py >> logs/_mega.log 2>&1", shell=True)
    print("[mega] aggregated", flush=True)


if __name__ == "__main__":
    main()
