"""GPU-pool runner for the front-strengthening cross experiments.
Uses ONLY idle GPUs (mem < 1GB) so it coexists with run_extra2 — starts on the
free GPUs now and expands as others free up. Each job = train then eval inline.
Skips runs already evaluated. After all 6 done: aggregate + README + push."""
import os, subprocess, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

JOBS = []  # (name, bs, extra, seed)  -- extra may override --in-ch (argparse last-wins)
for s in (0, 1, 2):
    JOBS.append((f"Bnode2_cross_frontwt_s{s}", 24, "--front-back-w 2.0", s))
    JOBS.append((f"Bnode2_cross_hitok_s{s}", 12, "--hi-tokens True", s))
    JOBS.append((f"Bnode2_cross_5chflip_s{s}", 24, "--in-ch 5 --flip-aug True", s))


def done(n): return os.path.exists(f"out/{n}/metrics_test.json")
def idle_gpus():
    o = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits"]).decode()
    return [int(l.split(",")[0]) for l in o.strip().splitlines() if int(l.split(",")[1]) < 1000]


def launch(g, name, bs, extra, seed):
    cmd = (f"python train.py --model cross --run-name {name} --seed {seed} --epochs 25 "
           f"--batch-size {bs} --n-rays 2048 --num-workers 6 --lr 3e-4 --in-ch 2 {extra} "
           f"&& python eval.py --run-name {name} --controls True")
    lf = open(f"logs/{name}.log", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env)
    print(f"[{time.strftime('%H:%M:%S')}] launched {name} gpu{g} pid {p.pid}", flush=True)
    return p


def main():
    pending = [j for j in JOBS if not done(j[0])]
    running = {}  # gpu -> (proc, name)
    print(f"[front_pool] {len(pending)} jobs (idle-GPU pool)", flush=True)
    while pending or running:
        for g in list(running):
            p, n = running[g]
            if p.poll() is not None:
                print(f"[front_pool] {'OK' if done(n) else 'EXIT'} {n} gpu{g}", flush=True)
                running.pop(g)
        if pending:
            for g in idle_gpus():
                if g in running or not pending:
                    continue
                name, bs, extra, seed = pending.pop(0)
                running[g] = (launch(g, name, bs, extra, seed), name)
        time.sleep(20)
    print("[front_pool] all done -> aggregate + push", flush=True)
    subprocess.run("python agg_full.py > logs/_agg_extra3.log 2>&1", shell=True)
    subprocess.run("python update_readme.py >> logs/_extra3.log 2>&1", shell=True)
    push = ("TOKEN=$(grep github.com ~/.git-credentials|head -1|sed -E 's#^https?://##;s#@github.com.*##;s#^[^:]*:##'); "
            "git add -A && git -c user.name=Kang-ChangWoo -c user.email=branden.c.w.kang@gmail.com "
            "commit -q -m 'Add front-strengthening cross (sector-weighted loss, hi-res tokens)\n\n"
            "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>' "
            "&& git -c credential.helper='!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f' "
            "push origin main >> logs/_extra3.log 2>&1")
    subprocess.run(push, shell=True, executable="/bin/bash")
    print("[front_pool] complete", flush=True)


if __name__ == "__main__":
    main()
