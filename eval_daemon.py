"""Auto-eval daemon: as each training run finishes (train_done.json), run its test
evaluation -> metrics_test.json (imp/fm) or prob_eval.json (prob). Runs concurrently
with the training scheduler; uses any GPU with >=20G free (one eval per GPU), so it
piggybacks on training GPUs safely and fans out once training frees them.

Exits when every trained run is evaluated AND no training is still running.

Run: setsid bash -c 'python eval_daemon.py > logs/_evald.log 2>&1' </dev/null &
"""
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

from scheduler import JOBS   # name -> kind/eval mapping

KIND = {j["name"]: j["kind"] for j in JOBS}
EVAL = {"imp": ("eval.py", "metrics_test.json", "--controls True"),
        "fm":  ("eval_fullmap.py", "metrics_test.json", "--controls True"),
        "prob":("eval_prob.py", "prob_eval.json", "")}

MIN_FREE_MIB = 20000


def trained(name):
    return os.path.exists(os.path.join("out", name, "train_done.json"))


def evaluated(name):
    _, art, _ = EVAL[KIND[name]]
    return os.path.exists(os.path.join("out", name, art))


def gpu_free():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]).decode()
    return {int(l.split(",")[0]): int(l.split(",")[1]) for l in out.strip().splitlines()}


def training_alive():
    try:
        subprocess.check_output(["pgrep", "-f", "scheduler.py"])
        return True
    except subprocess.CalledProcessError:
        pass
    try:
        subprocess.check_output(["pgrep", "-f", "train_fullmap.py|train.py|train_prob.py"])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    running = {}   # gpu -> (proc, name)
    print(f"[evald] start, {len(JOBS)} jobs known", flush=True)
    while True:
        # reap
        for g in list(running):
            proc, name = running[g]
            if proc.poll() is not None:
                print(f"[evald] {'OK' if evaluated(name) else 'FAIL(rc=%d)'%proc.returncode} "
                      f"eval {name} (gpu{g})", flush=True)
                running.pop(g)
        # candidates: trained, not evaluated, not currently evaluating
        inflight = {n for _, n in running.values()}
        todo = [j["name"] for j in JOBS
                if trained(j["name"]) and not evaluated(j["name"]) and j["name"] not in inflight]
        if not todo and not running and not training_alive():
            print("[evald] all trained runs evaluated; exiting", flush=True)
            break
        # dispatch onto GPUs with enough free memory (not already running our eval)
        if todo:
            free = gpu_free()
            for g, mib in sorted(free.items(), key=lambda x: -x[1]):
                if not todo:
                    break
                if g in running or mib < MIN_FREE_MIB:
                    continue
                name = todo.pop(0)
                script, art, ctrl = EVAL[KIND[name]]
                cmd = f"python {script} --run-name {name} {ctrl}".strip()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                lf = open(os.path.join("logs", f"eval_{name}.log"), "w")
                proc = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env)
                running[g] = (proc, name)
                print(f"[evald] EVAL {name} on gpu{g} (free {mib}MiB)  [{len(todo)} more queued]", flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
