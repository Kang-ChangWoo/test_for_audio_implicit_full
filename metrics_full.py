import sys, json, torch
from eval_fullmap import load, evrun
from data import make_loader
run = sys.argv[1]
device = torch.device("cuda")
try:
    m, cfg, extra = load(f"out/{run}", device)
    loader = make_loader(cfg, "test", shuffle=False)
    r = evrun(m, loader, cfg, extra, device)
    json.dump({"run": run, **r}, open(f"out/{run}/metrics_ext.json", "w"), indent=2)
    print(f"{run} OK MAE_plain={r['MAE_plain']:.4f} AbsRel={r['AbsRel']:.4f} d1={r['delta1']:.3f} d2={r['delta2']:.3f} d3={r['delta3']:.3f}", flush=True)
except Exception as e:
    print(f"{run} FAIL {repr(e)[:150]}", flush=True)
