import json, os, glob, collections, re
def load(run):
    p=f"out/{run}/metrics_test.json"
    return json.load(open(p))["test"] if os.path.exists(p) else None
def comp(d): return d["RMSE"]/1.6 + (1-d["delta1"])/0.46 + 0.3*d["AbsRel"]/0.4
def fam(base):
    ds=[load(os.path.basename(os.path.dirname(f))) for f in glob.glob(f"out/{base}_s*/metrics_test.json")]
    ds=[d for d in ds if d]
    if not ds: return None
    n=len(ds); mn=lambda k: sum(d[k] for d in ds)/n
    return dict(MAE=mn("MAE"),RMSE=mn("RMSE"),AbsRel=mn("AbsRel"),delta1=mn("delta1"),n=n)
def row(lab,base):
    d=fam(base)
    if not d: print(f"{lab:22} (미완)"); return
    print(f"{lab:22} MAE={d['MAE']:.4f} RMSE={d['RMSE']:.4f} AbsRel={d['AbsRel']:.4f} d1={d['delta1']:.3f} comp={comp(d):.3f} (n={d['n']})")
print("=== E22 3-seed 확정 + EMA/coarse-sa 2x2 factorial ===")
print("# comp=RMSE/1.6+(1-d1)/0.46+0.3*AbsRel/0.4 (낮을수록 우수)")
row("[neither] RayDPT+E2","R_raydpt_e2")
row("[EMA only]","Q6_emaonly")
row("[coarse-sa only]","Q6_csaonly")
row("[both] E22","Q5_e22_coarsesa")
row("(ref) U-Net8","Bnode2_unet8_5chflip")
