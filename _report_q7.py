import json,glob,os
def comp(d): return d["RMSE"]/1.6+(1-d["delta1"])/0.46+0.3*d["AbsRel"]/0.4
def load(run):
    p=f"out/{run}/metrics_test.json"; return json.load(open(p))["test"] if os.path.exists(p) else None
UNET=dict(MAE=0.8933,RMSE=1.4361,AbsRel=0.5308,delta1=0.455); UNET["comp"]=comp(UNET)
print("=== Q7 RMSE-balanced RayDPT vs U-Net8 (RMSE 1.436 / comp %.3f) ==="%UNET["comp"])
print("# balanced 승리 = RMSE<1.436 AND d1>=0.455 AND comp<2.481")
rows=[]
for f in sorted(glob.glob("out/Q7_*/metrics_test.json")):
    run=os.path.basename(os.path.dirname(f)); d=load(run)
    win = "WIN" if (d["RMSE"]<1.436 and d["delta1"]>=0.455) else ""
    rows.append((comp(d),run,d,win))
for c,run,d,win in sorted(rows):
    print(f"{run:30} MAE={d['MAE']:.4f} RMSE={d['RMSE']:.4f} AbsRel={d['AbsRel']:.4f} d1={d['delta1']:.3f} comp={c:.3f} {win}")
print(f"완주 {len(rows)}/10")
