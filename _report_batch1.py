import json, glob, os
fams = {"batvision":"BatVision","preunet":"pretrained UNet","previt":"pretrained ViT"}
print("=== finalv2 첫 배치 결과 (planar depth, 수정된 metric, 3-seed) ===")
print(f"{'model':18}{'MAE_plain':>10}{'RMSE':>8}{'AbsRel':>8}{'d1':>7}  seeds")
for key,name in fams.items():
    ds=[]
    for f in sorted(glob.glob(f"out/finalv2_{key}_2ch_s*/metrics_test.json")):
        try: ds.append(json.load(open(f))["test"])
        except: pass
    if not ds: print(f"{name:18}  (미완)"); continue
    n=len(ds); avg={k:sum(d[k] for d in ds)/n for k in ['MAE_plain','RMSE','AbsRel','delta1']}
    print(f"{name:18}{avg['MAE_plain']:10.4f}{avg['RMSE']:8.4f}{avg['AbsRel']:8.4f}{avg['delta1']:7.3f}  n={n}")
