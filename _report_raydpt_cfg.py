import json, glob, sys, statistics as st
cfg = sys.argv[1]   # raydpt_2ch_ray | raydpt_5ch_ray | raydpt_2ch_noray | raydpt_5ch_noray
S=0.86
def agg(pat):
    ds=[json.load(open(f))["test"] for f in sorted(glob.glob(pat))]; n=len(ds)
    if not n: return None,0
    return {k:(sum(d[k] for d in ds)/n, st.pstdev([d[k] for d in ds]) if n>1 else 0)
            for k in ['MAE_plain','RMSE','AbsRel','delta1']}, n
P,nP = agg(f"out/finalv2_{cfg}_s*/metrics_test.json")
# radial 기준 = RayDPT champion P_b3 (both-win 대표) + best-per-metric
R = json.load(open("out/P_b3_s0/metrics_test.json"))["test"]
print(f"=== RayDPT {cfg} : PLANAR(n={nP}) vs RADIAL(P_b3 champion) ===")
print(f"{'metric':10}{'planar':>16}{'radial(P_b3)':>14}{'planar÷.86':>12}   판정(scale-inv)")
for k,inv in [('MAE_plain',1),('RMSE',1),('AbsRel',1),('delta1',0)]:
    pm,ps=P[k]; rm=R[k]
    sn = f"{pm/S:.3f}" if k in ('MAE_plain','RMSE') else "—"
    if k in ('MAE_plain','RMSE'): note="스케일영향"
    else:
        better=(pm<rm) if inv else (pm>rm); note=("planar✅" if better else "radial✅")+" (진짜)"
    print(f"{k:10}{pm:9.4f}±{ps:.3f}{rm:>14.4f}{sn:>12}   {note}")
