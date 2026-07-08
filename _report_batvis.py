import json, glob, statistics as st
def agg(pat):
    ds=[json.load(open(f))["test"] for f in sorted(glob.glob(pat))]
    n=len(ds)
    return {k:(sum(d[k] for d in ds)/n, st.pstdev([d[k] for d in ds]) if n>1 else 0) for k in
            ['MAE_plain','MAE','RMSE','AbsRel','delta1']}, n
pl,npl = agg("out/finalv2_batvision_2ch_s*/metrics_test.json")
ra,nra = agg("out/B2_batvis_s*/metrics_test.json")
print(f"=== BatVision: planar(finalv2, n={npl}) vs radial(n={nra}) ===")
print(f"{'metric':12}{'planar':>16}{'radial':>16}   판정")
for k,inv in [('MAE_plain',1),('MAE',1),('RMSE',1),('AbsRel',1),('delta1',0)]:
    pm,ps=pl[k]; rm,rs=ra[k]
    if k in ('MAE_plain','MAE','RMSE'):
        note="스케일영향(0.86x)-직접비교X"
    else:
        better = (pm<rm) if inv else (pm>rm)
        note=("planar 우위 ✅" if better else "radial 우위") + "  (scale-invariant=진짜비교)"
    print(f"{k:12}{pm:8.4f}±{ps:.3f}{rm:9.4f}±{rs:.3f}   {note}")
# 스케일 보정 MAE (planar/0.86 로 radial-스케일 환산)
pmp=pl['MAE_plain'][0]; print(f"\nplanar MAE_plain {pmp:.4f} / 0.86 = {pmp/0.86:.4f} (radial-스케일 환산) vs radial {ra['MAE_plain'][0]:.4f}")
