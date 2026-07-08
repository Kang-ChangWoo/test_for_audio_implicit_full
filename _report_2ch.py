import json, glob, statistics as st
pairs=[("BatVision","finalv2_batvision_2ch","B2_batvis"),
       ("pretrained UNet","finalv2_preunet_2ch","B2_presnet"),
       ("pretrained ViT","finalv2_previt_2ch","B2_pvit")]
def agg(pat):
    fs=sorted(glob.glob(pat)); ds=[json.load(open(f))["test"] for f in fs]; n=len(ds)
    if not n: return None,0
    return {k:(sum(d[k] for d in ds)/n, st.pstdev([d[k] for d in ds]) if n>1 else 0) for k in
            ['MAE_plain','AbsRel','delta1']}, n
print("=== planar(finalv2) vs radial — scale-invariant(AbsRel/δ1)이 진짜 비교 ===")
print(f"{'model':16}{'MAE_plain(pl/ra)':>22}{'AbsRel(pl/ra)':>20}{'δ1(pl/ra)':>18}")
for name,pl,ra in pairs:
    P,nP=agg(f"out/{pl}_s*/metrics_test.json"); R,nR=agg(f"out/{ra}_s*/metrics_test.json")
    if not P: print(f"{name:16}  (planar 미완)"); continue
    mp=f"{P['MAE_plain'][0]:.3f}/{R['MAE_plain'][0]:.3f}"
    ar=f"{P['AbsRel'][0]:.3f}/{R['AbsRel'][0]:.3f}"
    d1=f"{P['delta1'][0]:.3f}/{R['delta1'][0]:.3f}"
    inv = "planar✅" if P['AbsRel'][0]<R['AbsRel'][0] else "radial"
    print(f"{name:16}{mp:>22}{ar:>20}{d1:>18}  AbsRel:{inv}")
    print(f"{'':16}  scale환산 MAE {P['MAE_plain'][0]/0.86:.3f} vs radial {R['MAE_plain'][0]:.3f}  (n_pl={nP})")
