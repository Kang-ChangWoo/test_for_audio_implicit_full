import json, glob
def agg(pat):
    ds=[json.load(open(f))['test'] for f in sorted(glob.glob(pat))]; n=len(ds)
    if not n: return None
    return {k:sum(d[k] for d in ds)/n for k in ['RMSE','AbsRel','delta1','delta2','delta3']},n
rows=[
 ('BatVision','out/finalv2_batvision_2ch_s*/metrics_test.json'),
 ('pretrained UNet','out/finalv2_preunet_2ch_s*/metrics_test.json'),
 ('pretrained ViT','out/finalv2_previt_2ch_s*/metrics_test.json'),
 ('EchoDiffusion','out/finalv2_echodiff_2ch_s*/metrics_test.json'),
 ('RayDPT 2ch (ray)','out/finalv2_raydpt_2ch_ray_s*/metrics_test.json'),
 ('RayDPT 5ch (ray)','out/finalv2_raydpt_5ch_ray_s*/metrics_test.json'),
 ('RayDPT 2ch champ+TTA','out/finalv3_raydpt_2ch_champ_s*/metrics_test.json'),
 ('RayDPT 2ch champ+E51','out/finalv3_raydpt_2ch_champ_e51_s*/metrics_test.json'),
 ('RayDPT 5ch champ','out/finalv3_raydpt_5ch_champ_s*/metrics_test.json'),
 ('RayDPT 2ch g8clean','out/finalv3_raydpt_2ch_g8clean_s*/metrics_test.json'),
 ('RayDPT 2ch g8global(+e8)','out/finalv3_raydpt_2ch_g8global_s*/metrics_test.json'),
 ('RayDPT mres champ','out/finalv3_raydpt_mres_champ_s*/metrics_test.json'),
 ('RayDPT mres champ+E51','out/finalv3_raydpt_mres_champ_e51_s*/metrics_test.json'),
]
print('=== FULL TABLE (PLANAR, n=3) — RMSE/AbsRel/delta ===')
print('%-26s %7s %7s %6s %6s %6s' % ('model','RMSE','AbsRel','d1','d2','d3'))
print('-'*62)
for name,pat in rows:
    r=agg(pat)
    if not r: print('%-26s (미완)'%name); continue
    m,n=r
    print('%-26s %7.3f %7.3f %6.3f %6.3f %6.3f' % (name,m['RMSE'],m['AbsRel'],m['delta1'],m['delta2'],m['delta3']))
