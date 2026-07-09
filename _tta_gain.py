import torch
from eval_fullmap import load, evrun
from data import make_loader
run="finalv3_raydpt_2ch_champ_s0"; dev="cuda"
m,cfg,extra=load(f"out/{run}", dev)
te=make_loader(cfg,"test",shuffle=False)
for tta in [False, True]:
    cfg.eval_tta_flip=tta
    r=evrun(m, te, cfg, extra, dev)
    print(f"  TTA={str(tta):5} : MAE_plain={r['MAE_plain']:.4f} RMSE={r['RMSE']:.4f} AbsRel={r['AbsRel']:.4f} d1={r['delta1']:.3f}")
