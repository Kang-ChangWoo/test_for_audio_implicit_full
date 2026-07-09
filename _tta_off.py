import torch
from eval_fullmap import load, evrun
from data import make_loader
m,cfg,extra=load("out/finalv3_raydpt_2ch_champ_s0","cuda")
te=make_loader(cfg,"test",shuffle=False)
cfg.eval_tta_flip=False
r=evrun(m,te,cfg,extra,"cuda")
print(f"TTA=False: MAE_plain={r['MAE_plain']:.4f} RMSE={r['RMSE']:.4f} AbsRel={r['AbsRel']:.4f} d1={r['delta1']:.3f}")
