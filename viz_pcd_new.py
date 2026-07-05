"""~10 point clouds from best-metric models -> out/pcd_new/*.ply"""
import os
import eval_fullmap as ev_fm
import viz_contenders as vc
from viz_contenders import preds_for, best_seed, DEV
from save_pcd import erp_dirs, save_ply

vc.WANT = {"val": 5, "test": 5}   # +10 val/test scenes
PCD = "out/pcd_new"; os.makedirs(PCD, exist_ok=True)
# best-metric models
MODELS = [
    ("E22",          "Q5_e22_coarsesa"),     # best honest comp
    ("coarse-sa",    "Q6_csaonly"),          # best d1
    ("UNet8_normal", "U_unet8_normal"),      # best RMSE
    ("GCC_UNet8",    "Bnode2_gcc_unet8"),    # best MAE
]

def main():
    cfg = ev_fm.load(f"out/{best_seed('Bnode2_unet8_5chflip')}", DEV)[1]
    scenes = vc.pick_scenes(cfg)
    H, W = cfg.img_h, cfg.img_w; dirs = erp_dirs(H, W)
    n = 0
    for s in scenes:                                       # GT clouds
        tag = f"{s['split']}_{s['key'].replace('/','-')}"
        save_ply(f"{PCD}/{tag}__GT.ply", s["gt"], dirs, s["vmax"]); n += 1
    for lab, base in MODELS:
        r = best_seed(base)
        if r is None: print(f"[skip] {base}"); continue
        preds = preds_for(r, "fm", scenes)
        for s in scenes:
            tag = f"{s['split']}_{s['key'].replace('/','-')}"
            save_ply(f"{PCD}/{tag}__{lab}.ply", preds[s["key"]], dirs, s["vmax"]); n += 1
        print(f"[pcd] {lab} <- {r}", flush=True)
    print(f"[done] {n} PLY -> {PCD}", flush=True)

if __name__ == "__main__": main()
