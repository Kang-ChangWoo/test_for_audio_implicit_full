#!/bin/bash
# Decisive strong-ray test on the NON-overfitting ngf=64 baseline (the only arm
# where ray showed a faint positive: A18_raymod64reg). Crank the modulation:
#   scale 0.1 -> 0.4,  single e3  ->  e2+e3 (both 16x32 and 8x16 skips).
# Same recipe as A18 (ngf64, lr 2e-3, batch 64, wd 5e-4). Baseline = A18_unet64reg
# (s2 added here so the comparison is a clean 3 seeds). GPUs 0-3.
run () {
  GPU=$1; RUN=$2; ARCH=$3; SEED=$4; shift 4
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --arch $ARCH --run-name $RUN \
    --seed $SEED --ngf 64 --dim 256 --n-heads 8 --n-cross 2 \
    --coarse-h 8 --coarse-w 16 --epochs 25 --batch-size 64 --lr 2e-3 \
    --weight-decay 5e-4 --num-workers 6 "$@" > logs/$RUN.log 2>&1 &
  echo "launched $RUN ($ARCH) on GPU$GPU (pid $!)"
}
# strong ray: scale 0.4, e2+e3
run 0 A19_raymodStrong_s0 unet_raymod 0 --ray-mod-scale 0.4 --ray-mod-stage e2+e3
run 1 A19_raymodStrong_s1 unet_raymod 1 --ray-mod-scale 0.4 --ray-mod-stage e2+e3
run 2 A19_raymodStrong_s2 unet_raymod 2 --ray-mod-scale 0.4 --ray-mod-stage e2+e3
# complete the matched plain-U-Net baseline to 3 seeds
run 3 A18_unet64reg_s2    unet        2
wait
echo "ALL_STRONG_DONE"
