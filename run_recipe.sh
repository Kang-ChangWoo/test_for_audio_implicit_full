#!/bin/bash
# Recipe fix for the diagnosed OVERFIT (ngf96 val rises after ~ep10): go back to
# right-size ngf=64 + stronger weight decay, and test whether ray modulation adds
# value on a baseline that is NOT overfitting. Uses the free GPUs 2,3,6,7.
# Recipe matches the original Aunet (lr 2e-3, batch 64) + wd 5e-4.
run () {
  GPU=$1; RUN=$2; ARCH=$3; SEED=$4; shift 4
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --arch $ARCH --run-name $RUN \
    --seed $SEED --ngf 64 --dim 256 --n-heads 8 --n-cross 2 \
    --coarse-h 8 --coarse-w 16 --ray-mod-scale 0.1 \
    --epochs 25 --batch-size 64 --lr 2e-3 --weight-decay 5e-4 --num-workers 6 "$@" \
    > logs/$RUN.log 2>&1 &
  echo "launched $RUN ($ARCH) on GPU$GPU (pid $!)"
}
run 2 A18_unet64reg_s0    unet         0
run 3 A18_raymod64reg_s0  unet_raymod  0
run 6 A18_unet64reg_s1    unet         1
run 7 A18_raymod64reg_s1  unet_raymod  1
wait
echo "ALL_RECIPE_DONE"
