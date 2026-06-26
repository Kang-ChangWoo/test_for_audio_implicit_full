#!/bin/bash
# Does correct L/R flip-augmentation help? A20/A21 = A18/A19 recipe + --flip-aug.
# Compare to no-aug full-val: A18_unet64reg_fv (0.9226), A19_raymodStrong_fv (0.9215).
# 6 GPUs, one wave. (Correct aug = depth/mask width-flip + L/R channel swap; NO spec flip.)
run () { GPU=$1; RUN=$2; ARCH=$3; SEED=$4; shift 4
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --arch $ARCH --run-name $RUN \
    --seed $SEED --ngf 64 --dim 256 --n-heads 8 --n-cross 2 --coarse-h 8 --coarse-w 16 \
    --epochs 25 --batch-size 64 --lr 2e-3 --weight-decay 5e-4 --flip-aug True --num-workers 6 "$@" \
    > logs/$RUN.log 2>&1 &
  echo "launched $RUN ($ARCH) GPU$GPU pid $!"; }
run 0 A20_unet64_aug_s0   unet        0
run 1 A20_unet64_aug_s1   unet        1
run 2 A20_unet64_aug_s2   unet        2
run 3 A21_raymodStrong_aug_s0 unet_raymod 0 --ray-mod-scale 0.4 --ray-mod-stage e2+e3
run 4 A21_raymodStrong_aug_s1 unet_raymod 1 --ray-mod-scale 0.4 --ray-mod-stage e2+e3
run 5 A21_raymodStrong_aug_s2 unet_raymod 2 --ray-mod-scale 0.4 --ray-mod-stage e2+e3
wait
echo ALL_FLIPAUG_DONE
