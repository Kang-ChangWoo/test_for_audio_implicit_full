#!/bin/bash
# Full-val re-run of the ngf96 pair (original recipe: lr 1e-3, batch 48, wd 1e-4,
# scale 0.1 e3). _fv suffix. 4 GPUs (0-3), 2 waves.
run () {
  GPU=$1; RUN=$2; ARCH=$3; SEED=$4; shift 4
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --arch $ARCH --run-name $RUN \
    --seed $SEED --ngf 96 --dim 256 --n-heads 8 --n-cross 2 \
    --coarse-h 8 --coarse-w 16 --ray-mod-scale 0.1 \
    --epochs 25 --batch-size 48 --lr 1e-3 --weight-decay 1e-4 --num-workers 6 "$@" \
    > logs/$RUN.log 2>&1 &
  echo "launched $RUN ($ARCH) on GPU$GPU (pid $!)"
}
run 0 A15_bigunet_fv_s0   unet        0
run 1 A16_raymod_fv_s0    unet_raymod 0
run 2 A15_bigunet_fv_s1   unet        1
run 3 A16_raymod_fv_s1    unet_raymod 1
wait; echo WAVE1_DONE
run 0 A15_bigunet_fv_s2   unet        2
run 1 A16_raymod_fv_s2    unet_raymod 2
wait
echo ALL_NGF96_FV_DONE
