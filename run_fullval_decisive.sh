#!/bin/bash
# Re-run the decisive ngf64 comparison with FULL-VAL early-stopping (N_VAL=None).
# Same recipe as A18/A19 (lr 2e-3, batch 64, wd 5e-4). _fv suffix so old (subset
# early-stop) results are kept for side-by-side. 4 GPUs (0-3), 2 waves.
run () {
  GPU=$1; RUN=$2; ARCH=$3; SEED=$4; shift 4
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --arch $ARCH --run-name $RUN \
    --seed $SEED --ngf 64 --dim 256 --n-heads 8 --n-cross 2 \
    --coarse-h 8 --coarse-w 16 --epochs 25 --batch-size 64 --lr 2e-3 \
    --weight-decay 5e-4 --num-workers 6 "$@" > logs/$RUN.log 2>&1 &
  echo "launched $RUN ($ARCH) on GPU$GPU (pid $!)"
}
# wave 1
run 0 A18_unet64reg_fv_s0    unet        0
run 1 A18_unet64reg_fv_s1    unet        1
run 2 A19_raymodStrong_fv_s0 unet_raymod 0 --ray-mod-scale 0.4 --ray-mod-stage e2+e3
run 3 A19_raymodStrong_fv_s1 unet_raymod 1 --ray-mod-scale 0.4 --ray-mod-stage e2+e3
wait; echo WAVE1_DONE
# wave 2 (seed 2)
run 0 A18_unet64reg_fv_s2    unet        2
run 1 A19_raymodStrong_fv_s2 unet_raymod 2 --ray-mod-scale 0.4 --ray-mod-stage e2+e3
wait
echo ALL_FULLVAL_DONE
