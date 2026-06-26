#!/bin/bash
# A15 vs A16: the capacity-vs-ray-modulation comparison.
#   A15_bigunet      strong plain U-Net (ngf=96)            -> capacity control
#   A16_raymod8x16   same U-Net + ray-conditioned FiLM      -> the claim
# A16 beating A15 (beyond seed std) is what licenses "ray queries add a directional
# inductive bias", not "bigger U-Net wins". args: GPU RUN ARCH SEED EXTRA...
run () {
  GPU=$1; RUN=$2; ARCH=$3; SEED=$4; shift 4
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --arch $ARCH --run-name $RUN \
    --seed $SEED --ngf 96 --dim 256 --n-heads 8 --n-cross 2 \
    --coarse-h 8 --coarse-w 16 --ray-mod-scale 0.1 \
    --epochs 25 --batch-size 48 --lr 1e-3 --weight-decay 1e-4 --num-workers 8 "$@" \
    > logs/$RUN.log 2>&1 &
  echo "launched $RUN ($ARCH) on GPU$GPU (pid $!)"
}

# Only GPUs 0-3 are free here (4-7 busy with other jobs), so run 1 job/GPU in
# two waves. Wave 1: seeds 0,1 of both arms (the core comparison).
run 0 A15_bigunet_s0     unet         0
run 1 A16_raymod8x16_s0  unet_raymod  0
run 2 A15_bigunet_s1     unet         1
run 3 A16_raymod8x16_s1  unet_raymod  1
wait
echo "WAVE1_DONE"
# Wave 2: seed 2 of both arms.
run 0 A15_bigunet_s2     unet         2
run 1 A16_raymod8x16_s2  unet_raymod  2
wait
echo "ALL_RAYMOD_DONE"
