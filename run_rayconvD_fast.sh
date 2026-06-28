#!/bin/bash
# Dense ray grid (64x128 = 8192 points) rayconv + 5ch + flip, GPUs 0,1,2.
# Denser ray feature map -> conv only x4 upsample (vs x16 for the sparse 16x32).
cd "$(dirname "$0")"
u(){ g=$1; n=$2; s=$3
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train_fullmap.py --arch rayconv --run-name $n --seed $s --epochs 25 --batch-size 8 \
      --num-workers 6 --lr 2e-3 --in-ch 5 --coarse-h 64 --coarse-w 128 --flip-aug True \
    && python eval_fullmap.py --run-name $n --controls True" > logs/$n.log 2>&1 &
  echo "launched $n gpu$g pid $!"; }
u 0 Bnode2_rayconv5d_s0 0
u 1 Bnode2_rayconv5d_s1 1
u 2 Bnode2_rayconv5d_s2 2
wait
echo "RAYCONV5D FAST DONE $(date)"
