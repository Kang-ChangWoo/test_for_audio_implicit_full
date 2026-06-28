#!/bin/bash
# Priority fast lane: cross + 8d-pix2pix-encoder + 5ch + flip, dedicated GPUs 0,1,2.
cd "$(dirname "$0")"
u(){ g=$1; n=$2; s=$3
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train.py --model cross --run-name $n --seed $s --epochs 25 --batch-size 16 \
      --n-rays 2048 --num-workers 6 --lr 3e-4 --in-ch 5 --cross-enc unet --ngf 64 --flip-aug True \
    && python eval.py --run-name $n --controls True" > logs/$n.log 2>&1 &
  echo "launched $n gpu$g pid $!"; }
u 0 Bnode2_cross_unetenc5_s0 0
u 1 Bnode2_cross_unetenc5_s1 1
u 2 Bnode2_cross_unetenc5_s2 2
wait
echo "UNETENC5 FAST DONE $(date)"
