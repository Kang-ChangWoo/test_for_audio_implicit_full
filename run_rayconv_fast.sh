#!/bin/bash
cd "$(dirname "$0")"
u(){ g=$1; n=$2; s=$3
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train_fullmap.py --arch rayconv --run-name $n --seed $s --epochs 25 --batch-size 16 \
      --num-workers 6 --lr 2e-3 --in-ch 5 --coarse-h 16 --coarse-w 32 --flip-aug True \
    && python eval_fullmap.py --run-name $n --controls True" > logs/$n.log 2>&1 &
  echo "launched $n gpu$g pid $!"; }
u 0 Bnode2_rayconv5_s0 0
u 1 Bnode2_rayconv5_s1 1
u 2 Bnode2_rayconv5_s2 2
wait; echo "RAYCONV5 FAST DONE $(date)"
