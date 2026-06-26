#!/bin/bash
# ERP-valid PE for ViT: fourier-xyz / SH / both, vs planar (A22). 3 seeds each.
# Same recipe as A22 (fine-tune, flip-aug, lr 3e-4). sh_order 6, fourier_bands 6.
run () { GPU=$1; RUN=$2; PE=$3; SEED=$4
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --arch vit --run-name $RUN \
    --vit-pe $PE --sh-order 6 --fourier-bands 6 --seed $SEED \
    --epochs 25 --batch-size 32 --lr 3e-4 --weight-decay 5e-4 --flip-aug True --num-workers 6 \
    > logs/$RUN.log 2>&1 &
  echo "launched $RUN (pe=$PE) GPU$GPU pid $!"; }
run 0 A23_vit_fourier_s0 fourier 0
run 1 A23_vit_fourier_s1 fourier 1
run 2 A23_vit_fourier_s2 fourier 2
run 3 A23_vit_sh_s0 sh 0
run 4 A23_vit_sh_s1 sh 1
run 5 A23_vit_sh_s2 sh 2
run 6 A23_vit_both_s0 both 0
run 7 A23_vit_both_s1 both 1
wait; echo WAVE1_DONE
run 0 A23_vit_both_s2 both 2
wait; echo ALL_VITPE_DONE
