#!/bin/bash
# A22: pretrained ViT-B/16 audio->depth (model_vit.py), fine-tuned, with correct
# flip-aug + full-val. lr 3e-4 (ViT fine-tune). Compare to best so far (plain+aug 0.9130).
run () { GPU=$1; RUN=$2; SEED=$3; shift 3
  CUDA_VISIBLE_DEVICES=$GPU nohup python train_fullmap.py --arch vit --run-name $RUN \
    --seed $SEED --epochs 25 --batch-size 32 --lr 3e-4 --weight-decay 5e-4 \
    --flip-aug True --num-workers 6 "$@" > logs/$RUN.log 2>&1 &
  echo "launched $RUN GPU$GPU pid $!"; }
run 0 A22_vit_aug_s0 0
run 1 A22_vit_aug_s1 1
run 2 A22_vit_aug_s2 2
wait
echo ALL_VIT_DONE
