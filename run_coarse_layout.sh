#!/bin/bash
# Coarse-layout heads on the strong U-Net8 (+5ch+flip) encoder vs the dense-decoder baseline.
# Goal: keep the winning encoder, replace the dense per-pixel decoder with a band-limited
# coarse/SH/ray-coarse head. Judge by layout metrics (MAE_low, SHcoefL1, sector MAE), not
# only MAE_plain. 3 seeds each; same in_ch=5 / flip / unet_downs=8 as the baseline.
# (In practice these run via mega_pool.py across all GPUs; this script is the explicit recipe.)
cd "$(dirname "$0")"
COMMON="--in-ch 5 --unet-downs 8 --flip-aug True --epochs 25 --num-workers 6 --lr 2e-3 --ngf 64"
run () { g=$1; arch=$2; name=$3; seed=$4; bs=$5; shift 5
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train_fullmap.py --arch $arch --run-name $name --seed $seed $COMMON --batch-size $bs $* \
    && python eval_fullmap.py --run-name $name --controls True" > logs/$name.log 2>&1 &
  echo "launched $name ($arch) gpu$g pid $!"; }

for s in 0 1 2; do
  run $((s))   unet_coarse     C_unet8_coarse16_5chflip_s$s   $s 48 --coarse-head-h 16 --coarse-head-w 32
  run $((s+3)) unet_coarse     C_unet8_coarse32_5chflip_s$s   $s 48 --coarse-head-h 32 --coarse-head-w 64
  wait
  run $((s))   unet_sh         C_unet8_sh4_5chflip_s$s        $s 48 --coarse-sh-order 4
  run $((s+3)) unet_sh         C_unet8_sh6_5chflip_s$s        $s 48 --coarse-sh-order 6
  wait
  run $((s))   unet_raycoarse  C_unet8_raycoarse16_5chflip_s$s $s 32 --ray-coarse-h 16 --ray-coarse-w 32
  run $((s+3)) unet_coarse_res C_unet8_coarseres_5chflip_s$s  $s 48
  wait
done
echo "COARSE_LAYOUT DONE"
python agg_full.py > logs/_agg_coarse.log 2>&1
