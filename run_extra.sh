#!/bin/bash
# Controlled comparison (queued AFTER the main sweep so it doesn't fight for GPUs):
#   B_unet8nolog : baseline-faithful pix2pix U-Net  (radial depth, NO input log,
#                  8 downsamples -> 1x2 global bottleneck at 256x512, masked L1)
#   B_cross_nolog: our best model (cross-attn implicit) under the SAME setting (no log)
# Each job = train then eval (inline), so it does not depend on the eval daemon.
cd "$(dirname "$0")"
ML=logs/_extra.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ML"; }

say "EXTRA queued — waiting for main scheduler to finish (free GPUs) ..."
while pgrep -f scheduler.py >/dev/null 2>&1; do sleep 60; done
say "main scheduler done"
while [ ! -f /root/implicit_full_cache/ic2_256x512_nolog/train_spec.npy ]; do
  say "waiting for no-log cache ..."; sleep 60; done
say "no-log cache ready — launching 6 jobs"

fmjob () { g=$1; n=$2; s=$3
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train_fullmap.py --arch unet --run-name $n --seed $s --epochs 25 \
      --batch-size 64 --num-workers 6 --lr 2e-3 --in-ch 2 --unet-downs 8 --log-spec False \
    && python eval_fullmap.py --run-name $n --controls True" > logs/$n.log 2>&1 &
  echo "launched $n (unet8 nolog) gpu$g pid $!"; }

impjob () { g=$1; n=$2; s=$3
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train.py --model cross --run-name $n --seed $s --epochs 25 \
      --batch-size 24 --n-rays 2048 --num-workers 6 --lr 3e-4 --in-ch 2 --log-spec False \
    && python eval.py --run-name $n --controls True" > logs/$n.log 2>&1 &
  echo "launched $n (cross nolog) gpu$g pid $!"; }

fmjob 0 B_unet8nolog_s0 0
fmjob 1 B_unet8nolog_s1 1
fmjob 2 B_unet8nolog_s2 2
impjob 3 B_cross_nolog_s0 0
impjob 4 B_cross_nolog_s1 1
impjob 5 B_cross_nolog_s2 2
wait
say "EXTRA jobs trained+evaluated — aggregating"
python agg_full.py > logs/_agg_extra.log 2>&1
say "EXTRA DONE -> RESULTS_full.md updated (B_unet8nolog vs B_cross_nolog)"
