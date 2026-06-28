#!/bin/bash
# Combine the winning levers + richer input, on the best base (8d global-bottleneck U-Net):
#  #1 Bnode2_unet8_5chflip      : 8d U-Net + 5ch(phase) + flip-aug          (untested combo of all 3 winners)
#  #2 Bnode2_unet8_5chflip_w20  : same + 20m audio window (richer input = later reflections)
# Queued after run_extra3(front_pool)+run_extra4(vitenc) so GPUs are free. eval via eval_fullmap.
cd "$(dirname "$0")"
ML=logs/_extra5.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ML"; }

say "extra5 queued — waiting for run_extra3 + run_extra4 to finish ..."
while pgrep -f run_extra3.sh >/dev/null 2>&1 || pgrep -f front_pool.py >/dev/null 2>&1 \
   || pgrep -f run_extra4.sh >/dev/null 2>&1; do sleep 60; done
while [ ! -f /root/implicit_full_cache/ic5_256x512_w20/train_spec.npy ]; do
  say "waiting for ic5_w20 cache ..."; sleep 60; done
say "clear — launching #1 (5ch+flip) and #2 (+20m window)"

u () { g=$1; n=$2; s=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train_fullmap.py --arch unet --run-name $n --seed $s --epochs 25 --batch-size 48 \
      --num-workers 6 --lr 2e-3 --ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True $* \
    && python eval_fullmap.py --run-name $n --controls True" > logs/$n.log 2>&1 &
  echo "launched $n gpu$g pid $!"; }

# #1: 8d U-Net + 5ch + flip (window 10, ic5 cache)
u 0 Bnode2_unet8_5chflip_s0 0
u 1 Bnode2_unet8_5chflip_s1 1
u 2 Bnode2_unet8_5chflip_s2 2
# #2: + 20m window (richer input, ic5_w20 cache)
u 3 Bnode2_unet8_5chflip_w20_s0 0 --audio-window-m 20
u 4 Bnode2_unet8_5chflip_w20_s1 1 --audio-window-m 20
u 5 Bnode2_unet8_5chflip_w20_s2 2 --audio-window-m 20
wait
say "extra5 jobs done — aggregate + README + push"
python agg_full.py > logs/_agg_extra5.log 2>&1
python update_readme.py >> "$ML" 2>&1
TOKEN=$(grep github.com ~/.git-credentials | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" \
    commit -q -m "Add 8d-U-Net + 5ch + flip combo and +20m-window (richer input)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
  && git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" \
     push origin main >> "$ML" 2>&1 && say "extra5 pushed" || say "extra5 nothing to push"
say "extra5 complete"
