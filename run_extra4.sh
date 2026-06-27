#!/bin/bash
# cross with a pretrained ViT-B/16 ENCODER (ViT patch tokens as ray-attention keys/values)
# vs the conv-encoder cross. Matched to cross_flip (2ch, log, +flip-aug) so the only diff
# is the encoder. Tests whether ViT's richer tokens help once flip-aug is equalised
# (latest result: ViT's edge was flip-aug, not pretrain — this checks the token side).
# Queued LAST: waits for scheduler + run_extra2 + run_extra3.
cd "$(dirname "$0")"
ML=logs/_extra4.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ML"; }

say "extra4 queued — waiting for scheduler + run_extra2 + run_extra3 ..."
while pgrep -f scheduler.py >/dev/null 2>&1 || pgrep -f run_extra2.sh >/dev/null 2>&1 \
   || pgrep -f run_extra3.sh >/dev/null 2>&1; do sleep 60; done
say "clear — launching cross+ViT-encoder (3 seeds, gpu0-2)"

job () { g=$1; n=$2; s=$3
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train.py --model cross --run-name $n --seed $s --epochs 25 --batch-size 16 \
      --n-rays 2048 --num-workers 6 --lr 3e-4 --in-ch 2 --cross-enc vit --flip-aug True \
    && python eval.py --run-name $n --controls True" > logs/$n.log 2>&1 &
  echo "launched $n gpu$g pid $!"; }

job 0 Bnode2_cross_vitenc_s0 0
job 1 Bnode2_cross_vitenc_s1 1
job 2 Bnode2_cross_vitenc_s2 2
wait
say "extra4 jobs done — aggregate + README + push"
python agg_full.py > logs/_agg_extra4.log 2>&1
python update_readme.py >> "$ML" 2>&1
TOKEN=$(grep github.com ~/.git-credentials | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" \
    commit -q -m "Add cross with pretrained ViT-B/16 encoder (ViT tokens as ray K/V)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
  && git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" \
     push origin main >> "$ML" 2>&1 && say "extra4 pushed" || say "extra4 nothing to push"
say "extra4 complete"
