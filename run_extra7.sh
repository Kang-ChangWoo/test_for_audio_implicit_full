#!/bin/bash
# Address cross's "discrete" look: rays attend EACH OTHER via ray self-attention
# (crossself) + the one winning lever (flip-aug). Never combined before.
#   Bnode2_crossself_flip : --model crossself --flip-aug True  (2ch log)
# Self-attention couples neighbouring rays -> smoother, spatially coherent field.
# Queued after run_extra6.
cd "$(dirname "$0")"
ML=logs/_extra7.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ML"; }

say "extra7 queued — waiting for run_extra6 ..."
while pgrep -f run_extra6.sh >/dev/null 2>&1; do sleep 60; done
say "clear — launching crossself + flip (3 seeds)"

j () { g=$1; n=$2; s=$3
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train.py --model crossself --run-name $n --seed $s --epochs 25 --batch-size 16 \
      --n-rays 2048 --num-workers 6 --lr 3e-4 --in-ch 2 --flip-aug True \
    && python eval.py --run-name $n --controls True" > logs/$n.log 2>&1 &
  echo "launched $n gpu$g pid $!"; }

j 0 Bnode2_crossself_flip_s0 0
j 1 Bnode2_crossself_flip_s1 1
j 2 Bnode2_crossself_flip_s2 2
wait
say "extra7 jobs done — aggregate + push"
python agg_full.py > logs/_agg_extra7.log 2>&1
python update_readme.py >> "$ML" 2>&1
TOKEN=$(grep github.com ~/.git-credentials | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" \
    commit -q -m "Add crossself + flip (ray self-attention for spatial coherence)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
  && git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" \
     push origin main >> "$ML" 2>&1 && say "extra7 pushed" || say "extra7 nothing to push"
say "extra7 complete"
