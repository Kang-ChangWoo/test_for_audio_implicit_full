#!/bin/bash
# Wait until the MAIN sweep (scheduler + eval daemon + all train/eval jobs on THIS
# node) finishes, then: aggregate -> update README best-model block -> commit + push.
# (B_ comparison runs on another node write to the shared out/, so agg_full/README
#  pick them up automatically if they're done by then.)
cd "$(dirname "$0")"
ML=logs/_finalize.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ML"; }
busy () {
  pgrep -f "scheduler.py"  >/dev/null && return 0
  pgrep -f "eval_daemon.py" >/dev/null && return 0
  pgrep -f "train_fullmap.py|train.py|train_prob.py" >/dev/null && return 0
  pgrep -f "eval_fullmap.py|eval_prob.py" >/dev/null && return 0
  pgrep -f " eval.py " >/dev/null && return 0
  return 1
}

say "finalize: waiting for main sweep to finish ..."
while busy; do sleep 60; done
sleep 20
say "main sweep done -> aggregate + update README + push"

python agg_full.py    > logs/_agg_final.log 2>&1
python update_readme.py >> "$ML" 2>&1

TOKEN=$(grep github.com ~/.git-credentials | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" \
    commit -q -m "Final results: README best-model + RESULTS_full (main sweep complete)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
    && git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" \
       push origin main >> "$ML" 2>&1 \
    && say "finalize: pushed to origin/main" \
    || say "finalize: nothing to commit / push skipped"

{ echo "ALL DONE $(date)"; echo "evaluated: $(ls out/*/metrics_test.json out/*/prob_eval.json 2>/dev/null|wc -l)"; } > logs/_ALL_DONE
say "finalize: complete"
