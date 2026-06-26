#!/bin/bash
cd "$(dirname "$0")"
echo "[chain] waiting for ALL_FULLVAL_DONE (ngf64) ..."
while ! grep -q ALL_FULLVAL_DONE logs/_fullval_launch.log 2>/dev/null; do sleep 15; done
echo "[chain] ngf64 done -> launching ngf96 full-val pair"
bash run_fullval_ngf96.sh
