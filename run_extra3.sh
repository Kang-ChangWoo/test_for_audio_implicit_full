#!/bin/bash
# Front-strengthening cross (sector-weighted loss + hi-res tokens) via an idle-GPU
# pool — starts on free GPUs immediately (coexists with run_extra2), expands as
# GPUs free. Kept as run_extra3.sh so run_extra4's dependency wait still holds.
cd "$(dirname "$0")"
python front_pool.py >> logs/_extra3.log 2>&1
