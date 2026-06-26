#!/bin/bash
# Master orchestration for test_for_audio_implicit_full: run EVERY experiment from
# scratch at FULL resolution (256x512, radial depth, scene_split), reading the actual
# dataset files (materialised once into a LOCAL full-res cache for speed — NOT the
# old 64x128 low-res cache). Each run_*.sh manages its own GPU fan-out + `wait`, so
# we call them sequentially (no cross-script GPU oversubscription).
# Dependency order:
#   * in_ch=2 cache BEFORE Phase A
#   * a9plus BEFORE a14            (A14 warm-starts from out/A9_fullmap_s*)
#   * in_ch=3,5 caches BEFORE a13/a14/seeds (RIR feature runs)
cd "$(dirname "$0")"
ML=logs/_master.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ML"; }

say "MASTER START"

# ---- Phase 0: full-res LOCAL cache for in_ch=2 (idempotent; one-time NFS read) ----
say "ensuring in_ch=2 full-res cache ..."
python build_fullcache.py --in-ch 2 --num-workers 24 --splits val test train >> "$ML" 2>&1
say "ic2 cache ready"

# ---- Phase A: in_ch=2 experiments ----
for s in run_stage1.sh run_stage3_cross.sh run_stage245.sh run_a9plus.sh \
         run_a6sec.sh run_tips.sh run_unet.sh run_unet_raymod.sh \
         run_recipe.sh run_raymod_strong.sh run_fullval_decisive.sh \
         run_fullval_ngf96.sh run_flipaug.sh run_vit.sh run_vit_pe.sh run_prob.sh; do
  say ">>> START $s"
  bash "$s" >> "$ML" 2>&1
  say "<<< DONE  $s"
done

# ---- drain: run_prob.sh detaches (no `wait`); ensure no train job is still live ----
say "draining any lingering training processes (e.g. prob) ..."
while pgrep -f "train_prob.py|train_fullmap.py|train.py" >/dev/null 2>&1; do sleep 20; done
say "all training processes drained"

# ---- Phase A2: full-res caches for the RIR feature runs (in_ch=3 and 5) ----
say "building in_ch=3 / in_ch=5 full-res caches ..."
python build_fullcache.py --in-ch 3 --num-workers 24 --splits val test train >> "$ML" 2>&1
python build_fullcache.py --in-ch 5 --num-workers 24 --splits val test train >> "$ML" 2>&1
say "ic3/ic5 caches ready"

# ---- Phase B: RIR (in_ch 3/5) + remaining mixed seeds ----
for s in run_a13.sh run_a14.sh run_seeds.sh; do
  say ">>> START $s"
  bash "$s" >> "$ML" 2>&1
  say "<<< DONE  $s"
done

say "MASTER DONE — all experiments finished"
