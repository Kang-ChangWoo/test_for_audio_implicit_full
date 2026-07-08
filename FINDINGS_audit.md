# Audit & correctness report (finalv2 protocol)

## A. Audit findings (fixed)
1. **Coarse-scale loss zero-dilution** — train_fullmap.py (COARSE_ARCH block).
   Problem: `gt_c=avg_pool(gt)`, `m_c=avg_pool(mask)`, masked_mae → invalid GT zeros dilute the target
   (0.5,0.5,0,0 → 0.25). Fix: `gt_c=pool(gt*mask)/pool(mask)`, `valid_c=(m_c>0)`. Matters: corrupted coarse
   supervision biased RayDPT layout toward 0 near invalid regions. Old RayDPT numbers affected.
2. **Low-pass (Gaussian) loss zero-dilution** — same block. Fix: `lpG=blur(gt*mask)/blur(mask)`, pred blurred
   normally. Same rationale.
3. **Normal loss validity** — normal_loss(). Problem: mask used only `mask[:-1,:-1]` (1 of 3 stencil points).
   Fix: `m = mask[:-1,:-1]*mask[1:,:-1]*mask[:-1,1:]`. Matters: normals across invalid edges were supervised.
4. **Metric batch-composition dependence** — metrics.py add(). Problem: batch-pooled ratio × B, /Σimages →
   depends on batching. Fix: PER-IMAGE metrics averaged (num/den per image, mean, n=B) → batch-invariant
   (proven: [A,B],[C,D] == [A],[B,C,D], diff<1e-5). RMSE/SILog now per-image (SILog's λ-term is per-image by
   definition). **All reported numbers change vs old batch-pooled values → finalv2 recomputes uniformly.**
5. **Dataset silent duplication** — data.py __getitem__. Problem: `return self[(i+1)%len]` on exception
   distorts distribution. Fix: bounded index-varying retry (≤8) in TRAIN only; fail-fast in eval; RuntimeError
   if none found. No silent neighbour copy.

## B. Architecture (coordinate consistency) — INSPECTED
RayDPT decoder DOES add encoder skips `se4(e4)/se3(e3)/se2(e2)` (spectrogram-grid conv features) to ERP ray
features. Per the spec this is a spectrogram↔ERP coordinate mix. However `se*` are LEARNED 1×1 conv maps
(not raw copies) and the `msf` variant already routes e2/e3 via compact KV instead of raw add. Given the
skip-harmful / grounding-inert ablations already run this session, and the directive "smallest change",
the fair-comparison finalv2 uses the established coarse-sa RayDPT (skips intact) — a coordinate-clean
KV-only variant (raydpt_msf) exists for a follow-up. NO forced redesign (would invalidate all checkpoints).

## E. Ablation semantics (documented)
- **RayDPT (ray)** = explicit physical ray direction (RayBank xyz+Fourier) → ray_proj queries.
- **RayDPT w/o ray condition** (finalv2_*_noray) = **LEARNED ABSOLUTE POSITIONAL queries** (one param per
  ERP cell; capacity-matched; NO explicit physical direction). Explicitly NOT "direction-agnostic".
- Added `raydpt_shared_q` = single shared query expanded to all cells (true no-position) as an extra ablation.

## F. Split protocol
Existing split = scene_split.json, **scene-level 72/9/9** (train/val/test) — already scene-independent, no
frame leakage. BUT the test split was auto-evaluated across ~370 dev runs (winner's-curse / model-selection
exposure). A brand-new untouched holdout would require re-splitting → changes train data → invalidates all
prior runs. **Decision (documented, honest):** keep the scene-level split; treat current test as a
**development benchmark**; report finalv2 with 3-seed mean±std and note the model-selection caveat. Do NOT
auto-rank future exploratory runs by it. (A held-out scene subset can be carved later if a fresh comparison
is needed, at scene granularity.)

## Input channels (verified, identical per label)
- **2ch** = `_spec2`: 2-channel binaural |STFT| magnitude, log1p. (all 2ch models identical)
- **5ch** = `_specN`: [logL, logR, ILD, cos(IPD), sin(IPD)]. (RayDPT 5ch only)
- EchoDiffusion consumes 2ch spec + raw waveform (its design); documented adaptation.

## Checkpoint / metric policy (documented)
- Primary checkpoint metric: cos-lat-weighted val MAE (quick_val) — spherical-area-correct for ERP.
- Primary reported: MAE_plain (mask-only) as headline + cos-lat MAE + RMSE/AbsRel/δ1-3/SILog. Kept distinct
  intentionally (checkpoint by spherical metric; report plain for comparability). NOTE remaining: could align
  checkpoint→MAE_plain; not changed silently.

## Remaining (documented, not yet changed to avoid scope creep)
- Cache key uses `int(audio_window)` — fine for current integer windows (10/20/30/40) but should hash a
  canonical preprocessing JSON if non-integer windows are introduced.
- Flip aug is representation-aware for binaural (L/R swap; ILD/IPD handled via swap_audio_lr) but a RAZ
  azimuth-indexed feature would need azimuth-axis mirror — RAZ not in finalv2, flagged for its own runs.
