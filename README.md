# test_for_audio_implicit_full — full-resolution (256×512) audio→ERP-depth

> **Full-res variant:** reads the actual `erp_depth_radial` files at **256×512**
> (no 64×128 cache downsampling), like `baseline`. Local full-res cache + 8-GPU
> scheduler + auto-eval. Radial depth, scene_split 72/9/9, masked L1 (fullmap) /
> per-ray L1 (implicit), cos-lat weighted metrics.

## ⭐ Current best model (256×512, test split, MAE_plain ↓ = masked MAE [m])

<!-- BEST:START -->
**Best (lowest test MAE_plain) = the ray-conditioned cross-attention *implicit* model** — per-ray queries that cross-attend the audio tokens, predicting depth for each ERP ray direction instead of decoding a pixel map.

| rank | model | MAE_plain [m] ↓ | seeds |
|---|---|---|---|
| 🥇 1 | Bnode2_unet8_5chflip (Bnode2_unet8_5chflip) | 0.7463 ± 0.0038 | 2 |
| 2 | Bnode2_unet8_5chflip_w20 (Bnode2_unet8_5chflip_w20) | 0.7480 ± 0.0017 | 2 |
| 3 | B_unet8_5ch (B_unet8_5ch) | 0.7528 ± 0.0019 | 3 |
| 4 | B_unet8nolog_aug (B_unet8nolog_aug) | 0.7582 ± 0.0036 | 3 |
| 5 | A22_vit_aug (ViT-B/16 (planar PE)) | 0.7605 ± 0.0025 | 3 |
| 6 | Bnode2_cross_flip (Bnode2_cross_flip) | 0.7613 ± 0.0019 | 3 |
| 7 | A23_vit_both (ViT-B/16 (SH+Fourier)) | 0.7616 ± 0.0036 | 3 |
| 8 | A23_vit_sh (ViT-B/16 (SH PE)) | 0.7618 ± 0.0013 | 3 |
| 9 | Bnode2_unet8nolog (Bnode2_unet8nolog) | 0.7628 ± 0.0033 | 3 |
| 10 | B_unet8nolog (U-Net 8-down, no-log (baseline-faithful)) | 0.7637 ± 0.0029 | 3 |
| 11 | B_cross_nolog (cross implicit, no-log (matched)) | 0.7682 | 1 |
| 12 | Bnode2_cross_5chflip (Bnode2_cross_5chflip) | 0.7697 ± 0.0019 | 2 |
| 13 | A4_ffmask (cross + far-mask) | 0.7711 | 1 |
| 14 | A5_crossMic (cross + mic-PE) | 0.7724 | 1 |
| 15 | A23_vit_fourier (ViT-B/16 (Fourier PE)) | 0.7724 ± 0.0135 | 3 |
| 16 | Bnode2_cross_nolog (Bnode2_cross_nolog) | 0.7735 ± 0.0037 | 3 |
| 17 | A3_crossSH (cross + SH-PE) | 0.7738 | 1 |
| 18 | T_mlpskip (T_mlpskip) | 0.7748 | 1 |
| 19 | T_film (T_film) | 0.7763 | 1 |
| 20 | T_sector (T_sector) | 0.7777 | 1 |
| 21 | A6_crossself (cross + ray self-attn) | 0.7804 ± 0.0049 | 3 |
| 22 | A4_cross (cross-attn implicit) | 0.7805 ± 0.0028 | 3 |
| 23 | A6sec (A6sec) | 0.7810 ± 0.0040 | 3 |
| 24 | T_progpe (T_progpe) | 0.7831 | 1 |
| 25 | T_all (T_all) | 0.7831 | 1 |
| 26 | A12_film (A12_film) | 0.7931 | 1 |
| 27 | A14_rir5 (A14_rir5) | 0.7939 | 1 |
| 28 | A19_raymodStrong (A19_raymodStrong) | 0.7940 ± 0.0076 | 3 |
| 29 | A19_raymodStrong_fv (A19_raymodStrong_fv) | 0.7944 ± 0.0047 | 3 |
| 30 | A10_cross (A10_cross) | 0.7947 | 1 |
| 31 | A14_frozen (A14_frozen) | 0.7959 | 1 |
| 32 | A13_ild3 (A13_ild3) | 0.7981 | 1 |
| 33 | A9_fullmap (full-map decoder (global bottleneck)) | 0.7988 ± 0.0014 | 3 |
| 34 | Bnode2_cross5ch (Bnode2_cross5ch) | 0.7994 ± 0.0043 | 3 |
| 35 | A8_hybrid (A8_hybrid) | 0.8004 | 1 |
| 36 | A13_mag2 (2ch mag (RIR ctrl)) | 0.8039 | 1 |
| 37 | A11_shaux (A11_shaux) | 0.8040 ± 0.0072 | 3 |
| 38 | A2_raymlp (RayMLP (global latent)) | 0.8047 ± 0.0063 | 3 |
| 39 | A14_logmag (A14_logmag) | 0.8057 ± 0.0022 | 2 |
| 40 | A16_raymod8x16 (A16_raymod8x16) | 0.8077 ± 0.0038 | 3 |
| 41 | A16_raymod_fv (A16_raymod_fv) | 0.8084 ± 0.0130 | 3 |
| 42 | A13_ipd5 (5ch RIR (+phase/IPD)) | 0.8084 | 1 |
| 43 | A18_raymod64reg (A18_raymod64reg) | 0.8121 ± 0.0079 | 2 |
| 44 | A20_unet64_aug (A20_unet64_aug) | 0.8166 ± 0.0034 | 3 |
| 45 | A21_raymodStrong_aug (A21_raymodStrong_aug) | 0.8179 ± 0.0311 | 3 |
| 46 | A15_bigunet_fv (A15_bigunet_fv) | 0.8220 ± 0.0073 | 3 |
| 🔻 47 | A18_unet64reg_fv (A18_unet64reg_fv) | 0.8222 ± 0.0027 | 3 |
| 48 | A15_bigunet (pix2pix U-Net (ngf96)) | 0.8225 ± 0.0065 | 3 |
| 49 | Bnode2_hybrid5ch (Bnode2_hybrid5ch) | 0.8233 ± 0.0070 | 3 |
| 🔻 50 | Aunet (pix2pix U-Net) | 0.8287 ± 0.0034 | 3 |
| 🔻 51 | A18_unet64reg (pix2pix U-Net (reg)) | 0.8290 ± 0.0050 | 3 |
| 52 | A4_cross_shuf (A4_cross_shuf) | 0.9793 | 1 |
| 53 | A2_shuf (shuffle-audio (control)) | 0.9823 ± 0.0004 | 2 |
| 54 | A1_rayonly (ray-only prior (control)) | 0.9832 ± 0.0067 | 2 |

**Headline:** best robust model = **B_unet8_5ch** at **0.753 ± 0.002 m** (3 seeds). pix2pix U-Net = 0.822 m. **Resolution inverts the 64×128 ranking** — there the U-Net was best (~0.775) and RayMLP worst; at full 256×512 the cross-attention *implicit* model is best and the U-Net is the worst real model. Implicit/coordinate models emit a **band-limited** field that sits at the audio observability ceiling (resolution-robust); the U-Net chases fine detail audio cannot predict and that full-res GT exposes (degrades). Full per-metric table: see `RESULTS_full.md`.
<!-- BEST:END -->

---

Tests whether a **ray-conditioned implicit depth function** beats the existing
global-bottleneck encoder–decoder at binaural-audio → ERP radial depth, by
**decomposing the hypothesis** and breaking each sub-question in order rather
than building one big model.

```
binaural spec ──► audio encoder ──► global latent z  /  tokens
ERP ray dir r  ─► [xyz | Fourier-PE | SH basis | ear-axis mic-PE] ─► ray query q
q  (×audio)  ──► depth(r)            # implicit: predict per-ray, not a full map
```

## Hypothesis ladder (run in order)
| Q | question | runs |
|---|---|---|
| Q1 | implicit fn uses audio at all? | A1 ray-only prior vs A2 RayMLP (+shuffled control) |
| Q2 | SH/Fourier ray-PE give inductive bias? | A2 vs A3 (`--use-sh-pe`) on low-freq metrics |
| Q3 | ear-axis mic-PE helps binaural use? | A3 vs A5 (`--use-mic-pe`), L/R-swap test |
| Q4 | ray self-attn corrects unobservable rays? | A5 vs A6 (`--model crossself`) |
| Q5 | SH-coarse + residual cuts mean-blob? | A5 vs A8 (`--model hybrid`) |

## Data / reuse
- Reuses `../test_for_audio_better/cache` (spec 2×64×128 log-mag binaural,
  depth radial /max_depth∈[0,1], mask). No data prep.
- Dataset is **listener-centred & self-emitting** (active echolocation): source≈origin
  (per-ray source-PE degenerate → dropped); ears are a fixed `±y` rig (`head_r`),
  giving a legitimate mic-PE that drives the L/R-swap mirror test.

## Models (`--model`)
`rayonly` · `raymlp` · `cross` · `crossself` · `hybrid`.
Ray-feature flags: `--use-xyz --use-fourier-pe --use-sh-pe --use-mic-pe`.
Head: `--use-depth-bins`. Controls: `--audio-mode {stereo,mono,left,right,none}`,
`--shuffle-audio True`.

## Run
```bash
bash run_stage1.sh          # Q1 gate, 2 seeds
python agg_stage1.py        # verdict table
python eval.py --run-name A2_raymlp_s0 --controls True
```
Training supervises N random rays/sample; eval predicts the full grid by chunking.

## Metrics (`metrics.py`)
MAE/RMSE/AbsRel/δ<1.25/SILog (cos-lat weighted) + **layout** metrics that matter
for the SH/implicit claim: low-pass MAE, SH-coefficient L1 error, sector MAE.
Controls in `eval.py`: mono/left/right/shuffle + L/R-swap mirror consistency.
