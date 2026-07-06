# FINDINGS — AAAI paper (EchoRay / ray decoding)

Running log of the gap-closing pass. Headline metric = **masked plain MAE @ 256×512, test**
(`MAE_plain`); also report cos-lat weighted `MAE`. Seeds {0,1,2}. New runs prefixed `F_`.

## Task 0 — Verification (read-only) ✅

### Champion config — `Q8_csa_wrel05_normal10` (effective, from train_done.json)
| field | value | note |
|---|---|---|
| arch / branch | `raydpt` (model_raydpt.RayDPT) | coarse-sa decoder |
| in_ch / audio_src | **5** / binaural | 5ch = [logL, logR, ILD, cosIPD, sinIPD] (NOT 2ch log-mag) |
| flip_aug | **True** | L/R mirror aug |
| use_xyz / use_fourier_pe / fourier_bands | True / True / 6 | ray bank = xyz(3) + Fourier(36) = 39-D |
| **use_sh_pe / use_mic_pe** | **False / False** | ⭐ **CONFIRMED: SH-PE and mic-PE are OFF** (xyz+Fourier only) |
| raydpt_coarse_sa / coarse_sa_geo | True / True | global 16×32 ray↔ray self-attn + cos-ang-dist bias |
| lsa_mode / win32 / win64 | spherical / 5 / 3 | local spherical window attn |
| ray_cross_layers | 2 | per-ray cross-attn depth |
| **losses** | w_dense 1.0, **w_rel 0.05**, **w_normal 0.1**, w_low 0.5, w_coarse_layout 1.0 | ⚠️ prompt said w_rel=0.5/normal=1.0 — ACTUAL is 0.05 / 0.1 |
| amp / lr / batch / epochs / n_rays | bf16 / 4e-4 / 24 / 25 / 2048 | |

**Key answer:** the champion uses **xyz + Fourier ray-PE only; SH-PE and mic-PE are OFF** — matches config.py defaults. Task 1 re-launches this verbatim. (Loss weights corrected to the true 0.05 / 0.1.)

### Q9_ground flag mapping (all on `raydpt` coarse-sa branch — never cross-compare to other arches)
| run | raydpt_noray | coarse_sa_geo | lsa_mode | meaning |
|---|---|---|---|---|
| full (=Q6_csaonly) | False | True(default) | spherical | all grounding ON |
| noquery | **True** | True | spherical | learned direction-agnostic ray queries |
| nocsageo | False | **False** | spherical | CSA without cos-ang-dist bias |
| planarlsa | False | True | **planar** | local attn without spherical wrap/bias |
| none | **True** | **False** | **planar** | fully geometry-agnostic |

All Q9 runs share the coarse-sa RayDPT branch (same as champion minus its loss add-ons), so grounding cells are directly comparable.

### Note on metric scale
Prompt reference numbers (champion 0.7397 plain MAE) come from the github `RESULTS_full.md`
aggregation. Live `metrics_test.json` in this working tree reports a `MAE` field on a
different (cos-lat / normalized) convention; Task 3 re-derives `MAE_plain` uniformly so
all rows are comparable. Verdicts below are reported in the prompt's plain-MAE band.

## Task 1 & 2 — launched ✅ (running)
- **Task 1 champion multi-seed**: `F_champion_s1`, `F_champion_s2` — champion config verbatim
  (raydpt + coarse-sa + w_rel0.05 + w_normal0.1, xyz+Fourier, SH/mic OFF), seeds 1/2.
- **Task 2 raymlp+CSA** (`cross_mode="global"`): `F_raymlpcsa_s{0,1,2}` — per-ray cross-attn
  replaced by a SINGLE mean-pooled global audio code (concat+MLP on ray-bank features),
  CSA + head unchanged. **Param count: champion(cross) 24.80M vs global 24.91M (+0.4%, within ±15%).**
  Verdict pending eval (plain-MAE band: ≤0.755 → cross-attn not load-bearing; 0.755–0.77 → modest; ≥0.77 → necessary).

## Published baselines — added (running)
- **Channel-adapt pretrained backbones** (`baseline/models/pretrain/`): 1×1 conv (in_ch→3ch pseudo-RGB)
  + pretrained ViT-B/16 / ResNet-50 + decoder. `B_pvit_s{0,1,2}` (89.9M), `B_presnet_s{0,1,2}` (30.0M),
  5ch input, deploy (ImageNet weights). Wired via model_baseline.py (file-path import to avoid the
  baseline package __init__ → data.sh_basis clash).
- **EchoDiffusion**:
  - *architecture re-train* — already present as `E_echo_unet_s{0,1,2}` / `E_echo_ray_s{0,1,2}`
    (wav2vec2 scene encoder + cross-attn port, model_echo.py), completed (~0.91 comp-metric).
    A faithful port also exists in sibling `test_for_audio_better/model_echodiff.py` (echodiff_s0/s1 trained).
  - *pretrained-parameter deploy* — needs the upstream wjzhang-ai/EchoDiffusion checkpoint (not in tree);
    flagged for Task-5-style feasibility check (external download). TODO before claiming this baseline.

## Memo — EchoDiffusion "pretrained deploy" feasibility (investigate-only)

**Question:** can we deploy the original wjzhang-ai/EchoDiffusion pretrained weights on our data as a baseline?

**Findings**
1. **Local:** no original-author checkpoint in tree. All `*echodiff*` .pth files are OUR retrained
   weights (test_for_audio_better/out/echodiff_s0, echorange_radial/*). Our `echodiffusion.py`
   loads NO external weights except frozen wav2vec2.
2. **Upstream (github.com/wjzhang-ai/EchoDiffusion):** pretrained weights ARE released (one Google
   Drive file, id 15MLo6jRcxtDE-xNHwRy5lpVAwz1pBCAY), trained on Replica + a "Matterport extension".
   **BUT only *test-code snippets* are released — "full research code is not released."** Input
   preprocessing (spectrogram params, sample rate, channel order, resolution, depth normalization)
   is undocumented.

**Verdict: NOT feasible as a valid baseline. Three independent blockers:**
- **(a) State-dict mismatch.** Our `baseline/comparison_methods/echodiffusion/echodiffusion.py` is a
  *re-implementation* (their full code is withheld), so their .pth layer names/shapes will not load
  into our module without reverse-engineering their exact architecture. High effort, no guarantee.
- **(b) Undocumented input pipeline.** Zero-shot inference needs their exact spectrogram/wave
  preprocessing, which is not released → any deployed number would run on mismatched inputs = invalid.
- **(c) Rendering distribution shift.** Their "Matterport" is the Visual-Echoes/SoundSpaces rendering,
  not our matterport3d_0303renew binaural rendering (different simulator, mic rig, spectra). Even a
  correctly-loaded model would be cross-distribution zero-shot, not a fair comparison.

**Decision:** report EchoDiffusion via **architecture re-trained on our data** (`B_echodiff_s{0,1,2}`,
faithful port; plus our `E_echo_*` variant), which is the standard and defensible way to include a
comparison method under a shared protocol. Pretrained-deploy is documented here as infeasible/invalid;
the only pretrained parameters the architecture actually uses (frozen wav2vec2) ARE deployed in our
retrain. No download performed (would not change the verdict). File size unverified (GDrive, likely <1GB).

## Memo UPDATE — EchoDiffusion pretrained-deploy: EMPIRICALLY VERIFIED infeasible
Downloaded the authors' GDrive weights (1.6 GB ZIP; <5 GB) = {mp3d/checkpoint_10.pth, replica/checkpoint_150.pth}.
The **mp3d** checkpoint is the relevant one (Matterport). Compatibility with our re-implementation:
- **394 / 1190 tensors match (key+shape) ≈ 33%**; 778 missing, 832 unexpected.
- Matched = ResNet encoder (encoder.layer1-4) + ASPP-ASFF. **UNMATCHED = the entire diffusion UNet core**:
  ckpt has `encoder.unet.unet.diffusion_model.*` (a full Stable-Diffusion LDM UNet) + `lm_head`,
  `masked_spec_embed`; our port's `diffusion_unet.py` is a lightweight re-implementation with different nesting.
- Loading strict=False leaves the generative core randomly initialised → the pretrained model does NOT run.
**Verdict (verified, not assumed): pretrained-param deploy is not possible with the released (partial) code.**
Reconstructing their exact SD-LDM UNet = the withheld "full research code"; even then it is cross-rendering.
Table row "EchoDiffusion (pretrained param)" = **N/A (released weights architecturally incompatible with
available code)**. EchoDiffusion is reported via architecture re-train (B_echodiff) under our shared protocol.

## Final comparison tables (test, 256×512, masked; δk = ratio<1.25^k)

### Table 1 — all-2ch FAIR comparison (isolates architecture; same 2ch input)
baselines = 2ch log-mag spectrogram; EchoDiffusion = 2ch spec + raw wave; RayDPT = 2ch.

| method | input | MAE_plain↓ | MAE↓ | AbsRel↓ | RMSE↓ | δ1↑ | δ2↑ | δ3↑ | n |
|---|---|---|---|---|---|---|---|---|---|
| pretrained UNet (ResNet50) | 2ch | 0.7962 | 0.948 | 0.600 | 1.472 | 0.425 | 0.647 | 0.781 | 3 |
| pretrained ViT (ViT-B/16) | 2ch | 0.7619 | 0.909 | 0.564 | **1.424** | 0.442 | 0.665 | 0.796 | 3 |
| BatVision | 2ch | **0.7597** | 0.905 | 0.537 | 1.443 | **0.446** | 0.667 | 0.797 | 3 |
| EchoDiffusion (retrain) | 2ch+wave | 0.8020 | 0.962 | 0.625 | 1.457 | 0.407 | 0.630 | 0.772 | 3 |
| EchoDiffusion (pretrained param) | — | N/A | — | — | — | — | — | — | — |
| **RayDPT champion (ours)** | 2ch | 0.7630 | 0.906 | **0.5235** | 1.443 | 0.445 | 0.667 | 0.797 | 3 |

**Read:** at equal 2ch input, RayDPT ties BatVision/ViT on MAE_plain (0.763 vs 0.760/0.762, within seed std)
and wins only AbsRel. EchoDiffusion / pretrained-UNet clearly worse.

### Table 2 — native-input comparison (each method's intended input; RayDPT uses full 5ch)
| method | input | MAE_plain↓ | MAE↓ | AbsRel↓ | RMSE↓ | δ1↑ | δ2↑ | δ3↑ | n |
|---|---|---|---|---|---|---|---|---|---|
| pretrained UNet (ResNet50) | 2ch | 0.7962 | 0.948 | 0.600 | 1.472 | 0.425 | 0.647 | 0.781 | 3 |
| pretrained ViT | 2ch | 0.7619 | 0.909 | 0.564 | 1.424 | 0.442 | 0.665 | 0.796 | 3 |
| BatVision | 2ch | 0.7597 | 0.905 | 0.537 | 1.443 | 0.446 | 0.667 | 0.797 | 3 |
| EchoDiffusion | 2ch+wave | 0.8020 | 0.962 | 0.625 | 1.457 | 0.407 | 0.630 | 0.772 | 3 |
| **RayDPT champion (ours)** | **5ch** | **0.7572** | 0.896 | **0.5005** | 1.439 | 0.449 | 0.670 | 0.800 | 3 |
| **└ best seed** | 5ch | **0.7397** | 0.880 | 0.504 | **1.413** | 0.461 | 0.679 | 0.806 | 1 |

**Read:** RayDPT's edge comes from (a) richer 5ch input (2ch 0.763 → 5ch 0.757, best-seed 0.740) and
(b) AbsRel-specialised loss, NOT from the ray/spherical decoding per se — consistent with the grounding
ablations (ray-grounding inert) and the observability-ceiling finding.

### Verdict summary
- **Task 2 (cross-attn):** NOT load-bearing — raymlp+CSA (global audio code) ties the champion.
- **Ray-grounding:** inert across 3 ablations (no-ray query = learned query; planar ≥ spherical; all-off ≈ full).
- **Architecture vs input:** at equal 2ch, RayDPT ≈ BatVision/ViT; the gain is input-richness + AbsRel loss.
- **EchoDiffusion pretrained deploy:** verified infeasible (released SD-LDM weights incompatible with re-impl; 33% overlap).
