# test_for_audio_implicit_full — full-resolution (256×512) audio→ERP-depth

> **Full-res variant:** reads the actual `erp_depth_radial` files at **256×512**
> (no 64×128 cache downsampling), like `baseline`. Local full-res cache + 8-GPU
> scheduler + auto-eval. Radial depth, scene_split 72/9/9, masked L1 (fullmap) /
> per-ray L1 (implicit), cos-lat weighted metrics.

## ⭐ Current best model (256×512, test split, MAE_plain ↓ = masked MAE [m])

**Best = the ray-conditioned cross-attention *implicit* model** (`--model cross`,
`train.py` / `model.py`): per-ray queries that **cross-attend the audio tokens**,
predicting depth for each ERP ray direction rather than decoding a pixel map.

| rank | model | MAE_plain | seeds | note |
|---|---|---|---|---|
| 🥇 | **A4_cross** (cross-attn implicit) | **0.781 ± 0.003** | 3 | best robust |
| 🥇 | A6_crossself (cross + ray self-attn) | 0.780 ± 0.005 | 3 | tied |
| · | A5_crossMic / A4_ffmask / A3_crossSH | 0.771–0.774 | 1 | single-seed, same family |
| | A9 full-map decoder | 0.799 ± 0.001 | 3 | global-bottleneck |
| | A2 RayMLP (global latent) | 0.805 ± 0.006 | 3 | |
| 🔻 | Aunet / A18 (pix2pix U-Net) | 0.829 ± 0.004 | 3 | **worst** real model |
| (ctrl) | shuffle-audio / ray-only | ~0.98 | 2 | audio-ablated → confirms audio is used |

**Key finding — resolution inverts the ranking.** At 64×128 the pix2pix **U-Net was
best (~0.775)** and RayMLP worst. At full 256×512 it flips: **cross-attention implicit
is best (~0.78), U-Net worst (~0.83)**. Coordinate/implicit models output a
**band-limited** field and sit at the audio observability ceiling → resolution-robust;
the U-Net chases fine detail that audio cannot predict and that full-res GT exposes →
degrades. (Interim: ViT / RIR(5-ch phase) / probabilistic / a baseline-faithful
8-down U-Net comparison are still training; see `RESULTS_full.md`.)

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
