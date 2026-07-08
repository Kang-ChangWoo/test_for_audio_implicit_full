# Research queue — Cue-Factorized Acoustic Representation + Cue-Specific Ray Routing

Extends the existing RayDPT queue. **Not auto-launched in bulk** — Priority-1 (CF_*) added to the
scheduler; Priority-2/3 recorded here as hypotheses (some need extra flags, listed below), to be
promoted only if a P1 signal appears. Current experiments/results preserved (append-only).

## Context: what already exists (avoid duplication)
- Input layout (data.py `_specN`): **ch0,1 = magnitude** (logL, logR); **ch2,3,4 = spatial** (ILD, cosIPD, sinIPD). cos/sin-IPD already bounded — keep as-is (no raw-phase regression).
- **Not the same as any prior experiment:** Q11 (per-channel *zero-out* ablation), `raydpt_msf` (multi-scale KV *concat*), `raydpt_resampler` (learned-latent KV). None separates cues into branches or routes K vs V by cue → **cue factorization + K/V routing is genuinely new.**
- Ray cross-attn: `CrossBlock.forward(q, kv)` uses kv for BOTH K and V. New `CueCrossBlock(q,k,v)` added for separate sources.

## Minimally-invasive implementation (done, behind flags — RayDPT intact when off)
- **model_raydpt.py**: `CueEncoder` (2/3-ch cue input → 16×32×dim coarse tokens, 4 stride-2 convs); `CueCrossBlock` (separate K/V, per-branch LayerNorm). RayDPT: `cue_stems` (Group A two-stem input → main encoder) + `cue_route` (Group B: Z_mag/Z_spatial/Z_fused; routed K/V at **coarse F16 only**, token-aligned 512 → no resolution confound). Routing applied in the canonical `else` decode branch (champion path).
- **config.py**: `cue_route, kv_key_source∈{fused,spatial,magnitude}, kv_value_source∈{...}, cue_stems, cue_cmag, cue_cspatial`.
- eval unchanged (RayDPT dispatch reads cue flags from saved cfg).
- **Param note**: baseline 24.80M; cue_stems +0.08M; cue_route +2.0M (two CueEncoders) → the +2M is a **capacity confound** → F1/F2/F3 controls mandatory before any claim.

---
## PRIORITY 1 (queued as CF_*) — does cue organization help at all?
Parent baseline for ALL: **Q8_csa_wrel05_normal10** (coarse-sa + w_rel0.05 + w_normal0.1, no cue; DONE). Controlled vars: identical recipe/decoder/losses/schedule; only audio-repr + F16 K/V source vary.

| # | name | hypothesis | K / V | complexity | falsification |
|---|---|---|---|---|---|
| 1 | CF_stems | cue-specific input stems help encoding | (fused) | Low | ≈ baseline within seed noise |
| 2 | CF_route_ff | cue branches + fused K/V (capacity/pipeline control) | fused/fused | Low | ≈ baseline → gain (if any) not from cues |
| 3 | CF_route_sf | **spatial cues better for association (K)** | spatial/fused | Low-Med | not > ff and not > mf |
| 4 | CF_route_mf | control vs #3 (magnitude as key) | magnitude/fused | Low-Med | — |
| 5 | CF_route_sm | **directional-K + range-V functional roles** | spatial/magnitude | Med | ≈ CF_route_ms |
| 6 | CF_route_ms | **reverse-role control** (critical) | magnitude/spatial | Med | if ≈ #5 → cues have NO distinct role |

**Supporting result for the direction:** CF_route_sf > CF_route_ff AND > CF_route_mf (spatial keys help association); and CF_route_sm > CF_route_ms (distinct functional roles). **Falsification:** no consistent spatial-vs-magnitude key difference, or CF_route_sm ≈ CF_route_ms.

Dependencies: run 1–6 first; each is one training run (~champion cost). Confirm winners with 3 seeds only after a signal.

---
## PRIORITY 2 (controls — promote ONLY if P1 improves; need extra flags)
- **F3 capacity-matched single encoder** — widen ngf so params ≈ cue_route (+2M). *Flag: `--ngf` up.* Mandatory before claiming factorization helps.
- **F4 / A3 shared-encoder + separate K/V heads** — Z=E(all); K=P_k(Z), V=P_v(Z) (no separate cue encoders). *Needs flag `cue_adapter=True` (adapters on shared e4 instead of CueEncoder).* Distinguishes "cue-specific reps" from "role-specific projection."
- **F2 duplicate-input two-branch** — both cue branches receive ALL 5ch. *Needs flag `cue_dup_input=True`.* Tests capacity/ensemble vs semantics.
- **F1 random channel split** — e.g. {logL,cosIPD} vs {logR,ILD,sinIPD}. *Needs flag `cue_random_split=True`.* Tests semantic grouping vs "just multiple branches."

Falsification for the whole direction: if F1/F2/F3/F4 match the semantic cue split → reframe as "role-specific routing / extra capacity," NOT physical cue separation.

---
## PRIORITY 3 (later; only after clear P1 signal)
- **A2 late two-stream encoders** (full separate encoders + multi-scale 1×1 fusion) — needs capacity-matched baseline (F3).
- **C fusion ablations** for Z_fused: C1 concat+linear (default) → C2 add → C3 scalar gate → C4 channel gate. Start C1; don't over-optimize fusion before cue-usefulness is established.
- **D dual cross-attention** (parallel spatial/mag attention → combine): D1 concat, D2 scalar α, D3 ray-dependent gate (defer — adds another ray-conditioning mechanism → interpretation risk).
- **E multi-scale cue routing** (coarse=mag / mid=fused / fine=spatial etc.): E1 all-fused, E2/E3 role assignments, E4 learned per-scale mix. Requires cue tokens at 32×64 / 64×128 (currently F16-only) — extends CueEncoder to finer scales.

## Normalization / attention cautions (implemented / to watch)
- CueCrossBlock uses **separate LayerNorm on K and V branches** (spatial vs magnitude have different activation scales → fair comparison).
- cue_stems: mag/spatial processed by separate conv stems before fusion (branch-local stats preserved). Do NOT add a single LayerNorm across heterogeneous raw channels.
- All K/V variants share identical ray queries, attn dim, heads, token resolution (16×32) — only K/V source differs.

## First-try order
CF_stems → CF_route_ff → CF_route_sf / CF_route_mf → CF_route_sm / CF_route_ms.
Read: (a) does any beat the same-recipe baseline? (b) spatial-K vs magnitude-K difference? (c) sm vs ms (role) difference? Only then promote P2 controls.
