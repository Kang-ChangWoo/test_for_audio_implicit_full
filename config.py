"""Config for the ray-conditioned implicit audio->ERP-depth experiment.

Hypothesis decomposition (break in order, do NOT jump to one big model):
  Q1  ray-conditioned implicit fn  >  existing global encoder-decoder?
  Q2  SH / Fourier ray-PE give real inductive bias (lower LOW-FREQ error)?
  Q3  ear-axis mic-PE helps the model exploit binaural (ILD/IPD) cues?
  Q4  ray self-attention corrects unobservable rays from neighbours?
  Q5  hybrid SH-coarse + implicit-residual reduces the mean-blob collapse?

Data reality (confirmed): the dataset is listener-centred and SELF-EMITTING
(active echolocation). There is NO per-sample p_L/p_R/p_s metadata, BUT:
  * source == listener == origin  -> a per-ray source-PE is degenerate (dropped)
  * the two ears are a FIXED known rig -> we place them at +/- y (head radius
    `head_r` m) and feature each ray by its geometry to each ear. This is the
    legitimate "mic PE". It also drives the L/R-swap mirror test.

Reuses test_for_audio_better's cache (spec/depth/mask, radial depth /max_depth in
[0,1]) so no data prep is needed.
"""

import argparse
from types import SimpleNamespace

DEFAULTS = dict(
    # --- data (reuse the better-experiment cache; no rebuild) ---
    # full variant: data IS local -> use the local data path referenced by the
    # original test_for_audio_implicit (relative sibling cache). Full 64x128 res,
    # no downscaling to fit data size.
    dataset_dir="/root/storage/matterport3d_0303renew",
    cache_dir="/root/implicit_full_cache",   # LOCAL-disk full-res cache base (in_ch/HxW keyed)
    # FULL resolution, loaded directly from files like baseline (images_size 256x512).
    # The real erp_depth_radial files are 512x1024; we resize to this (no 64x128 cache).
    img_h=256, img_w=512, max_depth=10.0, sample_rate=48000,
    depth_type="radial",      # "radial" (erp_depth_radial, distance) | "planar" (erp_depth, conventional z)
    log_spec=True,            # 2ch magnitude spectrogram: log1p (True) vs raw (False)
    audio_window_m=10.0,      # audio truncation window [m round-trip basis]; >max_depth = richer (later reflections)
    audio_src="binaural",     # binaural | foa | gcc (5ch+GCC-PHAT, 6ch) | wave (5ch+raw waveform)
    wave_ch=8, wave_ngf=32,   # WaveUNet: raw-waveform 1D-CNN global embedding width

    # --- input channels (2 = log-mag binaural; 5 = RIR spatial feature, A13) ---
    in_ch=2,
    # --- model selection ---
    # rayonly | raymlp | cross | crossself | hybrid
    model="raymlp",
    width=48, embed_dim=128, dim=192, audio_dim=256,

    # --- ray feature flags (the modular ablation knobs) ---
    use_xyz=True,             # raw unit direction (3)
    use_fourier_pe=True,      # Fourier PE of xyz
    fourier_bands=6,          # -> 3*2*bands dims
    use_sh_pe=False,          # spherical-harmonic ray basis
    sh_order=4,               # -> (sh_order+1)**2 dims
    use_mic_pe=False,         # ear-axis (binaural) geometry features
    head_r=0.0875,            # head radius [m] for the +/- y ear rig
    # front/back are the cone-of-confusion (spectral-only) directions; two levers:
    front_back_w=1.0,         # sector-weighted loss: upweight front+back rays (1.0=off)
    hi_tokens=False,          # tap audio tokens at /4 (finer spectrum) vs /8 for cross-attn
    # ray TV-smoothness: predict on a fixed grid + total-variation penalty (anti-discrete)
    ray_tv_w=0.0, ray_tv_grid_h=64, ray_tv_grid_w=128,
    cross_enc="conv",         # cross/crossself audio encoder: conv (default) | vit (ViT-B/16 tokens)
    # --- coarse-layout heads on the U-Net8 encoder (model_unet_coarse.py) ---
    coarse_head_h=16, coarse_head_w=32,   # coarse ERP depth-head resolution
    coarse_sh_order=4,                    # SH order for unet_sh head
    residual_scale=0.1, residual_h=32, residual_w=64,   # constrained low-pass residual
    ray_coarse_h=16, ray_coarse_w=32,     # coarse ray-token grid for unet_raycoarse
    ray_cross_layers=2, ray_self_layers=1,
    raydpt_win32=5, raydpt_win64=3,       # RayDPT local spherical-attention window per scale
    raydpt_lite=False,            # 2-scale (32,64) lite Ray-DPT variant (single ray cross-attn)
    raydpt_full_decode=True,      # learned upsample 64x128->256x512 (+e1 skip); always on (vs bilinear x4)
    raydpt_msf=False,             # multi-scale-KV fusion: F32<-cat(e4,e3), F64<-cat(e4,pooled e3,pooled e2);
                                  # drop raw e2/e3 DPT skip-add (coord-mismatch) -> ray cross-attends compact KV
    # Perceiver/Q-Former-style acoustic resampler: learned latents compress multi-scale
    # acoustic tokens into a compact scene memory; physical ERP ray queries read it.
    raydpt_resampler=False, resampler_latents=64, resampler_layers=3,
    raydpt_noray=False,       # w/o ray: LEARNED ABSOLUTE POSITIONAL queries (NOT direction-agnostic)
    raydpt_shared_q=False,    # + shared query (single vector, true no-position) when raydpt_noray
    lsa_mode="spherical",         # local spherical attn ablation: spherical|nobias|planar|off           # ablation: replace physical spherical ray-direction queries
                                  # with LEARNED direction-agnostic queries (no ray conditioning)

    # coarse-arch loss weights (only applied for arch in the coarse family)
    w_dense=1.0, w_coarse_layout=1.0, w_low=0.5, w_tv_res=0.01,

    # --- attention sizes (cross / self models) ---
    n_heads=4, n_cross=2, n_self=2,

    # --- implicit-field tuning tips (applied to cross/raymlp) ---
    ray_mlp_skip=False,       # tip6: coord re-injection skip-MLP for ray embedding
    ray_film=False,           # tip5: FiLM (global audio -> gamma,beta) on ray tokens after cross
    prog_pe=False,            # tip3: progressive coarse->fine Fourier PE band opening
    sector_sample=False,      # tip4: sector/near/grad-balanced ray sampling (vs uniform)

    # --- depth head ---
    use_depth_bins=False,     # log-depth bin classification + expected value
    n_bins=64,

    # --- hybrid SH-coarse + residual (model=hybrid) ---
    hybrid_sh_order=3,        # coarse spherical geometry order
    w_coarse=0.5,             # aux loss on SH-coarse depth
    w_res=0.02,               # residual L1 magnitude penalty

    # --- architecture (train_fullmap.py): fullmap(A9 no-skip) | unet(pix2pix skip)
    #     | unet_raymod(A16: strong U-Net + ray-conditioned sparse FiLM modulation) ---
    arch="fullmap",
    ngf=64, unet_downs=6,     # pix2pix U-Net width / #downsamples (64x128 -> 1x2 at 6)
    # --- A16 ray-conditioned U-Net modulation (model_unet_raymod.py) ---
    ray_mod_scale=0.1,        # FiLM strength s; 0.0 => plain U-Net (capacity-only control)
    ray_mod_stage="e3",       # encoder stage modulated: e3(8x16 grid) | e2(16x32 grid)
    # --- A22 pretrained ViT-B/16 (arch="vit", model_vit.py) ---
    vit_pretrained=True,      # load ImageNet ViT-B/16 weights (cached locally)
    vit_freeze=False,         # freeze encoder, train adapter+decoder only
    vit_pe="planar",          # patch PE: planar(interp ImageNet) | fourier | sh | both (ERP-valid)
    # --- A9-A12 full-map decoder + audio correction (train_fullmap.py) ---
    correction="none",        # none(A9) | cross(A10) | sh(A11) | film(A12) | cross_sup(A14)
    coarse_h=16, coarse_w=32,  # coarse grid for the cross-residual branch
    corr_sh_order=3,          # SH order for the sh-correction / aux
    w_sh_aux=1.0,             # weight on audio->SH-coef auxiliary loss (A11)
    # --- A14 supervised residual corrector ---
    init_decoder="",          # warm-start decoder weights from this run (e.g. A9_fullmap_s0)
    freeze_decoder=False,     # freeze D0 -> pure "is the A0 residual audio-predictable?" test
    res_scale=0.3,            # bound on tanh residual (normalised depth)
    w_res_sup=1.0,            # supervise Dcorr to lowpass(GT - D0)
    w_tv=0.01,                # total-variation reg on Dcorr
    chan_norm=False,          # per-channel train-set input normalisation (for 5ch A13/A14)
    w_swap_eq=0.0,            # tip8: weak swap-equivariance reg: f(swap_LR(x)) ~ mirror(f(x))
    # --- 3D-space auxiliary losses (on the ERP point cloud p = depth * ray_dir) ---
    w_normal=0.0,             # type-3: surface-normal cosine loss (edge-aware, anti-blob)
    w_chamfer=0.0,            # type-2: subsampled symmetric Chamfer distance in 3D
    chamfer_k=1024,           # points subsampled per sample for Chamfer
    # --- best training recipe ported from sibling repo audioresearch_audio (E2) ---
    w_rel=0.0,                # relative-depth (AbsRel-style) loss weight; E2 best = 0.1
    amp=False,                # AMP bf16 autocast (E0b: enables bs32, faster, small gain)
    # --- per-scene SCALE guide: match predicted mean to GT mean (RMSE-optimal central
    # tendency). oracle mean-match = -5.4% RMSE. does NOT touch spatial distribution. ---
    w_scale=0.0,
    w_silog=0.0,              # scale-invariant log loss (audioresearch_audio E4); combine w/ w_rel
    # --- EXPERIMENTS.md champion stack (E14/E22/E27/E29/E34) ---
    w_grad=0.0,               # E34: edge-aware gradient-matching loss (champion += 0.05)
    w_depth_gamma=0.0,        # per-pixel weighting: dense L1 * gt**gamma (>0 far/RMSE, <0 near/AbsRel)
    berhu=False,              # reverse-Huber main loss (error-magnitude weighting)
    w_ema=0.0,                # E14: weight-EMA decay for eval/checkpoint (0=off, best=0.995)
    raydpt_coarse_sa=False,   # E22/E27: global ray<->ray self-attn at 16x32 + cos-ang-dist bias
    cross_mode="cross",       # ray<->audio: cross (per-ray retrieval) | global (single mean-pooled audio code, no cross-attn)
    coarse_sa_geo=True,
    coarse_sa_blocks=1,       # E51: # of post-fusion geo self-attn blocks on m16 (2=champion)
    berhu_low=False,          # E117: berHu on low-pass loss term only (RMSE lever, no main-term slide)
    # --- Cue-factorized acoustic representation + cue-specific ray K/V routing (minimally
    #     invasive: RayDPT intact when off). ch0,1=mag(logL,logR); ch2,3,4=spatial(ILD,cos/sinIPD). ---
    cue_route=False,          # enable lightweight cue-specific coarse branches + configurable K/V at F16
    kv_key_source="fused",    # ray-attn KEY source: fused | spatial | magnitude
    kv_value_source="fused",  # ray-attn VALUE source: fused | spatial | magnitude
    cue_stems=False,          # Group A: two-stem input adapter (mag/spatial) before the MAIN encoder
    cue_cmag=32, cue_cspatial=32,   # stem widths (ratio experiments: 1:1 / 2:1 / 1:2)
    cue_dup_input=False,      # F2 control: both cue branches see all 5ch (capacity/ensemble test)
    cue_random_split=False,   # F1 control: non-semantic channel split
    cue_adapter=False,        # F4/A3 control: cue reps via adapters on shared e4 (no separate cue encoders)
    cue_fused_mode="kv4",     # C fusion for Z_fused: kv4 | concat | add | gate
    cue_dual=False,           # D1: parallel spatial+magnitude cross-attention then combine
    rayvit_mode="single",     # RayViT encoder: single | multiscale | hybrid
    zero_chan=-1,             # 5ch input ablation: zero out channel idx (0=logL,1=logR,2=ILD,3=cosIPD,4=sinIPD; -1=none)       # ray-grounding ablation: cos-ang-dist bias in coarse-sa (False=plain global SA)
    raydpt_gated_skip=False,  # E29: gated (vs raw-add) DPT encoder skips
    # --- EchoBin: distance-binned binaural directional weak guide (model_echo.EchoBin) ---
    echo_kbins=32, echo_dmax=8.0,

    # --- probabilistic coarse head (train_prob.py): model the coarse-layout AMBIGUITY ---
    # Finding (oracle decomp): error = coarse-layout multi-modality + unobservable fine detail.
    # K diverse coarse hypotheses (relaxed-WTA) + per-pixel Laplace scale (aleatoric uncertainty).
    prob_k=5,                 # number of hypotheses
    prob_eps=0.05,            # relaxed-WTA weight on non-winner heads (keeps them alive)
    prob_w_nll=0.2,           # weight on Laplace NLL (uncertainty calibration)
    prob_coarse=True,         # band-limit heads (avg-pool to prob_head res then upsample)
    prob_head_h=16, prob_head_w=32,   # coarse head resolution (enforces smoothness)

    # --- ray sampling ---
    n_rays=2048,              # rays supervised per sample per step
    eval_chunk=4096,          # rays per forward at full-grid eval

    # --- training ---
    # flip_aug: physically-correct L/R mirror augmentation for THIS data (per-sample,
    # p=0.5): depth/mask width-flip + audio L/R channel swap. NOT the SoundSpaces
    # 3-step (that also flips the spectrogram width=TIME, which breaks our models).
    flip_aug=False,
    epochs=25, batch_size=64, lr=2e-3, weight_decay=1e-4,
    num_workers=10, seed=0, device="cuda",
    out_dir="out", run_name="run",

    # --- input controls (negative controls live here so they are logged) ---
    audio_mode="stereo",      # stereo | mono | left | right | none
    shuffle_audio=False,      # break audio<->scene pairing (control B)
    mask_farfield=False,      # drop >=10m clamp pixels from the TRAIN loss (ablation)
)


def get_cfg():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_','-')}", type=lambda s: s == "True", default=v)
        else:
            p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    return SimpleNamespace(**vars(p.parse_args()))
