"""Unified idle-GPU pool: runs ALL remaining experiments across every free GPU.
Replaces the serial run_extra*/front_pool chain. Done-check skips finished runs
(metrics_test.json / prob_eval.json); cache-check defers jobs whose local cache
isn't built yet. One job per idle GPU (mem<1500MiB). Each job = train then eval.
"""
import os, subprocess, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
CK = "/root/implicit_full_cache"
EP = "--epochs 25 --num-workers 6"

def imp(name, model, seed, lr, extra, bs, cache=None, ev="--controls False"):
    cmd = (f"python train.py --model {model} --run-name {name} --seed {seed} {EP} "
           f"--batch-size {bs} --n-rays 2048 --lr {lr} {extra} "
           f"&& python eval.py --run-name {name} {ev}")
    return dict(name=name, cmd=cmd, cache=cache, art="metrics_test.json")

def fm(name, seed, arch, lr, extra, bs, cache=None):
    cmd = (f"python train_fullmap.py --arch {arch} --run-name {name} --seed {seed} {EP} "
           f"--batch-size {bs} --lr {lr} {extra} "
           f"&& python eval_fullmap.py --run-name {name} --controls False")
    return dict(name=name, cmd=cmd, cache=cache, art="metrics_test.json")

def prob(name, seed, extra, bs, cache=None):
    cmd = (f"python train_prob.py --run-name {name} --seed {seed} {EP} --batch-size {bs} "
           f"--lr 2e-3 {extra} && python eval_prob.py --run-name {name}")
    return dict(name=name, cmd=cmd, cache=cache, art="prob_eval.json")

IC5 = f"{CK}/ic5_256x512"; IC5W = f"{CK}/ic5_256x512_w20"; FOA = f"{CK}/ic4_256x512_foa"
IC_GCC = f"{CK}/ic6_256x512_gcc"; IC_WAVE = f"{CK}/ic5_256x512_wave"
IC5W30 = f"{CK}/ic5_256x512_w30"; IC5W40 = f"{CK}/ic5_256x512_w40"
IC2 = f"{CK}/ic2_256x512"; IC3 = f"{CK}/ic3_256x512"
IC2P = f"{CK}/ic2_256x512_planar"; IC5P = f"{CK}/ic5_256x512_planar"; IC_WAVEP = f"{CK}/ic2_256x512_planar_wave"
IC_RAZ = f"{CK}/ic13_256x512_raz"
JOBS = []
for s in (0, 1, 2):
    # --- front-strengthening (anti-discreteness) ---  (front-weighted-loss removed: no effect)
    JOBS += [imp(f"Bnode2_cross_hitok_s{s}", "cross", s, "3e-4", "--in-ch 2 --hi-tokens True", 12)]
    JOBS += [imp(f"Bnode2_cross_5chflip_s{s}", "cross", s, "3e-4", "--in-ch 5 --flip-aug True", 24, IC5)]
    # --- ViT encoder for cross / pix2pix U-Net encoder for cross (front-strong tokens) ---
    JOBS += [imp(f"Bnode2_cross_vitenc_s{s}", "cross", s, "3e-4", "--in-ch 2 --cross-enc vit --flip-aug True", 16)]
    JOBS += [imp(f"Bnode2_cross_unetenc_s{s}", "cross", s, "3e-4", "--in-ch 2 --cross-enc unet --ngf 64 --flip-aug True", 16)]
    JOBS += [imp(f"Bnode2_cross_unetenc5_s{s}", "cross", s, "3e-4", "--in-ch 5 --cross-enc unet --ngf 64 --flip-aug True", 16, IC5)]
    # --- #1 combo + #2 richer window (U-Net) ---
    JOBS += [fm(f"Bnode2_unet8_5chflip_s{s}", s, "unet", "2e-3", "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True", 48, IC5)]
    JOBS += [fm(f"Bnode2_unet8_5chflip_w20_s{s}", s, "unet", "2e-3", "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --audio-window-m 20", 48, IC5W)]
    # --- #3a uncertainty ---
    JOBS += [prob(f"P_5ch_k5_s{s}", s, "--prob-k 5 --in-ch 5", 32, IC5)]
    # --- aggregation suite ---
    JOBS += [imp(f"Bnode2_crossself_flip_s{s}", "crossself", s, "3e-4", "--in-ch 2 --flip-aug True", 16)]
    JOBS += [imp(f"Bnode2_cross_hitokflip_s{s}", "cross", s, "3e-4", "--in-ch 2 --hi-tokens True --flip-aug True", 12)]
    JOBS += [imp(f"Bnode2_crossself_hitokflip_s{s}", "crossself", s, "3e-4", "--in-ch 2 --hi-tokens True --flip-aug True", 8)]
# --- NEW: ray-sampling sweep (cross+flip), 2 seeds each ---
for s in (0, 1):
    for nr in (1024, 4096, 8192):
        bs = {1024: 24, 4096: 16, 8192: 8}[nr]
        JOBS.append(imp(f"Bnode2_cross_flip_nr{nr}_s{s}", "cross", s, "3e-4",
                        f"--in-ch 2 --flip-aug True --n-rays {nr}", bs))
# (ray-sampling jobs carry a 2nd --n-rays in extra; argparse last-wins so nr applies.)
# --- NEW: FOA (ambisonic, 4ch) richer input -- QUEUED LAST (runs only after all else) ---
for s in (0, 1, 2):
    JOBS.append(fm(f"Bnode2_foa_unet8_s{s}", s, "unet", "2e-3", "--ngf 64 --unet-downs 8 --in-ch 4 --audio-src foa --flip-aug True", 48, FOA))
    JOBS.append(imp(f"Bnode2_foa_cross_s{s}", "cross", s, "3e-4", "--in-ch 4 --audio-src foa --flip-aug True", 16, FOA))


# --- coarse-layout heads on U-Net8 encoder (band-limited; ray as 16x32 coarse field) ---
for s in (0, 1, 2):
    CL = "--in-ch 5 --unet-downs 8 --flip-aug True"
    JOBS.append(fm(f"C_unet8_coarse16_5chflip_s{s}", s, "unet_coarse", "2e-3", CL + " --coarse-head-h 16 --coarse-head-w 32", 48, IC5))
    JOBS.append(fm(f"C_unet8_coarse32_5chflip_s{s}", s, "unet_coarse", "2e-3", CL + " --coarse-head-h 32 --coarse-head-w 64", 48, IC5))
    JOBS.append(fm(f"C_unet8_sh4_5chflip_s{s}", s, "unet_sh", "2e-3", CL + " --coarse-sh-order 4", 48, IC5))
    JOBS.append(fm(f"C_unet8_sh6_5chflip_s{s}", s, "unet_sh", "2e-3", CL + " --coarse-sh-order 6", 48, IC5))
    JOBS.append(fm(f"C_unet8_raycoarse16_5chflip_s{s}", s, "unet_raycoarse", "2e-3", CL + " --ray-coarse-h 16 --ray-coarse-w 32", 32, IC5))
    JOBS.append(fm(f"C_unet8_coarseres_5chflip_s{s}", s, "unet_coarse_res", "2e-3", CL, 48, IC5))
    JOBS.append(fm(f"Bnode2_rayconv5d_s{s}", s, "rayconv", "2e-3", "--in-ch 5 --coarse-h 64 --coarse-w 128 --flip-aug True", 8, IC5))

# --- cross_align: high-res audio feature (e2 64x128) + ray cross-attn + conv smoothing ---
# Fixes ray "discreteness": each ray gets its own aligned local feature + neighbour
# coupling via conv, instead of only global tokens. Judge vs cross_flip / U-Net.
for s in (0, 1, 2):
    JOBS.append(fm(f"C_cross_align_5chflip_s{s}", s, "cross_align",
                   "3e-4", "--in-ch 5 --flip-aug True --ray-cross-layers 2", 24, IC5))

# --- richer-input bets: GCC-PHAT (waveform-derived ITD, 6ch) + raw-waveform WaveUNet ---
# GCC-PHAT recovers the fine binaural timing log-mag throws away (handedness/range);
# WaveUNet feeds the RAW waveform through a 1D-CNN global prior (EchoDiffusion-style).
for s in (0, 1, 2):
    JOBS.append(fm(f"Bnode2_gcc_unet8_s{s}", s, "unet", "2e-3",
                   "--ngf 64 --unet-downs 8 --in-ch 6 --audio-src gcc --flip-aug True", 48, IC_GCC))
    JOBS.append(fm(f"Bnode2_wave_unet8_s{s}", s, "wave", "2e-3",
                   "--ngf 64 --unet-downs 8 --in-ch 5 --audio-src wave --flip-aug True", 40, IC_WAVE))

# --- RayDPT (canonical): shared ray-proj + LEARNED full-decode (upsample 64->256
# +e1 skip) + local spherical attention. full-decode is default-on now. ---
for s in (0, 1, 2):
    JOBS.append(fm(f"C_raydpt_5chflip_s{s}", s, "raydpt", "3e-4",
                   "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2", 16, IC5))
# Ray-DPT-lite: 2-scale (32,64), single ray cross-attn + e2 skip + local spherical attn.
# Staged variant to isolate the fusion gain; loss dense+0.5*coarse+0.5*low.
for s in (0, 1, 2):
    JOBS.append(fm(f"C_raydptlite_5chflip_s{s}", s, "raydpt", "3e-4",
                   "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --raydpt-lite True "
                   "--ray-cross-layers 2 --w-coarse-layout 0.5", 16, IC5))

# --- EchoDiffusion "success-type" transfer (model_echo.py): frozen wav2vec2 scene
# encoder + CIDE class-embedding + cross-attention conditioning. NO 5ch, NO flip
# (in-ch 2 binaural spec + raw waveform). UNet-backbone vs Ray-backbone variants. ---
for s in (0, 1, 2):
    JOBS.append(fm(f"E_echo_unet_s{s}", s, "echo_unet", "2e-3",
                   "--in-ch 2 --audio-src wave", 24, IC_WAVE))
    JOBS.append(fm(f"E_echo_ray_s{s}", s, "echo_ray", "3e-4",
                   "--in-ch 2 --audio-src wave --ray-cross-layers 2", 12, IC_WAVE))

# --- 3D-space auxiliary losses on the winning U-Net8 (in-ch5+flip): type-2 Chamfer
# vs type-3 surface-normal. Edge-aware / anti-blob test; expect RMSE/shape change. ---
JOBS.append(fm("U_unet8_normal_s0", 0, "unet", "2e-3",
               "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --w-normal 0.1", 48, IC5))
JOBS.append(fm("U_unet8_chamfer_s0", 0, "unet", "2e-3",
               "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --w-chamfer 0.1", 48, IC5))
# per-scene SCALE guide (pred mean -> gt mean; RMSE-targeted, oracle ceiling -5.4%)
JOBS.append(fm("U_unet8_scale1_s0", 0, "unet", "2e-3",
               "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --w-scale 1.0", 48, IC5))
JOBS.append(fm("U_unet8_scale2_s0", 0, "unet", "2e-3",
               "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --w-scale 2.0", 48, IC5))
# distance-binned binaural directional WEAK guide (ITD-preserving, zero-init cross-attn)
JOBS.append(fm("E_echo_bin_s0", 0, "echo_bin", "2e-3",
               "--in-ch 2 --audio-src wave --echo-kbins 32 --echo-dmax 8.0", 24, IC_WAVE))
# RayDPT multi-scale-KV fusion: F64 cross-attends compact e4+pooled-e3+pooled-e2
# memory (no raw e2 skip-add). bs12+amp for the heavier F64 (8192 Q x 1536 KV).
JOBS.append(fm("C_raydpt_msf_s0", 0, "raydpt", "3e-4",
               "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 "
               "--raydpt-msf True --amp True", 12, IC5))
JOBS.append(fm("C_raydpt_noray_s0", 0, "raydpt", "3e-4",
               "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --raydpt-noray True", 16, IC5))
# RayDPT + acoustic Perceiver resampler: learned latents compress multi-scale acoustic
# tokens -> compact scene memory; physical ERP ray queries read it (Q x 64 latents).
JOBS.append(fm("C_raydpt_rsmp_s0", 0, "raydpt", "3e-4",
               "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 "
               "--raydpt-resampler True --resampler-latents 64 --resampler-layers 3 --amp True", 16, IC5))

# --- BEST training recipe ported from sibling repo audioresearch_audio (E2 best):
# AMP-bf16 + bs32 + lr4e-4 + w_rel=0.1. Applied to the current active models. ---
JOBS.append(fm("R_raydpt_e2_s0", 0, "raydpt", "4e-4",
               "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --w-rel 0.1", 32, IC5))
JOBS.append(fm("R_echo_unet_e2_s0", 0, "echo_unet", "4e-4",
               "--in-ch 2 --audio-src wave --amp True --w-rel 0.1", 32, IC_WAVE))
JOBS.append(fm("R_echo_ray_e2_s0", 0, "echo_ray", "4e-4",
               "--in-ch 2 --audio-src wave --ray-cross-layers 2 --amp True --w-rel 0.1", 16, IC_WAVE))

# --- 20 RayDPT improvement jobs (audioresearch_audio-inspired: loss reweighting +
# recipe). base = amp-bf16 + lr4e-4 + w_rel/w_silog sweeps + arch(msf/lite) x recipe. ---
_RB = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True"
_Q = [  # (name, lr, extra, bs)
    ("Q_rd_rel05_s0",        "4e-4", _RB + " --w-rel 0.05", 16),
    ("Q_rd_rel13_s0",        "4e-4", _RB + " --w-rel 0.13", 16),
    ("Q_rd_rel15_s0",        "4e-4", _RB + " --w-rel 0.15", 16),
    ("Q_rd_silog5_s0",       "4e-4", _RB + " --w-silog 0.5", 16),
    ("Q_rd_silog25_s0",      "4e-4", _RB + " --w-silog 0.25", 16),
    ("Q_rd_rel10silog25_s0", "4e-4", _RB + " --w-rel 0.1 --w-silog 0.25", 16),
    ("Q_rd_rel10silog5_s0",  "4e-4", _RB + " --w-rel 0.1 --w-silog 0.5", 16),
    ("Q_rd_rel10_lr5e4_s0",  "5e-4", _RB + " --w-rel 0.1", 16),
    ("Q_rd_rel10_lr3e4_s0",  "3e-4", _RB + " --w-rel 0.1", 16),
    ("Q_rd_rel10_xl3_s0",    "4e-4", _RB + " --ray-cross-layers 3 --w-rel 0.1", 16),
    ("Q_rd_rel10_wcl05_s0",  "4e-4", _RB + " --w-rel 0.1 --w-coarse-layout 0.5", 16),
    ("Q_rd_rel10_wlow1_s0",  "4e-4", _RB + " --w-rel 0.1 --w-low 1.0", 16),
    ("Q_rd_rel10_normal_s0", "4e-4", _RB + " --w-rel 0.1 --w-normal 0.1", 16),
    ("Q_rd_rel10_chamfer_s0","4e-4", _RB + " --w-rel 0.1 --w-chamfer 0.1", 16),
    ("Q_rdlite_rel10_s0",    "4e-4", _RB + " --raydpt-lite True --w-coarse-layout 0.5 --w-rel 0.1", 16),
    ("Q_rdmsf_rel10_s0",     "4e-4", _RB + " --raydpt-msf True --w-rel 0.1", 12),
    ("Q_rdmsf_rel10silog25_s0","4e-4", _RB + " --raydpt-msf True --w-rel 0.1 --w-silog 0.25", 12),
    ("Q_rdmsf_rel10_s1",     "4e-4", _RB + " --raydpt-msf True --w-rel 0.1", 12),
    ("R_raydpt_e2_s1",       "4e-4", _RB + " --w-rel 0.1", 32),   # complete E2 3-seed
    ("R_raydpt_e2_s2",       "4e-4", _RB + " --w-rel 0.1", 32),
]
for _nm, _lr, _ex, _bs in _Q:
    _seed = int(_nm.rsplit("_s", 1)[1])
    JOBS.append(fm(_nm, _seed, "raydpt", _lr, _ex, _bs, IC5))

# --- drop FOA (ambisonic 4ch input confirmed worse than binaural: foa_unet8 ~0.992) ---

























# --- M (max-margin): maximise min(RMSE-margin, AbsRel-margin) over baselines. RMSE is the
# bottleneck (ViT 1.424 strong), so push RMSE->1.40 via ALL inward levers (E51 2-3 block +
# berHu-low + heavy normal/grad/scale) while holding AbsRel<0.51 (moderate w_rel). ---
_M = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True --berhu-low True"
_MC = [
  ("M1",  "--coarse-sa-blocks 2 --w-rel 0.03 --w-normal 0.15 --w-grad 0.05 --w-scale 0.05"),
  ("M2",  "--coarse-sa-blocks 2 --w-rel 0.04 --w-normal 0.2 --w-scale 0.1"),
  ("M3",  "--coarse-sa-blocks 3 --w-rel 0.03 --w-normal 0.15 --w-grad 0.05"),
  ("M4",  "--coarse-sa-blocks 2 --w-rel 0.03 --w-normal 0.2 --w-grad 0.05 --w-scale 0.1"),
  ("M5",  "--coarse-sa-blocks 2 --w-rel 0.04 --w-normal 0.15 --w-scale 0.05 --w-depth-gamma 0.2"),
  ("M6",  "--coarse-sa-blocks 2 --w-rel 0.04 --w-normal 0.2 --w-grad 0.05 --w-scale 0.05"),
  ("M7",  "--coarse-sa-blocks 3 --w-rel 0.04 --w-normal 0.2 --w-grad 0.05 --w-scale 0.05"),
  ("M8",  "--coarse-sa-blocks 2 --w-rel 0.03 --w-normal 0.25 --w-grad 0.05"),
  ("M9",  "--coarse-sa-blocks 2 --w-rel 0.04 --w-normal 0.15 --w-grad 0.03 --w-scale 0.15"),
  ("M10", "--coarse-sa-blocks 2 --w-rel 0.035 --w-normal 0.18 --w-grad 0.05 --w-scale 0.08 --w-ema 0.995"),
  ("M11", "--coarse-sa-blocks 2 --w-rel 0.03 --w-normal 0.2 --w-scale 0.15 --w-depth-gamma 0.15"),
  ("M12", "--coarse-sa-blocks 3 --w-rel 0.035 --w-normal 0.2 --w-grad 0.05 --w-scale 0.1"),
]
for _nm,_ex in _MC:
    JOBS.append(fm(f"{_nm}_s0", 0, "raydpt", "4e-4", _M+" "+_ex, 24, IC5))





# --- FINAL v3 multi-res STFT (S19, repo global champion): 2ch magnitude x 3 windows (6ch),
# champion loss stack, NO TTA (eval-only ~0.002 gain here, excluded per request). ---
IC6MRES = f"{CK}/ic6_256x512_planar_mres"
_V3M = ("--in-ch 6 --audio-src mres --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True "
        "--depth-type planar --w-rel 0.1 --w-normal 0.15 --w-grad 0.05 --w-ema 0.995")
for _s in (0, 1, 2):
    JOBS.append(fm(f"finalv3_raydpt_mres_champ_s{_s}",     _s, "raydpt", "4e-4", _V3M, 24, IC6MRES))
    JOBS.append(fm(f"finalv3_raydpt_mres_champ_e51_s{_s}", _s, "raydpt", "4e-4", _V3M + " --coarse-sa-blocks 2", 24, IC6MRES))
# --- 5ch baselines (planar): BatVision/pUNet/pViT on ic5_planar; EchoDiffusion on ic5_planar_wave.
# Complements the 2ch finalv2 baselines for a channel-matched comparison vs RayDPT 5ch. ---
IC5PW = f"{CK}/ic5_256x512_planar_wave"
for _s in (0, 1, 2):
    JOBS.append(fm(f"finalv2_batvision_5ch_s{_s}", _s, "batvis",   "2e-3", "--in-ch 5 --flip-aug True --depth-type planar", 32, IC5P))
    JOBS.append(fm(f"finalv2_preunet_5ch_s{_s}",   _s, "presnet",  "3e-4", "--in-ch 5 --flip-aug True --depth-type planar", 24, IC5P))
    JOBS.append(fm(f"finalv2_previt_5ch_s{_s}",    _s, "pvit",     "3e-4", "--in-ch 5 --flip-aug True --depth-type planar", 16, IC5P))
    JOBS.append(fm(f"finalv2_echodiff_5ch_s{_s}",  _s, "echodiff", "2e-3", "--in-ch 5 --audio-src wave --depth-type planar", 16, IC5PW))
# --- FINAL v3: repo champion recipes (auto_audio_depth_estimation) applied to PLANAR, 2ch-focused.
# Champion stack = w_rel0.1 (E2) + EMA0.995 (E16) + coarse-geo-self-attn + grad-loss (E34/P_b3),
# + E127/E128 TTA (eval-time L/R-flip). Variants: E51 (2-block coarse-sa), E117 (berHu-low RMSE lever). ---
_V3 = ("--flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True --depth-type planar "
       "--w-rel 0.1 --w-normal 0.15 --w-grad 0.05 --w-ema 0.995 --eval-tta-flip True")
for _s in (0, 1, 2):
    JOBS.append(fm(f"finalv3_raydpt_2ch_champ_s{_s}",      _s, "raydpt", "4e-4", "--in-ch 2 " + _V3, 24, IC2P))
    JOBS.append(fm(f"finalv3_raydpt_2ch_champ_e51_s{_s}",  _s, "raydpt", "4e-4", "--in-ch 2 " + _V3 + " --coarse-sa-blocks 2", 24, IC2P))
    JOBS.append(fm(f"finalv3_raydpt_2ch_champ_bhlow_s{_s}",_s, "raydpt", "4e-4", "--in-ch 2 " + _V3 + " --berhu-low True", 24, IC2P))
    JOBS.append(fm(f"finalv3_raydpt_5ch_champ_s{_s}",      _s, "raydpt", "4e-4", "--in-ch 5 " + _V3, 24, IC5P))
# --- FINAL v2 (protocol-corrected: masked coarse/low-pass/normal losses, batch-invariant metrics,
# documented no-ray = learned-positional). 8 model families x 3 seeds, requested order. HIGHEST. ---
_RB = "--flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True --w-rel 0.05"
for _s in (0, 1, 2):
    JOBS.append(fm(f"finalv2_batvision_2ch_s{_s}", _s, "batvis",   "2e-3", "--in-ch 2 --flip-aug True --depth-type planar", 32, IC2P))
    JOBS.append(fm(f"finalv2_preunet_2ch_s{_s}",   _s, "presnet",  "3e-4", "--in-ch 2 --flip-aug True --depth-type planar", 24, IC2P))
    JOBS.append(fm(f"finalv2_previt_2ch_s{_s}",    _s, "pvit",     "3e-4", "--in-ch 2 --flip-aug True --depth-type planar", 16, IC2P))
    JOBS.append(fm(f"finalv2_echodiff_2ch_s{_s}",  _s, "echodiff", "2e-3", "--in-ch 2 --audio-src wave --depth-type planar", 16, IC_WAVEP))
    JOBS.append(fm(f"finalv2_raydpt_2ch_ray_s{_s}",   _s, "raydpt", "4e-4", "--in-ch 2 --depth-type planar " + _RB, 24, IC2P))
    JOBS.append(fm(f"finalv2_raydpt_5ch_ray_s{_s}",   _s, "raydpt", "4e-4", "--in-ch 5 --depth-type planar " + _RB, 24, IC5P))
    JOBS.append(fm(f"finalv2_raydpt_2ch_noray_s{_s}", _s, "raydpt", "4e-4", "--in-ch 2 --depth-type planar " + _RB + " --raydpt-noray True", 24, IC2P))
    JOBS.append(fm(f"finalv2_raydpt_5ch_noray_s{_s}", _s, "raydpt", "4e-4", "--in-ch 5 --depth-type planar " + _RB + " --raydpt-noray True", 24, IC5P))
# --- CF (Cue-Factorization, PRIORITY 1): cue-specific input stems (Group A) + configurable
# ray K/V routing at coarse F16 (Group B). Parent baseline = Q8_csa_wrel05_normal10 (SAME recipe,
# no cue). Minimally invasive: RayDPT intact; only audio-repr + F16 K/V source change. n=1 first. ---
_CF = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True --w-rel 0.05 --w-normal 0.1"
_CFC = [
  ("CF_stems",    "--cue-stems True"),                                              # A1 two-stem input
  ("CF_route_ff", "--cue-route True --kv-key-source fused --kv-value-source fused"),        # B0 control
  ("CF_route_sf", "--cue-route True --kv-key-source spatial --kv-value-source fused"),      # B1
  ("CF_route_mf", "--cue-route True --kv-key-source magnitude --kv-value-source fused"),    # B2 control vs B1
  ("CF_route_sm", "--cue-route True --kv-key-source spatial --kv-value-source magnitude"),  # B3 role hypothesis
  ("CF_route_ms", "--cue-route True --kv-key-source magnitude --kv-value-source spatial"),  # B4 reverse control
]
for _nm,_ex in _CFC:
    JOBS.append(fm(f"{_nm}_s0", 0, "raydpt", "4e-4", _CF+" "+_ex, 24, IC5))

# --- CF PRIORITY 2 (controls) + PRIORITY 3 (fusion/dual/ratio). Paired with the main
# hypothesis routing (spatial-K/fused-V) so controls are interpretable. Promote-by-signal. ---
_CF2 = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True --w-rel 0.05 --w-normal 0.1"
_SF = "--cue-route True --kv-key-source spatial --kv-value-source fused"
_CFP = [
  # P2 controls
  ("CF_capmatch", "--ngf 72"),                                          # F3 capacity-matched single-enc (no cue)
  ("CF_shared_sf", _SF + " --cue-adapter True"),                        # F4/A3 shared-adapter
  ("CF_dup_sf",    _SF + " --cue-dup-input True"),                      # F2 duplicate-input
  ("CF_rand_sf",   _SF + " --cue-random-split True"),                  # F1 random split
  # P3 fusion (C) / dual (D) / ratio
  ("CF_fuse_add",  _SF + " --cue-fused-mode add"),
  ("CF_fuse_gate", _SF + " --cue-fused-mode gate"),
  ("CF_fuse_concat", _SF + " --cue-fused-mode concat"),
  ("CF_dual",      "--cue-route True --cue-dual True"),                 # D1 parallel spatial+mag attn
  ("CF_stems_r21", "--cue-stems True --cue-cmag 64 --cue-cspatial 32"), # mag-rich stem 2:1
  ("CF_stems_r12", "--cue-stems True --cue-cmag 32 --cue-cspatial 64"), # spatial-rich stem 1:2
]
for _nm,_ex in _CFP:
    JOBS.append(fm(f"{_nm}_s0", 0, "raydpt", "4e-4", _CF2+" "+_ex, 24, IC5))
# --- RV (RayViT): pretrained ViT encoder (fine-tuned) + ray-conditioned cross-attn decoder.
# 3 modes + anticipated-problem variants: frozen ViT (overfit), scratch (ImageNet-prior control),
# hybrid+champion-loss / +2block (fine-detail + best recipe). lr3e-4, bs12 (86M ViT heavy). ---
_RV = "--arch rayvit --in-ch 5 --flip-aug True --ray-cross-layers 2 --raydpt-coarse-sa True"
_CH = "--w-rel 0.05 --w-normal 0.15 --w-grad 0.05"
_RC = [
  ("RV_single",        _RV + " --rayvit-mode single"),
  ("RV_multiscale",    _RV + " --rayvit-mode multiscale"),
  ("RV_hybrid",        _RV + " --rayvit-mode hybrid"),
  ("RV_single_frozen", _RV + " --rayvit-mode single --vit-freeze True"),        # anticipate overfit
  ("RV_hybrid_frozen", _RV + " --rayvit-mode hybrid --vit-freeze True"),
  ("RV_single_scratch",_RV + " --rayvit-mode single --vit-pretrained False"),   # ImageNet-prior control
  ("RV_hybrid_champ",  _RV + " --rayvit-mode hybrid " + _CH),                    # + champion loss
  ("RV_hybrid_2block", _RV + " --rayvit-mode hybrid --coarse-sa-blocks 2 " + _CH),
]
for _nm,_ex in _RC:
    JOBS.append(fm(f"{_nm}_s0", 0, "rayvit", "3e-4", _ex, 12, IC5))
# --- S (SOTA): both-win frontier (P_b3/P_b4/P_r2 recipes) + 2 NEW audioresearch_audio levers:
# E51 post-fusion geo self-attn x2 blocks (reopened frontier at convergence) + E117 berHu-on-
# LOWPASS-term-only (berHu RMSE lever without main-term frontier slide). Aim: RMSE & AbsRel both SOTA. ---
_S = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
_PB3="--w-rel 0.05 --w-normal 0.15 --w-grad 0.05"; _PB4="--w-rel 0.03 --w-normal 0.1 --w-scale 0.05"; _PR2="--w-rel 0.05 --w-normal 0.1 --w-grad 0.05 --w-scale 0.1"
_SC = [
  ("S_e51_pb3",  f"--coarse-sa-blocks 2 {_PB3}"),
  ("S_e51_pb4",  f"--coarse-sa-blocks 2 {_PB4}"),
  ("S_e51_pr2",  f"--coarse-sa-blocks 2 {_PR2}"),
  ("S_bhlow_pb3",f"--berhu-low True {_PB3}"),
  ("S_bhlow_pb4",f"--berhu-low True {_PB4}"),
  ("S_bhlow_pr2",f"--berhu-low True {_PR2}"),
  ("S_full_pb3", f"--coarse-sa-blocks 2 --berhu-low True {_PB3}"),
  ("S_full_pb4", f"--coarse-sa-blocks 2 --berhu-low True {_PB4}"),
  ("S_full_pr2", f"--coarse-sa-blocks 2 --berhu-low True {_PR2}"),
  ("S_full_a",   f"--coarse-sa-blocks 2 --berhu-low True --w-rel 0.05 --w-normal 0.15 --w-grad 0.05 --w-scale 0.1"),
  ("S_ema_pb3",  f"--w-ema 0.995 {_PB3}"),
  ("S_ema_full", f"--coarse-sa-blocks 2 --berhu-low True --w-ema 0.995 {_PB3}"),
  ("S_ema_e51",  f"--coarse-sa-blocks 2 --w-ema 0.995 {_PB3}"),
  ("S_bhlow_wlow075", f"--berhu-low True --w-low 0.75 {_PB3}"),
  ("S_bhlow_wlow10",  f"--berhu-low True --w-low 1.0 {_PB3}"),
  ("S_abs_g05",  f"--coarse-sa-blocks 2 --berhu-low True --w-depth-gamma -0.5 --w-normal 0.2"),
  ("S_abs_rel08",f"--coarse-sa-blocks 2 --berhu-low True --w-rel 0.08 --w-normal 0.2"),
  ("S_3block_pb3",  f"--coarse-sa-blocks 3 {_PB3}"),
  ("S_3block_full", f"--coarse-sa-blocks 3 --berhu-low True {_PB3}"),
  ("S_kitchen",  f"--coarse-sa-blocks 2 --berhu-low True --w-ema 0.995 --w-rel 0.05 --w-normal 0.15 --w-grad 0.05 --w-scale 0.1"),
]
for _nm,_ex in _SC:
    JOBS.append(fm(f"{_nm}_s0", 0, "raydpt", "4e-4", _S+" "+_ex, 24, IC5))
# --- P (Pareto/polarity): 20 configs spanning the RMSE<->AbsRel frontier. Each keeps a
# COUNTER-lever so both stay below the baseline thresholds (RMSE<1.424, AbsRel<0.537) while
# maximising spread. RMSE-lean: heavy structure(normal/grad/scale/low/berhu/gamma+) + light rel;
# AbsRel-lean: heavy rel/gamma- + strong normal to hold RMSE. coarse-sa RayDPT 5ch base. ---
_P = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
_PC = [
  # RMSE-leaning (low RMSE, keep AbsRel<0.537 via light rel)
  ("P_r1", "--w-rel 0.03 --w-normal 0.2 --w-scale 0.1"),
  ("P_r2", "--w-rel 0.03 --w-normal 0.1 --w-grad 0.05 --w-scale 0.1"),
  ("P_r3", "--w-rel 0.03 --w-normal 0.2 --w-low 0.75"),
  ("P_r4", "--w-rel 0.03 --w-scale 0.2 --w-normal 0.1"),
  ("P_r5", "--berhu True --w-rel 0.05"),
  ("P_r6", "--w-depth-gamma 0.3 --w-rel 0.05 --w-normal 0.1"),
  # balanced knee
  ("P_b1", "--w-rel 0.05 --w-normal 0.1 --w-grad 0.05"),
  ("P_b2", "--w-rel 0.05 --w-normal 0.1 --w-scale 0.1"),
  ("P_b3", "--w-rel 0.05 --w-normal 0.15 --w-grad 0.05"),
  ("P_b4", "--w-rel 0.04 --w-normal 0.1 --w-scale 0.05"),
  ("P_b5", "--w-rel 0.05 --w-grad 0.1 --w-scale 0.1"),
  # AbsRel-leaning (low AbsRel, keep RMSE<1.424 via strong normal)
  ("P_a1", "--w-rel 0.1 --w-normal 0.15"),
  ("P_a2", "--w-rel 0.1 --w-normal 0.2 --w-scale 0.1"),
  ("P_a3", "--w-depth-gamma -0.5 --w-normal 0.15"),
  ("P_a4", "--w-depth-gamma -0.5 --w-normal 0.2 --w-scale 0.1"),
  ("P_a5", "--w-rel 0.13 --w-normal 0.2 --w-grad 0.05"),
  ("P_a6", "--w-depth-gamma -0.3 --w-normal 0.1 --w-rel 0.05"),
  # extreme-but-guarded (max polarity + strong counter-lever)
  ("P_x1", "--berhu True --w-rel 0.1 --w-normal 0.1"),
  ("P_x2", "--w-depth-gamma -1.0 --w-normal 0.2 --w-scale 0.1"),
  ("P_x3", "--w-depth-gamma 0.5 --w-rel 0.1 --w-normal 0.1"),
]
for _nm,_ex in _PC:
    JOBS.append(fm(f"{_nm}_s0", 0, "raydpt", "4e-4", _P + " " + _ex, 24, IC5))
# --- metric-best RayDPT (5ch) 3-seed confirm: best-RMSE (berhu weighting) + best-AbsRel
# (depth-gamma -1.0). champion coarse-sa base; single-metric-optimised (frontier extremes). ---
_B = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
for _sd in (1, 2):
    JOBS.append(fm(f"F3_bestRMSE_s{_sd}",   _sd, "raydpt", "4e-4", _B + " --berhu True", 24, IC5))
    JOBS.append(fm(f"F3_bestAbsRel_s{_sd}", _sd, "raydpt", "4e-4", _B + " --w-depth-gamma -1.0", 24, IC5))
# --- Our RayDPT champion at NATIVE-BASELINE 2ch spectrogram input (isolates ARCHITECTURE
# vs input richness: all-2ch fair comparison against B2_* baselines). Champion loss recipe. ---
_CH2 = "--ngf 64 --unet-downs 8 --in-ch 2 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True --w-rel 0.05 --w-normal 0.1"
for s in (0, 1, 2):
    JOBS.append(fm(f"F2_raydpt_s{s}", s, "raydpt", "4e-4", _CH2, 24, IC2))

# --- Comparison-method baselines with their NATIVE 2ch spectrogram input (fair): pretrained
# UNet/ViT/BatVision take a 2ch log-mag binaural spectrogram (EchoDiffusion already uses
# 2ch spec + raw wave). Supersedes the 5ch runs for the paper comparison table. ---
for s in (0, 1, 2):
    JOBS.append(fm(f"B2_presnet_s{s}", s, "presnet", "3e-4", "--in-ch 2 --flip-aug True", 24, IC2))
    JOBS.append(fm(f"B2_pvit_s{s}",    s, "pvit",    "3e-4", "--in-ch 2 --flip-aug True", 16, IC2))
    JOBS.append(fm(f"B2_batvis_s{s}",  s, "batvis",  "2e-3", "--in-ch 2 --flip-aug True", 32, IC2))

# --- Comparison methods (faithful ports): BatVision (Christensen, pix2pix UNet) + EchoDiffusion
# (wjzhang-ai: CIDE/wav2vec2 + ASPP-ASFF + diffusion UNet, 2ch spec + raw wave). 3-seed each. ---
for s in (0, 1, 2):
    JOBS.append(fm(f"B_batvis_s{s}",   s, "batvis",   "2e-3", "--in-ch 5 --flip-aug True", 32, IC5))
    JOBS.append(fm(f"B_echodiff_s{s}", s, "echodiff", "2e-3", "--in-ch 2 --audio-src wave", 16, IC_WAVE))

# --- Published-baseline pretrained backbones (channel-adapt: 1x1 conv in_ch->3ch pseudo-RGB
# + pretrained ViT-B/16 / ResNet-50 + decoder). "change only the channels" deploy baselines. ---
for s in (0, 1, 2):
    JOBS.append(fm(f"B_pvit_s{s}",    s, "pvit",    "3e-4", "--in-ch 5 --flip-aug True", 16, IC5))
    JOBS.append(fm(f"B_presnet_s{s}", s, "presnet", "3e-4", "--in-ch 5 --flip-aug True", 24, IC5))

# --- AAAI final (F_): Task1 champion multi-seed + Task2 raymlp+CSA (global audio code,
# no per-ray cross-attn) decisive ablation. Champion = Q8_csa_wrel05_normal10 config. ---
_CH = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True --w-rel 0.05 --w-normal 0.1"
JOBS.append(fm("F_champion_s1", 1, "raydpt", "4e-4", _CH, 24, IC5))                       # Task1
JOBS.append(fm("F_champion_s2", 2, "raydpt", "4e-4", _CH, 24, IC5))
for s in (0, 1, 2):
    JOBS.append(fm(f"F_raymlpcsa_s{s}", s, "raydpt", "4e-4", _CH + " --cross-mode global", 24, IC5))  # Task2

# --- Q17: ray/spherical GROUNDING that could actually help (grounds the AUDIO physics,
# not the redundant output grid). (a) RayBank ear-geometry mic-PE + SH-PE on the ray
# query (immediate). (b) NEW range-azimuth steered acoustic image input (ToF+azimuth
# physics, in_ch=13) on U-Net8 + coarse-sa champion (needs raz cache). ---
_C = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True --w-rel 0.03"
JOBS.append(fm("Q17_csa_micpe_s0", 0, "raydpt", "4e-4", _C + " --use-mic-pe True", 24, IC5))
JOBS.append(fm("Q17_csa_shpe_s0",  0, "raydpt", "4e-4", _C + " --use-sh-pe True", 24, IC5))
JOBS.append(fm("Q17_csa_micsh_s0", 0, "raydpt", "4e-4", _C + " --use-mic-pe True --use-sh-pe True", 24, IC5))
_RZ = "--ngf 64 --unet-downs 8 --in-ch 13 --audio-src raz --flip-aug True"
JOBS.append(fm("Q17_unet_raz_s0", 0, "unet", "2e-3", _RZ, 40, IC_RAZ))                                  # range-azimuth on U-Net
_RZC = "--ngf 64 --unet-downs 8 --in-ch 13 --audio-src raz --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
JOBS.append(fm("Q17_csa_raz_s0", 0, "raydpt", "4e-4", _RZC + " --w-rel 0.03", 20, IC_RAZ))              # + coarse-sa
JOBS.append(fm("Q17_csa_raz_normal_s0", 0, "raydpt", "4e-4", _RZC + " --w-rel 0.05 --w-normal 0.1", 20, IC_RAZ))  # + champion loss

# --- Q16: ADDITIVE input-channel sweep (2->3->5), 3-seed. Complements Q11 leave-one-out.
# in_ch2 = [logL,logR] (magnitude only); in_ch3 = +ILD (level cue); in_ch5 = +cos/sin-IPD
# (phase). in_ch5 = Bnode2_unet8_5chflip (done, n=3). Tests "how much does each added
# channel group actually buy" -> justify channel count OR show insensitivity (obs. ceiling). ---
for s in (0, 1, 2):
    JOBS.append(fm(f"Q16_unet_ic2_s{s}", s, "unet", "2e-3", "--ngf 64 --unet-downs 8 --in-ch 2 --flip-aug True", 48, IC2))
    JOBS.append(fm(f"Q16_unet_ic3_s{s}", s, "unet", "2e-3", "--ngf 64 --unet-downs 8 --in-ch 3 --flip-aug True", 48, IC3))

# --- Q15: LONG-WINDOW input for RMSE. default window=10m covers depth<=5m only; longer
# windows keep LATER reflections (far surfaces + higher-order bounces = room geometry) =
# the far-depth info RMSE needs. base = coarse-sa RayDPT (+normal, the RMSE combo). ---
_CN = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True --w-rel 0.05 --w-normal 0.1"
_CB = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
JOBS.append(fm("Q15_csa_norm_w20_s0", 0, "raydpt", "4e-4", _CN + " --audio-window-m 20", 24, IC5W))    # cache ready now
JOBS.append(fm("Q15_csa_base_w20_s0", 0, "raydpt", "4e-4", _CB + " --audio-window-m 20", 24, IC5W))
JOBS.append(fm("Q15_csa_norm_w30_s0", 0, "raydpt", "4e-4", _CN + " --audio-window-m 30", 24, IC5W30))   # after cache build
JOBS.append(fm("Q15_csa_norm_w40_s0", 0, "raydpt", "4e-4", _CN + " --audio-window-m 40", 24, IC5W40))
JOBS.append(fm("Q15_csa_base_w30_s0", 0, "raydpt", "4e-4", _CB + " --audio-window-m 30", 24, IC5W30))   # pure window effect
JOBS.append(fm("Q15_unet_norm_w30_s0", 0, "unet", "2e-3", "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --w-normal 0.1 --audio-window-m 30", 48, IC5W30))

# --- Q14: alternative WEIGHTING schemes (not scalar aux, but per-pixel/error weighting).
# depth-gamma: dense L1 * gt**gamma (gamma>0 upweights FAR -> targets RMSE; <0 NEAR ->
# AbsRel, a smoother alternative to w_rel). berhu: reverse-Huber. base = coarse-sa RayDPT. ---
_C = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
_Q14 = [
    ("Q14_gamma_p05_s0",       _C + " --w-depth-gamma 0.5"),    # far-weighted -> RMSE
    ("Q14_gamma_p10_s0",       _C + " --w-depth-gamma 1.0"),
    ("Q14_gamma_p15_s0",       _C + " --w-depth-gamma 1.5"),
    ("Q14_gamma_n05_s0",       _C + " --w-depth-gamma -0.5"),   # near-weighted -> AbsRel (vs w_rel)
    ("Q14_gamma_n10_s0",       _C + " --w-depth-gamma -1.0"),
    ("Q14_berhu_s0",           _C + " --berhu True"),           # robust (paper loss)
    ("Q14_berhu_rel03_s0",     _C + " --berhu True --w-rel 0.03"),
    ("Q14_gamma_p05_rel03_s0", _C + " --w-depth-gamma 0.5 --w-rel 0.03"),   # far-weight + AbsRel = both?
    ("Q14_gamma_p10_scale10_s0", _C + " --w-depth-gamma 1.0 --w-scale 0.1"),
]
for _nm,_ex in _Q14:
    JOBS.append(fm(_nm, 0, "raydpt", "4e-4", _ex, 24, IC5))

# --- Q13: LOSS ablation on a FIXED base (coarse-sa RayDPT, amp, lr4e-4). Isolate each
# loss term's marginal contribution. Already-done on this base: none(Q7_csa_norel),
# w_rel 0.03/0.05/0.1, w_scale0.1, w_normal/chamfer, w_low sweep. Fill gaps: w_silog,
# w_grad (alone), structural terms (w_low=0, w_coarse_layout=0), + key combos. ---
_C = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
_Q13 = [
    ("Q13_loss_silog25_s0",       _C + " --w-silog 0.25"),
    ("Q13_loss_silog5_s0",        _C + " --w-silog 0.5"),
    ("Q13_loss_grad05_s0",        _C + " --w-grad 0.05"),
    ("Q13_loss_grad10_s0",        _C + " --w-grad 0.1"),
    ("Q13_loss_wlow0_s0",         _C + " --w-low 0.0"),          # remove low-pass (E11: helps RMSE?)
    ("Q13_loss_wcoarse0_s0",      _C + " --w-coarse-layout 0.0"),# remove coarse-layout supervision
    ("Q13_loss_rel03_grad05_s0",  _C + " --w-rel 0.03 --w-grad 0.05"),
    ("Q13_loss_rel03_normal10_s0",_C + " --w-rel 0.03 --w-normal 0.1"),
    ("Q13_loss_rel03_silog25_s0", _C + " --w-rel 0.03 --w-silog 0.25"),
    ("Q13_loss_rel03_scale10_s0", _C + " --w-rel 0.03 --w-scale 0.1"),
]
for _nm,_ex in _Q13:
    JOBS.append(fm(_nm, 0, "raydpt", "4e-4", _ex, 24, IC5))

# --- Q12: FAIR CONTROL. Isolate the coarse-sa architecture vs U-Net by giving U-Net8
# the SAME loss recipe as the Q7 champion. (a) U-Net + w_rel0.03 at U-Net own lr (2e-3),
# (b) U-Net + w_rel0.03 + amp + lr4e-4 (Q7 EXACT recipe minus arch), (c) + w_scale.
# 3-seed both the control (a) and the Q7 champion for a publishable mean+/-std. ---
_UB = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True"
for s in (0, 1, 2):
    JOBS.append(fm(f"Q12_unet_wrel03_s{s}", s, "unet", "2e-3", _UB + " --w-rel 0.03", 48, IC5))   # fair: U-Net own recipe + w_rel
JOBS.append(fm("Q12_unet_wrel03_amp_s0", 0, "unet", "4e-4", _UB + " --w-rel 0.03 --amp True", 48, IC5))   # exact Q7 recipe, arch=unet
JOBS.append(fm("Q12_unet_wrel03_wscale10_s0", 0, "unet", "2e-3", _UB + " --w-rel 0.03 --w-scale 0.1", 48, IC5))
# Q7 champion 3-seed confirm (s0 done)
_C = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
JOBS.append(fm("Q7_csa_wrel03_s1", 1, "raydpt", "4e-4", _C + " --w-rel 0.03", 24, IC5))
JOBS.append(fm("Q7_csa_wrel03_s2", 2, "raydpt", "4e-4", _C + " --w-rel 0.03", 24, IC5))

# --- Q11: 5ch input per-channel ablation (zero out one channel). channels =
# [0 logL, 1 logR, 2 ILD, 3 cosIPD, 4 sinIPD]. base = champion U-Net8 5ch+flip.
# reference (none-zeroed) = Bnode2_unet8_5chflip (n=3, MAE 0.893). ---
_UB = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True"
for _i,_nm in [(0,"logL"),(1,"logR"),(2,"ILD"),(3,"cosIPD"),(4,"sinIPD")]:
    JOBS.append(fm(f"Q11_zc{_i}_{_nm}_s0", 0, "unet", "2e-3", _UB + f" --zero-chan {_i}", 48, IC5))

# --- Q10: "simple ViT" check. Pretrained ViT-B/16 (A22/A23) already lost to U-Net8
# (~0.905 vs 0.886). Test simpler forms on the fair 5ch input: frozen backbone (only
# 3.8M trainable adapter+decoder = simplest), from-scratch (no ImageNet prior). ---
_V = "--arch vit --in-ch 5 --flip-aug True"
JOBS.append(fm("Q10_vit_frozen_s0",  0, "vit", "3e-4", _V + " --vit-freeze True", 32, IC5))     # simplest: frozen backbone
JOBS.append(fm("Q10_vit_scratch_s0", 0, "vit", "3e-4", _V + " --vit-pretrained False", 32, IC5)) # no ImageNet prior
JOBS.append(fm("Q10_vit_5ch_s0",     0, "vit", "3e-4", _V, 32, IC5))                              # pretrained on fair 5ch input

# --- Q9: ray-GROUNDING ablation. Is grounding decode in physical ray/spherical geometry
# meaningful? 3 grounding sources: (1) ray-dir queries (RayBank), (2) spherical local attn,
# (3) coarse-sa cos-ang bias. full-grounding = Q6_csaonly (done, n=2). Turn OFF one-at-a-time
# + ALL-off (geometry-agnostic). base = coarse-sa + amp + lr4e-4 + w_rel0.1. ---
_G = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --w-rel 0.1 --raydpt-coarse-sa True"
JOBS.append(fm("Q9_ground_noquery_s0",   0, "raydpt", "4e-4", _G + " --raydpt-noray True", 24, IC5))          # off (1)
JOBS.append(fm("Q9_ground_planarlsa_s0", 0, "raydpt", "4e-4", _G + " --lsa-mode planar", 24, IC5))            # off (2)
JOBS.append(fm("Q9_ground_nocsageo_s0",  0, "raydpt", "4e-4", _G + " --coarse-sa-geo False", 24, IC5))        # off (3)
JOBS.append(fm("Q9_ground_none_s0", 0, "raydpt", "4e-4", _G + " --raydpt-noray True --lsa-mode planar --coarse-sa-geo False", 24, IC5))  # ALL off
JOBS.append(fm("Q9_ground_none_s1", 1, "raydpt", "4e-4", _G + " --raydpt-noray True --lsa-mode planar --coarse-sa-geo False", 24, IC5))
JOBS.append(fm("Q9_ground_full_s2", 2, "raydpt", "4e-4", _G, 24, IC5))   # 3rd seed of full (Q6_csaonly has s0,s1)

# --- Q8: 20 RMSE-balanced follow-ups. coarse-sa base (proven), NO EMA (hurts RMSE),
# new axes: 3D normal/chamfer loss (best-RMSE U-Net used normal), other inputs (GCC
# RMSE 1.425 / w20), local-spherical window sweep (free lever per EXPERIMENTS.md),
# w_low low-pass + w_scale mean-match + low-lr anneal (E20 best-RMSE) + gated combos. ---
_C  = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
_CG = "--ngf 64 --unet-downs 8 --in-ch 6 --audio-src gcc --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
_CW = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --audio-window-m 20 --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
_Q8 = [
    # 3D/structural losses (RMSE + shape)
    ("Q8_csa_normal10_s0",           "4e-4", _C + " --w-normal 0.1", 24, "IC5"),
    ("Q8_csa_wrel05_normal10_s0",    "4e-4", _C + " --w-rel 0.05 --w-normal 0.1", 24, "IC5"),
    ("Q8_csa_chamfer10_s0",          "4e-4", _C + " --w-chamfer 0.1", 24, "IC5"),
    ("Q8_csa_wrel05_normal05_wscale10_s0","4e-4", _C + " --w-rel 0.05 --w-normal 0.05 --w-scale 0.1", 24, "IC5"),
    # other inputs (GCC has good RMSE / w20 richer window)
    ("Q8_gcc_csa_wrel05_s0",         "4e-4", _CG + " --w-rel 0.05", 16, "IC_GCC"),
    ("Q8_gcc_csa_norel_s0",          "4e-4", _CG, 16, "IC_GCC"),
    ("Q8_gcc_csa_wscale10_s0",       "4e-4", _CG + " --w-scale 0.1", 16, "IC_GCC"),
    ("Q8_w20_csa_wrel05_s0",         "4e-4", _CW + " --w-rel 0.05", 24, "IC5W"),
    # local spherical-attention window sweep (free lever)
    ("Q8_csa_win7_s0",               "4e-4", _C + " --raydpt-win32 7 --raydpt-win64 5", 24, "IC5"),
    ("Q8_csa_win9_s0",               "4e-4", _C + " --raydpt-win32 9 --raydpt-win64 5", 24, "IC5"),
    ("Q8_csa_wrel05_win7_s0",        "4e-4", _C + " --w-rel 0.05 --raydpt-win32 7 --raydpt-win64 5", 24, "IC5"),
    # w_low low-pass sweep (RMSE lever)
    ("Q8_csa_wlow075_s0",            "4e-4", _C + " --w-low 0.75", 24, "IC5"),
    ("Q8_csa_wlow15_s0",             "4e-4", _C + " --w-low 1.5", 24, "IC5"),
    ("Q8_csa_wrel03_wlow075_s0",     "4e-4", _C + " --w-rel 0.03 --w-low 0.75", 24, "IC5"),
    # low-lr anneal (E20 best-RMSE)
    ("Q8_csa_lr3e4_s0",              "3e-4", _C, 24, "IC5"),
    ("Q8_csa_wrel05_lr3e4_s0",       "3e-4", _C + " --w-rel 0.05", 24, "IC5"),
    # w_scale mean-match sweep (RMSE)
    ("Q8_csa_wscale20_s0",           "4e-4", _C + " --w-scale 0.2", 24, "IC5"),
    ("Q8_csa_wscale30_s0",           "4e-4", _C + " --w-scale 0.3", 24, "IC5"),
    # combined best-RMSE levers (E29 gated + low-pass/normal)
    ("Q8_csa_gated_wlow075_wrel05_s0","4e-4", _C + " --raydpt-gated-skip True --w-low 0.75 --w-rel 0.05", 24, "IC5"),
    ("Q8_csa_gated_normal10_wrel05_s0","4e-4", _C + " --raydpt-gated-skip True --w-normal 0.1 --w-rel 0.05", 24, "IC5"),
]
_CA = {"IC5": IC5, "IC_GCC": IC_GCC, "IC5W": IC5W}
for _nm,_lr,_ex,_bs,_ca in _Q8:
    JOBS.append(fm(_nm, 0, "raydpt", _lr, _ex, _bs, _CA[_ca]))

# --- Q7: RayDPT RMSE-balanced sweep. Insight: RayDPT-orig already beats U-Net8 RMSE
# (1.420<1.436); coarse-sa fixes d1/AbsRel; w_rel HURTS RMSE (near-pixel) + EMA hurts
# RMSE in our factorial. So: coarse-sa base, NO EMA, minimal w_rel, + RMSE levers
# (w_low low-pass, w_scale mean-match, gated skips E29). Goal: beat U-Net8 balanced. ---
_C = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --raydpt-coarse-sa True"
_Q7 = [
    ("Q7_csa_norel_s0",            _C),                                        # pure arch (best-RMSE bet, no gaming)
    ("Q7_csa_wrel03_s0",           _C + " --w-rel 0.03"),
    ("Q7_csa_wrel05_s0",           _C + " --w-rel 0.05"),
    ("Q7_csa_gated_norel_s0",      _C + " --raydpt-gated-skip True"),          # E22+E29 arch, no w_rel
    ("Q7_csa_gated_wrel05_s0",     _C + " --raydpt-gated-skip True --w-rel 0.05"),
    ("Q7_csa_norel_wscale10_s0",   _C + " --w-scale 0.1"),                     # mean-match -> RMSE
    ("Q7_csa_wrel05_wscale10_s0",  _C + " --w-rel 0.05 --w-scale 0.1"),
    ("Q7_csa_wrel05_wlow10_s0",    _C + " --w-rel 0.05 --w-low 1.0"),          # stronger low-pass -> RMSE
    ("Q7_csa_gated_wrel05_grad05_s0", _C + " --raydpt-gated-skip True --w-rel 0.05 --w-grad 0.05"),
    ("Q7_csa_wrel03_wscale10_s0",  _C + " --w-rel 0.03 --w-scale 0.1"),
]
for _nm,_ex in _Q7:
    JOBS.append(fm(_nm, 0, "raydpt", "4e-4", _ex, 24, IC5))

# --- E22 3-seed confirm + 2x2 factorial ablation isolating EMA vs coarse-geo-self-attn.
# corners already have: R_raydpt_e2 (neither), Q5_e22 (both). add the two middles. ---
_E22 = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --w-rel 0.1 --w-ema 0.995 --raydpt-coarse-sa True"
_EMAONLY = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --w-rel 0.1 --w-ema 0.995"
_CSAONLY = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --w-rel 0.1 --raydpt-coarse-sa True"
JOBS.append(fm("Q5_e22_coarsesa_s1", 1, "raydpt", "4e-4", _E22, 24, IC5))     # E22 3-seed
JOBS.append(fm("Q5_e22_coarsesa_s2", 2, "raydpt", "4e-4", _E22, 24, IC5))
JOBS.append(fm("Q6_emaonly_s0", 0, "raydpt", "4e-4", _EMAONLY, 24, IC5))       # EMA only (no coarse-sa)
JOBS.append(fm("Q6_emaonly_s1", 1, "raydpt", "4e-4", _EMAONLY, 24, IC5))
JOBS.append(fm("Q6_csaonly_s0", 0, "raydpt", "4e-4", _CSAONLY, 24, IC5))       # coarse-sa only (no EMA)
JOBS.append(fm("Q6_csaonly_s1", 1, "raydpt", "4e-4", _CSAONLY, 24, IC5))

# --- EXPERIMENTS.md CHAMPION stack (E22->E29->E34). recipe: amp-bf16 + bs + lr4e-4 +
# w_rel0.1 + weight-EMA(0.995). E22 coarse geo self-attn; E29 +gated skips; E34 +grad. ---
_QE = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True --w-rel 0.1 --w-ema 0.995 --raydpt-coarse-sa True"
JOBS.append(fm("Q5_e22_coarsesa_s0", 0, "raydpt", "4e-4", _QE, 24, IC5))
JOBS.append(fm("Q5_e29_gated_s0",    0, "raydpt", "4e-4", _QE + " --raydpt-gated-skip True", 24, IC5))
JOBS.append(fm("Q5_e34_champion_s0", 0, "raydpt", "4e-4", _QE + " --raydpt-gated-skip True --w-grad 0.05", 24, IC5))

# --- audioresearch_audio BEST-SET trio (faithful recipe: amp-bf16 + bs32 + lr4e-4).
# E2(w_rel=0.1) already = R_raydpt_e2. Add E4(w_silog=0.5) + E0c(base, no aux). ---
_E = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --amp True"
JOBS.append(fm("Q3_e4_silog5_s0",  0, "raydpt", "4e-4", _E + " --w-silog 0.5", 32, IC5))
JOBS.append(fm("Q3_e0c_base_s0",   0, "raydpt", "4e-4", _E, 32, IC5))

# --- spherical-attention ablation (vs C_raydpt_5chflip = spherical baseline) ---
_LSAB = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2"
for _m in ("off", "planar", "nobias"):
    JOBS.append(fm(f"C_raydpt_lsa{_m}_s0", 0, "raydpt", "3e-4", _LSAB + f" --lsa-mode {_m}", 16, IC5))

# --- 20 improvement variants (Q2): champion U-Net8/GCC get the loss recipe (w_rel/
# w_silog) they never had; + no-ray(learned query)+recipe; + w20/downs/lite combos. ---
_UB = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True"
_GB = "--ngf 64 --unet-downs 8 --in-ch 6 --audio-src gcc --flip-aug True"
_WB = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --audio-window-m 20"
_NR = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --ray-cross-layers 2 --raydpt-noray True"
_LT = "--ngf 64 --unet-downs 8 --in-ch 5 --flip-aug True --raydpt-lite True --ray-cross-layers 2 --w-coarse-layout 0.5"
_Q2 = [
    ("Q2_unet_rel10_s0","unet","2e-3",_UB+" --w-rel 0.1",48,IC5),
    ("Q2_unet_rel05_s0","unet","2e-3",_UB+" --w-rel 0.05",48,IC5),
    ("Q2_unet_rel13_s0","unet","2e-3",_UB+" --w-rel 0.13",48,IC5),
    ("Q2_unet_silog25_s0","unet","2e-3",_UB+" --w-silog 0.25",48,IC5),
    ("Q2_unet_silog5_s0","unet","2e-3",_UB+" --w-silog 0.5",48,IC5),
    ("Q2_unet_rel10silog25_s0","unet","2e-3",_UB+" --w-rel 0.1 --w-silog 0.25",48,IC5),
    ("Q2_unet_rel05silog25_s0","unet","2e-3",_UB+" --w-rel 0.05 --w-silog 0.25",48,IC5),
    ("Q2_unet_rel10_normal_s0","unet","2e-3",_UB+" --w-rel 0.1 --w-normal 0.1",48,IC5),
    ("Q2_unet_rel10_chamfer_s0","unet","2e-3",_UB+" --w-rel 0.1 --w-chamfer 0.1",48,IC5),
    ("Q2_unet_downs7_s0","unet","2e-3","--ngf 64 --unet-downs 7 --in-ch 5 --flip-aug True",48,IC5),
    ("Q2_unet_rel10_amp_s0","unet","2e-3",_UB+" --w-rel 0.1 --amp True",48,IC5),
    ("Q2_unet_rel10_s1","unet","2e-3",_UB+" --w-rel 0.1",48,IC5),
    ("Q2_gcc_rel10_s0","unet","2e-3",_GB+" --w-rel 0.1",48,IC_GCC),
    ("Q2_gcc_silog25_s0","unet","2e-3",_GB+" --w-silog 0.25",48,IC_GCC),
    ("Q2_gcc_rel10silog25_s0","unet","2e-3",_GB+" --w-rel 0.1 --w-silog 0.25",48,IC_GCC),
    ("Q2_w20_rel10_s0","unet","2e-3",_WB+" --w-rel 0.1",48,IC5W),
    ("Q2_noray_rel10_s0","raydpt","3e-4",_NR+" --w-rel 0.1",16,IC5),
    ("Q2_noray_rel10silog25_s0","raydpt","3e-4",_NR+" --w-rel 0.1 --w-silog 0.25",16,IC5),
    ("Q2_noray_s1","raydpt","3e-4",_NR,16,IC5),
    ("Q2_rdlite_silog25_s0","raydpt","3e-4",_LT+" --w-silog 0.25",16,IC5),
]
for _nm,_ar,_lr,_ex,_bs,_ca in _Q2:
    _sd=int(_nm.rsplit("_s",1)[1]); JOBS.append(fm(_nm,_sd,_ar,_lr,_ex,_bs,_ca))

# explicit front-of-queue ordering: just-added RayDPT runs FIRST, then the other
# richer-input / research-focus jobs, then everything else (stable within a rank).
FRONT = [  # FINAL v3 (repo champion recipes on planar, 2ch-focus) — highest
         "finalv3_raydpt_2ch_champ", "finalv3_raydpt_2ch_champ_e51", "finalv3_raydpt_2ch_champ_bhlow", "finalv3_raydpt_5ch_champ",
         "finalv3_raydpt_mres_champ", "finalv3_raydpt_mres_champ_e51",
         "finalv2_batvision_5ch", "finalv2_preunet_5ch", "finalv2_previt_5ch", "finalv2_echodiff_5ch",
         # FINAL v2 (requested order)
         "finalv2_batvision", "finalv2_preunet", "finalv2_previt", "finalv2_echodiff",
         "finalv2_raydpt_2ch_ray", "finalv2_raydpt_5ch_ray", "finalv2_raydpt_2ch_noray", "finalv2_raydpt_5ch_noray",
         # -2) AAAI final runs + published baselines (highest)
         "F_champion", "F_raymlpcsa", "CF_", "RV_", "M1","M2","M3","M4","M5","M6","M7","M8","M9","M10","M11","M12", "S_", "P_r", "P_b", "P_a", "P_x", "F3_bestRMSE", "F3_bestAbsRel", "F2_raydpt", "B2_presnet", "B2_pvit", "B2_batvis", "B_pvit", "B_presnet", "B_batvis", "B_echodiff",
         # -1) fair control + Q7 3-seed confirm + pending ablations (HIGHEST)
         "Q12_unet", "Q7_csa_wrel03", "Q11_zc", "Q9_ground", "Q17_csa", "Q17_unet", "Q16_unet", "Q15_csa", "Q15_unet", "Q10_vit", "Q14_gamma", "Q14_berhu", "Q13_loss", "Q8_",
         # 0) no-ray ablation
         "C_raydpt_noray",
         # 1) best-param reproduction from local audioresearch_audio (E2 recipe)
         "R_raydpt_e2",
         # 2) U-Net-based experiments next (pure U-Net8 + U-Net8-backbone echo)
         "U_unet8_scale1", "U_unet8_scale2", "U_unet8_chamfer", "U_unet8_normal",
         "E_echo_unet", "E_echo_bin", "R_echo_unet_e2",
         # 3) ray-decoder variants after
         "C_raydpt_msf", "C_raydpt_rsmp", "E_echo_ray", "R_echo_ray_e2",
         "C_raydpt_5chflip", "C_raydptlite", "Bnode2_gcc_",
         "Bnode2_wave_", "C_cross_align", "C_unet8", "rayconv5d", "cross_unetenc"]
def _rank(n):
    for i, p in enumerate(FRONT):
        if p in n:
            return i
    return len(FRONT)
JOBS = sorted(JOBS, key=lambda j: _rank(j["name"]))
# --- Planar conversion: all exploratory (non-finalv2) jobs run in PLANAR depth (radial reservation
# changed to planar). Planar caches already built (ic5/ic2/ic2_wave _planar). finalv2 unchanged.
# Disable with PLANAR_EXPLORATORY=0. ---
_PLANAR_CACHE = {f"{CK}/ic5_256x512": f"{CK}/ic5_256x512_planar",
                 f"{CK}/ic2_256x512": f"{CK}/ic2_256x512_planar",
                 f"{CK}/ic5_256x512_wave": f"{CK}/ic2_256x512_planar_wave"}
if os.environ.get("PLANAR_EXPLORATORY", "1") == "1":
    for j in JOBS:
        if j["name"].startswith("finalv2") or "--depth-type" in j["cmd"]:
            continue
        j["cmd"] = j["cmd"].replace("python train_fullmap.py ",
                                    "python train_fullmap.py --depth-type planar ", 1)
        if j.get("cache") in _PLANAR_CACHE:
            j["cache"] = _PLANAR_CACHE[j["cache"]]
_only = os.environ.get("MEGA_ONLY")
if _only:
    JOBS = [j for j in JOBS if j["name"].startswith(_only)]   # temp priority isolation

# explicit drops: pulled from the queue (e.g. underperforming, free the slot for RayDPT)
DROP = {"Bnode2_foa_unet8_s0","Bnode2_foa_unet8_s1","Bnode2_foa_unet8_s2","Bnode2_foa_cross_s0","Bnode2_foa_cross_s1","Bnode2_foa_cross_s2",
        "Bnode2_cross_hitok_s1",   # killed slow 6-pass eval (non-contender); free GPU, no re-train
        "C_cross_align_5chflip_s2",
        "C_cross_align_5chflip_s0", "C_cross_align_5chflip_s1",   # killed: non-contender, free GPU for RayDPT
        "Bnode2_crossself_flip_s0", "Bnode2_cross_vitenc_s0"}     # killed eval; keep best.pth, skip re-run
JOBS = [j for j in JOBS if j["name"] not in DROP]


# optional split: restrict to a GPU subset and/or skip a name substring (run elsewhere)
ALLOW = set(int(x) for x in os.environ.get("MEGA_GPUS", "0,1,2,3,4,5,6,7").split(","))
SKIP = os.environ.get("MEGA_SKIP", "")
if SKIP:
    JOBS = [j for j in JOBS if SKIP not in j["name"]]


def running_elsewhere(name):
    """True if a train process for this run already exists (restart-safe: don't dup)."""
    try:
        return bool(subprocess.check_output(["pgrep", "-f", f"run-name {name} "]).strip())
    except subprocess.CalledProcessError:
        return False


def done(j): return os.path.exists(os.path.join("out", j["name"], j["art"]))
def cache_ready(j): return j["cache"] is None or os.path.exists(j["cache"] + "/train_spec.npy")
def idle_gpus():
    o = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"]).decode()
    return [int(l.split(",")[0]) for l in o.strip().splitlines()
            if int(l.split(",")[1]) < 1500 and int(l.split(",")[0]) in ALLOW]


def main():
    running = {}
    print(f"[mega] {len(JOBS)} jobs total", flush=True)
    while True:
        for g in list(running):
            p, n = running[g]
            if p.poll() is not None:
                print(f"[mega] finished {n} gpu{g}", flush=True); running.pop(g)
        pend = [j for j in JOBS if not done(j) and j["name"] not in {n for _, n in running.values()}
                and cache_ready(j) and not running_elsewhere(j["name"])]
        nd = [j for j in JOBS if not done(j)]
        if not nd and not running:
            print("[mega] ALL DONE", flush=True); break
        if pend:
            for g in idle_gpus():
                if g in running or not pend:
                    continue
                j = pend.pop(0)
                lf = open(f"logs/{j['name']}.log", "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                running[g] = (subprocess.Popen(j["cmd"], shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env), j["name"])
                print(f"[mega] launch {j['name']} gpu{g} ({len([x for x in JOBS if not done(x)])} left)", flush=True)
        time.sleep(20)
    subprocess.run("python agg_full.py > logs/_agg_mega.log 2>&1; python update_readme.py >> logs/_mega.log 2>&1", shell=True)
    print("[mega] aggregated", flush=True)


if __name__ == "__main__":
    main()
