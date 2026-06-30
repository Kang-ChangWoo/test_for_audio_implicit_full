"""Build results.tsv summarising EVERY completed experiment in out/ (seed-averaged
by family). Columns: name, MAE, RMSE, AbsRel, d1, n_seeds, verdict, description.
description = WHAT was changed (no judgement).  verdict = the assessment.
Baseline reference: Bnode2_unet8_5chflip (U-Net8, 5ch RIR + L/R-flip) = best, MAE 0.886.
"""
import json, glob, os, re, collections

# what-was-changed (description) per family. baseline = pix2pix U-Net8, 5ch RIR
# input [logL,logR,ILD,cosIPD,sinIPD], L/R-flip aug, masked-MAE.
DESC = {
 "Bnode2_unet8_5chflip": "U-Net8 (pix2pix, 8 down) + 5ch RIR + L/R-flip aug [BASELINE]",
 "Bnode2_unet8_5chflip_w20": "baseline + 20m audio window (later reflections)",
 "Bnode2_gcc_unet8": "U-Net8 + GCC-PHAT lag map input (6ch, waveform ITD)",
 "U_unet8_normal": "baseline + 3D surface-normal cosine aux loss (w=0.1)",
 "B_unet8_5ch": "U-Net8 + 5ch RIR, no flip aug",
 "A22_vit_aug": "pretrained ViT-B/16 encoder + aug",
 "B_unet8nolog_aug": "U-Net8 + raw (non-log) magnitude spec + aug",
 "Bnode2_cross_flip": "ray-conditioned cross-attention (rays attend audio tokens), 2ch + flip",
 "Bnode2_wave_unet8": "U-Net8 + raw-waveform 1D-CNN global embedding (EchoDiffusion-lite)",
 "A23_vit_sh": "ViT encoder + spherical-harmonic ray PE",
 "A23_vit_both": "ViT encoder + Fourier+SH ray PE",
 "C_raydptlite_5chflip": "RayDPT-lite: 2-scale ray-conditioned DPT decoder, 5ch+flip",
 "Bnode2_cross_unetenc5": "cross-attention + U-Net token encoder, 5ch",
 "C_raydpt_5chflip": "RayDPT-full: multi-scale ray DPT + learned full-decode, 5ch+flip",
 "B_cross_nolog": "cross-attention + raw (non-log) spec",
 "B_unet8nolog": "U-Net8 + raw (non-log) spec, no aug",
 "Bnode2_unet8nolog": "U-Net8 + raw (non-log) spec",
 "Bnode2_cross_5chflip": "cross-attention, 5ch + flip",
 "A3_crossSH": "cross-attention + SH ray PE",
 "A4_ffmask": "cross-attention + far-field (>=10m) mask in loss",
 "A5_crossMic": "cross-attention + ear-axis (binaural) mic PE",
 "Bnode2_cross_nolog": "cross-attention + raw (non-log) spec",
 "T_film": "tip: FiLM (global audio -> gamma,beta) on ray tokens",
 "A23_vit_fourier": "ViT encoder + Fourier ray PE",
 "T_mlpskip": "tip: ray-coordinate re-injection skip-MLP",
 "T_sector": "tip: sector/near-balanced ray sampling",
 "A4_cross": "cross-attention base, 2ch",
 "A6sec": "cross+self-attn + sector-weighted (front/back) loss",
 "A6_crossself": "cross-attention + ray self-attention",
 "Bnode2_cross_unetenc": "cross-attention + U-Net token encoder, 2ch",
 "T_all": "tip: all implicit-field tricks combined",
 "T_progpe": "tip: progressive coarse->fine Fourier PE",
 "C_unet8_raycoarse16_5chflip": "U-Net8 encoder + coarse ray-token field 16x32 head",
 "A19_raymodStrong": "U-Net + strong ray-conditioned FiLM modulation",
 "A14_rir5": "supervised residual corrector on frozen D0, 5ch",
 "Bnode2_cross5ch": "cross-attention, 5ch, no flip",
 "A12_film": "full-map decoder + FiLM audio correction",
 "A8_hybrid": "SH-coarse(audio) + implicit residual",
 "A19_raymodStrong_fv": "strong ray-FiLM modulation (full-val eval)",
 "A14_frozen": "frozen-decoder residual test (is A0 residual audio-predictable?)",
 "A10_cross": "full-map decoder + cross-attn residual correction",
 "C_unet8_sh6_5chflip": "U-Net8 + SH order-6 coarse head",
 "A13_ild3": "3ch input [logL,logR,ILD] (drop IPD/phase)",
 "C_unet8_sh4_5chflip": "U-Net8 + SH order-4 coarse head",
 "C_unet8_coarse16_5chflip": "U-Net8 + coarse 16x32 depth head",
 "A9_fullmap": "A0-style full-map decoder (no skip), no correction",
 "A13_mag2": "2ch magnitude only (no phase) variant",
 "A11_shaux": "full-map decoder + audio->SH-coef auxiliary loss",
 "C_unet8_coarseres_5chflip": "U-Net8 coarse head + constrained low-pass residual",
 "A2_raymlp": "implicit ray-MLP: f(global audio, ray) -> depth",
 "A14_logmag": "supervised residual corrector, log-mag input",
 "C_unet8_coarse32_5chflip": "U-Net8 + coarse 32x64 depth head",
 "A16_raymod8x16": "ray-modulated U-Net at e3 (8x16 grid)",
 "A16_raymod_fv": "ray-modulated U-Net (full-val eval)",
 "A13_ipd5": "5ch incl. IPD phase variant",
 "A18_raymod64reg": "ray-mod U-Net at 64 grid + regularisation",
 "A20_unet64_aug": "wide U-Net (ngf64, fewer downs) + aug",
 "Bnode2_hybrid5ch": "SH-coarse + implicit residual, 5ch",
 "A21_raymodStrong_aug": "strong ray-FiLM modulation + aug",
 "A18_unet64reg_fv": "wide U-Net + reg (full-val eval)",
 "A15_bigunet_fv": "bigger U-Net (full-val eval)",
 "A15_bigunet": "bigger U-Net (more capacity)",
 "Aunet": "plain U-Net depth baseline (A-series)",
 "A18_unet64reg": "wide U-Net + regularisation",
 "Bnode2_rayconv5d": "dense ray-conv decoder at 64x128, 5ch",
 "A4_cross_shuf": "cross-attention + SHUFFLED audio (break scene pairing) [CONTROL]",
 "A2_shuf": "ray-MLP + SHUFFLED audio [CONTROL]",
 "A1_rayonly": "ray prior only, NO audio [CONTROL]",
}


def verdict(fam, mae, rmse, ar):
    if fam in ("A4_cross_shuf", "A2_shuf", "A1_rayonly"):
        return "control: audio removed/broken -> expected worst; confirms model uses audio"
    if mae <= 0.893:
        return "best tier (~SOTA); global U-Net/GCC encoder wins"
    if mae <= 0.905:
        return "competitive; near baseline (ViT / window / wave variants)"
    if mae <= 0.918:
        return "cross/RayDPT ceiling (~0.91); ray-conditioning does not beat U-Net"
    if mae <= 0.935:
        return "below baseline; tips/PE/coarse-token tweaks do not help"
    if mae <= 0.965:
        return "worse; aux heads (SH/coarse/raymod) and residual correctors hurt"
    if mae <= 1.05:
        return "much worse; bigger U-Nets / dense ray-conv lose accuracy"
    return "broken-level (control)"


def main():
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in glob.glob("out/*/metrics_test.json"):
        n = os.path.basename(os.path.dirname(m)); fam = re.sub(r"_s\d+$", "", n)
        try: d = json.load(open(m))["test"]
        except Exception: continue
        for k in ("MAE", "RMSE", "AbsRel", "delta1"):
            if k in d: g[fam][k].append(d[k])
    rows = []
    for fam, a in g.items():
        if "MAE" not in a: continue
        mean = lambda k: sum(a[k]) / len(a[k])
        rows.append((fam, mean("MAE"), mean("RMSE"), mean("AbsRel"), mean("delta1"), len(a["MAE"])))
    rows.sort(key=lambda r: r[1])
    out = ["name\tMAE\tRMSE\tAbsRel\td1\tn_seeds\tverdict\tdescription"]
    for fam, mae, rmse, ar, d1, n in rows:
        out.append(f"{fam}\t{mae:.4f}\t{rmse:.4f}\t{ar:.4f}\t{d1:.4f}\t{n}\t"
                   f"{verdict(fam, mae, rmse, ar)}\t{DESC.get(fam, '(undocumented)')}")
    open("results.tsv", "w").write("\n".join(out) + "\n")
    miss = [fam for fam, *_ in rows if fam not in DESC]
    print(f"[results.tsv] {len(rows)} experiments written; undocumented: {miss}")


if __name__ == "__main__":
    main()
