"""Echolocation time-of-flight viz: link binaural waveform <-> ERP depth/RGB.

Active echolocation (source==listener==origin): a surface at radial distance d
returns its echo at round-trip time t = 2d/c. So waveform sample n maps to a
distance shell d(n) = c * (n/sr) / 2.

Per scene we draw:
  (top)  L/R waveform vs distance(m)/time(ms), with vertical cursors, and the
         GEOMETRY echo envelope overlaid = cos-lat-weighted pixel count per
         distance bin (i.e. how much surface area returns at each time) -- the
         "pixel count stacked over time, like the waveform".
  (grid) ERP RGB per cursor distance, with the shell pixels (|depth-d|<band)
         highlighted -> which region of the scene returns at that instant.
"""
import os
import numpy as np
import soundfile as sf
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT = "/root/storage/matterport3d_0303renew"
FIG = "out/figs"; os.makedirs(FIG, exist_ok=True)
C = 343.0          # speed of sound (m/s)
SR = 48000
DMAX_VIZ = 12.0    # plot distance window (m round-trip basis)
BAND = 0.20        # shell half-thickness (m)
SCENES = [("17DRP5sb8fy", "010"), ("17DRP5sb8fy", "005"), ("1LXtFkjw3qL", "000")]
CURSORS = [1.5, 2.5, 3.5, 5.0, 7.0]    # distances (m) to snapshot


def load(scene, idx):
    rgb = np.array(Image.open(f"{ROOT}/{scene}/erp_rgb/erp_{idx}.png").convert("RGB"))
    depth = np.nan_to_num(np.load(f"{ROOT}/{scene}/erp_depth_radial/erp_depth_{idx}.npy").astype(np.float32))
    wav, sr = sf.read(f"{ROOT}/{scene}/audio_wav/audio_{idx}.wav")
    wav = np.asarray(wav).T                       # (2,N)
    if depth.shape != rgb.shape[:2]:
        depth = np.array(Image.fromarray(depth).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
    return rgb, depth, wav, sr


def geometry_envelope(depth, H, W, nbins, dmax, alpha=0.0, eps=0.3):
    """Expected-echo histogram over distance bins.

    weight(p) = cos(phi_p) / (D(p)+eps)^alpha
      - cos(phi_p): ERP solid-angle correction (pixel area ~ cos latitude)
      - 1/(D+eps)^alpha: round-trip distance attenuation (alpha=2 amplitude-ish,
        alpha=4 energy-ish). alpha=0 = pure cos-weighted surface area.
    """
    el = (np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi)
    cosphi = np.cos(el)[:, None] * np.ones((1, W))           # solid-angle weight
    valid = (depth > 0.05) & (depth < dmax)
    w = cosphi[valid] / (depth[valid] + eps) ** alpha
    db = (depth[valid] / dmax * nbins).astype(int).clip(0, nbins - 1)
    return np.bincount(db, weights=w, minlength=nbins)


def main():
    for scene, idx in SCENES:
        try:
            rgb, depth, wav, sr = load(scene, idx)
        except Exception as e:
            print(f"[skip] {scene}/{idx}: {e}"); continue
        H, W = depth.shape
        nmax = int(DMAX_VIZ * 2 * sr / C)                    # samples within DMAX_VIZ
        L = wav[0, :nmax]; R = wav[1, :nmax]
        env = np.abs(L) + np.abs(R)
        k = 41
        env_s = np.convolve(env, np.ones(k) / k, mode="same")
        # --- calibration: the RIR direct-path peak (global max in first 6ms) is the
        # t0 reference. reflections map to d = c*(t - t0)/2.  (data is already an
        # impulse response, NOT a chirp -> no deconvolution needed/possible.) ---
        n0 = int(np.argmax(env_s[:int(sr * 0.006)]))
        t0_ms = n0 / sr * 1e3
        distw = C * ((np.arange(nmax) - n0) / sr) / 2        # direct-peak-calibrated distance
        dist = C * (np.arange(nmax) / sr) / 2                # true round-trip distance (geometry)
        # geometry: cos-weighted area, and distance-attenuated expected echo (a=2,4)
        sm = lambda x: np.convolve(x, np.ones(k) / k, mode="same")
        geo0 = sm(geometry_envelope(depth, H, W, nmax, DMAX_VIZ, alpha=0))
        geo2 = sm(geometry_envelope(depth, H, W, nmax, DMAX_VIZ, alpha=2))
        geo4 = sm(geometry_envelope(depth, H, W, nmax, DMAX_VIZ, alpha=4))

        def nz(c, x):                                        # normalise to peak within 0.3-5m
            w = (x > 0.3) & (x < 5.0)
            return c / (c[w].max() + 1e-9)

        ncur = len(CURSORS)
        fig = plt.figure(figsize=(2.2 * ncur, 7.5))
        gs = fig.add_gridspec(3, ncur, height_ratios=[1.1, 1.2, 1.4], hspace=0.4, wspace=0.08)

        # --- row 0: RIR waveform, direct-peak-calibrated ---
        axw = fig.add_subplot(gs[0, :])
        axw.plot(distw, L, lw=0.4, color="#1f77b4", alpha=0.6, label="L")
        axw.plot(distw, R, lw=0.4, color="#d62728", alpha=0.6, label="R")
        axw.plot(distw, env_s, lw=1.5, color="k", label="|RIR| env")
        axw.axvline(0, color="purple", ls=":", lw=1.2, label=f"direct peak t₀={t0_ms:.2f}ms")
        for cd in CURSORS:
            axw.axvline(cd, color="green", ls="--", lw=1)
        axw.set_xlim(-0.3, DMAX_VIZ); axw.set_xlabel("distance d = c·(t−t₀)/2  (m, direct-peak calibrated)")
        axw.set_ylabel("RIR amp"); axw.legend(fontsize=7, loc="upper right")
        axw.set_title(f"{scene}/{idx}  —  impulse response (RIR) vs distance", fontsize=10)

        # --- row 1: expected-echo (cos-area + distance attenuation a=2,4) vs RIR env ---
        axg = fig.add_subplot(gs[1, :])
        axg.fill_between(dist, nz(geo0, dist), color="#ff7f0e", alpha=0.45, label="depth dist = cos-area (α=0)")
        axg.plot(dist, nz(geo0, dist), lw=1.8, color="#ff7f0e")
        axg.plot(dist, nz(geo2, dist), lw=1.0, color="gray", ls="--", alpha=0.6, label="cos-area /d² (α=2, ref)")
        axg.plot(distw, nz(env_s, distw), lw=1.6, color="k", alpha=0.8, label="|RIR| env")
        axg.axvline(0, color="purple", ls=":", lw=1.2)
        for cd in CURSORS:
            axg.axvline(cd, color="green", ls="--", lw=1)
        axg.set_xlim(-0.3, DMAX_VIZ); axg.set_xlabel("distance (m)")
        axg.set_ylabel("normalised (peak@0.3–5m)"); axg.legend(fontsize=7, loc="upper right")

        # --- row 2: ERP RGB with the shell highlighted at each cursor ---
        el = (np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi)
        for j, cd in enumerate(CURSORS):
            ax = fig.add_subplot(gs[2, j])
            base = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
            shell = np.abs(depth - cd) < BAND
            ov = base.copy()
            ov[shell] = (np.array([60, 255, 60]))            # bright green shell
            ax.imshow(ov)
            ax.set_xticks([]); ax.set_yticks([])
            npx = int(shell.sum())
            ax.set_title(f"d={cd}m  t={cd*2/C*1e3:.0f}ms\n{npx}px", fontsize=7)
        out = f"{FIG}/fig_echo_tof_{scene}_{idx}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
