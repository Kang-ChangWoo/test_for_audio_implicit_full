"""Validate the premise: per-distance-bin binaural ITD encodes the lateral
direction of surfaces at that distance.

For each distance bin d_k (= time window of the RIR after direct-peak onset):
  audio_lat(d_k) = ITD from windowed L/R cross-correlation -> lateral cue in [-1,1]
                   (ITD * c / ear_sep ~ projection onto the interaural +/-y axis)
  gt_lat(d_k)    = cos-lat-weighted mean of ray dir_y over GT pixels in that shell
                   (ears at +/-y, so dir_y is the interaural-axis projection)
If the two curves track -> ITD per bin really carries surface direction -> the
distance-binned binaural model is worth building.
"""
import os
import numpy as np
import soundfile as sf
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/root/storage/matterport3d_0303renew"
FIG = "out/figs"; os.makedirs(FIG, exist_ok=True)
C = 343.0
EAR = 0.175            # interaural separation (m) = 2*head_r
DMAX = 6.0
K = 24                 # distance bins
SCENES = [("17DRP5sb8fy", "010"), ("17DRP5sb8fy", "005"), ("1LXtFkjw3qL", "000")]


def load(scene, idx):
    rgb = np.array(Image.open(f"{ROOT}/{scene}/erp_rgb/erp_{idx}.png").convert("RGB"))
    depth = np.nan_to_num(np.load(f"{ROOT}/{scene}/erp_depth_radial/erp_depth_{idx}.npy").astype(np.float32))
    wav, sr = sf.read(f"{ROOT}/{scene}/audio_wav/audio_{idx}.wav"); wav = np.asarray(wav).T
    if depth.shape != rgb.shape[:2]:
        depth = np.array(Image.fromarray(depth).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
    return depth, wav, sr


def itd_lateral(Lw, Rw, sr, max_lag):
    """lag (samples) of max cross-correlation in +/-max_lag -> lateral cue [-1,1]."""
    if len(Lw) < 8:
        return np.nan, 0.0
    Lw = Lw - Lw.mean(); Rw = Rw - Rw.mean()
    cc = np.correlate(Lw, Rw, mode="full")
    mid = len(Rw) - 1
    lo, hi = mid - max_lag, mid + max_lag + 1
    seg = cc[lo:hi]
    if seg.size == 0 or not np.isfinite(seg).any():
        return np.nan, 0.0
    lag = np.argmax(seg) - max_lag                       # +lag: L later than R
    itd = lag / sr
    lat = np.clip(itd * C / EAR, -1, 1)                  # interaural-axis projection
    energy = float(np.abs(Lw).mean() + np.abs(Rw).mean())
    return lat, energy


def main():
    fig, axes = plt.subplots(len(SCENES), 1, figsize=(9, 3.0 * len(SCENES)))
    if len(SCENES) == 1:
        axes = [axes]
    for ax, (scene, idx) in zip(axes, SCENES):
        depth, wav, sr = load(scene, idx)
        H, W = depth.shape
        el = (np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi)[:, None]
        az = ((np.arange(W) + 0.5) / W * 2 * np.pi)[None, :]
        diry = np.cos(el) * np.sin(az)                   # interaural (+/-y) component per pixel
        cosphi = np.cos(el) * np.ones((1, W))
        L, R = wav[0], wav[1]
        env = np.convolve(np.abs(L) + np.abs(R), np.ones(41) / 41, mode="same")
        n0 = int(np.argmax(env[:int(sr * 0.006)]))       # direct-peak onset
        max_lag = int(np.ceil(EAR / C * sr)) + 2
        edges = np.linspace(0, DMAX, K + 1)
        dc = 0.5 * (edges[:-1] + edges[1:])
        a_lat, a_en, g_lat = [], [], []
        for k in range(K):
            d0, d1 = edges[k], edges[k + 1]
            s0 = n0 + int(d0 * 2 * sr / C); s1 = n0 + int(d1 * 2 * sr / C)
            lat, en = itd_lateral(L[s0:s1], R[s0:s1], sr, max_lag)
            a_lat.append(lat); a_en.append(en)
            m = (depth >= d0) & (depth < d1)
            w = cosphi[m]
            g_lat.append(float((diry[m] * w).sum() / (w.sum() + 1e-9)) if m.any() else np.nan)
        a_lat = np.array(a_lat); a_en = np.array(a_en); g_lat = np.array(g_lat)
        a_en = a_en / (a_en.max() + 1e-9)
        # correlation where both valid and there is echo energy
        ok = np.isfinite(a_lat) & np.isfinite(g_lat) & (a_en > 0.1)
        r = np.corrcoef(a_lat[ok], g_lat[ok])[0, 1] if ok.sum() > 2 else np.nan

        ax.axhline(0, color="gray", lw=0.6)
        ax.plot(dc, g_lat, "-o", color="#ff7f0e", ms=3, label="GT shell lateral (mean dir_y)")
        ax.scatter(dc, a_lat, s=20 + 120 * a_en, color="k", alpha=0.7,
                   label="audio ITD lateral (size∝echo energy)")
        ax.set_xlim(0, DMAX); ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("distance (m)"); ax.set_ylabel("lateral  (−y … +y)")
        ax.set_title(f"{scene}/{idx}   corr(audio-ITD, GT-lateral) = {r:+.2f}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out = f"{FIG}/fig_bin_direction.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
