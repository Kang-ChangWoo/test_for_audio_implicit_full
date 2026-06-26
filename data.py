"""Full-resolution RAW loader: binaural audio -> ERP radial depth, read directly
from the dataset files (NO 64x128 cache).

Mirrors test_for_audio_better.RawDataset but at cfg.img_h x cfg.img_w (e.g.
256x512, exactly like baseline) — the actual erp_depth_radial files are 512x1024,
so this just resizes to the requested size instead of collapsing to 64x128.

Returns: spec (in_ch,H,W), depth (1,H,W) radial/max_depth in [0,1], mask (1,H,W),
key. in_ch=2 -> log-mag binaural spectrogram; in_ch=5 -> RIR spatial feature
[logL, logR, ILD, cos(IPD), sin(IPD)] computed on the fly (no cache_rir needed).

Split: scene_split.json in dataset_dir (the existing implicit/better split).
Control transforms (audio_mode / swap / shuffle) are channel-count aware and kept
identical to the old cache loader so every ablation behaves the same.
"""

import json
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", message=".*torchaudio.*", category=UserWarning)

SPLIT = "scene_split.json"
_C = 340.0
_NFFT, _HOP, _WIN = 512, 160, 400


class RawDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg, self.root, self.split = cfg, cfg.dataset_dir, split
        self.H, self.W, self.md, self.sr = cfg.img_h, cfg.img_w, cfg.max_depth, cfg.sample_rate
        self.in_ch = getattr(cfg, "in_ch", 2)
        self.log_spec = getattr(cfg, "log_spec", True)   # 2ch mag: log1p vs raw
        self.cut = int(2.0 * self.md / _C * self.sr)
        self.spec = T.Spectrogram(n_fft=_NFFT, win_length=_WIN, hop_length=_HOP, power=1.0)
        self._win = torch.hann_window(_WIN)
        scenes = json.load(open(os.path.join(self.root, SPLIT)))[split]
        self.samples = self._list(scenes)
        print(f"[{split}] {len(self.samples)} samples / {len(scenes)} scenes "
              f"(raw {self.H}x{self.W}, in_ch={self.in_ch})", flush=True)

    def _list(self, scenes):
        out = []
        for s in scenes:
            adir = os.path.join(self.root, s, "audio_wav")
            ddir = os.path.join(self.root, s, "erp_depth_radial")
            if not (os.path.isdir(adir) and os.path.isdir(ddir)):
                continue
            for af in sorted(f for f in os.listdir(adir) if f.endswith(".wav")):
                idx = af.replace("audio_", "").replace(".wav", "")
                if os.path.exists(os.path.join(ddir, f"erp_depth_{idx}.npy")):
                    out.append((s, idx))
        lim = os.environ.get("DEBUG_LIMIT")          # fast end-to-end smoke only
        if lim:
            out = out[:int(lim)]
        return out

    def __len__(self):
        return len(self.samples)

    def _wave(self, s, idx):
        wav, sr = torchaudio.load(
            os.path.join(self.root, s, "audio_wav", f"audio_{idx}.wav"), backend="soundfile")
        if sr != self.sr:
            wav = T.Resample(sr, self.sr)(wav)
        return wav[:, :self.cut]

    def _spec2(self, wav):
        """2ch binaural magnitude spectrogram (log1p or raw), resized to (H,W)."""
        sp = self.spec(wav)                                    # (2,F,T') magnitude
        if self.log_spec:
            sp = torch.log1p(sp)
        return F.interpolate(sp.unsqueeze(0), (self.H, self.W), mode="nearest").squeeze(0).float()

    def _specN(self, wav, n):
        """RIR spatial feature, first n of [logL,logR,ILD,cosIPD,sinIPD] (keeps phase/ITD).
        n=3 -> [logL,logR,ILD]; n=5 -> full ILD+IPD."""
        x = wav
        if x.shape[1] < self.cut:
            x = F.pad(x, (0, self.cut - x.shape[1]))
        st = torch.stft(x, _NFFT, _HOP, _WIN, self._win, return_complex=True)   # (2,F,T')
        L, R = st[0], st[1]
        eps = 1e-6
        lmag = torch.log1p(L.abs()); rmag = torch.log1p(R.abs())
        ild = torch.log(L.abs() + eps) - torch.log(R.abs() + eps)
        ipd = torch.angle(L * torch.conj(R))
        feat = torch.stack([lmag, rmag, ild, torch.cos(ipd), torch.sin(ipd)], 0)[:n]  # (n,F,T')
        return F.interpolate(feat.unsqueeze(0), (self.H, self.W), mode="nearest").squeeze(0).float()

    def _depth(self, s, idx):
        d = np.nan_to_num(np.load(os.path.join(self.root, s, "erp_depth_radial",
                                               f"erp_depth_{idx}.npy")).astype(np.float32))
        d[d < 0] = 0.0; d[d > self.md] = self.md
        t = F.interpolate(torch.from_numpy(d)[None, None], (self.H, self.W),
                          mode="nearest").squeeze(0)
        return t / self.md, (t > 0).float()

    def __getitem__(self, i):
        s, idx = self.samples[i]
        try:
            wav = self._wave(s, idx)
            spec = self._spec2(wav) if self.in_ch == 2 else self._specN(wav, self.in_ch)
            depth, mask = self._depth(s, idx)
        except Exception as e:
            print(f"[skip {s}/{idx}] {e}", flush=True)
            return self[(i + 1) % len(self)]
        return {"spec": spec, "depth": depth, "mask": mask, "key": f"{s}/{idx}"}


def collate(b):
    return {k: ([x[k] for x in b] if k == "key" else torch.stack([x[k] for x in b]))
            for k in b[0]}


# --- full-resolution LOCAL-disk cache --------------------------------------- #
# The dataset lives on NFS; reading 28.8k wav/depth files per epoch is the
# bottleneck (GPUs sit at 0%). We materialise the SAME full-res tensors RawDataset
# produces (256x512, radial, in_ch-aware) into contiguous local .npy memmaps once,
# then mmap-read them. This is NOT the old 64x128 low-res cache — it is the actual
# file resolution, just on fast local disk. Cache is keyed by (in_ch, HxW).

def _cache_dir(cfg):
    base = getattr(cfg, "cache_dir", "") or "/root/implicit_full_cache"
    tag = "" if getattr(cfg, "log_spec", True) else "_nolog"
    return os.path.join(base, f"ic{getattr(cfg,'in_ch',2)}_{cfg.img_h}x{cfg.img_w}{tag}")


def _cache_paths(cdir, split):
    return ({k: os.path.join(cdir, f"{split}_{k}.npy") for k in ("spec", "depth", "mask")},
            os.path.join(cdir, f"{split}_keys.json"))


def cache_exists(cfg, split):
    paths, kp = _cache_paths(_cache_dir(cfg), split)
    return os.path.exists(kp) and all(os.path.exists(p) for p in paths.values())


def build_cache(cfg, split):
    cdir = _cache_dir(cfg); os.makedirs(cdir, exist_ok=True)
    paths, kp = _cache_paths(cdir, split)
    ds = RawDataset(cfg, split); N = len(ds); H, W, C = cfg.img_h, cfg.img_w, getattr(cfg, "in_ch", 2)
    dt = {"spec": np.float16, "depth": np.float16, "mask": np.uint8}
    sh = {"spec": (N, C, H, W), "depth": (N, 1, H, W), "mask": (N, 1, H, W)}
    mm = {k: np.lib.format.open_memmap(paths[k] + ".tmp", mode="w+", dtype=dt[k], shape=sh[k]) for k in dt}
    keys = [None] * N
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate)
    i = 0
    for b in dl:
        n = b["spec"].shape[0]
        for k in dt:
            mm[k][i:i+n] = b[k].numpy().astype(dt[k])
        keys[i:i+n] = b["key"]; i += n
        if i % 3200 == 0:
            print(f"  cache[{split} ic{C}] {i}/{N}", flush=True)
    for k in mm:
        mm[k].flush(); os.rename(paths[k] + ".tmp", paths[k])     # atomic publish
    json.dump(keys, open(kp, "w"))
    print(f"[cache] {split} ic{C}: {N} -> {cdir}", flush=True)


class CachedDataset(Dataset):
    def __init__(self, cfg, split):
        paths, kp = _cache_paths(_cache_dir(cfg), split)
        self.keys = json.load(open(kp))
        self.arr = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
        print(f"[{split}] {len(self.keys)} (cache {_cache_dir(cfg)})", flush=True)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        d = {k: torch.from_numpy(np.ascontiguousarray(self.arr[k][i])).float() for k in self.arr}
        d["key"] = self.keys[i]
        return d


def make_dataset(cfg, split):
    if cache_exists(cfg, split):
        return CachedDataset(cfg, split)
    return RawDataset(cfg, split)


def make_loader(cfg, split, shuffle):
    ds = make_dataset(cfg, split)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, collate_fn=collate,
                      drop_last=shuffle, pin_memory=True)


def chan_stats_raw(cfg, device, n=512):
    """Per-channel mean/std over a sample of the RAW train set (for chan_norm).
    Replaces the old cache-file based stats; works at any resolution / in_ch."""
    import copy
    ds = RawDataset(copy.copy(cfg), "train")
    n = min(n, len(ds))
    acc = [torch.stack([ds[j]["spec"] for j in range(i, min(i + 64, n))])
           for i in range(0, n, 64)]
    t = torch.cat(acc, 0)                                   # (n, C, H, W)
    mean = t.mean((0, 2, 3)).view(1, -1, 1, 1).to(device)
    std = t.std((0, 2, 3)).clamp(min=1e-4).view(1, -1, 1, 1).to(device)
    return mean, std


# --- negative-control / ablation input transforms (channel-count aware) -------

def apply_audio_mode(spec, mode):
    """2ch = [L,R] log-mag; 3ch = [Lmag,Rmag,ILD]; 5ch = [Lmag,Rmag,ILD,cosIPD,sinIPD]."""
    C = spec.shape[1]
    if mode == "stereo":
        return spec
    if mode == "none":
        return torch.zeros_like(spec)
    if C == 2:
        if mode == "mono":
            m = spec.mean(1, keepdim=True); return m.expand(-1, 2, -1, -1).clone()
        if mode == "left":
            return spec[:, 0:1].expand(-1, 2, -1, -1).clone()
        if mode == "right":
            return spec[:, 1:2].expand(-1, 2, -1, -1).clone()
    if C in (3, 5) and mode == "mono":
        y = spec.clone()
        mag = 0.5 * (spec[:, 0:1] + spec[:, 1:2])
        y[:, 0:1] = mag; y[:, 1:2] = mag; y[:, 2:3] = 0.0    # ILD removed
        if C >= 5:
            y[:, 3:4] = 1.0; y[:, 4:5] = 0.0                 # cos(IPD)=1, sin(IPD)=0
        return y
    raise ValueError(f"unsupported audio_mode={mode} for C={C}")


def swap_audio_lr(spec):
    """L<->R swap, channel-count aware (the L/R-mirror control).
    Negate the L-R-antisymmetric channels (ILD, sin(IPD)); cos(IPD) is symmetric."""
    C = spec.shape[1]
    if C == 2:
        return spec[:, [1, 0]]
    y = spec.clone()
    y[:, 0] = spec[:, 1]; y[:, 1] = spec[:, 0]
    if C >= 3:
        y[:, 2] = -spec[:, 2]                                # ILD -> -ILD
    if C >= 5:
        y[:, 3] = spec[:, 3]; y[:, 4] = -spec[:, 4]          # cosIPD same, sinIPD -> -sinIPD
    return y


def shuffle_audio_batch(spec, generator=None):
    """Control B: break the audio<->scene pairing within a batch (roll by 1)."""
    return torch.roll(spec, shifts=1, dims=0)
