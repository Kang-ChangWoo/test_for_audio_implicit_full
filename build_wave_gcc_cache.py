"""Build local full-res caches for the waveform experiments (CPU only).
  gcc : 6ch (5ch RIR + GCC-PHAT lag map)
  wave: 5ch RIR spec + raw binaural waveform (extra array)
val/test first (small), then train (large)."""
import copy
from types import SimpleNamespace
from config import DEFAULTS
from data import build_cache, cache_exists

JOBS = [("gcc", 6), ("wave", 5)]
for src, ic in JOBS:
    for split in ("val", "test", "train"):
        cfg = SimpleNamespace(**DEFAULTS); cfg.audio_src = src; cfg.in_ch = ic
        if cache_exists(cfg, split):
            print(f"[skip] {src} {split} exists", flush=True); continue
        print(f"[build] {src} {split}", flush=True)
        build_cache(cfg, split)
print("[cache] ALL DONE", flush=True)
