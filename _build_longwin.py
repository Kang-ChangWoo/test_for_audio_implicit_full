from types import SimpleNamespace
from config import DEFAULTS
from data import build_cache, cache_exists
for win in (30, 40):
    for sp in ("val", "test", "train"):   # val/test first (needed for eval), then train
        c = SimpleNamespace(**DEFAULTS); c.in_ch = 5; c.audio_window_m = float(win)
        if cache_exists(c, sp):
            print(f"[skip] w{win} {sp} exists", flush=True); continue
        print(f"[build] w{win} {sp}", flush=True); build_cache(c, sp)
print("[longwin cache] DONE", flush=True)
