from types import SimpleNamespace
from config import DEFAULTS
from data import build_cache, cache_exists
for sp in ("val", "test", "train"):
    c = SimpleNamespace(**DEFAULTS); c.in_ch = 13; c.audio_src = "raz"
    if cache_exists(c, sp): print(f"[skip] raz {sp}", flush=True); continue
    print(f"[build] raz {sp}", flush=True); build_cache(c, sp)
print("[raz cache] DONE", flush=True)
