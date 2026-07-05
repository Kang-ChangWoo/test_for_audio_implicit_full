from types import SimpleNamespace
from config import DEFAULTS
from data import build_cache, cache_exists
for sp in ("test", "train"):     # val already done
    c = SimpleNamespace(**DEFAULTS); c.audio_src = "foa"; c.in_ch = 4
    if cache_exists(c, sp):
        print(f"[skip] foa {sp} exists", flush=True); continue
    print(f"[build] foa {sp}", flush=True); build_cache(c, sp)
print("[foa cache] DONE", flush=True)
