import sys
from types import SimpleNamespace
from config import DEFAULTS
from data import build_cache, cache_exists
kind = sys.argv[1]
c = SimpleNamespace(**DEFAULTS); c.depth_type = "planar"; c.num_workers = 16
if kind == "ic2":  c.in_ch = 2; c.audio_src = "binaural"
elif kind == "ic5": c.in_ch = 5; c.audio_src = "binaural"
elif kind == "wave": c.in_ch = 2; c.audio_src = "wave"   # dir=ic2..._wave, spec 5ch + wave
for split in ["train", "val", "test"]:
    if cache_exists(c, split):
        print(f"[{kind}/{split}] exists, skip", flush=True); continue
    print(f"[{kind}/{split}] building (16w)...", flush=True)
    build_cache(c, split)
print(f"[{kind}] ALL DONE", flush=True)
