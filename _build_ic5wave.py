from types import SimpleNamespace
from config import DEFAULTS
from data import build_cache, cache_exists
c=SimpleNamespace(**DEFAULTS); c.depth_type="planar"; c.in_ch=5; c.audio_src="wave"; c.num_workers=16
for split in ["train","val","test"]:
    if cache_exists(c,split): print(f"{split} exists",flush=True); continue
    print(f"building {split}...",flush=True); build_cache(c,split)
print("ic5_planar_wave DONE",flush=True)
