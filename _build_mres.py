from types import SimpleNamespace
from config import DEFAULTS
from data import build_cache, cache_exists
c=SimpleNamespace(**DEFAULTS); c.depth_type="planar"; c.in_ch=6; c.audio_src="mres"; c.num_workers=16
for split in ["train","val","test"]:
    if cache_exists(c,split): print(f"{split} exists",flush=True); continue
    print(f"building {split}...",flush=True); build_cache(c,split)
print("ic6_planar_mres DONE",flush=True)
