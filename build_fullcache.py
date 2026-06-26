"""Materialise the full-resolution (256x512) RAW tensors to the LOCAL-disk cache.

Reads the actual dataset files once (NFS) and writes contiguous local .npy memmaps
so training reads are fast. Same resolution as the files (no downscaling) — this
is NOT the old 64x128 cache. Keyed by (in_ch, HxW) inside cfg.cache_dir.

  python build_fullcache.py --in-ch 2 --splits train val test
"""
import argparse
from config import get_cfg
import data as D

if __name__ == "__main__":
    # get_cfg parses --in-ch / --img-h / --img-w / --num-workers from config defaults;
    # add --splits on top.
    import sys
    splits = ["val", "test", "train"]
    if "--splits" in sys.argv:
        i = sys.argv.index("--splits")
        splits = []
        j = i + 1
        while j < len(sys.argv) and not sys.argv[j].startswith("--"):
            splits.append(sys.argv[j]); j += 1
        del sys.argv[i:j]
    cfg = get_cfg()
    for sp in splits:
        if D.cache_exists(cfg, sp):
            print(f"[skip] {sp} ic{cfg.in_ch} already cached", flush=True)
        else:
            D.build_cache(cfg, sp)
    print("BUILD_CACHE_DONE", flush=True)
