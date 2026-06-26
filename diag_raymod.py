"""Diagnostic for A16 UNetRayMod: is the ray modulation sample-specific or has it
collapsed to a fixed positional prior?

For a test batch we read the FiLM extras and report the per-sample std (variance
across the batch dim, averaged over channels/space):
    ray_map.std(0).mean(), gamma.std(0).mean(), beta.std(0).mean()
Near-zero std => the modulation is the same for every input => it is NOT using
audio/ray conditioning, just a learned positional bias (claim would be invalid).
We also report stereo-vs-shuffle gamma divergence: if shuffling the audio leaves
gamma unchanged, the modulation ignores audio.

  python diag_raymod.py --run-name A16_raymod8x16_s0
"""

import argparse
import torch

from eval_fullmap import load
from data import make_loader, apply_audio_mode, shuffle_audio_batch


@torch.no_grad()
def _extras(model, spec, extra, cfg):
    if spec.shape[1] > getattr(cfg, "in_ch", 2):
        spec = spec[:, :getattr(cfg, "in_ch", 2)]
    out = model(spec, extra.get("coarse_feat"), extra.get("sh_basis"))
    return out["extras"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--out-dir", default="out")
    p.add_argument("--n", type=int, default=256)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, extra = load(f"{args.out_dir}/{args.run_name}", device)
    if getattr(cfg, "arch", "") != "unet_raymod":
        print(f"[skip] {args.run_name} arch={getattr(cfg,'arch','?')} is not unet_raymod")
        return
    loader = make_loader(cfg, "test", shuffle=False)
    b = next(iter(loader))
    spec = b["spec"][: args.n].to(device)

    st = _extras(model, apply_audio_mode(spec, "stereo"), extra, cfg)
    sh = _extras(model, shuffle_audio_batch(apply_audio_mode(spec, "stereo")), extra, cfg)

    def smean(x):                       # mean over the per-sample (batch-dim) std
        return float(x.std(dim=0).mean())

    print(f"[{args.run_name}] ray_mod_scale={getattr(cfg,'ray_mod_scale','?')} "
          f"stage={getattr(cfg,'ray_mod_stage','?')} n={spec.size(0)}", flush=True)
    print(f"  sample-std  ray_map={smean(st['ray_map']):.4e}  "
          f"gamma={smean(st['gamma']):.4e}  beta={smean(st['beta']):.4e}", flush=True)
    print(f"  |tanh(gamma)|.mean={float(st['gamma'].tanh().abs().mean()):.4e}  "
          f"|beta|.mean={float(st['beta'].abs().mean()):.4e}  "
          f"(scale s={getattr(cfg,'ray_mod_scale','?')})", flush=True)
    # audio sensitivity: how much does gamma move when the audio is shuffled?
    dg = (st["gamma"] - sh["gamma"]).abs().mean() / st["gamma"].abs().mean().clamp(min=1e-9)
    print(f"  audio-sensitivity (|d gamma| on shuffle / |gamma|) = {float(dg):.4f}", flush=True)


if __name__ == "__main__":
    main()
