# all-2ch fair comparison (test 256x512, masked). baselines=2ch spec; EchoDiffusion=2ch spec+wave; RayDPT=2ch.

| method | input | MAE_plain | MAE | AbsRel | RMSE | d1 | d2 | d3 | n |
|---|---|---|---|---|---|---|---|---|---|
| pretrained UNet (ResNet50) | 2ch | 0.7962±0.010 | 0.9482 | 0.6002 | 1.4721 | 0.425 | 0.647 | 0.781 | 3 |
| pretrained ViT (ViT-B/16) | 2ch | 0.7619±0.008 | 0.9089 | 0.5638 | 1.4244 | 0.442 | 0.665 | 0.796 | 3 |
| BatVision | 2ch | 0.7597±0.006 | 0.9049 | 0.5374 | 1.4433 | 0.446 | 0.667 | 0.797 | 3 |
| EchoDiffusion (retrain) | 2ch+wave | 0.8020±0.012 | 0.9616 | 0.6247 | 1.4572 | 0.407 | 0.630 | 0.772 | 3 |
| EchoDiffusion (pretrained param) | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | — |
| RayDPT champion (ours) | 2ch | 0.7630±0.000 | 0.9063 | 0.5235 | 1.4426 | 0.445 | 0.667 | 0.797 | 3 |
