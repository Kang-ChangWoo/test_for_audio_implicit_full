# RESULTS_metrics_full — comparison table (test, 256x512, masked)

MAE_plain = masked plain MAE (headline); MAE = cos-lat weighted; delta_k = ratio<1.25^k.

| method | MAE_plain↓ | MAE↓ | AbsRel↓ | RMSE↓ | δ1↑ | δ2↑ | δ3↑ | n |
|---|---|---|---|---|---|---|---|---|
| pretrained UNet (ResNet50+UNet dec) | **0.8193**±0.007 | 0.9767 | 0.6360 | 1.4958 | 0.416 | 0.635 | 0.771 | 3 |
| pretrained ViT (ViT-B/16) | **0.7598**±0.004 | 0.9026 | 0.5589 | 1.4241 | 0.454 | 0.673 | 0.801 | 3 |
| BatVision (pix2pix UNet) | **0.7522**±0.007 | 0.8954 | 0.5371 | 1.4318 | 0.456 | 0.675 | 0.801 | 3 |
| EchoDiffusion (retrain, ours-data) | **0.8020**±0.012 | 0.9616 | 0.6247 | 1.4572 | 0.407 | 0.630 | 0.772 | 3 |
| EchoDiffusion (pretrained param) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | — |
| RayDPT (ours, champion) | **0.7572**±0.012 | 0.8956 | 0.5005 | 1.4386 | 0.449 | 0.670 | 0.800 | 3 |
| U-Net8 (strong baseline) | **0.7495**±0.006 | 0.8933 | 0.5308 | 1.4361 | 0.455 | 0.674 | 0.802 | 3 |
| Aunet (pix2pix baseline) | **0.8287**±0.003 | 0.9889 | 0.6574 | 1.5179 | 0.406 | 0.625 | 0.764 | 3 |
