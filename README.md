# test_for_audio_implicit_full — full-resolution (256×512) audio→ERP-depth

> **Full-res variant:** reads the actual `erp_depth_radial` files at **256×512**
> (no 64×128 cache downsampling), like `baseline`. Local full-res cache + 8-GPU
> scheduler + auto-eval. Radial depth, scene_split 72/9/9, masked L1 (fullmap) /
> per-ray L1 (implicit), cos-lat weighted metrics.

## ⭐ Current best model (256×512, test split, MAE_plain ↓ = masked MAE [m])

<!-- BEST:START -->
**Best (lowest test MAE_plain) = the ray-conditioned cross-attention *implicit* model** — per-ray queries that cross-attend the audio tokens, predicting depth for each ERP ray direction instead of decoding a pixel map.

| rank | model | MAE_plain [m] ↓ | seeds |
|---|---|---|---|
| 🥇 1 | Q8_csa_wrel05_normal10 (Q8_csa_wrel05_normal10) | 0.7397 | 1 |
| 2 | Q8_csa_wrel05_normal05_wscale10 (Q8_csa_wrel05_normal05_wscale10) | 0.7417 | 1 |
| 3 | Q8_csa_win7 (Q8_csa_win7) | 0.7420 | 1 |
| 4 | Q13_loss_rel03_scale10 (Q13_loss_rel03_scale10) | 0.7423 | 1 |
| 5 | Q13_loss_grad10 (Q13_loss_grad10) | 0.7424 | 1 |
| 6 | Q8_csa_gated_normal10_wrel05 (Q8_csa_gated_normal10_wrel05) | 0.7426 | 1 |
| 7 | Q8_csa_wlow15 (Q8_csa_wlow15) | 0.7431 | 1 |
| 8 | Q8_w20_csa_wrel05 (Q8_w20_csa_wrel05) | 0.7432 | 1 |
| 9 | Q13_loss_rel03_normal10 (Q13_loss_rel03_normal10) | 0.7433 | 1 |
| 10 | Q8_csa_wlow075 (Q8_csa_wlow075) | 0.7434 | 1 |
| 11 | Q13_loss_grad05 (Q13_loss_grad05) | 0.7441 | 1 |
| 12 | Q8_csa_wrel03_wlow075 (Q8_csa_wrel03_wlow075) | 0.7442 | 1 |
| 13 | Q7_csa_norel (Q7_csa_norel) | 0.7460 | 1 |
| 14 | Q15_unet_norm_w30 (Q15_unet_norm_w30) | 0.7461 | 1 |
| 15 | Q7_csa_norel_wscale10 (Q7_csa_norel_wscale10) | 0.7468 | 1 |
| 16 | Q8_csa_wscale30 (Q8_csa_wscale30) | 0.7468 | 1 |
| 17 | Q2_unet_rel10_normal (Q2_unet_rel10_normal) | 0.7469 | 1 |
| 18 | Q11_zc4_sinIPD (Q11_zc4_sinIPD) | 0.7469 | 1 |
| 19 | Q5_e34_champion (Q5_e34_champion) | 0.7472 | 1 |
| 20 | Q7_csa_wrel05_wscale10 (Q7_csa_wrel05_wscale10) | 0.7473 | 1 |
| 21 | Q17_csa_raz_normal (Q17_csa_raz_normal) | 0.7476 | 1 |
| 22 | Q7_csa_wrel03_wscale10 (Q7_csa_wrel03_wscale10) | 0.7478 | 1 |
| 23 | Q13_loss_wlow0 (Q13_loss_wlow0) | 0.7479 | 1 |
| 24 | Q13_loss_rel03_silog25 (Q13_loss_rel03_silog25) | 0.7482 | 1 |
| 25 | Q15_csa_base_w20 (Q15_csa_base_w20) | 0.7483 | 1 |
| 26 | Bnode2_gcc_unet8 (Bnode2_gcc_unet8) | 0.7483 ± 0.0043 | 3 |
| 27 | Q8_csa_win9 (Q8_csa_win9) | 0.7484 | 1 |
| 28 | Q17_csa_raz (Q17_csa_raz) | 0.7484 | 1 |
| 29 | Q8_csa_wrel05_win7 (Q8_csa_wrel05_win7) | 0.7484 | 1 |
| 30 | Q14_gamma_n05 (Q14_gamma_n05) | 0.7485 | 1 |
| 31 | Q8_csa_wscale20 (Q8_csa_wscale20) | 0.7487 | 1 |
| 32 | Q13_loss_silog25 (Q13_loss_silog25) | 0.7489 | 1 |
| 33 | Q11_zc0_logL (Q11_zc0_logL) | 0.7491 | 1 |
| 34 | Q10_vit_5ch (Q10_vit_5ch) | 0.7492 | 1 |
| 35 | Q5_e29_gated (Q5_e29_gated) | 0.7493 | 1 |
| 36 | Q8_csa_gated_wlow075_wrel05 (Q8_csa_gated_wlow075_wrel05) | 0.7494 | 1 |
| 37 | Bnode2_unet8_5chflip (Bnode2_unet8_5chflip) | 0.7495 ± 0.0055 | 3 |
| 38 | Q8_csa_wrel05_lr3e4 (Q8_csa_wrel05_lr3e4) | 0.7497 | 1 |
| 39 | Q7_csa_gated_wrel05 (Q7_csa_gated_wrel05) | 0.7498 | 1 |
| 40 | Q15_csa_norm_w30 (Q15_csa_norm_w30) | 0.7500 | 1 |
| 41 | Q14_berhu_rel03 (Q14_berhu_rel03) | 0.7501 | 1 |
| 42 | Q13_loss_rel03_grad05 (Q13_loss_rel03_grad05) | 0.7501 | 1 |
| 43 | Q7_csa_wrel05 (Q7_csa_wrel05) | 0.7504 | 1 |
| 44 | Q14_berhu (Q14_berhu) | 0.7506 | 1 |
| 45 | Q8_csa_lr3e4 (Q8_csa_lr3e4) | 0.7508 | 1 |
| 46 | Q15_csa_norm_w40 (Q15_csa_norm_w40) | 0.7511 | 1 |
| 47 | Q8_csa_normal10 (Q8_csa_normal10) | 0.7512 | 1 |
| 48 | Q11_zc3_cosIPD (Q11_zc3_cosIPD) | 0.7513 | 1 |
| 49 | Q12_unet_wrel03 (Q12_unet_wrel03) | 0.7513 ± 0.0026 | 3 |
| 50 | Bnode2_unet8_5chflip_w20 (Bnode2_unet8_5chflip_w20) | 0.7513 ± 0.0049 | 3 |
| 51 | Q7_csa_wrel03 (Q7_csa_wrel03) | 0.7516 ± 0.0069 | 3 |
| 52 | Q12_unet_wrel03_amp (Q12_unet_wrel03_amp) | 0.7516 | 1 |
| 53 | U_unet8_normal (U_unet8_normal) | 0.7518 | 1 |
| 54 | Q12_unet_wrel03_wscale10 (Q12_unet_wrel03_wscale10) | 0.7518 | 1 |
| 55 | Q13_loss_wcoarse0 (Q13_loss_wcoarse0) | 0.7521 | 1 |
| 56 | Q11_zc1_logR (Q11_zc1_logR) | 0.7521 | 1 |
| 57 | Q7_csa_gated_wrel05_grad05 (Q7_csa_gated_wrel05_grad05) | 0.7525 | 1 |
| 58 | B_unet8_5ch (B_unet8_5ch) | 0.7528 ± 0.0019 | 3 |
| 59 | Q7_csa_gated_norel (Q7_csa_gated_norel) | 0.7532 | 1 |
| 60 | Q6_csaonly (Q6_csaonly) | 0.7533 ± 0.0001 | 2 |
| 61 | Q8_csa_chamfer10 (Q8_csa_chamfer10) | 0.7535 | 1 |
| 62 | Q17_csa_shpe (Q17_csa_shpe) | 0.7536 | 1 |
| 63 | Q11_zc2_ILD (Q11_zc2_ILD) | 0.7537 | 1 |
| 64 | Q7_csa_wrel05_wlow10 (Q7_csa_wrel05_wlow10) | 0.7539 | 1 |
| 65 | Q17_csa_micpe (Q17_csa_micpe) | 0.7541 | 1 |
| 66 | Q14_gamma_p05_rel03 (Q14_gamma_p05_rel03) | 0.7541 | 1 |
| 67 | Q17_csa_micsh (Q17_csa_micsh) | 0.7542 | 1 |
| 68 | Q9_ground_planarlsa (Q9_ground_planarlsa) | 0.7542 | 1 |
| 69 | Q2_w20_rel10 (Q2_w20_rel10) | 0.7543 | 1 |
| 70 | Q5_e22_coarsesa (Q5_e22_coarsesa) | 0.7545 ± 0.0063 | 3 |
| 71 | Q2_unet_rel13 (Q2_unet_rel13) | 0.7549 | 1 |
| 72 | Q2_unet_rel10 (Q2_unet_rel10) | 0.7549 ± 0.0031 | 2 |
| 73 | Q16_unet_ic3 (Q16_unet_ic3) | 0.7558 ± 0.0024 | 3 |
| 74 | U_unet8_chamfer (U_unet8_chamfer) | 0.7560 | 1 |
| 75 | Q9_ground_full (Q9_ground_full) | 0.7564 | 1 |
| 76 | Q16_unet_ic2 (Q16_unet_ic2) | 0.7566 ± 0.0003 | 3 |
| 77 | Q2_unet_silog25 (Q2_unet_silog25) | 0.7571 | 1 |
| 78 | Q2_unet_rel10_amp (Q2_unet_rel10_amp) | 0.7573 | 1 |
| 79 | Q2_unet_rel05silog25 (Q2_unet_rel05silog25) | 0.7574 | 1 |
| 80 | Q8_gcc_csa_norel (Q8_gcc_csa_norel) | 0.7576 | 1 |
| 81 | Q17_unet_raz (Q17_unet_raz) | 0.7577 | 1 |
| 82 | Q2_unet_rel05 (Q2_unet_rel05) | 0.7581 | 1 |
| 83 | B_unet8nolog_aug (B_unet8nolog_aug) | 0.7582 ± 0.0036 | 3 |
| 84 | Q15_csa_base_w30 (Q15_csa_base_w30) | 0.7582 | 1 |
| 85 | Q8_gcc_csa_wrel05 (Q8_gcc_csa_wrel05) | 0.7583 | 1 |
| 86 | Q2_gcc_rel10 (Q2_gcc_rel10) | 0.7587 | 1 |
| 87 | Bnode2_cross_flip_nr1024 (Bnode2_cross_flip_nr1024) | 0.7593 ± 0.0019 | 2 |
| 88 | Q9_ground_nocsageo (Q9_ground_nocsageo) | 0.7595 | 1 |
| 89 | Q2_unet_rel10_chamfer (Q2_unet_rel10_chamfer) | 0.7600 | 1 |
| 90 | Q_rd_rel10_normal (Q_rd_rel10_normal) | 0.7602 | 1 |
| 91 | A22_vit_aug (ViT-B/16 (planar PE)) | 0.7605 ± 0.0025 | 3 |
| 92 | Q2_noray (Q2_noray) | 0.7607 | 1 |
| 93 | Q2_noray_rel10 (Q2_noray_rel10) | 0.7610 | 1 |
| 94 | E_echo_unet (E_echo_unet) | 0.7611 ± 0.0048 | 3 |
| 95 | E_echo_bin (E_echo_bin) | 0.7613 | 1 |
| 96 | Bnode2_wave_unet8 (Bnode2_wave_unet8) | 0.7613 ± 0.0036 | 3 |
| 97 | Bnode2_cross_flip (Bnode2_cross_flip) | 0.7613 ± 0.0019 | 3 |
| 98 | A23_vit_both (ViT-B/16 (SH+Fourier)) | 0.7616 ± 0.0036 | 3 |
| 99 | Q13_loss_silog5 (Q13_loss_silog5) | 0.7616 | 1 |
| 100 | Q3_e0c_base (Q3_e0c_base) | 0.7618 | 1 |
| 101 | A23_vit_sh (ViT-B/16 (SH PE)) | 0.7618 ± 0.0013 | 3 |
| 102 | Q2_unet_rel10silog25 (Q2_unet_rel10silog25) | 0.7624 | 1 |
| 103 | Q2_gcc_silog25 (Q2_gcc_silog25) | 0.7626 | 1 |
| 104 | Bnode2_unet8nolog (Bnode2_unet8nolog) | 0.7628 ± 0.0033 | 3 |
| 105 | Q9_ground_noquery (Q9_ground_noquery) | 0.7633 | 1 |
| 106 | C_raydpt_lsanobias (C_raydpt_lsanobias) | 0.7635 | 1 |
| 107 | B_unet8nolog (U-Net 8-down, no-log (baseline-faithful)) | 0.7637 ± 0.0029 | 3 |
| 108 | R_raydpt_e2 (R_raydpt_e2) | 0.7640 ± 0.0010 | 3 |
| 109 | Q9_ground_none (Q9_ground_none) | 0.7645 ± 0.0002 | 2 |
| 110 | Q6_emaonly (Q6_emaonly) | 0.7645 ± 0.0007 | 2 |
| 111 | Q14_gamma_p05 (Q14_gamma_p05) | 0.7647 | 1 |
| 112 | Q_rdlite_rel10 (Q_rdlite_rel10) | 0.7647 | 1 |
| 113 | Q2_unet_silog5 (Q2_unet_silog5) | 0.7649 | 1 |
| 114 | C_raydpt_5chflip (C_raydpt_5chflip) | 0.7653 ± 0.0022 | 3 |
| 115 | Q14_gamma_n10 (Q14_gamma_n10) | 0.7654 | 1 |
| 116 | C_raydptlite_5chflip (C_raydptlite_5chflip) | 0.7654 ± 0.0035 | 3 |
| 117 | Q_rd_rel10_xl3 (Q_rd_rel10_xl3) | 0.7655 | 1 |
| 118 | Q_rd_silog5 (Q_rd_silog5) | 0.7656 | 1 |
| 119 | Q_rd_rel10_chamfer (Q_rd_rel10_chamfer) | 0.7656 | 1 |
| 120 | Q_rd_rel05 (Q_rd_rel05) | 0.7656 | 1 |
| 121 | Q2_rdlite_silog25 (Q2_rdlite_silog25) | 0.7659 | 1 |
| 122 | Bnode2_cross_unetenc5 (Bnode2_cross_unetenc5) | 0.7660 ± 0.0021 | 3 |
| 123 | C_raydpt_lsaplanar (C_raydpt_lsaplanar) | 0.7663 | 1 |
| 124 | R_echo_unet_e2 (R_echo_unet_e2) | 0.7664 | 1 |
| 125 | C_raydpt_noray (C_raydpt_noray) | 0.7665 | 1 |
| 126 | Q_rd_rel10silog5 (Q_rd_rel10silog5) | 0.7667 | 1 |
| 127 | Q15_csa_norm_w20 (Q15_csa_norm_w20) | 0.7669 | 1 |
| 128 | Bnode2_cross_flip_nr4096 (Bnode2_cross_flip_nr4096) | 0.7670 ± 0.0017 | 2 |
| 129 | Q2_noray_rel10silog25 (Q2_noray_rel10silog25) | 0.7675 | 1 |
| 130 | Q_rd_rel10silog25 (Q_rd_rel10silog25) | 0.7675 | 1 |
| 131 | Q_rd_rel15 (Q_rd_rel15) | 0.7677 | 1 |
| 132 | Bnode2_cross_5chflip (Bnode2_cross_5chflip) | 0.7678 ± 0.0032 | 3 |
| 133 | Q3_e4_silog5 (Q3_e4_silog5) | 0.7679 | 1 |
| 134 | B_cross_nolog (cross implicit, no-log (matched)) | 0.7682 | 1 |
| 135 | Q_rd_rel10_wcl05 (Q_rd_rel10_wcl05) | 0.7685 | 1 |
| 136 | C_raydpt_msf (C_raydpt_msf) | 0.7688 | 1 |
| 137 | Bnode2_crossself_flip (Bnode2_crossself_flip) | 0.7695 ± 0.0051 | 2 |
| 138 | Q2_gcc_rel10silog25 (Q2_gcc_rel10silog25) | 0.7696 | 1 |
| 139 | Q2_unet_downs7 (Q2_unet_downs7) | 0.7696 | 1 |
| 140 | Q_rd_rel10_lr3e4 (Q_rd_rel10_lr3e4) | 0.7700 | 1 |
| 141 | A4_ffmask (cross + far-mask) | 0.7711 | 1 |
| 142 | Bnode2_cross_flip_nr8192 (Bnode2_cross_flip_nr8192) | 0.7718 ± 0.0031 | 2 |
| 143 | Q_rd_silog25 (Q_rd_silog25) | 0.7721 | 1 |
| 144 | C_raydpt_lsaoff (C_raydpt_lsaoff) | 0.7723 | 1 |
| 145 | A5_crossMic (cross + mic-PE) | 0.7724 | 1 |
| 146 | A23_vit_fourier (ViT-B/16 (Fourier PE)) | 0.7724 ± 0.0135 | 3 |
| 147 | U_unet8_scale1 (U_unet8_scale1) | 0.7726 | 1 |
| 148 | Bnode2_cross_nolog (Bnode2_cross_nolog) | 0.7735 ± 0.0037 | 3 |
| 149 | Q_rdmsf_rel10 (Q_rdmsf_rel10) | 0.7735 ± 0.0038 | 2 |
| 150 | A3_crossSH (cross + SH-PE) | 0.7738 | 1 |
| 151 | Q_rd_rel10_wlow1 (Q_rd_rel10_wlow1) | 0.7748 | 1 |
| 152 | T_mlpskip (T_mlpskip) | 0.7748 | 1 |
| 153 | Bnode2_cross_hitokflip (Bnode2_cross_hitokflip) | 0.7750 ± 0.0014 | 3 |
| 154 | Q8_gcc_csa_wscale10 (Q8_gcc_csa_wscale10) | 0.7753 | 1 |
| 155 | T_film (T_film) | 0.7763 | 1 |
| 156 | R_echo_ray_e2 (R_echo_ray_e2) | 0.7765 | 1 |
| 157 | T_sector (T_sector) | 0.7777 | 1 |
| 158 | Bnode2_cross_unetenc (Bnode2_cross_unetenc) | 0.7785 ± 0.0043 | 3 |
| 159 | A6_crossself (cross + ray self-attn) | 0.7804 ± 0.0049 | 3 |
| 160 | A4_cross (cross-attn implicit) | 0.7805 ± 0.0028 | 3 |
| 161 | A6sec (A6sec) | 0.7810 ± 0.0040 | 3 |
| 162 | E_echo_ray (E_echo_ray) | 0.7811 ± 0.0015 | 3 |
| 163 | C_raydpt_rsmp (C_raydpt_rsmp) | 0.7831 | 1 |
| 164 | T_progpe (T_progpe) | 0.7831 | 1 |
| 165 | T_all (T_all) | 0.7831 | 1 |
| 166 | C_unet8_raycoarse16_5chflip (C_unet8_raycoarse16_5chflip) | 0.7851 ± 0.0074 | 3 |
| 167 | Bnode2_cross_hitok (Bnode2_cross_hitok) | 0.7917 ± 0.0043 | 2 |
| 168 | A12_film (A12_film) | 0.7931 | 1 |
| 169 | A14_rir5 (A14_rir5) | 0.7939 | 1 |
| 170 | A19_raymodStrong (A19_raymodStrong) | 0.7940 ± 0.0076 | 3 |
| 171 | A19_raymodStrong_fv (A19_raymodStrong_fv) | 0.7944 ± 0.0047 | 3 |
| 172 | A10_cross (A10_cross) | 0.7947 | 1 |
| 173 | C_unet8_sh6_5chflip (C_unet8_sh6_5chflip) | 0.7950 ± 0.0107 | 3 |
| 174 | A14_frozen (A14_frozen) | 0.7959 | 1 |
| 175 | C_unet8_coarse16_5chflip (C_unet8_coarse16_5chflip) | 0.7962 ± 0.0147 | 3 |
| 176 | A13_ild3 (A13_ild3) | 0.7981 | 1 |
| 177 | Q_rdmsf_rel10silog25 (Q_rdmsf_rel10silog25) | 0.7987 | 1 |
| 178 | A9_fullmap (full-map decoder (global bottleneck)) | 0.7988 ± 0.0014 | 3 |
| 179 | Bnode2_cross5ch (Bnode2_cross5ch) | 0.7994 ± 0.0043 | 3 |
| 180 | C_unet8_sh4_5chflip (C_unet8_sh4_5chflip) | 0.7995 ± 0.0201 | 3 |
| 181 | C_unet8_coarseres_5chflip (C_unet8_coarseres_5chflip) | 0.7999 ± 0.0135 | 3 |
| 182 | A8_hybrid (A8_hybrid) | 0.8004 | 1 |
| 183 | Q_rd_rel10_lr5e4 (Q_rd_rel10_lr5e4) | 0.8004 | 1 |
| 184 | C_unet8_coarse32_5chflip (C_unet8_coarse32_5chflip) | 0.8009 ± 0.0055 | 3 |
| 185 | Bnode2_crossself_hitokflip (Bnode2_crossself_hitokflip) | 0.8025 ± 0.0123 | 3 |
| 186 | A13_mag2 (2ch mag (RIR ctrl)) | 0.8039 | 1 |
| 187 | A11_shaux (A11_shaux) | 0.8040 ± 0.0072 | 3 |
| 188 | A2_raymlp (RayMLP (global latent)) | 0.8047 ± 0.0063 | 3 |
| 189 | A14_logmag (A14_logmag) | 0.8057 ± 0.0022 | 2 |
| 190 | U_unet8_scale2 (U_unet8_scale2) | 0.8065 | 1 |
| 191 | A16_raymod8x16 (A16_raymod8x16) | 0.8077 ± 0.0038 | 3 |
| 192 | A16_raymod_fv (A16_raymod_fv) | 0.8084 ± 0.0130 | 3 |
| 193 | A13_ipd5 (5ch RIR (+phase/IPD)) | 0.8084 | 1 |
| 194 | Q14_gamma_p10_scale10 (Q14_gamma_p10_scale10) | 0.8108 | 1 |
| 195 | A18_raymod64reg (A18_raymod64reg) | 0.8121 ± 0.0079 | 2 |
| 196 | Q_rd_rel13 (Q_rd_rel13) | 0.8131 | 1 |
| 197 | Q10_vit_scratch (Q10_vit_scratch) | 0.8141 | 1 |
| 198 | A20_unet64_aug (A20_unet64_aug) | 0.8166 ± 0.0034 | 3 |
| 199 | A21_raymodStrong_aug (A21_raymodStrong_aug) | 0.8179 ± 0.0311 | 3 |
| 200 | Bnode2_foa_unet8 (Bnode2_foa_unet8) | 0.8195 ± 0.0006 | 2 |
| 201 | A15_bigunet_fv (A15_bigunet_fv) | 0.8220 ± 0.0073 | 3 |
| 🔻 202 | A18_unet64reg_fv (A18_unet64reg_fv) | 0.8222 ± 0.0027 | 3 |
| 203 | A15_bigunet (pix2pix U-Net (ngf96)) | 0.8225 ± 0.0065 | 3 |
| 204 | Bnode2_hybrid5ch (Bnode2_hybrid5ch) | 0.8233 ± 0.0070 | 3 |
| 205 | Q14_gamma_p10 (Q14_gamma_p10) | 0.8239 | 1 |
| 🔻 206 | Aunet (pix2pix U-Net) | 0.8287 ± 0.0034 | 3 |
| 🔻 207 | A18_unet64reg (pix2pix U-Net (reg)) | 0.8290 ± 0.0050 | 3 |
| 208 | Bnode2_cross_vitenc (Bnode2_cross_vitenc) | 0.8547 ± 0.0064 | 2 |
| 209 | Q10_vit_frozen (Q10_vit_frozen) | 0.8639 | 1 |
| 210 | Bnode2_rayconv5d (Bnode2_rayconv5d) | 0.8765 ± 0.0256 | 3 |
| 211 | Q14_gamma_p15 (Q14_gamma_p15) | 0.9012 | 1 |
| 212 | A4_cross_shuf (A4_cross_shuf) | 0.9793 | 1 |
| 213 | A2_shuf (shuffle-audio (control)) | 0.9823 ± 0.0004 | 2 |
| 214 | A1_rayonly (ray-only prior (control)) | 0.9832 ± 0.0067 | 2 |

**Headline:** best robust model = **Bnode2_gcc_unet8** at **0.748 ± 0.004 m** (3 seeds). pix2pix U-Net = 0.822 m. **Resolution inverts the 64×128 ranking** — there the U-Net was best (~0.775) and RayMLP worst; at full 256×512 the cross-attention *implicit* model is best and the U-Net is the worst real model. Implicit/coordinate models emit a **band-limited** field that sits at the audio observability ceiling (resolution-robust); the U-Net chases fine detail audio cannot predict and that full-res GT exposes (degrades). Full per-metric table: see `RESULTS_full.md`.
<!-- BEST:END -->

---

Tests whether a **ray-conditioned implicit depth function** beats the existing
global-bottleneck encoder–decoder at binaural-audio → ERP radial depth, by
**decomposing the hypothesis** and breaking each sub-question in order rather
than building one big model.

```
binaural spec ──► audio encoder ──► global latent z  /  tokens
ERP ray dir r  ─► [xyz | Fourier-PE | SH basis | ear-axis mic-PE] ─► ray query q
q  (×audio)  ──► depth(r)            # implicit: predict per-ray, not a full map
```

## Hypothesis ladder (run in order)
| Q | question | runs |
|---|---|---|
| Q1 | implicit fn uses audio at all? | A1 ray-only prior vs A2 RayMLP (+shuffled control) |
| Q2 | SH/Fourier ray-PE give inductive bias? | A2 vs A3 (`--use-sh-pe`) on low-freq metrics |
| Q3 | ear-axis mic-PE helps binaural use? | A3 vs A5 (`--use-mic-pe`), L/R-swap test |
| Q4 | ray self-attn corrects unobservable rays? | A5 vs A6 (`--model crossself`) |
| Q5 | SH-coarse + residual cuts mean-blob? | A5 vs A8 (`--model hybrid`) |

## Data / reuse
- Reuses `../test_for_audio_better/cache` (spec 2×64×128 log-mag binaural,
  depth radial /max_depth∈[0,1], mask). No data prep.
- Dataset is **listener-centred & self-emitting** (active echolocation): source≈origin
  (per-ray source-PE degenerate → dropped); ears are a fixed `±y` rig (`head_r`),
  giving a legitimate mic-PE that drives the L/R-swap mirror test.

## Models (`--model`)
`rayonly` · `raymlp` · `cross` · `crossself` · `hybrid`.
Ray-feature flags: `--use-xyz --use-fourier-pe --use-sh-pe --use-mic-pe`.
Head: `--use-depth-bins`. Controls: `--audio-mode {stereo,mono,left,right,none}`,
`--shuffle-audio True`.

## Run
```bash
bash run_stage1.sh          # Q1 gate, 2 seeds
python agg_stage1.py        # verdict table
python eval.py --run-name A2_raymlp_s0 --controls True
```
Training supervises N random rays/sample; eval predicts the full grid by chunking.

## Metrics (`metrics.py`)
MAE/RMSE/AbsRel/δ<1.25/SILog (cos-lat weighted) + **layout** metrics that matter
for the SH/implicit claim: low-pass MAE, SH-coefficient L1 error, sector MAE.
Controls in `eval.py`: mono/left/right/shuffle + L/R-swap mirror consistency.
