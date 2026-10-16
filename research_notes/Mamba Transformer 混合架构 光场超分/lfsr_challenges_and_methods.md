# NTIRE Light Field Image SR (LFSR) challenges 2023–2026, and the Transformer, Mamba and hybrid methods used in them

Research date: 2026-10-16. Access note: openaccess.thecvf.com, arxiv.org, ar5iv, alphaxiv and codalab could not be fetched directly from this environment (DNS/proxy refusals). Numbers from the NTIRE reports and from arXiv papers therefore come from search-engine extraction of those pages. They are marked "(search-extracted)" and should be spot-checked against the PDFs. GitHub READMEs (raw.githubusercontent.com) and the local LFTransMamba repo were read directly.

---

## Q1. What placed highly in each NTIRE LFSR edition (2023–2026), with which architectures, rules, training budgets and scores?

### Takeaway
The same team, OpenMeow (Kai Jin, Bigo / Zeqiang Wei, BUPT, et al.), won Track 1 in 2023, 2025 and 2026. It placed 3rd in Track 1 in 2024. The hybrid design changed over time:
- 2023: CNN+Transformer (DistgEPIT).
- 2024: winner was a scaled pure Transformer, BigEPIT by BNU&TMU-AI-TRY, ensembled with other models.
- 2025: Mamba+Transformer (LFTransMamba; LFTramba also placed 3rd).
- 2026: OpenMeow won all three tracks; its 2026 method is not public, but report text describes an LFTransMamba-style composed block.

So "Mamba+Transformer did best" holds only for 2025 and 2026. Mamba did not exist as a vision backbone in 2023, and the 2024 winner was a pure Transformer. Every winner relied on ensembles, TTA, disparity-aware training-data resampling and disparity-preserving tiled inference.

### Cited Findings

**Common rules and data (all editions)**
- Every edition is ×4 SR with bicubic downsampling on 5×5 views. Ranking is by average PSNR over the synthetic and real test halves; SSIM is reported but does not affect ranking — [NTIRE2026 branch README](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2026); [NTIRE 2024 README (commit a473ad7)](https://raw.githubusercontent.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/a473ad7/README.md)
- Training set: EPFL, HCInew, HCIold, INRIA and STFgantry, 144 scenes in total, all at 9×9 angular resolution (following DistgSSR). Validation and test sets each contain 16 synthetic scenes rendered with 3DS MAX and 16 real scenes captured with a Lytro ILLUM. LR inputs are 5×5 — [NTIRE2026 branch README](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2026)
- Metric protocol: per-SAI PSNR/SSIM averaged over A² views, then over the M scenes — [BasicLFSR README](https://github.com/ZhengyuLiang24/BasicLFSR)

**NTIRE 2023 (1st LFSR challenge; single fidelity track)** — Report: "NTIRE 2023 Challenge on Light Field Image Super-Resolution: Dataset, Methods and Results", Y. Wang, L. Wang, Z. Liang, J. Yang, R. Timofte, Y. Guo et al., CVPRW 2023, pp. 1320–1335, [arXiv:2304.10415](https://arxiv.org/abs/2304.10415)
- 148 registered participants. 11 teams beat the LF-InterNet baseline. The winner improved about 1 dB PSNR over DistgSSR. The report also introduced the NTIRE-2023 val/test dataset and the BasicLFSR toolbox — [arXiv:2304.10415 abstract (search-extracted)](https://arxiv.org/abs/2304.10415)
- NTIRE-2023 test spatial sizes: synthetic 500×500, real 624×432. For synthetic scenes, all virtual cameras have identical intrinsics and are co-planar with parallel optical axes — [BasicLFSR README](https://github.com/ZhengyuLiang24/BasicLFSR)
- **1st place: DistgEPIT**, team OpenMeow/BIGO. "DistgEPIT: Enhanced Disparity Learning for Light Field Image Super-Resolution", K. Jin, A. Yang, Z. Wei, S. Guo, M. Gao, X. Zhou, CVPRW 2023, pp. 1373–1383, [CVF page](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Jin_DistgEPIT_Enhanced_Disparity_Learning_for_Light_Field_Image_Super-Resolution_CVPRW_2023_paper.html)
  - Architecture: a hybrid of convolution (DistgSSR disentangling) and Transformer (EPIT) modules. It aims to "capture angular correlation, refine adjacent disparities and extract spatial features" — [repo README](https://github.com/OpenMeow/NTIRE23_LFSR_DistgEPIT)
  - Position-Sensitive Windowing (PSW) keeps disparity consistent between training and tiled inference. It gives about +0.2 dB average PSNR over conventional padding and windowing — [repo README](https://github.com/OpenMeow/NTIRE23_LFSR_DistgEPIT)
  - Training data: 5×5 views were generated from the 9×9 training LFs with "Central Selection", "Interval Selection" and "Uneven Selection" of views, i.e. angular resampling that changes the effective baseline and disparity — [repo README](https://github.com/OpenMeow/NTIRE23_LFSR_DistgEPIT)
  - Checkpoint names encode batch 4, LR 1e-4 or 5e-5, 100 epochs, HR patch 128 (LR 32), EMA 0.999, plus fine-tuning stages. Variants were "wider" and "deeper" DistgEPIT. Best local validation (with TTA) was 33.2494 dB mean (real 32.0270, synthetic 35.0832) — [repo README](https://github.com/OpenMeow/NTIRE23_LFSR_DistgEPIT)
  - Training ran on 4× RTX 2080Ti. LR was halved at epoch 80 and training stopped at 100 epochs; the paper reports 32.90 dB / 0.9468 for 4× — [DistgEPIT paper (search-extracted)](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Jin_DistgEPIT_Enhanced_Disparity_Learning_for_Light_Field_Image_Super-Resolution_CVPRW_2023_paper.pdf)
- Other 2023 participants, inferred from the co-author list of the report: SKKU (Duong et al., HLFSR line), USTC (Xiong group), TJU, BJTU, BNU (Chao et al., later the 2024 winners), BIT (Xia, Lu et al., later LFMamba), and Beihang (Cong, Sheng et al., LF-DET) — [BasicLFSR/NTIRE2023 bibtex in DistgEPIT README](https://github.com/OpenMeow/NTIRE23_LFSR_DistgEPIT). Their ranks could not be verified (see Gaps).

**NTIRE 2024 (2nd edition; two tracks)** — Report: "NTIRE 2024 Challenge on Light Field Image Super-Resolution: Methods and Results", Y. Wang, Z. Liang, Q. Chen, L. Wang, J. Yang, R. Timofte, Y. Guo et al., CVPRW 2024, pp. 6218–6234, DOI 10.1109/CVPRW63382.2024.00626, [CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.html)
- Tracks:
  - Track 1, "Fidelity Only": no efficiency limit.
  - Track 2, "Fidelity & Efficiency": ≤1M parameters and ≤20 GFLOPs (fvcore, 5×5×32×32 input). TTA cost counts toward FLOPs, and inference time is reported.
  - No external data or pretrained models were allowed.
  - Baselines: DistgSSR for Track 1, Bicubic for Track 2.
  - Source: [NTIRE 2024 README (commit a473ad7)](https://raw.githubusercontent.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/a473ad7/README.md)
- 125 registered and 9 teams beat the baselines. The report's main challenge statement is "how to exploit angular information across many views with differing disparities" — [research.fi record / CVF abstract (search-extracted)](https://research.fi/en/results/publication/0690534724)
- **Track 1 1st: BNU&TMU-AI-TRY, BigEPIT.** Test 30.80 dB / 0.9332, 11.04M params, 569.30 GFLOPs, Transformer. Improvement over DistgEPIT ≈ 0.14 dB — [NTIRE 2024 report table (search-extracted)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf)
  - Paper: "BigEPIT: Scaling EPIT for Light Field Image Super-Resolution", W. Chao et al., CVPRW 2024, pp. 6187–6197, [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chao_BigEPIT_Scaling_EPIT_for_Light_Field_Image_Super-Resolution_CVPRW_2024_paper.pdf)
  - Three strategies: compound depth+width scaling of EPIT; "augmented data resampling", which uses varying view-sampling intervals instead of only the central views; and a "high-precision test scheme".
  - The authors found PSNR rises with test patch size. They recommend full-size or larger patches without zero padding, because "zero padding and window partition will destroy the LF disparity structure".
  - Final submission: self-ensemble (flips/rotations) plus a multi-model average of BigEPIT, DistgEPIT (deeper/wider) and RR-HLFSR — [BigEPIT / NTIRE 2024 report (search-extracted)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chao_BigEPIT_Scaling_EPIT_for_Light_Field_Image_Super-Resolution_CVPRW_2024_paper.pdf)
  - BigEPIT notes that existing LF datasets have an imbalanced disparity distribution with few large-disparity samples. The 2024 report states that augmented sampling (central, even, uneven) explicitly increases large-disparity examples and improves robustness to disparity changes — [BigEPIT & NTIRE 2024 report (search-extracted)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf)
- **Track 1 3rd: OpenMeow, "PSW++"** on an LF-DET-based Transformer. Test 30.71 / 0.9323, 10.63M params, 353.52 GFLOPs — [NTIRE 2024 report table (search-extracted)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf); team confirms 3rd ("季军") in [OpenMeow profile](https://github.com/OpenMeow/.github)
  - Recipe: LF-DET variant, LR patch 32 / stride 16, batch 3, LR 2e-4, 90 epochs, EMA 0.999, then fine-tuning. Inference with the PSW++ tiler gave local validation 33.1377 dB — [PSW++ README](https://github.com/OpenMeow/NTIRE24_LFSR_PSWPP)
- **Track 2 2nd: team "Low-level visualist", LGFN.** CNN with local convolution modulation plus global attention; 0.45M params, 19.33 GFLOPs; test 30.05 dB / 0.924. The same team placed 7th in Track 1 — [LGFN, Z. Yu et al., CVPRW 2024, arXiv:2409.17759 (search-extracted)](https://arxiv.org/abs/2409.17759)
- Track 2 winner: not verified (see Gaps).

**NTIRE 2025 (3rd edition; three tracks)** — Report: "NTIRE 2025 Challenge on Light Field Image Super-Resolution: Methods and Results", Y. Wang et al., CVPRW 2025, pp. 1227–1246, [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Wang_NTIRE_2025_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2025_paper.pdf)
- Tracks:
  - Track 1 "Classic": standard training data only, no external data or pretrained models, no efficiency limit.
  - Track 2 "Efficiency": ≤1M params, ≤20 GFLOPs at 5×5×32×32, TTA counted.
  - Track 3 "Large Model": external data and pretrained models allowed, no efficiency limit; val/test sets may not be used for training.
  - Baselines: DistgSSR (Track 1), Bicubic (Track 2), EPIT (Track 3).
  - Source: [NTIRE 2025 README (commit 343a275)](https://raw.githubusercontent.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/343a275/README.md)
- 308 registered and 13 teams beat the baselines. The Track 1 winner improved 0.36 dB PSNR over the 2024 champion on the test set — [ECNU pure record of the report abstract](https://pure.ecnu.edu.cn/en/publications/ntire-2025-challenge-on-light-field-image-super-resolution-method/)
- **Track 1 1st, Track 3 1st, Track 2 2nd: OpenMeow, LFTransMamba.** "LFTransMamba: A Hybrid Mamba-Transformer Model for Light Field Image Super-Resolution", K. Jin, Z. Wei, A. Yang, D. Wu, M. Gao, X. Zhou, CVPRW 2025, pp. 1195–1204 — [CVF page](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/html/Jin_LFTransMamba_A_Hybrid_Mamba-Transformer_Model_for_Light_Field_Image_Super-Resolution_CVPRW_2025_paper.html); [local README](file:///home/user/LFTransMamba/README.md); [OpenMeow profile](https://github.com/OpenMeow/.github)
  - A secondary summary gives test PSNR of 31.16 dB (Track 1) and 31.22 dB (Track 3). Not verified against the report table — [search-extracted summary](https://mlanthology.org/cvprw/2025/jin2025cvprw-lftransmamba/)
  - The paper describes three components:
    - a hybrid of Transformer global context and Mamba long-range modelling, with a Transformer spatial-enhancement module for SAI features added on top of L²FMamba;
    - Masked Light Field Image Modeling (MLFIM), a masked-reconstruction training strategy "without additional modules or loss functions";
    - Enhanced PSW (EPSW), Gaussian-weighted aggregation of overlapping tiles that emphasises central pixels and reduces structural artifacts.
    - Source: [CVF abstract (search-extracted)](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/html/Jin_LFTransMamba_A_Hybrid_Mamba-Transformer_Model_for_Light_Field_Image_Super-Resolution_CVPRW_2025_paper.html)
  - Track 1 recipe (from the README):
    - Ensemble of 3 models: LF_DET_0214, a 160-channel pure-Transformer LF-DET variant, with patch 16/stride 8; plus LFTransMamba (D16, R1, K18, C80, T2) at patches 40/12 and 48/12. All use the psw++ tiler and TTA.
    - Training: LR patch 32, stride 8, batch 3, 30 epochs, LR 3e-4 (2e-4 for LF-DET), EMA 0.999, on dataset "LFSR.ALL4".
    - Local validation, single LFTransMamba: 33.8803 dB mean (real 32.7436, synthetic 35.5854). LF-DET_0214: 33.7795 dB.
    - Source: [local README](file:///home/user/LFTransMamba/README.md)
  - Track 2 recipe: the model is LFTransMamba_D16_R1_K6_C32_T0, i.e. T=0 with no Transformer blocks; the checkpoint is named "L2FMamba". Trained 100 epochs at LR 2e-4. Local validation 32.9525 dB without TTA. **The efficiency entry is effectively a pure-Mamba L²FMamba** — [local README](file:///home/user/LFTransMamba/README.md)
  - Track 3 recipe: LFTransMamba_D16_R1_K24_C80_T2, FP16, 30 epochs, on "LFSR.ALL4.EXTRA", ensembled with the Track 1 models. Local validation 33.9310 dB — [local README](file:///home/user/LFTransMamba/README.md)
  - "LFSR.ALL4.EXTRA" adds the external DLFD and SLFD datasets to the five standard sets. "LFSR.ALL4" contains the standard five sets each generated four times with different 5-of-9 view selections (central "matlab" default, 13579, 24568, 12589; 34567 for DLFD/SLFD) — [local lfsr.py lines 44–106](file:///home/user/LFTransMamba/lfsr.py)
- **Track 1 3rd and Track 2 5th: LFTramba.** "LFTramba: Comprehensive Information Learning for Light Field Image Super-Resolution via a Hybrid Transformer-Mamba Framework", H. Liu, X. Zhu, H. Zeng, J. Zhu, Y. Shi, J. Chen, J. Hou, CVPRW 2025, pp. 1137–1147 — [CVF page (search-extracted)](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/html/Liu_LFTramba_Comprehensive_Information_Learning_for_Light_Field_Image_Super-Resolution_via_CVPRW_2025_paper.html)
  - A Spatial-Angular Mamba Block alternates spatial and angular capture. An Epipolar Plane Transformer Block extracts disparity features from horizontal and vertical EPIs. So it is a second Mamba+Transformer entry in the 2025 top 3.
- **Also at CVPRW 2025: LFMix** — "LFMix: A Lightweight Hybrid Architecture for Light Field Super-Resolution", M. Yu, Z. Wu, D. Huang (ECNU).
  - Hybrid CNN-Transformer over SAI, MacPI and EPI representations. Its MixBlock combines spatial conv, spectral self-attention and learnable high-frequency filtering. Loss is pixel plus frequency (dual-domain).
  - Reports 32.99 dB with 0.98M params and 19.48 GFLOPs. Which dataset the 32.99 refers to, and its challenge rank, were not verified.
  - Source: [CVF PDF (search-extracted)](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Yu_LFMix_A_Lightweight_Hybrid_Architecture_for_Light_Field_Super-Resolution_CVPRW_2025_paper.pdf)

**NTIRE 2026 (4th edition; held at CVPR 2026)** — Report: "NTIRE 2026 Challenge on Light Field Image Super-Resolution: Methods and Results", Y. Wang, L. Wang, … R. Timofte et al., CVPRW 2026, [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Wang_NTIRE_2026_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2026_paper.pdf); [supplement](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/supplemental/Wang_NTIRE_2026_Challenge_CVPRW_2026_supplemental.pdf)
- Rules are identical to 2025: three tracks, Track 2 limited to 1M params and 20 GFLOPs with TTA counted, Track 3 allows external data and pretrained models. It moved to CodaBench. Dates: train/val release 2026-01-20, test release 2026-03-10, submission deadline 2026-03-17 — [NTIRE2026 branch README](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2026)
- 162 registered and 11 teams beat the baselines. The Track 1 winner improved 0.2 dB over the 2025 champion on the test set — [NTIRE 2026 report abstract (search-extracted)](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Wang_NTIRE_2026_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2026_paper.pdf)
- **OpenMeow ranked 1st in all three tracks.**
  - Track 1 entry: 95.20M params and 2733.5 GFLOPs. Track 2 entry: 0.56M params and 13.83 GFLOPs.
  - Report text (garbled extraction) describes the Track 1 configuration as K=48 composed blocks, C=160 channels and T=3 Transformer modules per block. That is the LFTransMamba parametrisation, scaled up from K18/C80/T2.
  - The team's GitHub profile lists 2026 champion ("冠军") in Classic, Efficiency and Large Model, with no code repo linked yet.
  - Sources: [NTIRE 2026 report (search-extracted)](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Wang_NTIRE_2026_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2026_paper.pdf); [OpenMeow profile](https://github.com/OpenMeow/.github)
- The 2026 report says recent LFSR methods increasingly use hybrid Mamba–Transformer or Mamba–CNN architectures, citing LFTransMamba as representative — [NTIRE 2026 report (search-extracted)](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Wang_NTIRE_2026_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2026_paper.pdf)
- **3rd in Tracks 1 and 3, 4th in Track 2: GTF.** "GTF: Omnidirectional EPI Transformer for Light Field Super-Resolution", Kunyu Li, Fei Wang, Lichao Zhang, Junjie Liu, Bihong Li, CVPRW 2026, [arXiv:2605.04581](https://arxiv.org/abs/2605.04581), [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Li_GTF_Omnidirectional_EPI_Transformer_for_Light_Field_Super-Resolution_CVPRW_2026_paper.pdf)
  - Models horizontal, vertical, 45° and 135° EPIs, with MacPI-based prior injection, adaptive directional fusion and a topology-preserving FFN.
  - 32.78 dB without inference-time enhancement. GTF-Tiny: 32.57 dB at 0.915M params and 19.81 GFLOPs.
  - The fidelity-track submission was a weighted average of GTF, LF_DET and LFTransMamba. The report lists the team as OpenXS (search-extracted).
  - Inference-trick ablation (search-extracted): vanilla 32.775 → EPSW 33.006 (+0.23 dB); TTA alone 32.896 (+0.12 dB).
- Another Track 2 entry is listed at 0.47M params and 19.51 GFLOPs (team not identified) — [NTIRE 2026 report (search-extracted)](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Wang_NTIRE_2026_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2026_paper.pdf)

**Validation-set reference scores (identical tables in the 2024, 2025 and 2026 READMEs)**

| Method | Avg PSNR | Syn PSNR | Real PSNR | Avg SSIM |
|---|---|---|---|---|
| Bicubic | 27.51 | 27.53 | 27.49 | 0.8714 |
| DistgSSR | 31.75 | 31.23 | 32.26 | 0.9424 |
| EPIT | 32.04 | 31.53 | 32.54 | 0.9447 |

Source: [NTIRE2026 README](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2026); [2024 README](https://raw.githubusercontent.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/a473ad7/README.md)

### Inferences
- **The test set has probably been reused since 2023 or 2024.** Three facts point this way:
  - The 2024, 2025 and 2026 READMEs share the same validation baseline table and the same OneDrive link.
  - The reports measure "improvement over last year's champion on the test set" (2025 +0.36 dB, 2026 +0.2 dB).
  - The arithmetic chains: 2024 winner 30.80 + 0.36 = 31.16, which matches the reported LFTransMamba Track 1 score. 2023's DistgEPIT ≈ 30.66 if BigEPIT's +0.14 is on the same set.
  - If true, the 2026 champion is about 31.36 dB on that test set. This is an inference, not a verified number.
- **Approximate winning trajectory on the hidden test (same-set assumption):** DistgEPIT (CNN+Transformer, 2023) ≈ 30.66 → BigEPIT ensemble (Transformer, 2024) 30.80 → LFTransMamba ensemble (Mamba+Transformer, 2025) ≈ 31.16 → OpenMeow 2026 (scaled LFTransMamba-style, about 95M params) ≈ 31.36. Gains per year are shrinking (+0.14, +0.36, +0.2 dB), and much of the 2026 gain comes with roughly 9× more params than the 2025 Track 1 model.
- **Winning recipes share non-architectural ingredients**, and these may matter as much as the backbone choice:
  - model ensembles that include a pure-Transformer LF-DET (2025 Track 1) or EPIT variants (2024);
  - 8× flip/rotation TTA;
  - EMA weights;
  - disparity-aware angular resampling of training data;
  - PSW / PSW++ / EPSW tiled inference (worth +0.2 dB per DistgEPIT; +0.23 dB per GTF's ablation).
- In the efficiency track the hybrid did not win in 2025: the OpenMeow entry was pure Mamba (T=0) and placed 2nd. Mamba's advantage is clearest in the parameter- and FLOP-constrained regime and in very large models; the Transformer branch was dropped when FLOPs were capped at 20G.

### Gaps
- The full per-team result tables for all four reports could not be read (CVF and arXiv blocked here). Not verified:
  - 2023 ranks 2 through 11;
  - 2024 Track 1 2nd place and the Track 2 winner;
  - 2025 Track 1 2nd place, the Track 2 winner (LFMix is a candidate, unverified) and Track 3 ranks 2 and 3;
  - 2026 2nd places and the exact OpenMeow 2026 architecture.
- No official per-team training budgets (GPU-hours, iterations) were found beyond those in team READMEs.
- 2024 participation by track is garbled in extraction ("10 and 6 teams reached final test phase").

---

## Q2. LF-specific Transformer and CNN baselines (LFT, DPT, DistgSSR, EPIT, LF-DET, HLFSR, LFSSR_SAV, SkimLFSR): what they model and how they score

### Takeaway
On the standard benchmarks and on the NTIRE-2023 test, the strongest pre-Mamba models are LF-DET and EPIT. Both are small Transformers (about 1.5–1.7M params) whose attention is laid out along LF structure: spatial/angular subspaces for LF-DET, EPIs for EPIT. EPIT's main selling point is robustness to disparity variation — exactly the property the NTIRE domain gap stresses.

### Cited Findings

**Benchmark scores (BasicLFSR, ×4, 5×5, PSNR/SSIM)** — [BasicLFSR README](https://github.com/ZhengyuLiang24/BasicLFSR)

| Method | Params | EPFL | HCInew | HCIold | INRIA | STFgantry | NTIRE-2023 test avg | NTIRE-2023 Lytro / Syn |
|---|---|---|---|---|---|---|---|---|
| LFT (Liang et al., SPL 2022, [arXiv:2108.07597](https://arxiv.org/abs/2108.07597)) | 1.163M | 29.255 | 31.462 | 37.630 | 31.205 | 31.860 | 29.769 / 0.9252 | 29.657 / 29.881 |
| DPT (Wang et al., AAAI 2022, [arXiv:2201.00346](https://arxiv.org/abs/2201.00346)) | 3.778M | 28.939 | 31.196 | 37.412 | 30.964 | 31.150 | 29.566 / 0.9226 | 29.360 / 29.771 |
| DistgSSR (Wang et al., TPAMI 2022, [arXiv:2202.10603](https://arxiv.org/abs/2202.10603)) | 3.582M | 28.992 | 31.380 | 37.563 | 30.994 | 31.649 | 29.637 / 0.9244 | 29.389 / 29.884 |
| LFSSR_SAV ([repo](https://github.com/Joechann0831/SAV_conv)) | 1.543M | 29.368 | 31.450 | 37.497 | 31.270 | 31.362 | 29.782 / 0.9250 | 29.713 / 29.850 |
| EPIT (Liang et al., ICCV 2023, [arXiv:2302.08058](https://arxiv.org/abs/2302.08058)) | 1.470M | 29.339 | 31.511 | 37.677 | 31.372 | 32.179 | 29.874 / 0.9259 | 29.718 / 30.030 |
| HLFSR-SSR ([repo](https://github.com/duongvinh/HLFSR-SSR)) | 13.87M | 29.196 | 31.571 | 37.776 | 31.241 | 31.641 | 29.830 / 0.9263 | 29.714 / 29.945 |
| LF-DET (Cong et al., IEEE TMM 2023) | 1.687M | 29.473 | 31.558 | 37.843 | 31.389 | 32.139 | 29.944 / 0.9260 | 29.911 / 29.976 |

- LF-DET's full title, "Exploiting Spatial and Angular Correlations With Deep Efficient Transformers for Light Field Image Super-Resolution" (R. Cong, H. Sheng, D. Yang, Z. Cui, R. Chen, IEEE TMM 2023), is from the citation block in [local LF_DET_0214.py](file:///home/user/LFTransMamba/models/LF_DET_0214.py). The OpenMeow team used a 160-channel LF-DET variant ("NTIRE2024: Efficiency Version" per the docstring) as an ensemble member in 2024 and 2025.
- **EPIT**: "Learning Non-Local Spatial-Angular Correlation for Light Field Image Super-Resolution", Z. Liang, Y. Wang, L. Wang, J. Yang, S. Zhou, Y. Guo, ICCV 2023, pp. 12376–12386 — [repo](https://github.com/ZhengyuLiang24/EPIT)
  - Argues existing networks fail to use long-range spatial-angular correlation and therefore degrade under large disparity variation. EPIT gets a global receptive field along epipolar lines.
  - The authors shear LFs by chosen amounts (e.g. 2 and 4 in the journal version) and show more consistent performance than other methods across disparity shifts, plus wider LAM attribution maps.
  - Source: [EPIT paper / summaries (search-extracted)](https://arxiv.org/abs/2302.08058)
- The journal extension of EPIT, "Diving Into Epipolar Transformers for Light Field Super-Resolution and Disparity Estimation" (IEEE TPAMI 2026), extends it to joint SR and disparity estimation — [DOI 10.1109/TPAMI.2026.3674997](https://doi.org/10.1109/tpami.2026.3674997)
- **SkimLFSR**: "Less is More: Skim Transformer for Light Field Image Super-resolution", Hu et al. (U. Sydney), [arXiv:2407.15329](https://arxiv.org/abs/2407.15329) (search-extracted)
  - Argues that attending to all SAIs indiscriminately causes "disparity entanglement". Uses multi-branch skim attention where each branch covers a specific disparity range over a subset of SAIs.
  - Uses 67% of the prior leading method's parameters and gains +0.63 dB (×2) and +0.35 dB (×4).
  - The authors attribute the gain to "disparity-aware characteristics and latent discriminative capability toward scene depth and camera configuration, learned implicitly from a regression-based task".
- LFMT's authors note that LF-DET is strongest on small-disparity scenes, attributing this to its multi-scale angular modelling — [arXiv:2509.04824 (search-extracted)](https://arxiv.org/abs/2509.04824)

### Inferences
- Benchmark rankings carry over well to the NTIRE-2023 test (LF-DET > EPIT > HLFSR/LFSSR_SAV > LFT > DistgSSR). Architectural gains measured on the standard five test sets did transfer to the challenge distribution; the domain gap did not reorder methods.
- The best Transformers are not generic. Their attention is restricted to LF subspaces (EPI, SAI, MacPI), which encodes the 4D geometry as structure in the architecture rather than in the loss. SkimLFSR's authors explicitly frame this as geometry learned implicitly from a regression objective, given a disparity-aware architecture.

### Gaps
- An "LFSAV" method distinct from LFSSR_SAV was not identified. The SAV paper's full title was not verified.
- The HLFSR journal title ("Light Field Image Super-Resolution Network via Joint Spatial-Angular and Epipolar Information", Duong et al., IEEE TCI 2023) appeared only as a link title in search and was not verified.

---

## Q3. LF Mamba and hybrid Mamba-Transformer methods: designs, ablations, efficiency, and the stated rationale for the combination

### Takeaway
LF Mamba papers consistently find that:
- naive 2D-vision Mamba (VMamba SS2D on flattened SAIs) is sub-optimal;
- Mamba must scan along LF subspaces (SAI, MacPI, EPI);
- EPI or geometry branches are worth about 0.2–0.5 dB.

Hybrid papers find Mamba+Transformer beats either alone by about 0.1–0.15 dB. The usual rationale: Mamba gives cheap long-range or global scanning, and attention fixes Mamba's weakness on fine local or EPI structure, or vice versa.

### Cited Findings

**LFMamba** — "LFMamba: Light Field Image Super-Resolution with State Space Model", W. Xia, Y. Lu, S. Wang, Z. Wang, P. Xia, T. Zhou (BIT), [arXiv:2406.12463](https://arxiv.org/abs/2406.12463) (2024); code at [stanley-313/LFMamba](https://github.com/stanley-313/LFMamba), uses mamba-ssm 1.1.2
- Applies SSMs to informative 2D slices of 4D LFs with an efficient SS2D block. A structure-learning module alternately applies SSM blocks on horizontal and vertical EPIs, three times, "because they are highly correlated with scene depth".
- Directly using the original SS2D in a VSS block "achieves sub-optimal results".
- Reported +0.14 dB average over LF-DET at ×4 (secondary summary).
- Source: [arXiv (search-extracted)](https://arxiv.org/abs/2406.12463)
- Ablation (Table II, search-extracted):
  - Full model: 2.30M params; EPFL 29.84, HCInew 31.70, HCIold 37.91, INRIA 31.81, STFgantry 31.85.
  - **w/o EPI** (EPI blocks replaced by five spatial-angular SSM blocks, 2.39M): −0.27 dB EPFL, −0.35 dB INRIA, but +0.16 dB STFgantry.
  - Replacing Mamba with Transformer modules: w/ LFT (1.65M) −0.14 EPFL; w/ EPIT (2.34M) −0.21; w/ LFT_EPIT (2.26M) −0.25. w/o fusion −0.21; w/ reverse scan −0.26.
  - Three basic SSM blocks gave no consistent gain over two.
  - Source: [arXiv:2406.12463 (search-extracted)](https://arxiv.org/abs/2406.12463)

**MLFSR** — "Mamba-based Light Field Super-Resolution with Efficient Subspace Scanning", R. Gao, Z. Xiao, Z. Xiong (USTC), ACCV 2024, [arXiv:2406.16083](https://arxiv.org/abs/2406.16083); code at [RSGao/MLFSR](https://github.com/RSGao/MLFSR)
- Tokenises 4D LFs into subspace sequences with bi-directional scanning on each subspace.
- A Mamba-based Global Interaction module handles global information; a Spatial-Angular Modulator adds local detail.
- Adds a **Transformer-to-Mamba (T2M) loss**, i.e. distillation from a Transformer teacher.
- Reports beating CNNs and being competitive with Transformers while being more efficient, with the efficiency advantage growing with input resolution.
- Motivation: Transformers' quadratic cost forces patch-based inference.
- Source: [ACCV PDF (search-extracted)](https://openaccess.thecvf.com/content/ACCV2024/papers/Gao_Mamba-based_Light_Field_Super-Resolution_with_Efficient_Subspace_Scanning_ACCV_2024_paper.pdf)

**L²FMamba** — "L²FMamba: Lightweight Light Field Image Super-Resolution With State Space Model", Z. Wei, K. Jin, Z. Hou, K. Song, X. Zhou, IEEE TCI vol. 11, pp. 816–826, 2025, DOI 10.1109/TCI.2025.3577338, [arXiv:2503.19253](https://arxiv.org/abs/2503.19253)
- The LF-VSSM block successively extracts three kinds of features: Intra-SSM (spatial, within SAIs), Inter-SSM (spatial-angular, between SAIs), and pixel-level spatial-angular (between LF pixels, i.e. MacPI). It can replace CNN or Transformer feature extractors in existing LFSR networks.
- Claims fewer params, lower complexity, faster inference and better PSNR.
- The ablation retains one layer type at a time; the numbers were not retrieved.
- Source: [arXiv (search-extracted)](https://arxiv.org/abs/2503.19253)

**LFTransMamba (CVPRW 2025): what the released code actually does** — [local models/LFTransMamba.py](file:///home/user/LFTransMamba/models/LFTransMamba.py)
- Each "composed block" is three sub-blocks in sequence, repeated K times: `[VSSBlock, VLSSBlock, LFSSBlock] * num_blocks` (lines ~2228–2232).
  - **VSSBlock (lines 2095–2197)**: T Transformer blocks (full multi-head self-attention, 4 heads, within each SAI's h×w tokens, flash-attention) followed by an SS2D Mamba scan on each SAI. This is the "spatial enhancement" Transformer on the intra-view subspace.
  - **VLSSBlock (lines 2007–2093)**: `SS2D4D`, the cross-scan in the SAI-array layout `(u h)(v w)` → sequences ordered `(u v)(h w)` and `(v u)(w h)` plus their reverses (CrossScan4D, lines 278–310). This is the inter-view/angular-major scan.
  - **LFSSBlock (lines 1918–2005)**: `SS2D3D` on the MacPI layout `(h u)(w v)` → sequences `(h w)(u v)` and `(w h)(v u)` plus reverses (CrossScan3D, lines 202–240; the Chinese comments say this corresponds to "spatial horizontal / LF horizontal" and "spatial vertical / LF vertical"). This is the pixel-level spatial-angular (macro-pixel) scan.
  - All three use a ConvFFN (`Conv_Mlp`) with learnable skip scales and DropPath up to 0.1.
- Other model details:
  - A learnable per-view angular embedding `ang_embed` of shape (1, C, U·V, 1, 1).
  - Hierarchical features from every composed block are concatenated, then a 3×3 conv, PixelShuffle ×4, and a global residual from per-SAI bicubic upsampling. The code comment says interpolation must be per SAI, not on the full SAI mosaic.
- **MLFIM is implemented as 35% random token masking** of shallow features with a learnable mask token during training (`mask_ratio=0.35`, lines 2378–2381). A secondary summary claimed 25%; the code says 0.35.
- Loss is plain L1 (`get_loss`, lines 2412–2420). There is no disparity, angular-consistency or EPI loss.
- Angular size 5 is hard-coded in the rearranges. `__SIZE__ = 32` is the training patch.

**LFTramba (CVPRW 2025) → LFMT (journal)** — "Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution", H. Liu, X. Zhu, H. Zeng, J. Zhu, J. Cao, J. Hou, [arXiv:2509.04824](https://arxiv.org/abs/2509.04824); Pattern Recognition (2026, volume/issue listing conflicts: vol. 180 Dec 2026 vs June 2026); code at hsliu01/LFMT (search-extracted)
- Argues pure-Mamba LF methods suffer from "coupling of 2D subspace scanning", with pixels redundantly modelled across subspaces. Proposes Subspace Simple Scanning (Sub-SS) and the Subspace Simple Mamba Block (SSMB).
- **Ablation: hybrid LFMT is +0.15 dB over pure Mamba and +0.12 dB over pure Transformer.** Three SA-RSMB plus three EPMB/EPTB blocks is optimal (rise then fall).
- **Rationale**: Mamba's limited state space misses some epipolar-plane structure, so Transformer blocks on EPIs are added in a later stage to compensate.
- Beats LF-DET by about 0.18 dB average PSNR at ×2.

**SMART** — "Slope-Guided Mamba and Angular-Refined Transformer for Light Field Super-Resolution", L. Jin, J. Huang, J. Lu, S. Wang, H. Sheng, J. Wu (Beihang), IEEE ICME 2026, [arXiv:2607.00965](https://arxiv.org/abs/2607.00965) (July 2026, search-extracted)
- A slope-guided Mamba whose adaptive scanning follows epipolar lines (EPI slope, i.e. disparity), an angular-refined Transformer, and an angular-modulated spatial module that injects angular priors.
- Claims +0.42 dB over prior methods on five benchmarks.
- **Ablations: angular prior +1.5–2.6 dB; slope-guided scanning about +0.5 dB average**, helping most in complex-depth regions. The size of the angular-prior gain suggests the baseline it is measured against lacked angular information entirely; verify in the paper.

**RASLF** — "RASLF: Representation-Aware State Space Model for Light Field Super-Resolution", Z. Wei, K. Jin, K. Song, X. Zhou, W. Chen, M. Xu (the OpenMeow/L²FMamba group), [arXiv:2603.16243](https://arxiv.org/abs/2603.16243) (2026, search-extracted)
- A Progressive Geometric Refinement block with a "Panoramic Epipolar Representation" for "explicit parallax-aware feature interaction".
- Representation-Aware Asymmetric Scanning adapts scan paths to each LF representation, "reducing redundancy while reinforcing geometric constraints".
- Dual-Anchor Aggregation filters redundant deep features.
- Motivation: existing SSM LFSR methods under-use the complementarity among LF representations.

**GTF (2026, Transformer, NTIRE 2026 participant)** — diagonal EPIs (45°/135°) in addition to horizontal and vertical. Argues prior Transformers (and LFTramba) confine EPI modelling to horizontal and vertical directions — [arXiv:2605.04581 (search-extracted)](https://arxiv.org/abs/2605.04581)

### Inferences
- **What each component is for**, by consensus across the papers:
  - Mamba scans along SAI-array or MacPI orderings give linear-cost global spatial-angular mixing. This is why Mamba dominates the 1M-param / 20 GFLOP regime: the 2025 OpenMeow Track 2 entry was pure Mamba.
  - Attention is placed where content-adaptive, non-causal matching matters: intra-SAI spatial detail (LFTransMamba's VSSBlock) or EPI/disparity lines (LFTramba/LFMT's EPTB, SMART's angular Transformer, EPIT).
  - The hybrid gain over either pure form is small (about 0.1–0.15 dB in LFMT) but consistent. In NTIRE terms that is about the size of a year-over-year championship margin.
- **The opposite design choices of LFTransMamba and LFTramba/LFMT both win**: LFTransMamba puts Transformers on spatial SAIs and Mamba on angular/MacPI; LFMT puts Mamba on spatial-angular and Transformers on EPIs. The benefit seems to come from mixing two different inductive biases plus subspace-aware tokenisation, not from one canonical split.
- The direction of research from 2025 to 2026 (SMART, RASLF, GTF, LFMT) is toward more explicit geometry in scanning or attention: slope-guided scans, epipolar panoramas, diagonal EPIs. That includes the OpenMeow group's own follow-up (RASLF). This runs against the idea that geometry can be dropped.

### Gaps
- The LFTransMamba paper's own ablation tables (Mamba-only vs +Transformer, MLFIM on/off, EPSW vs PSW) could not be retrieved. Only the abstract claims and the code are available. The user has the paper and the code and can read the tables directly.
- L²FMamba and MLFSR parameter, FLOP and PSNR tables were not retrieved.
- LFMamba's efficiency (FLOPs, runtime) numbers were not retrieved.

---

## Q4. Datasets and the train/test domain gap. Did physics- or geometry-aware designs help or hurt in the challenges?

### Takeaway
The documented gap is mostly about the **disparity distribution and synthetic renderer**, not the camera type:
- Half of every NTIRE val/test set is real Lytro ILLUM, the same camera family as the real training data (EPFL, INRIA).
- The other half is 3DS MAX-rendered synthetic content with co-planar virtual cameras.

Every top team addressed the gap with geometry-aware measures: disparity-diversifying angular resampling and disparity-preserving tiled inference. No evidence was found that geometry-aware design hurt. There is also no evidence that explicit disparity-estimation modules or angular-consistency losses were used by the winners. Winners trained with plain L1.

### Cited Findings
- **Training-set composition:** 144 scenes at 9×9, from EPFL, HCInew, HCIold, INRIA and STFgantry. HCInew and HCIold are synthetic; EPFL and INRIA are real Lytro Illum captures; STFgantry is a camera-array (gantry) dataset with high spatial resolution and a **large disparity range** — [NTIRE2026 README](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2026); [search-extracted dataset descriptions](https://github.com/ZhengyuLiang24/BasicLFSR)
- Reported disparity ranges (secondary comparison table, not verified against the original dataset papers): HCInew [−4, 4], HCIold [−3, 3], EPFL and INRIA [−1, 1]. Train/test splits in the same table: EPFL 70/10, INRIA 35/5, HCInew 20/4, HCIold 10/2 — [search-extracted, source table not pinned down](https://github.com/ZhengyuLiang24/BasicLFSR)
- **Val/test composition:** 16 synthetic scenes from 3DS MAX (co-planar virtual cameras with identical intrinsics) plus 16 real Lytro ILLUM scenes. Synthetic 500×500, real 624×432 — [BasicLFSR README](https://github.com/ZhengyuLiang24/BasicLFSR); [NTIRE2026 README](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2026)
- **Synthetic vs real difficulty:**
  - On the 2024–2026 validation set, the synthetic half is about 1 dB harder than the real half for both baselines: DistgSSR 31.23 vs 32.26, EPIT 31.53 vs 32.54 — [NTIRE2026 README](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2026)
  - On the NTIRE-2023 test, the halves are about equal (LF-DET Lytro 29.911 vs Syn 29.976) — [BasicLFSR README](https://github.com/ZhengyuLiang24/BasicLFSR)
- **Organisers' framing:** the main challenge is exploiting angular information "across many views with differing disparities" — [NTIRE 2024 report abstract (search-extracted)](https://research.fi/en/results/publication/0690534724)
- **Disparity imbalance is acknowledged.** BigEPIT states existing LF datasets have imbalanced disparity with scarce large-disparity data. The NTIRE 2024 report says augmented sampling (central, even, uneven) explicitly increases large-disparity examples and improves robustness to disparity changes — [BigEPIT / NTIRE 2024 report (search-extracted)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chao_BigEPIT_Scaling_EPIT_for_Light_Field_Image_Super-Resolution_CVPRW_2024_paper.pdf)
- **Winners used this geometry-based augmentation in every edition:**
  - 2023 DistgEPIT used central, interval and uneven selection — [README](https://github.com/OpenMeow/NTIRE23_LFSR_DistgEPIT)
  - 2024 BigEPIT used augmented resampling — [paper (search-extracted)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chao_BigEPIT_Scaling_EPIT_for_Light_Field_Image_Super-Resolution_CVPRW_2024_paper.pdf)
  - 2025 LFTransMamba's training protocol "LFSR.ALL4" uses four view-selection patterns per scene (default central, 13579 = every other view, i.e. 2× baseline and 2× disparity, 24568, 12589), plus 34567 for the external DLFD/SLFD in Track 3 — [local lfsr.py](file:///home/user/LFTransMamba/lfsr.py)
- **Disparity-preserving inference helped:**
  - PSW gave +0.2 dB over standard padding and windowing — [DistgEPIT README](https://github.com/OpenMeow/NTIRE23_LFSR_DistgEPIT)
  - BigEPIT found zero padding and window partitioning "destroy the LF disparity structure" and recommends full-size patches — [BigEPIT (search-extracted)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chao_BigEPIT_Scaling_EPIT_for_Light_Field_Image_Super-Resolution_CVPRW_2024_paper.pdf)
  - EPSW gave +0.23 dB in GTF's ablation — [arXiv:2605.04581 (search-extracted)](https://arxiv.org/abs/2605.04581)
- **Disparity-robust architectures:** EPIT is explicitly designed and evaluated for robustness to disparity shifts via shearing — [EPIT (search-extracted)](https://arxiv.org/abs/2302.08058). It became the Track 3 baseline and the backbone of the 2024 winner.
- Track 3 (2025+) allows external data. The 2025 winner added DLFD and SLFD, but the Track 3 local-validation gain over Track 1 was only about +0.05 dB (33.9310 vs 33.8803) — [local README](file:///home/user/LFTransMamba/README.md)

### Inferences
- **The premise "test-set camera parameters differ a lot from training" is only half right:**
  - Real test scenes come from the same Lytro ILLUM camera family as EPFL and INRIA, so their small-baseline disparity range should resemble the training data.
  - The synthetic 3DS MAX scenes come from a different renderer and camera rig, with an undocumented disparity range. That half is where baselines lose about 1 dB on validation, and it is the plausible source of gap.
- **The documented remedy is more geometry exposure, not less.** Resampling views to synthesise wider baselines teaches the network a broader disparity distribution, and disparity-consistent tiling avoids breaking epipolar structure at test time. Both are physics-motivated, even though the loss remains pure L1 regression.
- What did **not** appear in any winning solution examined: explicit disparity estimation or warping modules, angular-consistency or EPI-gradient losses, or physics-based priors on the camera model. Whether these would help or hurt in NTIRE was not tested in any source found. The "ignore explicit physics" part of the premise is consistent with what winners did, but it is not demonstrated to be better.
- **Small Track 3 gain from external data** (about +0.05 dB local, 31.16 → 31.22 test per secondary summary) suggests the remaining gap is not mainly closed by more data of other camera types. Architecture capacity, ensembles and disparity augmentation mattered more.

### Gaps
- No source quantified the disparity range of the NTIRE synthetic (3DS MAX) or real test scenes.
- No ablation was found that isolates the PSNR gain of angular-resampling augmentation on the NTIRE test. The 2024 report asserts robustness gains; DistgEPIT and BigEPIT ablation numbers for it were not retrieved.
- The disparity-range table (HCI ±3/±4, EPFL/INRIA ±1) comes from a secondary source and should be checked against the original dataset papers.
- No head-to-head study of explicit geometry losses or modules vs pure-regression models under the NTIRE protocol was found.

---

## Q5. Do EPI and spatial-angular disentangling — themselves structural or physical priors — remain important inside winning "pure regression" models?

### Takeaway
Yes. Every NTIRE winner, and every recent Mamba or hybrid paper, hard-codes LF geometry into tokenisation and scan order (SAI / MacPI / EPI). Ablations that remove EPI or angular structure lose about 0.2–0.35 dB (LFMamba) or more (SMART's angular prior). The "pure regression" in winning models is pure only in the loss. The architecture is a strong LF-structural prior.

### Cited Findings
- LFTransMamba's three sub-blocks each perform a geometry-specific reshaping before scanning or attention: intra-SAI spatial (Transformer + SS2D), SAI-array angular-major (SS2D4D), MacPI pixel-level spatial-angular (SS2D3D). It also has per-view angular embeddings and per-SAI bicubic residuals. The loss is only L1 — [local LFTransMamba.py](file:///home/user/LFTransMamba/models/LFTransMamba.py)
- L²FMamba's LF-VSSM is explicitly a three-stage disentangling (Intra-SAI, Inter-SAI, pixel-level MacPI) — [arXiv:2503.19253 (search-extracted)](https://arxiv.org/abs/2503.19253)
- LFMamba: removing the EPI-H/EPI-V SSM blocks costs −0.27 dB on EPFL and −0.35 dB on INRIA (but +0.16 dB on STFgantry). The authors call LF structure learning "necessary" — [arXiv:2406.12463 (search-extracted)](https://arxiv.org/abs/2406.12463)
- LFTramba/LFMT adds EPI-domain Transformer blocks specifically for disparity features. The hybrid with EPI attention beats pure Mamba by +0.15 dB — [arXiv:2509.04824 (search-extracted)](https://arxiv.org/abs/2509.04824)
- SMART: angular prior +1.5–2.6 dB; epipolar-slope-guided scanning +0.5 dB — [arXiv:2607.00965 (search-extracted)](https://arxiv.org/abs/2607.00965)
- DistgEPIT (2023 winner) is built on DistgSSR's disentangling plus EPIT's EPI Transformer, and is titled "Enhanced *Disparity* Learning" — [README](https://github.com/OpenMeow/NTIRE23_LFSR_DistgEPIT)
- BigEPIT (2024 winner) is a scaled EPI Transformer — [BigEPIT (search-extracted)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chao_BigEPIT_Scaling_EPIT_for_Light_Field_Image_Super-Resolution_CVPRW_2024_paper.pdf)
- SkimLFSR: disparity-range-specific attention branches, with depth and camera configuration learned implicitly from regression — [arXiv:2407.15329 (search-extracted)](https://arxiv.org/abs/2407.15329)
- GTF (2026 3rd place) adds diagonal EPIs and MacPI prior injection — [arXiv:2605.04581 (search-extracted)](https://arxiv.org/abs/2605.04581)

### Inferences
- The evidence supports a refined version of the user's view: **explicit physics (disparity estimation, warping, consistency losses) has not been needed to win, but implicit geometry is essential** — subspace tokenisation, EPI attention or scans, disparity-diverse training data, and disparity-preserving inference. Dropping these structural priors in favour of a generic 2D SR backbone is contradicted by:
  - LFMamba's statement that naive SS2D is sub-optimal;
  - the w/o-EPI ablations;
  - the fact that every winner from 2023 to 2026 is LF-structured.
- LFMamba's +0.16 dB on STFgantry when the EPI blocks were removed is one counter-signal. It hints that a fixed EPI design can be mismatched to large-baseline data. This fits the motivation for slope- or disparity-adaptive designs (SMART, SkimLFSR) and disparity augmentation, rather than for removing geometry altogether.

### Gaps
- No ablation was found that replaces LFTransMamba's MacPI and SAI-array scans with generic 2D scans on the full SAI mosaic, which would directly test the "pure regression" hypothesis on the NTIRE test. The user could run it: replace VLSSBlock/LFSSBlock with VSSBlock on the SAI mosaic, keep params and FLOPs matched, and compare on the NTIRE val synthetic and real halves separately.
