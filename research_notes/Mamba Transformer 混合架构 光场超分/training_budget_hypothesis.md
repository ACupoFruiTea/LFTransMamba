# Does the hybrid win because Mamba buys more training iterations? The training budget hypothesis for hybrid Mamba-Transformer LFSR models

Hypothesis under test (H, translated from the user): "SR is pure regression; regression needs enough iterations; Transformer-only models cost too much compute per unit time to iterate enough within the competition's time; Mamba's linear complexity allows many more iterations per unit time, and adding attention keeps attention's benefits so quality does not drop."
I split H into links: **H1** SR here is pure regression · **H2** quality is limited by the number of iterations · **H3** Transformer-only models are too slow per iteration at LFSR sizes · **H4** Mamba gives many more iterations per unit time · **H5** adding attention keeps quality · **H0** (implicit) the competition imposes a training-time budget.

Access note: arxiv.org and openaccess.thecvf.com were blocked by this sandbox's proxy (HTTP 403 on CONNECT). Paper-level numbers below therefore come from (a) search-engine extracts of those papers, flagged "(search extract)", (b) official GitHub training configs fetched directly from raw.githubusercontent.com, and (c) the local LFTransMamba repository (`/home/user/LFTransMamba`, public at github.com/OpenMeow/LFTransMamba). Everything under "Inferences" is my own reasoning or arithmetic, not a cited result.

---

## Q1. Does longer training or more iterations materially improve SR, and how long do SOTA SR and LFSR models train? (tests H1, H2)

### Takeaway
Partly yes. Single-image SR is documented to be **underfitting-limited**: longer training plus a better recipe gives roughly +0.1 to +0.44 dB. That is the same order as the margins between top NTIRE LFSR teams, so iteration count is not negligible. However, SOTA SR papers (Transformer and Mamba alike) train under almost identical, standardized iteration budgets (250k to 800k iterations at batch about 32). Their architecture comparisons are therefore made at **equal iterations**, so the field's Mamba-vs-Transformer gains cannot be explained by extra iterations. Under a fixed budget, a learning-rate schedule matched to that budget matters about as much as the raw count.

### Cited Findings
- **H1, pure regression: confirmed for this codebase.** LFTransMamba's training loss is a plain `torch.nn.L1Loss()` between the SR output and HR, with no perceptual, adversarial or auxiliary loss — [local: models/LFTransMamba.py `get_loss`, L2412–2419](/home/user/LFTransMamba/models/LFTransMamba.py). NTIRE LFSR tracks are ranked by PSNR, bicubic ×4 — [NTIRE 2024 LFSR report, Wang et al., CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf) (search extract); ranking by average test PSNR is stated in the 2026 edition — [NTIRE 2026 LFSR supplemental](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/supplemental/Wang_NTIRE_2026_Challenge_CVPRW_2026_supplemental.pdf) (search extract).
- **Underfitting dominates in SR.** *Revisiting RCAN: Improved Training for Image Super-Resolution* (Zudi Lin et al., arXiv 2201.11279, 2022): "underfitting is still the main problem restricting the model capability instead of overfitting… increasing training iterations clearly improves the model performance while applying regularization techniques generally degrades the predictions." With a proper recipe, RCAN (2018) "can outperform or match nearly all the CNN-based SR architectures published after RCAN" — [rcan-it README](https://github.com/zudi-lin/rcan-it) / [arXiv 2201.11279](https://arxiv.org/abs/2201.11279).
- RCAN-it reaches 39.88 dB on Manga109 ×2, **+0.44 dB over the original RCAN**. In the ablations, SiLU combined with longer training adds about **+0.1 dB** on average over the updated baseline — [arXiv 2201.11279](https://arxiv.org/abs/2201.11279) (search extract via [ResearchGate](https://www.researchgate.net/publication/358163338_Revisiting_RCAN_Improved_Training_for_Image_Super-Resolution)). I could not confirm a direct RCAN-it vs SwinIR table.
- RCAN-it's improved config: LAMB optimizer, base LR 3.2e-3, 64 samples per batch, `ITERATION_TOTAL: 160000 # longer training`, `ACT_MODE: silu` — [rcan-it configs/RCAN/RCAN_Improved.yaml](https://github.com/zudi-lin/rcan-it/blob/main/configs/RCAN/RCAN_Improved.yaml).
- **SOTA SISR schedules, from official configs:**
  - HAT ×4: ImageNet pretraining for `total_iter: 800000` (batch 4/GPU, gt 256, LR 2e-4, milestones 300k/500k/650k/700k/750k), then DF2K fine-tuning for `total_iter: 250000` at LR 1e-5, so **more than 1M iterations total** — [HAT train_HAT_SRx4_ImageNet_from_scratch.yml](https://github.com/XPixelGroup/HAT/blob/main/options/train/train_HAT_SRx4_ImageNet_from_scratch.yml), [HAT finetune yml](https://github.com/XPixelGroup/HAT/blob/main/options/train/train_HAT_SRx4_finetune_from_ImageNet_pretrain.yml).
  - DAT ×4: 500k iterations from scratch (commented default), or **250k** fine-tuning from the ×2 model at LR 1e-4 ("save half of training time if we finetune from x2"), batch 8/GPU, gt 256 — [DAT options/Train/train_DAT_x4.yml](https://github.com/zhengchen1999/DAT/blob/main/options/Train/train_DAT_x4.yml).
  - MambaIRv2 ×4: 8 GPUs × batch 4, gt 192, 500k from scratch (commented), or **250k** from ×2 at LR 1e-4, with **exactly the same milestones as DAT** [125k, 200k, 225k, 237.5k] — [MambaIR repo, train_MambaIRv2_SR_x4.yml](https://github.com/csguoh/MambaIR/blob/main/options/train/mambairv2/train_MambaIRv2_SR_x4.yml).
- **LFSR schedules.** BasicLFSR defaults: batch 4, LR 2e-4, step decay every 15 epochs, `--epoch 51` — [BasicLFSR option.py](https://github.com/ZhengyuLiang24/BasicLFSR/blob/main/option.py). LFTransMamba's official training commands:
  - Track 1 hybrid (`LFTransMamba_D16_R1_K18_C80_T2`): batch **3**, **30 epochs**, LR 3e-4, EMA 0.999, LR patch 32×32 (the HR crop is 128×128).
  - Track 1 attention-only model (`LF_DET_0214`): batch 3, **30 epochs**, LR 2e-4.
  - Track 2 (`..._K6_C32_T0`): **100 epochs**.
  - Track 3: 30 epochs on the `LFSR.ALL4.EXTRA` dataset with `--fp16`.
  — [local README §4 Training](/home/user/LFTransMamba/README.md); crop logic `patch_size*scale` in [local lfsr.py L235–262](/home/user/LFTransMamba/lfsr.py).
- **Budgeted training.** *Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints* (Mengtian Li, Ersin Yumer, Deva Ramanan, ICLR 2020, arXiv 1905.04753) finds that when the iteration budget is fixed, "it is critical to adjust the learning rate schedule according to the given budget." A simple **linear decay to zero** is robust and high-performing across ImageNet, Kinetics, COCO and Cityscapes ("budgeted convergence"). It beats adaptive-LR and restart methods in that setting — [arXiv 1905.04753](https://arxiv.org/abs/1905.04753), [ICLR 2020 poster](https://iclr.cc/virtual/2020/poster/1640).
- **Scaling-law evidence for SR is thin.** *Scaling Laws For Deep Learning Based Image Reconstruction* (arXiv 2209.13435, 2022) notes in its SR appendix that gains from very large training sets "do not seem to be as impressive as in high-level vision problems" — [arXiv 2209.13435](https://arxiv.org/abs/2209.13435) (search extract). In the NTIRE 2024 SR ×4 challenge, one team scaled training data to **25M images** on top of HAT — [arXiv 2404.09790](https://arxiv.org/abs/2404.09790) (search extract). The 2026 SST paper scales SR-Transformer training patches to 96×96 and data to **614,944 images** once FlashAttention is usable — [Lee et al., arXiv 2603.06738](https://arxiv.org/abs/2603.06738) (search extract).

### Inferences
- H1 holds for LFTransMamba: L1 regression, PSNR-ranked.
- H2 is **qualitatively supported**: SR is underfitting-limited, and +0.1 to +0.4 dB from recipe and length is the same scale as the NTIRE 2025 LFSR Track 1 winning margin (+0.36 dB over the previous year's champion, Q4).
- H2 also implies a testable prediction that **existing comparisons do not show**. SR papers compare architectures at matched schedules: MambaIRv2 and DAT have the same 250k/500k schedule, and LFTransMamba and LF_DET_0214 both use 30 epochs at batch 3. If H were the mechanism, the hybrid's advantage should **vanish at equal iterations** and appear only at equal wall-clock time. It does not vanish (Q3).
- LFTransMamba's 30-epoch schedule at batch 3 is short in absolute iterations. Its iteration count per epoch equals the number of training files ÷ 3 (`__len__ = len(self.targets)`, [local lfsr.py](/home/user/LFTransMamba/lfsr.py)). So the model is plausibly still underfit, and **longer training is a cheap experiment**, whatever the architecture.

### Gaps
- No 2024–2026 paper fitting a compute-optimal (model size × iterations) scaling law **specifically for SR or LFSR** was found.
- The exact number of training files and patches in `LFSR.ALL4` (hence the iterations per epoch) is not documented in the README.
- No public LFSR paper reports PSNR as a function of training length.

---

## Q2. What are the actual throughput and memory costs of Mamba vs attention at SR and LFSR token counts? (tests H3, H4)

### Takeaway
Linear-vs-quadratic only matters once sequences pass about **2K tokens** (the Mamba vs FlashAttention-2 kernel crossover) or **L > 6D** (the MambaOut FLOPs criterion).
- **SISR window attention** (64 to 256 tokens) is far below that threshold and is already linear in image size.
- **Per-view full attention** in LFTransMamba (1,024 tokens per view at training) is at or below the crossover. Mamba does **not** buy speed there, and 4-direction scans add cost.
- **Whole-LF 4D sequences** (25 views × 1,024 = 25,600 tokens at training) are far above the crossover. There Mamba is the only affordable global mixer.

So Mamba's linearity buys **reach** (global spatial-angular context per iteration), not more iterations for the same model. In addition:
- Vision Mamba's multi-direction scans are memory-bound and needed heavy kernel engineering to be competitive.
- The main reason SR Transformers were slow (relative-position bias blocking FlashAttention) was fixed in 2026 without Mamba.

### Cited Findings
- **Kernel crossover.**
  - *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (Albert Gu, Tri Dao, arXiv 2312.00752, 2023): the fused selective scan (N=16) is faster than FlashAttention-2 (causal) **beyond sequence length 2K**, and 20–40× faster than a naive PyTorch scan — [arXiv 2312.00752](https://arxiv.org/abs/2312.00752) (search extract; Fig. 8 / App. E.5).
  - *Transformers are SSMs* (Mamba-2/SSD; Tri Dao, Albert Gu, ICML 2024, arXiv 2405.21060): SSD is faster than FlashAttention-2 **from 2K onward** and 2–8× faster than Mamba-1's fused scan, because it uses matmul units — [arXiv 2405.21060](https://arxiv.org/abs/2405.21060) (search extract).
  - *Log-Linear Attention* (arXiv 2506.04761, 2025): a log-linear Mamba-2 kernel beats FlashAttention-2 (forward+backward) only **beyond 8K**. The crossover is implementation-dependent — [arXiv 2506.04761](https://arxiv.org/abs/2506.04761) (search extract).
- **FLOPs criterion.** *MambaOut: Do We Really Need Mamba for Vision?* (Weihao Yu, Xinchao Wang, CVPR 2025, arXiv 2405.07992):
  - A Transformer block (MLP ratio 4) costs 24D²L + 4DL² FLOPs. The quadratic/linear ratio is **L/(6D)**, so a task is "long-sequence" when **L > 6D**: τ = 2304 tokens for ViT-S (D=384) and 4608 for ViT-B (D=768).
  - ImageNet at 224² (196 tokens) is not long-sequence. COCO/ADE20K detection and segmentation are.
  - MambaOut (with the SSM removed) beats visual Mamba models on ImageNet but **falls short** of them on detection/segmentation.
  — [arXiv 2405.07992](https://arxiv.org/abs/2405.07992), [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.pdf) (search extract). A GitHub issue notes that with the 2DL² attention-FLOP convention the threshold doubles to 12D — [MambaOut issue #205](https://github.com/yuweihao/MambaOut/issues/205).
- **Vision-scan overheads.** *VMamba: Visual State Space Model* (Yue Liu et al., NeurIPS 2024, arXiv 2401.10166): the authors state that "low throughput and high memory overhead present significant challenges" for VMamba. The gains came from engineering:
  - Triton cross-scan/merge: +41 img/s.
  - fp16-in/fp32-out selective-scan kernel: training throughput 165 → 184.
  - Replacing an einsum with a linear layer: +174 img/s.
  After these fixes, VMamba-B reaches 646 vs Swin-B 458 img/s, and VMamba-T 1,686 vs Swin-T 1,244 img/s — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/baa2da9ae4bfed26520bb61d259a3653-Paper-Conference.pdf), [arXiv 2401.10166](https://arxiv.org/abs/2401.10166) (search extract).
- *Demystify Mamba in Vision: A Linear Attention Perspective* (Dongchen Han et al., NeurIPS 2024, arXiv 2405.16605): adding Mamba's forget gate to linear attention raises ImageNet top-1 from 77.6 to 78.4, but **throughput drops from 1152 to 743**, because the gate requires recurrent computation. The authors show it can be replaced by suitable positional encoding (MLLA) — [arXiv 2405.16605](https://arxiv.org/abs/2405.16605), [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e618724ac897c6cf3fbfb273f8695d67-Abstract.html) (search extract).
- MambaIR's own complexity framing: it "exhibits linear complexity with input resolution which is similar to the efficient attention techniques such as SwinIR". It claims **parity** with window attention, not superiority — [Hang Guo et al., MambaIR, ECCV 2024, arXiv 2402.15648](https://arxiv.org/abs/2402.15648) (search extract).
- **Why SR Transformers were slow.** *Rank-factorized Implicit Neural Bias: Scaling Super-Resolution Transformer with FlashAttention* (Dongheon Lee et al., arXiv 2603.06738, 2026): relative positional bias (RPB) in SR Transformers "prevents them from leveraging hardware-efficient attention kernels such as FlashAttention". Replacing RPB with RIB:
  - SST trains **2.1× faster with 24.6% lower memory** than a prior method at 64×64 patches.
  - It reports 2.1× training / 2.9× inference speed-ups vs the RPB-based PFT, and 35.63 dB on Urban100 ×2.
  — [arXiv 2603.06738](https://arxiv.org/abs/2603.06738) (search extract; preprint, sources differ slightly on the inference speed-up).
- **Runtime evidence for Mamba vs Transformer SR is inconsistent.**
  - One 2025 table lists full-benchmark inference at about 50 min for MambaIRv2, 58 min for SwinIR and 57 min for HAT (hardware and workload unclear) — [arXiv 2506.14541](https://arxiv.org/abs/2506.14541) (search extract).
  - Q-MambaIR reports full-precision MambaIR at 198.21 ms on Jetson Orin, with no SwinIR/HAT row — [arXiv 2503.21970](https://arxiv.org/abs/2503.21970) (search extract).
  - I found **no clean apples-to-apples GPU training-throughput comparison** of MambaIR/MambaIRv2 vs SwinIR/HAT.
- **Where LFTransMamba spends attention vs Mamba (local code):**
  - `TransBlock` performs **full (non-windowed) multi-head self-attention within each sub-aperture view**: tokens are `(b u v) h w c → b (h w) c`, `use_flashatten=True`, 2 blocks (`T=2`) per `VSSBlock`.
  - The layers are `[VSSBlock, VLSSBlock, LFSSBlock] × K`:
    - `VSSBlock`: per-view attention plus a 2D SS2D scan.
    - `VLSSBlock`: SS2D4D over the SAI-array layout `b (u h) (v w)`.
    - `LFSSBlock`: SS2D3D over the macro-pixel layout `b (h u) (w v)`.
  - Both 4D/3D scans run over sequences of length 25·h·w.
  - Track 1 uses K=18, C=80, d_state=16, T=2, which gives 54 layers including 36 attention blocks.
  — [local models/LFTransMamba.py L62–151, L1918–2192, L2200–2240](/home/user/LFTransMamba/models/LFTransMamba.py); the name parsing `D, R, K, C, T` is in [local lfsr.py L740](/home/user/LFTransMamba/lfsr.py).
- The efficiency-track Track 2 model is `LFTransMamba_D16_R1_K6_C32_T0`, which has **zero Transformer blocks** — [local README](/home/user/LFTransMamba/README.md). The NTIRE LFSR efficiency track caps models at **<1M params and <20 GFLOPs measured on a 5×5×32×32 input** (stated in the 2024 and 2026 reports) — [NTIRE 2026 LFSR report](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Wang_NTIRE_2026_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2026_paper.pdf), [NTIRE 2024 LFSR report](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf) (search extract).

### Inferences
- **Arithmetic** with MambaOut's formula, D = C = 80, forward pass, per sequence per attention layer:

| Sequence (LFTransMamba-relevant) | L | L/(6D) | quadratic term 4DL² | linear term 24D²L |
|---|---|---|---|---|
| SwinIR 8×8 window | 64 | 0.13 | 0.001 GF | 0.010 GF |
| HAT 16×16 window | 256 | 0.53 | 0.021 GF | 0.039 GF |
| LFTransMamba per-view, train (32×32) | 1,024 | 2.1 | 0.34 GF (×25 views = 8.4 GF per LF) | 0.16 GF |
| per-view, inference patch 48×48 | 2,304 | 4.8 | 1.70 GF | 0.35 GF |
| SISR 64×64 global | 4,096 | 8.5 | 5.37 GF | 0.63 GF |
| **Hypothetical global 4D attention, train (5×5×32×32)** | 25,600 | **53** | **210 GF** (25× the per-view total) | 3.9 GF |
| Hypothetical global 4D attention, inference (5×5×48×48) | 57,600 | 120 | 1,062 GF | 8.8 GF |

  - Without FlashAttention, a fp16 attention matrix for L=25,600 with 4 heads is about 5.2 GB per sample per layer.
  - Implication: **H3 is false for windowed and per-view attention but true for global 4D attention.** Window attention is linear-dominated. Per-view 1,024-token attention with FlashAttention sits below the about 2K Mamba/FlashAttention-2 crossover, so a Mamba scan of the same view is not faster (and SS2D runs 4 directional scans).
  - Only the cross-view 4D mixing is where "Transformer-only is too expensive" holds. That is a statement about **which context the model can afford to see per step**, not about iterations per second for an equally capable model.
- **H4 (Mamba allows many more iterations) is weakly supported at best.** No source reports that hybrid LFSR models train faster in wall-clock than attention baselines. Vision-scan kernels are memory-bound (VMamba, Demystify). Locally, the hybrid and the attention-only `LF_DET_0214` were trained for the **same 30 epochs**, so the team did not convert any speed advantage into extra iterations.
- **In the efficiency track, the binding constraint is the inference FLOPs cap.** With C=32, one per-view full-attention layer costs about 3.36 GF (quadratic) + 0.63 GF (linear) ≈ 4 GF per LF. That is about 20% of the 20 GF cap per layer, which explains the T=0 (pure Mamba) Track 2 design. This is a genuine "linear complexity wins" mechanism, but it operates through an **inference FLOPs rule**, not training iterations.

### Gaps
- There is no measured iterations/sec or GPU-memory comparison of LF_DET_0214 vs LFTransMamba (or any LFSR Mamba vs Transformer pair). This is a 10-minute local benchmark the user can run; it was not run here because there is no GPU and the CUDA selective-scan extension is not built.
- Kernel crossover numbers come from LLM-style causal 1D benchmarks on A100/H100. No published crossover exists for **bidirectional multi-direction 2D/4D vision scans** vs FlashAttention-2/3 at 1K–60K tokens.
- I could not fetch MambaIR/MambaIRv2's own runtime tables (arXiv blocked).

---

## Q3. Is Mamba's advantage better explained by inductive bias, receptive field or global context than by training speed? Are there equal-iteration comparisons? (tests H5 and the mechanism of H)

### Takeaway
The available comparisons show **hybrid/Mamba gains at matched training schedules**, which is the opposite of what an "it is just more iterations" mechanism predicts. This holds for MambaIR vs SwinIR, for LFMT's hybrid vs pure-Mamba vs pure-Transformer ablation, and for the local LFTransMamba vs LF_DET pair. The better-supported explanations are:
1. affordable global or long-range context (ERF);
2. complementary biases: content-based attention within a view, plus ordered, decaying, forget-gated propagation across the 4D LF;
3. on short sequences, Mamba's SSM brings little (MambaOut) and can be emulated by linear attention plus positional priors (Demystify).

### Cited Findings
- **MambaIR vs SwinIR (similar budgets).** MambaIR "outperforms Transformer-based baseline SwinIR by up to **0.36 dB**, using similar computational cost but with global receptive field" — [MambaIR README](https://github.com/csguoh/MambaIR). MambaIR-light beats SwinIR-light by up to 0.34 dB on Manga109 ×4 with similar params and MACs, and the paper includes ERF visualizations — [arXiv 2402.15648](https://arxiv.org/abs/2402.15648) (search extract). A "+0.45 dB" figure circulates only on aggregator sites, so treat it as unverified — [EmergentMind topic page](https://www.emergentmind.com/topics/mambair).
- MambaIRv2 and DAT share an identical schedule (250k iterations and milestones, see Q1), so published SR comparisons are made at equal iterations — [MambaIRv2 config](https://github.com/csguoh/MambaIR/blob/main/options/train/mambairv2/train_MambaIRv2_SR_x4.yml), [DAT config](https://github.com/zhengchen1999/DAT/blob/main/options/Train/train_DAT_x4.yml).
- **LFSR-specific hybrid ablation.** *Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution* (LFMT; Haosong Liu, Xiancheng Zhu, Huanqiang Zeng, Jianqing Zhu, Jiuwen Cao, Junhui Hou; arXiv 2509.04824, 2025): the hybrid gains about **+0.15 dB over a pure-Mamba** variant and about **+0.12 dB over a pure-Transformer** variant (Table VI). The authors attribute this to combining state-space and self-attention modeling, which enlarges the receptive field. No training time is reported — [arXiv 2509.04824](https://arxiv.org/abs/2509.04824) (search extract; absolute PSNRs not retrieved).
- **Local equal-epoch pair.** `LF_DET_0214` is a pure-attention model (`Attention`/`Block` classes, no SSM — [local models/LF_DET_0214.py](/home/user/LFTransMamba/models/LF_DET_0214.py)). It was trained 30 epochs at batch 3 on LFSR.ALL4. The hybrid was trained 30 epochs at batch 3 on the same data. Public-validation results with TTA (and different inference patch sizes, 16 vs 40):

| | mean PSNR | real PSNR | synthetic PSNR |
|---|---|---|---|
| LF_DET_0214 (attention-only) | 33.7795 | 32.6072 | 35.5381 |
| LFTransMamba (hybrid) | 33.8803 (**+0.10**) | 32.7436 (**+0.14**) | 35.5854 (**+0.05**) |

  — [local README §3](/home/user/LFTransMamba/README.md). Caveats: the LRs differ (2e-4 vs 3e-4), parameter counts are not listed, and the inference patch settings differ.
- **Short vs long sequences.** MambaOut shows that removing the SSM *helps* on short-sequence ImageNet but *hurts* on long-sequence detection/segmentation — [arXiv 2405.07992](https://arxiv.org/abs/2405.07992) (search extract).
- **What actually drives Mamba's gains.** Demystify Mamba: the forget gate and the block design are "the two core contributors to Mamba's superiority". The forget gate's benefit can be obtained with positional encoding in non-autoregressive vision. This means the gain is an **inductive bias (decay/position prior)**, not compute efficiency — [arXiv 2405.16605](https://arxiv.org/abs/2405.16605) (search extract).
- **Hybrid backbones in vision.** *MambaVision: A Hybrid Mamba-Transformer Vision Backbone* (Ali Hatamizadeh, Jan Kautz, CVPR 2025) places self-attention in the final stages. It argues VMamba's receptive field stays constrained by its cross-scan paths, and reports 84.2% top-1 (vs ConvNeXt-B 83.8%, Swin-B 83.5%) with better throughput — [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf) (search extract).
- **LFTransMamba's own stated rationale.** It "integrates Transformer-based global context modeling with Mamba-based efficient long-range dependency capture". It adds Masked Light Field Image Modeling (MLFIM) "without introducing additional modules or loss functions", plus Enhanced Position-Sensitive Windowing (EPSW), a Gaussian-weighted aggregation of overlapping inference patches — [Kai Jin et al., CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/html/Jin_LFTransMamba_A_Hybrid_Mamba-Transformer_Model_for_Light_Field_Image_Super-Resolution_CVPRW_2025_paper.html) (search extract). The paper's stated motivation is modeling capability and efficiency, not training-iteration count. Its full ablation tables were not retrieved.

### Inferences
- **H5 is supported, but with the wrong mechanism.** Hybrids do not lose quality vs pure attention; at equal iterations they **gain** (+0.10 to +0.15 dB in the LFSR evidence above). Because the gain appears at matched iterations, "more iterations per unit time" is **not needed** to explain it.
- A mechanism consistent with the code: attention handles **content-adaptive dense matching within each view** (self-similarity, texture), at 1,024 tokens where FlashAttention is cheap. The two SSM scan orderings — **macro-pixel order** (angular-first, SS2D3D) and **SAI-array order** (spatial-first, SS2D4D) — propagate information across all 25,600 spatial-angular tokens at linear cost.
  - This is a learned, input-selective analogue of the EPI / angular "disentangling" operators that dominate LFSR design.
  - The selective decay (forget gate) provides a distance prior along each ordering.
  - Neither component alone gives both. That matches the LFMT result that the hybrid beats both pure variants.
- The cleanest direct test of H, at the user's scale, is to train LF_DET_0214 (or a T-only LFTransMamba) for **equal wall-clock** and for 2–3× more epochs (e.g. 60 to 90), using a budget-aware linear-to-zero schedule.
  - If attention-only catches up with the hybrid given more iterations, H gains support.
  - If it plateaus below, the gap is architectural.
  - Ablate T ∈ {0,1,2} and drop the 3D/4D scans to see which component carries the gain.

### Gaps
- No LFSR paper reports PSNR vs **wall-clock time** curves for Mamba vs Transformer vs hybrid. Without that, H's specific mechanism cannot be confirmed or definitively ruled out at equal wall-clock.
- LFMT's absolute PSNRs, parameter counts and training settings for the three variants were not retrieved.
- LFTransMamba's paper ablations (MLFIM ratio, T, scan types) were not retrieved (CVF blocked).

---

## Q4. Do NTIRE-style challenges impose training-time budgets, or only inference constraints? (tests H0)

### Takeaway
No source shows a **training-time or iteration limit** in NTIRE LFSR rules. The rules constrain:
- inference-side model size and FLOPs (efficiency track: <1M params, <20 GFLOPs);
- the allowed training data (external data only in the Large-Model track);
- ranking, which is by test PSNR.

The "unit time" in the hypothesis is therefore the **development window × the team's GPUs**. That is a real resource constraint, but one the team allocates freely: the winning team spent part of it on ensembles and TTA rather than longer schedules.

### Cited Findings
- **NTIRE 2025 LFSR** (Yingqian Wang et al., CVPRW 2025): three tracks (classic, efficiency, large model); 308 registrants, 13 teams beat the baselines. The top Track 1 entry improves **+0.36 dB** over the previous year's champion on the test set — [MLAnthology entry](https://mlanthology.org/cvprw/2025/wang2025cvprw-ntire), [CVPRW 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Wang_NTIRE_2025_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2025_paper.pdf) (search extract).
- The 2025 Large-Model track "permits the use of external training data while removing the restrictions on model size and computational complexity" — [NTIRE 2025 LFSR report](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Wang_NTIRE_2025_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2025_paper.pdf) (search extract).
- **Efficiency-track caps:** "solutions with model size larger than 1M or FLOPs larger than 20G will not be ranked", with FLOPs measured on a 5×5×32×32 input. This is stated for 2026 and the same constraints for 2024 — [NTIRE 2026 LFSR report](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Wang_NTIRE_2026_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2026_paper.pdf), [NTIRE 2024 LFSR report](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf) (search extract). The 2025 values are presumed identical but not confirmed. A ResearchGate snippet mentioning "<30M params / <4.0T FLOPs" for the 2025 report is **unverified** and may refer to a Track 1 limit — [ResearchGate listing](https://www.researchgate.net/publication/395611057_NTIRE_2025_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_Results).
- The 2026 Track 3 wording allows "external training data and pretrained models… no efficiency limitation" — [NTIRE 2026 supplemental](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/supplemental/Wang_NTIRE_2026_Challenge_CVPRW_2026_supplemental.pdf) (search extract).
- **How the winning team spent its compute** — [local README §3](/home/user/LFTransMamba/README.md):
  - Track 1 submission = an **ensemble of 3 models** (LF_DET_0214 at patch 16 plus LFTransMamba at patches 40 and 48) with TTA.
  - Track 3 = an ensemble of the Track 1 and Track 3 models, with fp16 + TTA.
  - Track 1 models were trained for 30 epochs each.

### Inferences
- **H0 is not supported as a rule.** There is no mandated training-time budget. The real binding constraints are:
  1. the inference FLOPs/params cap (Track 2), which strongly favors linear-cost token mixers and explains the pure-Mamba T=0 entry;
  2. the team's GPU-hours over a weeks-long development phase.
- Under constraint (2), a faster architecture *could* translate into more iterations. However, the evidence shows the team trained the hybrid and the attention model for the **same** number of epochs, and trained the big Track 1/3 models for **fewer** epochs (30) than the small Track 2 model (100). So compute was spent on capacity, multiple models and ensembling, not on squeezing more iterations out of one model.

### Gaps
- I did not retrieve the 2025 LFSR challenge timeline (dates of the development and test phases) or the CodaLab terms pages (CodaLab was unreachable).
- I found no statements from teams about how many GPU-hours they used.

---

## Q5. Does fitting the training regression better also help generalization under camera or domain shift?

### Takeaway
The NTIRE LFSR test set is half **real Lytro** and half **synthetic** scenes under the same bicubic degradation, so the shift is in scene and camera content, not degradation. Real-scene PSNR is about 3 dB lower. Locally, the hybrid's gain over attention-only is **larger on real scenes (+0.14 dB) than synthetic (+0.05 dB)** at equal epochs. Masked-modeling regularization (used in LFTransMamba) is the documented route to better generalization in restoration. There is a tension with RCAN-it's in-distribution finding that regularization hurts. Overall, the evidence points to priors and regularization mattering for out-of-distribution performance at least as much as raw iteration count.

### Cited Findings
- NTIRE LFSR validation and test sets each contain 16 synthetic scenes (3DS MAX, co-planar camera arrays) and 16 real scenes (Lytro Illum) — [BasicLFSR README (NTIRE 2023 dataset)](https://github.com/ZhengyuLiang24/BasicLFSR); the same composition is used for 2024 — [NTIRE 2024 LFSR report](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf) (search extract); see also [NTIRE 2023 LFSR, arXiv 2304.10415](https://arxiv.org/abs/2304.10415).
- Local validation PSNRs: real about 32.6–32.8 dB vs synthetic about 35.5–35.65 dB across all LFTransMamba variants. The hybrid vs attention-only gaps are real +0.14 and synthetic +0.05 (Q3). The Track 3 model (extra data) vs the Track 1 hybrid is real +0.04, synthetic +0.06 — [local README](/home/user/LFTransMamba/README.md).
- *Masked Image Training for Generalizable Deep Image Denoising* (Haoyu Chen, Jinjin Gu et al., CVPR 2023, arXiv 2303.13132) masks random input pixels and masks features inside self-attention during training. It reports better generalization to unseen noise and direct applicability to real-world scenarios — [arXiv 2303.13132](https://arxiv.org/abs/2303.13132), [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Masked_Image_Training_for_Generalizable_Deep_Image_Denoising_CVPR_2023_paper.html) (search extract).
- LFTransMamba's MLFIM: in code, `random_masking(mask_buffer, mask_ratio=0.35)` replaces **35%** of per-view feature tokens with a learned `mask_token` during training only — [local models/LFTransMamba.py L2339–2385](/home/user/LFTransMamba/models/LFTransMamba.py). A secondary summary of the paper says 25%, so the paper and the released code may differ — [LFTransMamba CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/html/Jin_LFTransMamba_A_Hybrid_Mamba-Transformer_Model_for_Light_Field_Image_Super-Resolution_CVPRW_2025_paper.html) (search extract).
- Counterpoint for in-distribution bicubic SR: regularization "generally degrades the predictions" — [Revisiting RCAN](https://github.com/zudi-lin/rcan-it).
- Domain-gap work in LFSR (zero-shot LFSR; LytroZoom; LF-DMnet for real degradations) is cited in the NTIRE 2024 report and a 2025 review — [NTIRE 2024 LFSR report](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf), [LFSR critical review, arXiv 2510.07879](https://arxiv.org/abs/2510.07879) (search extract).

### Inferences
- Better fitting of the training regression (more iterations) primarily improves in-distribution PSNR. Because the test set is half real-Lytro, priors that encode **LF geometry** matter more than squeezing out the last training loss. Examples are cross-view 4D scans that follow disparity-consistent orderings, and masked reconstruction that forces the use of angular redundancy. The larger real-scene gain of the hybrid fits this, though n=1 comparison with confounds.
- MLFIM deliberately makes the training task *harder*: 35% of tokens are hidden, so the loss goes down more slowly per iteration. That is in tension with H's premise that the goal is maximal regression fitting per unit time. The winning recipe trades fitting speed for robustness.
- Ensembles of architecturally different models (attention-only LF_DET plus the hybrid) also reduce variance on out-of-distribution scenes. This is an alternative reason the team's final scores were high that is unrelated to iteration count.

### Gaps
- No ablation was found isolating MLFIM's effect on real vs synthetic PSNR.
- There is no study of iteration count vs out-of-distribution (real-Lytro) PSNR in LFSR.
- I could not verify *Reflash Dropout in Image Super-Resolution* (Kong et al., CVPR 2022) in this session, though it is likely relevant to "regularization improves SR generalization".

---

## Q6. Overall verdict on each link of the hypothesis, and alternative or complementary explanations

### Takeaway
The hypothesis gets the **ingredients right but the causal mechanism mostly wrong**. SR is regression (H1). It is underfitting-limited, so iterations matter (H2). Hybrids do keep, and in fact improve, quality (H5). But the evidence does not support "Transformers are too slow per iteration at LFSR sizes" (H3) for windowed or per-view attention. Nor does it support "Mamba wins by enabling more iterations" (H4) or "competitions impose a training-time budget" (H0). The stronger explanation is that **linear cost buys global 4D spatial-angular context per iteration, plus inference-FLOPs headroom, at equal iterations**, combined with complementary inductive biases and the recipe (masking, EMA, TTA, ensembles).

### Cited Findings (summary table; sources as in Q1–Q5)

| Link | Verdict | Key evidence |
|---|---|---|
| H1 SR is pure regression | **Supported** | L1 loss only ([local code](/home/user/LFTransMamba/models/LFTransMamba.py)); PSNR ranking ([NTIRE 2024 LFSR](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf)) |
| H2 needs many iterations | **Supported (qualitatively)** | Underfitting dominates; +0.1 to +0.44 dB from recipe/length ([Revisiting RCAN](https://arxiv.org/abs/2201.11279)); SOTA SR uses 250k to more than 1M iterations ([HAT](https://github.com/XPixelGroup/HAT/blob/main/options/train/train_HAT_SRx4_ImageNet_from_scratch.yml), [DAT](https://github.com/zhengchen1999/DAT/blob/main/options/Train/train_DAT_x4.yml), [MambaIRv2](https://github.com/csguoh/MambaIR/blob/main/options/train/mambairv2/train_MambaIRv2_SR_x4.yml)); the schedule must match the budget ([Budgeted Training](https://arxiv.org/abs/1905.04753)) |
| H3 Transformer-only too slow per unit time | **Partially supported / mostly contradicted** | False for window attention (L/6D < 1) and for per-view attention at 1K tokens with FlashAttention (below the about 2K crossover: [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-2](https://arxiv.org/abs/2405.21060)); true only for global attention over the 4D LF (25.6K to 57.6K tokens; L/6D = 53 to 120 per [MambaOut](https://arxiv.org/abs/2405.07992)); SR-Transformer slowness was largely an RPB/no-FlashAttention issue ([SST/RIB 2026](https://arxiv.org/abs/2603.06738)) |
| H4 Mamba lets you iterate many more times | **Weak / unsupported** | No LFSR wall-clock evidence; vision scans memory-bound ([VMamba](https://arxiv.org/abs/2401.10166), [Demystify](https://arxiv.org/abs/2405.16605)); the hybrid was trained for the same 30 epochs as the attention baseline ([local README](/home/user/LFTransMamba/README.md)) |
| H5 attention keeps quality | **Supported, but via complementary bias** | Hybrid > pure Mamba (+0.15) and > pure Transformer (+0.12) at matched settings ([LFMT](https://arxiv.org/abs/2509.04824)); local hybrid vs attention-only +0.10 dB at equal epochs |
| H0 competition has a training-time budget | **Not supported** | Rules cap inference params/FLOPs (efficiency track) and data (Track 3), not training time ([NTIRE 2025](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Wang_NTIRE_2025_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2025_paper.pdf), [NTIRE 2026](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Wang_NTIRE_2026_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2026_paper.pdf)) |

### Inferences
**Alternative and complementary explanations, ranked by how well the evidence supports them:**
1. **Affordable global 4D spatial-angular context, with linear cost buying "reach" rather than "speed".**
   - Whole-LF sequences (25,600 tokens at training, up to 57,600 at inference) are 53 to 120× past MambaOut's L > 6D threshold.
   - Global attention there costs about 210 GF per layer per sample, versus linear-time scans.
   - Mamba makes cross-view, all-pairs-reachable mixing feasible at every layer.
   - This is the long-sequence regime where MambaOut concedes Mamba helps.
2. **Complementary inductive biases.**
   - Attention = content-adaptive matching within each view.
   - SSM = ordered propagation with selective decay (a forget-gate/positional prior; Demystify) along two LF-aware orderings (macro-pixel and SAI-array).
   - The LFMT hybrid beats both pure variants at matched training.
3. **Inference-FLOPs headroom under challenge caps.** In the efficiency track (<20 GF on 5×5×32×32), one C=32 per-view attention layer costs about 4 GF, so linear mixers allow more depth and width under the cap. This explains the pure-Mamba T=0 Track 2 design.
   - Track 1 had no published cap here (not verified for 2025). There, larger patches at inference (40 or 48 vs training 32) are also cheap for scans.
4. **Training recipe and regularization, independent of architecture.**
   - MLFIM masking (35% in code), EMA 0.999, a budget-aware LR schedule, TTA, EPSW Gaussian patch fusion, and fp16 for the large model.
   - Revisiting RCAN shows that recipe alone can be worth 0.1 to 0.4 dB.
5. **Ensembling heterogeneous architectures.** The final Track 1 and Track 3 submissions ensemble an attention-only model with hybrids, adding diversity gains that iteration count cannot explain.
6. **Capacity and data.** Track 3 adds external data (`LFSR.ALL4.EXTRA`) and a larger K=24 model, giving about +0.05 dB mean over the Track 1 hybrid locally.
7. **(The user's H4 mechanism) More iterations per wall-clock time.** Possible in principle for whole-LF global mixing, but **not demonstrated** anywhere found. It is contradicted at the per-view level by kernel crossover data, and not exercised by the team (equal epochs).

**Re-stated, evidence-consistent version of the hypothesis** (my inference): "SR fidelity is an underfitting-limited regression, so compute matters. In LFSR the useful context is the whole 4D light field (tens of thousands of tokens). Global attention over that is prohibitively expensive per step, while Mamba's linear scans make cross-view global mixing affordable at every layer. Keeping cheap per-view attention preserves content-adaptive matching. The hybrid therefore wins *at the same number of iterations* because each iteration sees more relevant context with complementary priors, and in FLOPs-capped tracks because it fits under the cap."

### Gaps
- The decisive experiment (equal wall-clock vs equal iterations: LF_DET / T-only vs hybrid, with 2–3× longer schedules and a linear-to-zero LR) has not been published for LFSR. It is cheap to run in this repo.
- No public measurements exist of per-iteration time or memory for SS2D3D/SS2D4D scans vs FlashAttention per-view attention at 1K/25.6K tokens on the same GPU.
- The full texts of the NTIRE 2025 LFSR report, LFTransMamba and LFMT could not be read directly (arXiv/CVF blocked by the proxy). Numeric claims attributed to them come from search extracts and should be spot-checked against the PDFs.
