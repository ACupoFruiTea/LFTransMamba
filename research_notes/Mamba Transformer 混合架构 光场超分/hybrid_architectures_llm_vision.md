# Hybrid Mamba/SSM + Attention Architectures in Language Modeling and High-Level Vision: Designs, Ablations, and Analyses of Why Hybrids Work

Verification legend, used throughout. WebFetch could not reach arxiv.org or huggingface.co in this session (DNS failure), so evidence came from web-search extraction of the primary pages.
- **[V]** means the claim was confirmed this session in search results that quote or summarise the primary paper, model card or report.
- **[R]** means the claim comes from the paper's own abstract or tables as I recall them. I did not re-check these this session, so check the number against the linked arXiv page before quoting it.

Years: arXiv first-posting year; the venue is given where known. Today: 2026-10-16.

---

## Q1. LLM hybrids and their analyses (Jamba, NVIDIA studies, Zamba, Samba, Hymba, Nemotron-H, Falcon-H1, MiniMax, Qwen3-Next, Kimi Linear, MAD, systematic studies, distillation)

### Takeaway
Across production hybrids and controlled studies, the pattern is consistent. A small share of global softmax attention layers is enough to match or beat a pure Transformer on quality, with large KV-cache and throughput savings. Typical shares are about 7–25%, i.e. 1:3 to 1:9 attention:SSM. Quality is fairly insensitive to the exact ratio within that band; Jamba, for example, found 1:3 and 1:7 about equal. Placement matters more for specific skills than for perplexity. Retrieval depends on only a handful of attention layers. Parallel (intra-layer) head mixing is a strong alternative to interleaving, especially at long context.

### Cited Findings

**Summary table of major LLM hybrids**

| Model (org, year) | Sequence mixers | Attention share / layout | Headline claim | Source |
|---|---|---|---|---|
| Jamba (AI21, 2024; ICLR 2025) | Mamba-1 + softmax attn + MoE | 1 attn : 7 Mamba per 8-layer block | 52B total/12B active; 256K ctx; fits one 80GB GPU [R]; 1:3 ≈ 1:7 quality [V] | [arXiv 2403.19887](https://arxiv.org/abs/2403.19887) |
| Jamba-1.5 (AI21, 2024) | same, scaled | 1:7 kept ("found optimal") [V] | Large 94B active/398B total; Mini 12B/52B; 256K effective ctx [R] | [arXiv 2408.12570](https://arxiv.org/abs/2408.12570) |
| Mamba-2-Hybrid 8B (NVIDIA, 2024) | Mamba-2 + attn + MLP | 43% Mamba-2, **7% attn**, 50% MLP [V] | +2.65 avg over 8B Transformer on 12 tasks; up to 8× faster generation (predicted) [V] | [arXiv 2406.07887](https://arxiv.org/abs/2406.07887) |
| Zamba 7B (Zyphra, 2024) | Mamba + **one shared** attn block | shared attn+MLP block re-applied periodically | attention benefits "at minimal parameter cost" [R] | [arXiv 2405.16712](https://arxiv.org/abs/2405.16712) |
| Zamba2 1.2B/2.7B/7.4B (Zyphra, 2024) | Mamba-2 + 2 alternating shared attn blocks (+LoRA) | shared | [R] | [arXiv 2411.15242](https://arxiv.org/abs/2411.15242) |
| Samba 3.8B (Microsoft, 2024; ICLR 2025) | Mamba + **sliding-window attn (2048)** + MLP, layer-wise | interleaved | 3.73× throughput vs GQA Transformer at 128K prompt; 3.64× when generating 64K; 4K-trained → 1M zero-shot extrapolation [V] | [arXiv 2406.07522](https://arxiv.org/abs/2406.07522) |
| Hymba 1.5B (NVIDIA, 2024; ICLR 2025) | **parallel** attn heads + SSM heads in each layer; meta tokens; mostly SWA, few full-attn layers | intra-layer | beats Llama-3.2-3B by 1.32% avg acc; 11.67× smaller cache; 3.49× throughput [R] | [arXiv 2411.13676](https://arxiv.org/abs/2411.13676) |
| Nemotron-H 8B/56B/47B (NVIDIA, 2025) | Mamba-2 + few attn + FFN | ~8% attn layers [R] | up to 3× faster inference than similar-size Transformers (Qwen-2.5, Llama-3.1) at similar accuracy [R] | [arXiv 2504.03624](https://arxiv.org/abs/2504.03624) |
| Nemotron Nano 2 9B (NVIDIA, 2025) | Mamba-2 hybrid | mostly Mamba-2 | up to 6× throughput vs Qwen3-8B in reasoning-length generation [R] | [arXiv 2508.14444](https://arxiv.org/abs/2508.14444) |
| Falcon-H1 0.5B–34B (TII, 2025) | **parallel** attn + Mamba-2 in each block | intra-layer, tunable channel split | 34B claimed to match or beat ~70B models [R] | [arXiv 2507.22448](https://arxiv.org/abs/2507.22448) |
| MiniMax-Text-01 (MiniMax, 2025) | lightning (linear) attn + softmax attn, MoE | **1 softmax after every 7 lightning; 80 layers** [V] | 456B total/45.9B active [R]; 100% on 4M-token vanilla NIAH (self-reported) [V] | [arXiv 2501.08313](https://arxiv.org/abs/2501.08313); [GitHub](https://github.com/MiniMax-AI/MiniMax-01) |
| MiniMax-M1 (2025) | same hybrid, reasoning RL | 7:1 | 25% of DeepSeek-R1 FLOPs at 100K-token generation [R] | [arXiv 2506.13585](https://arxiv.org/abs/2506.13585) |
| Qwen3-Next-80B-A3B (Alibaba, Sep 2025) | **Gated DeltaNet + gated (softmax) attn**, MoE | **3:1**; 48 layers = 36 GDN + 12 gated attn [V, secondary] | ratio "determined through extensive experimentation" but no public ablation found [V] | [deeplearning.ai thread](https://community.deeplearning.ai/t/qwen3-next-employs-hybrid-attention-for-long-context-inputs/879461); [Raschka gallery](https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/) |
| Qwen3.5 / Qwen3.6 (Alibaba, 2026) | GDN + gated attn hybrid (Qwen3.6-35B-A3B: 3:1) [V, secondary] | 3:1 | Qwen3.5 used as a hybrid testbed in 2609.04434 [V] | [mlabonne, Medium](https://medium.com/@mlabonne/qwen3-5-nobody-agrees-on-attention-anymore-4709e1bd014b); [arXiv 2609.04434](https://arxiv.org/abs/2609.04434) |
| Kimi Linear 48B-A3B (Moonshot, Oct 2025) | **Kimi Delta Attention (KDA)** + MLA | **3 KDA : 1 MLA**, layer-wise [V] | up to 75% less KV cache; up to 6× decoding throughput at 1M ctx; 6.3× faster TPOT vs MLA [V] | [arXiv 2510.26692](https://arxiv.org/abs/2510.26692); [GitHub](https://github.com/MoonshotAI/Kimi-Linear) |
| IBM Granite 4.0-H (Oct 2025) | Mamba-2 + attn (some MoE) | **~9:1 Mamba-2:attn** [V] | >70% RAM reduction for long-context/multi-session inference (vendor claim) [V] | [MarkTechPost](https://www.marktechpost.com/2025/10/02/ibm-released-new-granite-4-0-models-with-a-novel-hybrid-mamba-2-transformer-architecture-drastically-reducing-memory-use-without-sacrificing-performance/); [The Decoder](https://the-decoder.com/ibms-granite-4-0-family-of-hybrid-models-uses-much-less-memory-during-inference/) |
| Griffin / RecurrentGemma (Google DeepMind, 2024) | RG-LRU + **local** MQA | 2 recurrent : 1 local-attn [R] | matches Llama-2 with >6× fewer training tokens [R] | [arXiv 2402.19427](https://arxiv.org/abs/2402.19427) |
| Hunyuan-TurboS (Tencent, 2025) | Mamba-2 + attn + MoE FFN | attention a small minority of 128 layers [R] | 56B active/560B total [R] | [arXiv 2505.15431](https://arxiv.org/abs/2505.15431) |
| Jet-Nemotron 2B/4B (NVIDIA, 2025) | JetBlock (linear) + a few searched full-attn layers | 2–3 full-attn layers placed by search [V] | reported ~50× generation speed-up at long context [R; secondary title](https://medium.com/data-science-in-your-pocket/nvidia-jet-nemotron-end-of-transformers-50x-faster-llms-b35b1366f923) | [arXiv 2508.15884](https://arxiv.org/abs/2508.15884) |

**Jamba (Lieber et al., AI21, 2024; ICLR 2025, "Jamba: Hybrid Transformer-Mamba Language Models")**
- The attention-to-Mamba ratio "may be 1:3 or 1:7 with virtually no performance difference." 1:7 was chosen as "the most compute-efficient variant amongst the best performing variants" [V]. Sources: [arXiv 2403.19887](https://arxiv.org/abs/2403.19887); [OpenReview ICLR 2025 PDF](https://openreview.net/pdf?id=JFPaD7lpBD).
- Pure Mamba struggles with in-context learning, while the attention–Mamba hybrid succeeds and its attention layers develop **induction heads**. The paper shows this through attention visualisation [V]. Source: [arXiv 2403.19887](https://arxiv.org/abs/2403.19887).
- Pure Mamba also failed to follow the few-shot answer format on tasks such as IMDB, QuAC and NarrativeQA. Search corroborated this only from a secondary blog [Medium](https://sulbhajain.medium.com/jamba-a-hybrid-transformer-mamba-language-model-with-mixture-of-experts-506281f2398e), so treat it as **unverified** against the paper.
- Jamba-1.5 states the 1:7 ratio "was found optimal in our work on Jamba and similar ratios was also confirmed as successful in follow-up work" [V]. Source: [arXiv 2408.12570](https://arxiv.org/abs/2408.12570).
- Recalled Jamba design details [R], from [arXiv 2403.19887](https://arxiv.org/abs/2403.19887):
  - No explicit positional embedding is needed, because Mamba layers supply positional information.
  - RMSNorm was added inside the Mamba layers for large-scale training stability.
  - The KV cache at 256K context is about 4GB, against about 32GB for Mixtral-8x7B.
  - Long-context throughput is about 3× Mixtral's.

**"An Empirical Study of Mamba-based Language Models" (Waleffe et al., NVIDIA, 2024)**
- Setup: 8B Mamba, Mamba-2, Transformer and hybrid models trained on up to 3.5T tokens with identical hyperparameters. The hybrid is 43% Mamba-2, 7% self-attention and 50% MLP [V]. Source: [arXiv 2406.07887](https://arxiv.org/abs/2406.07887).
- Pure SSMs lag on tasks that need strong **copying or in-context learning**: five-shot MMLU, Phonebook Lookup and long-context reasoning [V]. Source: [arXiv 2406.07887](https://arxiv.org/abs/2406.07887).
- The hybrid "exceeds the 8B-parameter Transformer on all 12 standard tasks (+2.65 points on average) and is predicted to be up to 8× faster when generating tokens" [V]. Source: [arXiv 2406.07887](https://arxiv.org/abs/2406.07887); checkpoint at [HF nvidia/mamba2-hybrid-8b-3t-4k](https://huggingface.co/nvidia/mamba2-hybrid-8b-3t-4k).
- Ratio and placement ablations [V]: validation loss was minimised at a small attention fraction, and spreading attention and MLP layers evenly through the depth performed about as well as other layouts tried. Sources: [arXiv 2406.07887 PDF](https://arxiv.org/pdf/2406.07887); [search summary](https://arxiv.org/pdf/2406.07887).
- Recalled detail [R]: the 8B hybrid is 24 Mamba-2 + 4 attention + 28 MLP layers, and the first layer is not attention. Variants extended to 16K, 32K and 128K context matched or exceeded the Transformer on average on long-context tasks. Source: [arXiv 2406.07887](https://arxiv.org/abs/2406.07887).

**Mamba-2 / SSD paper (Dao & Gu, ICML 2024)**
- The hybrid ablation found that about 10% attention layers mixed with SSD layers beat both pure Mamba-2 and Transformer++ at the same size [R]. Source: [arXiv 2405.21060](https://arxiv.org/abs/2405.21060).

**Samba (Ren et al., Microsoft, 2024; ICLR 2025)**
- Samba interleaves Mamba, sliding-window attention (window 2048) and MLP layer-wise. The 3.8B model was trained on 3.2T tokens. It gets 3.73× higher throughput than GQA Transformers on 128K prompts and 3.64× when generating 64K tokens, and extrapolates from 4K training to 1M zero-shot with improved Proof-Pile perplexity [V]. Sources: [arXiv 2406.07522](https://arxiv.org/abs/2406.07522); [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/84a7fc24ed52e8eff514c33e8ac76ea3-Paper-Conference.pdf); [review](https://artgor.medium.com/paper-review-samba-simple-hybrid-state-space-models-for-efficient-unlimited-context-language-69a00d4a3822).
- The design rationale is that Samba "selectively compresses a given sequence into recurrent hidden states while still maintaining the ability to precisely recall memories with the attention mechanism" [R]. Source: [arXiv 2406.07522](https://arxiv.org/abs/2406.07522).
- Follow-up: SambaY, "Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation" (Ren et al., 2025), used in Phi-4-mini-flash-reasoning. It adds a Gated Memory Unit that shares memory readouts across layers and reports up to ~10× decoding throughput on long generations [R]. Source: [arXiv 2507.06607](https://arxiv.org/abs/2507.06607).

**Hymba (Dong et al., NVIDIA, 2024; ICLR 2025)**
- Attention and SSM heads run in parallel inside every layer. The paper frames this as "attention heads provide high-resolution recall and SSM heads enable efficient context summarization" [R; also secondary V]. Sources: [arXiv 2411.13676](https://arxiv.org/abs/2411.13676); [NVIDIA blog](https://developer.nvidia.com/blog/hymba-hybrid-head-architecture-boosts-small-language-model-performance/).
- Head-removal analysis [V]:
  - The first-layer SSM head is essential; removing it pushes accuracy to random guessing.
  - The relative importance of attention and SSM heads in the same layer is **input-adaptive**.
  - Removing one attention head or one SSM head costs about 0.24% or 1.1% on HellaSwag in the ICLR version, versus about 0.46% or 1.2% averaged in the arXiv version. The figures differ between versions, so quote with care.
  
  Sources: [arXiv 2411.13676](https://arxiv.org/pdf/2411.13676); [ICLR version](https://www.jankautz.com/publications/Hymba_ICLR25.pdf).
- Efficiency design [R]:
  - Most layers use sliding-window attention; only 3 layers (first, middle, last) keep full attention.
  - KV cache is shared across layers.
  - Learnable meta tokens are prepended to the input.
  
  Result: Hymba-1.5B beats Llama-3.2-3B by 1.32% average accuracy with an 11.67× smaller cache and 3.49× throughput. Source: [arXiv 2411.13676](https://arxiv.org/abs/2411.13676).

**Zamba / Zamba2 (Glorioso et al., Zyphra, 2024)**
- Zamba 7B is a Mamba backbone plus **a single shared attention module** whose weights are reused at several depths, giving "the benefits of attention at minimal parameter cost." It was trained on about 1T tokens [R]. Source: [arXiv 2405.16712](https://arxiv.org/abs/2405.16712).
- Zamba2 moves to Mamba-2, uses two alternating shared attention blocks with LoRA specialisation per call site, and concatenates the original embeddings into the shared-block input [R]. Source: [arXiv 2411.15242](https://arxiv.org/abs/2411.15242).

**Nemotron-H and follow-ons (NVIDIA, 2025)**
- Nemotron-H 8B and 56B replace most self-attention layers with Mamba-2 (about 8% attention). They report accuracy on par with or better than Qwen-2.5 and Llama-3.1 at similar size, with up to 3× faster inference. A 47B variant was compressed from the 56B with MiniPuzzle [R]. Source: [arXiv 2504.03624](https://arxiv.org/abs/2504.03624).
- Nemotron Nano 2 (9B) reports up to 6× throughput over Qwen3-8B in reasoning-length generation [R]. Source: [arXiv 2508.14444](https://arxiv.org/abs/2508.14444).

**Falcon-H1 (TII, 2025)**
- Parallel hybrid: each block runs attention and Mamba-2 side by side and concatenates their outputs, and the attention-vs-SSM channel allocation was tuned. Sizes range from 0.5B to 34B; the 34B is claimed to match or beat about 70B models [R]. Source: [arXiv 2507.22448](https://arxiv.org/abs/2507.22448).
- Falcon-H1 is one of the two hybrids dissected in the 2026 mechanistic study (2609.04434), described under Q3 [V].

**MiniMax-01 / M1 / M2 (MiniMax, 2025–2026)**
- MiniMax "observed that lightning attention demonstrates limited retrieval capabilities." Inserting softmax attention every 8th layer ("Hybrid-lightning") restores retrieval. Layout: 1 softmax layer after every 7 lightning layers, 80 layers total [V]. Sources: [MiniMax-01 report PDF](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf); [arXiv 2501.08313](https://arxiv.org/abs/2501.08313); [GitHub](https://github.com/MiniMax-AI/MiniMax-01).
- MiniMax-M1 keeps the hybrid and reports 25% of DeepSeek-R1's FLOPs at 100K generation length [R]. Source: [arXiv 2506.13585](https://arxiv.org/abs/2506.13585).
- MiniMax-M2 **reverted to full attention**; this is counter-evidence, detailed under Q4 [V]. Source: [arXiv 2605.26494](https://arxiv.org/pdf/2605.26494).

**Kimi Linear (Moonshot AI, Oct 2025)**
- KDA refines the gated delta rule with fine-grained (channel-wise) gating and is combined with global MLA at a uniform 3:1 ratio. The ablation found 3:1 gave the best quality–throughput trade-off. Hybridisation is layer-wise rather than head-wise "for training stability." Training used 1.4T tokens [V]. Sources: [arXiv 2510.26692](https://arxiv.org/abs/2510.26692); [GitHub](https://github.com/MoonshotAI/Kimi-Linear); [HF model card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct).
- An ablation snippet shows 3:1 against 0:1 (pure MLA), but the column labels were not recoverable, so no numbers are given here [V]. Source: [search summary of arXiv 2510.26692](https://www.emergentmind.com/papers/2510.26692).

**Qwen3-Next / Qwen3.5 (Alibaba, 2025–2026)**
- Layout is 3 Gated DeltaNet : 1 gated softmax attention, 36 + 12 over 48 layers. The gated attention adds an output gate, QK-norm and partial RoPE. Output gating is credited with removing attention sinks and massive activations [V, secondary]. Sources: [deeplearning.ai](https://community.deeplearning.ai/t/qwen3-next-employs-hybrid-attention-for-long-context-inputs/879461); [Raschka gated-attention page](https://sebastianraschka.com/llm-architecture-gallery/gated-attention/); [lilting.ch analysis](https://lilting.ch/en/articles/qwen3-next-gated-attention-sink).
- No public ratio ablation was found. Qwen says the ratio came from "extensive experimentation." Raschka notes the hybrid comes with "a bit of a modeling performance trade-off" [V]. Source: [Raschka hybrid-attention gallery](https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/).

**Jet-Nemotron / PostNAS (Gu et al., NVIDIA, 2025)**
- The paper starts from the observation that "retaining a few full-attention layers ... is essential for maintaining high accuracy on challenging tasks such as retrieval. However, the optimal placement of these layers remains unclear" [V]. Source: [arXiv 2508.15884](https://arxiv.org/html/2508.15884v1).
- Method: a once-for-all supernet built from a pretrained Transformer, then beam search over which layers keep full attention [V].
- Findings [V]:
  - **Only 2–3 layers are critical for retrieval.**
  - Layers critical for MMLU are not the same as those critical for retrieval.
  - Math shows more diffuse importance patterns.
  - The whole pipeline (not placement alone) gives +5.3 MMLU, +8.4 math, +7.8 retrieval and +3.2 commonsense.
  
  Sources: [arXiv 2508.15884](https://arxiv.org/html/2508.15884v1); [NVlabs GitHub](https://github.com/NVlabs/Jet-Nemotron).

**"Mechanistic Design and Scaling of Hybrid Architectures" (Poli et al., ICML 2024)**
- MAD is a suite of synthetic token-manipulation tasks: in-context recall, fuzzy recall, noisy recall, selective copying, compression and memorisation. MAD performance correlates with compute-optimal perplexity at scale [R].
- More than 500 language models from 70M to 7B were trained. Striped hybrids (e.g. Hyena or Mamba interleaved with attention) and sparse/MoE variants beat Transformer++, Hyena and Mamba in compute-optimal and over-trained scaling [R].

Source: [arXiv 2403.17844](https://arxiv.org/abs/2403.17844).

**Systematic hybrid-design studies (2025–2026)**
- *"Hybrid Architectures for Language Models: Systematic Analysis and Design Insights"* (arXiv 2510.04800, Oct 2025) compares **inter-layer (sequential)** and **intra-layer (parallel)** fusion of attention with Mamba. It covers language modelling, long context, scaling, and training and inference efficiency, and studies block ratio and the position of Transformer blocks [V]. Sources: [arXiv 2510.04800](https://arxiv.org/abs/2510.04800); [HTML](https://arxiv.org/html/2510.04800v1).
  - Its specific numeric recommendations could not be retrieved; see Gaps.
  - One third-party review judged its evidence for general recipes "thinner than the claims suggest" [V]. Source: [Moonlight review](https://www.themoonlight.io/en/review/hybrid-architectures-for-language-models-systematic-analysis-and-design-insights).
- *"A Systematic Analysis of Hybrid Linear Attention"* (Wang, Zhu, Eshraghian et al., July 2025) trained and released **72 models**: 340M on 20B tokens and 1.3B on 100B tokens, 6 linear-attention variants by 5 hybrid ratios [V]. Findings:
  - Selective gating, hierarchical recurrence and controlled forgetting are critical for hybrids to reach Transformer-level recall.
  - The recommended setup is HGRN-2 or GatedDeltaNet at a **linear:full ratio between 3:1 and 6:1**.
  - Ratio affects language modelling and recall differently.
  - **"Superior standalone linear models do not necessarily excel in hybrids."**
  
  Sources: [arXiv 2507.06457](https://arxiv.org/abs/2507.06457); [HF paper page](https://huggingface.co/papers/2507.06457).
- *"Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall and Language Modeling"* (Lee, Yu et al., BabyLM workshop 2025) [V]:
  - **Sequential hybrids are better on short context; parallel hybrids are better on long context.**
  - The best variant is parallel with a "merge-attention" fusion layer.
  - Continued training on paraphrase-augmented data improved recall more than architectural changes did.
  - Strong language-modelling or reasoning scores did not reliably imply strong recall.
  
  Sources: [arXiv 2510.26912](https://arxiv.org/abs/2510.26912); [ACL Anthology](https://aclanthology.org/2025.babylm-main.27/).
- *"Rethinking the Role of Efficient Attention in Hybrid Architectures"* (Qiao et al., Tsinghua/OpenBMB, June 2026; Google Scholar lists EMNLP 2026, unconfirmed) [V]:
  - The efficient module (SWA or recurrence) mainly sets **how fast** long-context ability emerges; different hybrids converge to similar long-context performance given enough training.
  - Full attention carries most long-range retrieval.
  - "Large-Window Laziness": bigger sliding windows delay the formation of retrieval heads in the full-attention layers.
  - Using NoPE only in the full-attention layers of a small-window hybrid improves long context at little short-context cost.
  
  Sources: [arXiv 2606.15378](https://arxiv.org/abs/2606.15378); [HTML](https://arxiv.org/html/2606.15378v1).
- *"HyLo"* (arXiv 2604.24715, 2026) is a long-context upcycling recipe pairing MLA with Mamba-2 or Gated DeltaNet. It notes that more MLA layers add attention capacity but also KV cache, so the ratio sets the quality/efficiency trade-off [V]. Source: [arXiv 2604.24715](https://arxiv.org/html/2604.24715v1).
- Other 2025–2026 hybrid-design papers found by title only and **not reviewed**:
  - "Mechanics of Long-Context Hybrid Models Part 1.1: From Hybrid Attention to Hybrid Position" ([arXiv 2610.10114](https://arxiv.org/html/2610.10114v1))
  - "Multilinguality in Hybrid Attention LLMs" ([arXiv 2609.35378](https://arxiv.org/html/2609.35378))
  - "Alleviating Forgetfulness of Linear Attention by Hybrid ..." ([arXiv 2510.20787](https://arxiv.org/pdf/2510.20787))
  - "Kwai Summary Attention Technical Report" ([arXiv 2604.24432](https://arxiv.org/pdf/2604.24432))
  - "InfoMamba: An Attention-Free Hybrid Mamba-Transformer Model" ([arXiv 2603.18031](https://arxiv.org/pdf/2603.18031))
  - "MaBERT: ... Interleaved Transformer Mamba Hybrid Encoder" ([arXiv 2603.03001](https://arxiv.org/pdf/2603.03001))

**Distillation from Transformers into hybrids**
- *"The Mamba in the Llama: Distilling and Accelerating Hybrid Models"* (J. Wang, Paliotta, May, Rush, Dao; NeurIPS 2024) reuses the attention layers' linear-projection weights to initialise Mamba layers. A distilled hybrid **keeping 1/4 of the attention layers** matches the teacher on chat benchmarks and beats hybrids trained from scratch on trillions of tokens. It adds hardware-aware speculative decoding [R]. Source: [arXiv 2408.15237](https://arxiv.org/abs/2408.15237).
- *MOHAWK*, "Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models" (Bick et al., 2024), distils in three stages (matrix orientation, hidden-state alignment, end-to-end). Phi-Mamba used a tiny fraction of standard pretraining data [R]. Source: [arXiv 2408.10189](https://arxiv.org/abs/2408.10189).
- *Zebra-Llama* (AMD, 2025) combines MLA and Mamba-2 hybrids distilled from Llama with very small token budgets [R]. Source: [arXiv 2505.17272](https://arxiv.org/abs/2505.17272).
- *Llamba* (Cartesia, 2025) is a pure distilled Mamba-2, a non-hybrid reference point [R]. Source: [arXiv 2502.14458](https://arxiv.org/abs/2502.14458).
- 2025–2026 papers on **which layers to keep as attention** when distilling, found by title only:
  - "Distilling to Hybrid Attention Models via KL-Guided Layer Selection" ([arXiv 2512.20569](https://arxiv.org/pdf/2512.20569))
  - "Hybrid Linear Attention Done Right: Efficient Distillation ..." ([arXiv 2601.22156](https://arxiv.org/pdf/2601.22156))
  - "Taylor-Calibrate: Principled Initialization for Hybrid Linear Attention Distillation" ([arXiv 2606.16429](https://arxiv.org/pdf/2606.16429))

**Precursors that motivated hybrids**
- *"Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks"* (Park et al., ICML 2024) [R]:
  - SSMs match Transformers on standard ICL regression and beat them on sparse parity, but fail at non-standard retrieval.
  - The proposed **MambaFormer** puts Mamba first (replacing positional encoding) and interleaves it with attention. It "surpass[es] individual models in tasks where they struggle independently."
  
  Source: [arXiv 2402.04248](https://arxiv.org/abs/2402.04248).
- *BASED* (Arora et al., ICML 2024) combines linear attention with small sliding-window attention to navigate the recall–throughput (state-size) trade-off [R]. Source: [arXiv 2402.18668](https://arxiv.org/abs/2402.18668).
- *Taipan* (2024) is Mamba-2 plus "Selective Attention Layers" that send only tokens needing long-range interaction through attention [R]. Source: [arXiv 2410.18572](https://arxiv.org/abs/2410.18572).
- *TransMamba* ("Flexibly Switching between Transformer and Mamba", 2025) shares parameters between attention and SSM and switches mechanism at "TransPoints" along the sequence [R; arXiv ID not re-verified]. Source: [arXiv 2503.24067](https://arxiv.org/abs/2503.24067).

### Inferences
- The quality benefit of hybrids seems to come from **having some** global softmax attention, i.e. a few retrieval-capable layers, more than from any particular ratio. Evidence:
  - Jamba saw no 1:3 vs 1:7 difference.
  - Systematic studies settle on 3:1–6:1 linear:full, and production models span 3:1 (Qwen3-Next, Kimi Linear) to about 9:1–12:1 (Granite 4.0, NVIDIA Mamba-2 hybrids).
  - Jet-Nemotron finds only 2–3 layers critical for retrieval.
- The stronger the linear or SSM mixer, the fewer attention layers are needed. Delta-rule and gated mixers (GDN, KDA) are used at 3:1, while Mamba-2 hybrids go to 7–9% attention. These are different mixers, so this is a correlation across papers, not a controlled result.
- Parallel (intra-layer) mixing (Hymba, Falcon-H1) and sequential interleaving (Jamba, Nemotron-H, Samba) both work. Two studies suggest parallel helps long context, while large production models mostly ship layer-wise designs, citing training stability (Kimi) and kernel simplicity.

### Gaps
- Exact numeric recommendations of arXiv 2510.04800 (best inter-layer ratio, best position for Transformer blocks, intra- vs inter-layer Pareto numbers) were not retrievable; both arXiv and HF were unreachable.
- Kimi Linear's ratio-ablation table could not be read; column headers were missing from the extracted text.
- No public ablation of Qwen3-Next's 3:1 ratio was found.
- Nemotron 3 (late 2025/2026 hybrid Mamba-Transformer MoE family) and Falcon-H1R or other 2026 releases were not verified this session.
- [R] items, especially Hymba, Nemotron-H, Falcon-H1, Zamba2, Hunyuan-TurboS, SambaY, Taipan, TransMamba and Zebra-Llama numbers, should be re-checked against arXiv before quoting.

---

## Q2. Vision hybrids (classification / detection / segmentation / video / VLM) and what ablations show about layer ordering and complementarity

### Takeaway
The clearest vision ablation, MambaVision (CVPR 2025), shows that **putting self-attention in the last layers of the later stages** beats random, early or interleaved placement on ImageNet. That means local or SSM mixing early and global attention late. Newer hybrids such as A2Mamba move beyond stacking toward tighter coupling, with SSM states interacting with attention maps. Video and VLM hybrids (Vamba, MaTVLM, LongLLaVA, TimeViper) use Mamba mainly for **efficiency on very long token sequences**. MambaOut warns that for plain ImageNet classification the SSM itself may be unnecessary.

### Cited Findings
**MambaVision: "MambaVision: A Hybrid Mamba-Transformer Vision Backbone" (Hatamizadeh & Kautz, NVIDIA; arXiv 2024, CVPR 2025)**
- Hierarchical design: stages 1–2 use CNN residual blocks; stages 3–4 use a redesigned Mamba mixer (non-causal, with an extra symmetric conv branch) followed by self-attention blocks [R]. Source: [arXiv 2407.08083](https://arxiv.org/abs/2407.08083).
- From the abstract: "equipping the Mamba architecture with several self-attention blocks at the final layers greatly improves the modeling capacity to capture long-range spatial dependencies" [V]. Source: [arXiv 2407.08083](https://arxiv.org/abs/2407.08083).
- Placement ablation (Table 5, ImageNet-1K) [V]:

  | Placement of self-attention | Top-1 |
  |---|---|
  | **Last half of each stage (MMMMSSSS)** | **82.3%** |
  | First half | 81.5% |
  | Interleaved | 81.4–81.6% |
  | Random | 81.3% |

  Sources: [arXiv PDF](https://arxiv.org/pdf/2407.08083); [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf).
- MambaVision is also evaluated on COCO detection/instance segmentation and ADE20K semantic segmentation, where it outperforms comparably sized backbones [V]. Exact AP and mIoU were not retrieved. Source: [Emergent Mind summary](https://www.emergentmind.com/papers/2407.08083).

**A2Mamba: "Attention-augmented State Space Models for Visual Recognition" (Lou, Fu, Yu; July 2025)**
- The authors argue earlier hybrids "simply stacked" attention and Mamba layers without interaction. Their MASS token mixer has SSM hidden states interact with **multi-scale attention matrices** through a cross-attention variant [V].
- Results [V]: A2Mamba-L reaches 86.1% top-1 on ImageNet-1K. A2Mamba-S beats MambaVision-B by +1.2 AP^b / +0.9 AP^m on COCO with 40% fewer parameters.

Sources: [arXiv 2507.16624](https://arxiv.org/abs/2507.16624); [HTML](https://arxiv.org/html/2507.16624v1).

**Heracles (Patro, Namboodiri, Agneeswaran; 2024)**
- Heracles is a hybrid SSM-Transformer for high-resolution images and time series. It uses SSMs in the earlier layers and attention only in the deeper layers, the same "SSM early, attention late" ordering as MambaVision [V]. Source: [arXiv 2403.18063](https://arxiv.org/abs/2403.18063).

**Mamba as linear attention in vision: "Demystify Mamba in Vision: A Linear Attention Perspective" (MLLA; Han et al., NeurIPS 2024)**
- Mamba's advantage over plain linear attention comes chiefly from the **forget gate** and the **macro block design**. For non-causal vision, the forget gate can be replaced by suitable positional encoding. The resulting MLLA beats vision-Mamba models [R]. Source: [arXiv 2405.16605](https://arxiv.org/abs/2405.16605).

**Video and vision-language hybrids**
- *Vamba*, "Understanding Hour-Long Videos with Hybrid Mamba-Transformers" (Ren et al., ICCV 2025), encodes video tokens with Mamba-2 blocks (linear in visual tokens) and keeps cross-attention for text [V]:
  - It handles more than 1024 frames at 640×360 on one GPU, versus 256 frames for Transformer LMMs.
  - It uses at least 50% less GPU memory and runs nearly 2× faster per training step.
  - It trains on 8×A800, versus 64 GPUs for LongVU and 24 for LongLLaVA.
  - On LVBench it scores 42.1%, versus 37.8% for LongVU and 42.0% for its Qwen2-VL-7B Transformer baseline. The improvement is stated as +4.3% on arXiv and +4.6% in the ICCV poster.
  - The authors note Mamba kernels are "under-optimized," so theoretical gains are not fully realised.
  
  Sources: [arXiv 2503.11579](https://arxiv.org/abs/2503.11579); [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Ren_Vamba_Understanding_Hour-Long_Videos_with_Hybrid_Mamba-Transformers_ICCV_2025_paper.pdf).
- *MaTVLM*, "Hybrid Mamba-Transformer for Efficient Vision-Language Modeling" (2025), replaces part of a pretrained VLM's decoder layers with Mamba-2 layers initialised from the attention weights, then distils from the original VLM. It reports faster inference and lower GPU memory at comparable accuracy [R; exact numbers not verified]. Source: [arXiv 2503.13440](https://arxiv.org/abs/2503.13440).
- *LongLLaVA* (2024) is a Jamba-style hybrid Mamba-Transformer MLLM for many-image and long-video inputs, scaling to about 1000 images on one 80GB GPU [R]. Source: [arXiv 2409.02747](https://arxiv.org/abs/2409.02747).
- *TimeViper*, "A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding" (Nov 2025), was found by title only [V title]. Source: [arXiv 2511.16595](https://arxiv.org/pdf/2511.16595).

**Non-hybrid reference point**
- *Mamba-ND* (Li et al., ECCV 2024) extends **pure** Mamba to N-D data by alternating scan orders. It is not a hybrid and is useful as a baseline [R]. Source: [arXiv 2402.05892](https://arxiv.org/abs/2402.05892).

### Inferences
- The vision ordering "local, conv or SSM early; global attention late" (MambaVision, Heracles) is the vision counterpart of LLM findings that a few attention layers do the precise global matching. Early layers benefit from cheap local or sequential mixing; deep, low-resolution layers benefit from exact all-pairs interaction.
  - LLMs differ: Waleffe found even distribution of attention through depth fine, and Hymba keeps full attention in the first, middle and last layers. Ordering conclusions therefore may not transfer across domains.
- For dense prediction, which has long token sequences, MambaOut's negative result for classification and its positive signal for detection/segmentation together suggest SSMs pay off when token counts are large. That is the regime of high-resolution and light-field SR, where angular × spatial token counts are large.
  - This is an inference about LFSR transfer, not a tested claim.
- A2Mamba's gains over MambaVision suggest **coupling** SSM and attention (attention maps steering SSM states) can beat plain stacking. This parallels the LLM finding that parallel or merged fusion helps long context (2510.26912).

### Gaps
- MambaVision's COCO and ADE20K tables, and whether the placement ablation was repeated on detection, were not retrieved.
- No 2025–2026 vision paper was found that systematically studies attention:SSM **ratio** in the way LLM papers do; only MambaVision's placement ablation was located.
- No detailed per-layer interpretability analysis (what attention vs SSM layers do) was found for vision hybrids.

---

## Q3. Recurring explanatory themes: why hybrids work

### Takeaway
The evidence strongly supports a **division of labour**. Recurrent or SSM layers act as a compressed, recency-biased running memory that carries context, style, language and position. A few attention layers perform **exact, content-addressed retrieval and copying**, which fixed-size states provably and empirically cannot do reliably. 2025–2026 interpretability work has now shown this causally: in trained hybrids, the attention layers carry the retrieval ("aggregate") step, and recurrent layers hand the key to them.

### Cited Findings
**Why SSMs need attention (the fixed-state limit)**
- *"Repeat After Me: Transformers are Better than State Space Models at Copying"* (Jelassi et al., ICML 2024) proves that a two-layer Transformer can copy exponentially long strings, while generalised SSMs are bounded by their fixed-size latent state [R]. Source: [arXiv 2402.01032](https://arxiv.org/abs/2402.01032).
- Waleffe et al. found pure SSMs lag specifically on copying/ICL tasks (five-shot MMLU, Phonebook, long-context reasoning), and that adding 7% attention more than closes the gap [V]. Source: [arXiv 2406.07887](https://arxiv.org/abs/2406.07887).
- MiniMax found lightning (linear) attention has "limited retrieval capabilities," fixed by a softmax layer every 8 layers [V]. Source: [MiniMax-01 report](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf).

**Gather-and-Aggregate: the retrieval circuit and how hybrids divide it**
- *"Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism"* (Bick, Xing, Gu; ICML 2025) [V]:
  - Retrieval in both Transformers and SSMs uses a two-stage circuit. A **Gather head** compresses a segment into a summary token; an **Aggregate head** attends from the query to the summary tokens and selects the right one.
  - In SSMs the mechanism exists but is spread across more heads with smoother patterns.
  - **"Pretrained hybrid models ... delegate the role of Aggregate Heads to attention."**
  - Replacing a single G&A head in a pretrained SSM with an attention variant boosts retrieval and benchmark scores.
  - In a layer-pruned Llama-3.1-8B, removing one such head drops MMLU from 66% to 25% while knowledge benchmarks barely change.
  
  Sources: [arXiv 2504.18574](https://arxiv.org/abs/2504.18574); [PMLR v267](https://proceedings.mlr.press/v267/bick25a.html); [code](https://github.com/goombalab/gather-and-aggregate).

**Causal 2026 analyses of trained hybrids**
- *"What Attention Recalls and Recurrence Controls in Hybrid Language Models"* (Afendulev, Dontsov, Tutubalina, Korznikov; Sep 2026) uses cache-level interventions on **Qwen3.5 and Falcon-H1** [V]:
  - Split-prefill keeps only the KV cache or only the recurrent state. State-swap combines one context's KV with another's recurrent state.
  - **Exact retrieval goes through attention**: the KV-only path keeps 64–98% of full accuracy, and the recurrent-only path gives zero.
  - **Language and persona live in the recurrent state**: KV-only generation drops to about 1% language accuracy.
  - State-swap shows the answer's *value* comes from attention and its *language* from recurrence.
  - Recurrent-only generation produces semantically or formally related words that never appeared, i.e. lossy, compressed memory.
  
  Source: [arXiv 2609.04434](https://arxiv.org/abs/2609.04434).
- *"The Key Handoff: Retrieval in Hybrid Language Models"* (Kale, Baser, Vishwanath; Sep 2026) studies 12 dense and hybrid models by patching hidden states between stories [V]:
  - Attention layers convert the key into an answer in every model.
  - In four hybrids, **recurrent layers carry the key forward to an attention layer**, where a few heads resolve it against in-context facts.
  - Writing a different fact into the recurrent memory after the last attention layer can still shift the answer (odds multiplier 1.3–2.7).
  
  Source: [arXiv 2609.32942](https://arxiv.org/abs/2609.32942).
- Hymba's head-removal study [V]: SSM heads, especially the first layer's, are critical for basic function, while attention heads matter in an input-dependent way. This fits "SSM = always-on summariser, attention = on-demand high-resolution recall." Source: [arXiv 2411.13676](https://arxiv.org/pdf/2411.13676).
- Jamba [V]: induction heads, the classic ICL/copy circuit, emerge in the hybrid's attention layers, while pure Mamba fails ICL. Source: [arXiv 2403.19887](https://arxiv.org/abs/2403.19887).

**SSM or recurrence as positional/recency prior, so attention can drop positional encodings**
- Jamba needs no explicit positional embeddings in its hybrid, and Mamba-first MambaFormer replaces positional encoding with a Mamba layer [R]. Sources: [arXiv 2403.19887](https://arxiv.org/abs/2403.19887); [arXiv 2402.04248](https://arxiv.org/abs/2402.04248).
- *Rethinking Efficient Attention in Hybrids* (2026) [V]:
  - NoPE on the full-attention layers of a small-window hybrid improves long context.
  - Larger SWA windows make full-attention layers "lazy" (Large-Window Laziness), delaying retrieval-head formation.
  - So the cheap local or recurrent path and the global path compete for the same job, and a weaker local path forces the global layers to learn retrieval.
  
  Source: [arXiv 2606.15378](https://arxiv.org/abs/2606.15378).
- A 2026 paper explicitly frames hybrids as "From Hybrid Attention to Hybrid Position" (title-only) [V title]. Source: [arXiv 2610.10114](https://arxiv.org/html/2610.10114v1).

**Synthetic-task evidence for complementarity**
- MambaFormer outperforms both pure models on tasks where each fails alone: Transformers fail sparse parity; Mamba fails non-standard retrieval [R]. Source: [arXiv 2402.04248](https://arxiv.org/abs/2402.04248).
- MAD [R]: hybrids win on the combined suite (recall, copy, compression, memorisation), and MAD scores predict scaling perplexity. Source: [arXiv 2403.17844](https://arxiv.org/abs/2403.17844).

**Efficiency mechanism**
- KV cache scales only with the few attention layers, while SSM/linear layers keep constant-size state:
  - Jamba: about 8× less KV than Mixtral at 256K [R].
  - Hymba: 11.67× cache reduction [R].
  - Kimi Linear: up to 75% less KV and 6× decoding at 1M [V].
  - Granite 4.0: >70% RAM reduction, vendor claim [V].
  - Samba: 3.73× prefill throughput [V].
  
  Sources: [arXiv 2403.19887](https://arxiv.org/abs/2403.19887); [arXiv 2411.13676](https://arxiv.org/abs/2411.13676); [Kimi-Linear GitHub](https://github.com/MoonshotAI/Kimi-Linear); [MarkTechPost](https://www.marktechpost.com/2025/10/02/ibm-released-new-granite-4-0-models-with-a-novel-hybrid-mamba-2-transformer-architecture-drastically-reducing-memory-use-without-sacrificing-performance/); [arXiv 2406.07522](https://arxiv.org/abs/2406.07522).

### Inferences
- The best-supported mechanistic story is "**recurrence carries and compresses; attention looks up exactly.**" Four independent 2024–2026 lines of evidence converge on it: G&A delegation, the split-prefill/state-swap interventions, the Key Handoff patching study, and Jamba's induction heads.
- For an image-restoration analogue such as LFTransMamba, the transferable analogue is this: SSM scans propagate smooth, long-range context cheaply across the spatial-angular sequence, while attention performs precise content-based matching, e.g. finding corresponding pixels across sub-aperture views or exploiting non-local self-similarity.
  - This is an inference by analogy; other researchers cover IR/SR-specific evidence.
- Because the efficient path and the attention path can compete (Large-Window Laziness), making the cheap path too strong may stop the attention layers from learning precise retrieval. This is a design consideration when choosing window sizes or scan ranges.

### Gaps
- No vision-specific causal or interpretability study (comparable to 2609.04434 or G&A) was found showing what attention vs SSM layers do inside MambaVision-style backbones.
- The 2026 interpretability papers (2609.04434, 2609.32942) are very recent preprints. They have not been peer reviewed, and their dates in search listings were inconsistent.

---

## Q4. Counter-evidence: no benefit, or gains only from efficiency

### Takeaway
There is meaningful counter-evidence:
- MiniMax **abandoned** its lightning-attention hybrid for M2 after hybrids that matched full attention on standard benchmarks showed **multi-hop reasoning deficits at scale**.
- MambaOut shows the SSM mixer is unnecessary for ImageNet classification.
- Several studies find hybrid *quality* differences shrink with enough training, so the durable advantage is often **efficiency**, not accuracy.

### Cited Findings
- **MiniMax-M2 reverted to full attention.** The M2 report says hybrid attention appeared to match full attention on standard benchmarks, "but at a larger scale, clear deficits emerged in complex multi-hop reasoning." M2 dropped the hybrid for full attention in all layers [V]. Sources: [arXiv 2605.26494, "The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence"](https://arxiv.org/pdf/2605.26494); [MiniMax-01 report for context](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf).
  - Secondary accounts say MiniMax called efficient attention not yet "production-ready." Validation on real code, math and multimodal tasks was required, not just benchmark parity [V, secondary]. Sources: [ai2.work blog](https://ai2.work/blog/minimax-s-lightning-attention-bet-efficiency-vs-us-frontier-labs); [aibase (Chinese)](https://www.aibase.com/zh/news/22592).
  - One secondary source says efficiency came back in M3 (June 2026) via **sparse** rather than linear attention. This is secondary only and unverified [V, secondary]. Source: [ai2.work blog](https://ai2.work/blog/minimax-s-lightning-attention-bet-efficiency-vs-us-frontier-labs).
  - A third site describes later MiniMax models (M2.x) as still hybrid lightning-based, which **conflicts** with the full-attention account; treat with caution [V]. Source: [Claude Market blog](https://claudemarket.ai/blog/best-minimax-models-2026).
- **MambaOut** ("Do We Really Need Mamba for Vision?", Yu & Wang, CVPR 2025) [V]:
  - Mamba blocks with the SSM removed (gated CNN) beat all visual Mamba models on ImageNet classification. The authors conclude Mamba is unnecessary there.
  - On COCO detection and ADE20K segmentation, MambaOut does *not* match state-of-the-art visual Mamba models, so SSMs may help long-sequence dense tasks.
  - The claim that Mamba suits autoregressive tasks is conceptual and was not tested.
  
  Sources: [arXiv 2405.07992](https://arxiv.org/abs/2405.07992); [CVPR 2025 poster](https://cvpr2023.thecvf.com/virtual/2025/poster/33055).
- **Hybrid quality converges given enough training** (2606.15378): different efficient-attention hybrids "eventually reach similar long-context performance." The efficient module mainly changes how quickly the ability emerges, so differences are largely in data efficiency and speed, not final capability [V]. Source: [arXiv 2606.15378](https://arxiv.org/abs/2606.15378).
- **Ratio insensitivity** (Jamba): 1:3 vs 1:7 showed "virtually no performance difference." 1:7 was chosen for compute efficiency, so beyond a minimum amount of attention the extra layers buy efficiency, not quality [V]. Source: [arXiv 2403.19887](https://arxiv.org/abs/2403.19887).
- **Trade-off acknowledged for production linear hybrids.** Raschka's survey notes Qwen3-Next's 3:1 hybrid "comes with a bit of a modeling performance trade-off," and the claimed optimal ratio has no published ablation [V]. Source: [Raschka gallery](https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/).
- **Standalone strength does not predict hybrid strength**: a better linear model is not automatically a better hybrid component [V]. Source: [arXiv 2507.06457](https://arxiv.org/abs/2507.06457).
- **LM scores do not imply recall.** In 2510.26912, strong language-modelling or reasoning scores did not reliably imply strong memory recall. A data fix (paraphrase augmentation) beat architectural recall fixes [V]. Source: [arXiv 2510.26912](https://arxiv.org/abs/2510.26912).
- **Efficiency is not always realised in practice** [V]:
  - Vamba reports that Mamba kernels are under-optimised on current hardware, so theoretical gains are not fully realised.
  - Waleffe's 8× generation speed-up is a *prediction*, not a measured benchmark.
  
  Sources: [arXiv 2503.11579](https://arxiv.org/abs/2503.11579); [arXiv 2406.07887](https://arxiv.org/abs/2406.07887).
- **Vision hybrid accuracy gains are small and partly within-baseline.** Vamba's LVBench 42.1% vs its Qwen2-VL-7B Transformer baseline's 42.0% is essentially parity on accuracy; the win is memory and speed [V]. Source: [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Ren_Vamba_Understanding_Hour-Long_Videos_with_Hybrid_Mamba-Transformers_ICCV_2025_paper.pdf).
- **Data-dependent caveat** (Hymba): a version trained only on public data showed a noticeable drop in 5-shot MMLU, per a secondary summary [V, secondary]. Source: [search summary of arXiv 2411.13676](https://deep-diver.github.io/ai-paper-reviewer/paper-reviews/2411.13676/).

### Inferences
- The consensus "hybrid ≥ Transformer" result mostly comes from **controlled pretraining comparisons at ≤8B scale and standard benchmarks** (Waleffe, Jamba ablations, systematic studies). The MiniMax-M2 reversal suggests standard benchmarks may miss multi-hop or long-horizon reasoning deficits at frontier scale. Hybrid quality claims should be validated on task-specific, hard evaluations, not perplexity or average benchmark score alone.
- In vision, MambaOut plus MambaVision's small placement deltas (about 1 point top-1) imply that hybrid benefits in high-level vision are modest and task-dependent. They appear mainly for long-sequence dense prediction, plausibly the regime of light-field SR, rather than short-sequence classification.
- Where a hybrid wins on accuracy, a good share of the gain can come from training recipe and data rather than from the architecture itself. Examples: Jet-Nemotron's gains are pipeline-wide; 2510.26912's data fix beat architectural fixes.

### Gaps
- MiniMax's original engineering post ("Why did M2 end up as a full attention model?", Hugging Face blog) could not be fetched. The specific experiments behind the reversal are reported here only via the M2 report snippet and secondary coverage.
- No controlled study was found showing a hybrid **underperforming** a pure Transformer at equal compute on standard pretraining benchmarks. Counter-evidence is about scale-dependent reasoning, classification-only vision, or convergence, not head-to-head losses.
- No counter-evidence was located specifically for video, VLM or detection hybrids beyond MambaOut and Vamba's accuracy parity.
