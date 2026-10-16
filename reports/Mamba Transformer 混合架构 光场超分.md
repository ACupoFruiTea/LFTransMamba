# 混合架构的增益更可能来自互补先验，而非更多迭代

确实有不少论文分析过 Mamba 与 Transformer 结合的优势。下面先给出五条结论速览，再分节展开。

1. **混合架构论文与分工解释**：语言模型方向的结论高度一致，并已有因果干预证据：SSM（Mamba）以线性代价压缩并传播长程上下文，softmax 注意力负责精确的、按内容寻址的匹配与"复制"。在多数受控对比（多为 ≤8B 规模、标准基准）中，混合模型在相同训练设置下能同时胜过两种纯架构（反例见后文 MiniMax-M2）。视觉和复原领域目前只有相关性证据。光场超分中，LFMT 的消融显示混合模型比纯 Mamba 高约 0.15 dB、比纯 Transformer 高约 0.12 dB（待核实；三种变体的训练设置和参数量未查到），这与上述分工解释相符（推断）。本仓库 README 中，混合模型在相同 epoch 下比纯注意力的 LF_DET_0214 高 0.10 dB（同分布 val_all；学习率和推理分块不同，按代码粗算 LF_DET_0214 的每步算力约为混合模型的 3 倍，不是严格对照）。
2. **Mamba 与 Transformer 的优劣**：Mamba 的核心优势是线性时间和常数大小的状态，但墙钟时间上大约要到 2K token 以上才真正比 FlashAttention 快（待核实，测于语言模型式负载）。代价是有损的固定状态（复制和检索能力弱）、因果扫描带来的顺序伪影，以及"局部像素遗忘"。二者结合的优势是让便宜的扫描承担大范围传播、让少量注意力承担精确匹配；但在 LFTransMamba 里注意力只在单个视图内，跨视图对应完全由扫描承担。
3. **迭代次数假设**：用户的假设可以概括为"原料对了，因果链需要改写"。超分确实是纯 L1 回归；单图超分有证据表明欠拟合是主要限制（RCAN-it），但光场超分尚无训练时长曲线，而且本仓库同时使用 35% 掩码和 DropPath 0.1 这类正则，说明作者并未把欠拟合当作唯一瓶颈。LFTransMamba 的注意力是在每个子孔径视图内对 1,024 个 token（训练时）做全局注意力，按 FLOPs 粗算约占全模型的一半；它与扫描在墙钟时间上谁更快目前没有实测（推断，待 E1 实测）。纯注意力的 LF_DET_0214 通过"视图内全局空间注意力 + 宏像素角度注意力"交替，同样在每次迭代中覆盖整个 4D 光场；Mamba 带来的是在单层内、以线性代价沿光场感知的扫描顺序对全部 25,600 个 token 做联合（而非因子分解的）混合，这能否带来收益要由等 FLOPs 对照判定（推断）。官方训练命令里，混合模型和纯注意力模型都训练 30 个 epoch；已读到的比赛规则未见训练时长限制，但开发窗口 × 团队 GPU 是真实的墙钟预算，团队把它花在了更大的模型和集成上。
4. **物理约束**："不考虑物理约束"也只对了一半。已核实的 2025 冠军 LFTransMamba 代码只用 L1 损失；在已查到的 2023–2025 冠军方案描述中也未见视差估计模块或一致性损失（2023、2024 年的损失函数未逐一核实，2026 年方案未公开）。但这些方案都把光场几何写进了 token 划分和扫描顺序，用视角重采样扩大视差分布，并用保持视差的分块推理（PSW/EPSW）。LFMamba 去掉 EPI 分支后 EPFL、INRIA 分别降 0.27、0.35 dB，但 STFgantry 反升 0.16 dB；保视差分块推理单项约 +0.2 dB（均待核实）；选视角重采样尚无公开消融。
5. **最值得做的事**：不是换一个"更快"的骨干，而是在本仓库里做三件事：等墙钟时间与等迭代次数的对照（对照组按 FLOPs 匹配）、与训练预算匹配的学习率调度、几何先验的逐项消融。做这些实验之前，先把本地 PSNR 改为 25 视角口径（见证据说明）。

阅读导航：混合架构论文与分工证据见"多篇论文指向互补分工"一节；Mamba 与 Transformer 的优劣及二者结合的优势见"Mamba 与 Transformer 的优劣"一节；用户假设的逐条检验、本仓库实验和比赛行动清单分别见"逐条检验用户假设""八个本仓库实验"和"结论"三节；全部文献列在文末"参考文献"。

*证据说明：调研环境无法直接访问 arxiv.org 和 openaccess.thecvf.com。凡是来自论文正文、经搜索引擎摘录、研究者回忆（[R]）或第三方摘要的数字，下文都标注"（待核实）"，引用前请对照 PDF。来自 GitHub README（直接抓取）和本仓库代码的数字已逐一核对，不加标注；按代码做的参数量和 FLOPs 估算标注"粗算"，需用 fvcore 实测核对。另需特别说明：本仓库 README 中的 `real_psnr / synth_psnr` 是 `val_all` 在五个标准测试集上的分组，EPFL、INRIA、STFgantry 记为 real，HCInew、HCIold 记为 synth（[lfsr.py L1043–1062](../lfsr.py)）。它反映的是同分布测试，而不是 NTIRE 验证集或测试集，不能用来衡量比赛的域差异。另外，`compute_psnr_ssim` 用 `zip(range(angular), range(angular))` 只计算 5 个对角视角（[lfsr.py L988–992](../lfsr.py)），README 中的 PSNR 是对角视角均值，不能与 BasicLFSR 的 25 视角均值直接比较；用于 0.05–0.15 dB 级别的比较时，也更容易受噪声影响（推断）。*

## 四届 NTIRE 冠军显示，混合架构从 2025 年起才领跑

用户的观察"几届比赛基本是 Mamba 结合 Transformer 效果最好"只对最近两届成立。2023 年 Mamba 还没有成为视觉骨干，冠军是 CNN 加 EPI-Transformer 的 DistgEPIT。2024 年的冠军是放大版的纯 Transformer BigEPIT。从 2025 年起，Mamba+Transformer 才占据第一，而且同一年第 3 名的 LFTramba 也是混合架构。下表汇总四届 Track 1 冠军的情况。

| 届次 | Track 1 冠军 | 主干类型 | 与架构无关的关键手段 | 测试集信息 |
|---|---|---|---|---|
| 2023 | DistgEPIT（OpenMeow） | CNN（DistgSSR 解耦）+ EPI Transformer | 中心、间隔、非均匀三种选视角方式生成 5×5 训练数据；PSW 分块推理比常规补零分窗高约 0.2 dB（[README](https://github.com/OpenMeow/NTIRE23_LFSR_DistgEPIT)） | 比 DistgSSR 高约 1 dB（待核实，[arXiv:2304.10415](https://arxiv.org/abs/2304.10415)） |
| 2024 | BigEPIT（BNU&TMU-AI-TRY） | 深度、宽度同时放大的纯 Transformer（EPIT） | 增广视角重采样；整图或大块推理；多模型集成加自集成 | 30.80 dB，11.04M 参数，569.30 GFLOPs（待核实，[NTIRE 2024 报告](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf)） |
| 2025 | LFTransMamba（OpenMeow） | Mamba + Transformer | ALL4：每个场景用四套 5-of-9 选视角方案；EPSW；3 路集成：LF_DET_0214（P16）+ 同一个 LFTransMamba checkpoint 的 P40、P48 两种分块推理；TTA；EMA（[README](../README.md)） | 比 2024 冠军高 0.36 dB；二手摘要给出约 31.16 dB（均待核实，[ECNU 记录](https://pure.ecnu.edu.cn/en/publications/ntire-2025-challenge-on-light-field-image-super-resolution-method/)、[摘要](https://mlanthology.org/cvprw/2025/jin2025cvprw-lftransmamba/)） |
| 2026 | OpenMeow（三个赛道全部第一） | 报告文字描述为 LFTransMamba 式放大，K=48、C=160、T=3（摘录残缺，待核实） | 方法与代码未公开（[团队主页](https://github.com/OpenMeow/.github)） | 95.20M 参数、2733.5 GFLOPs；比 2025 冠军高 0.2 dB（待核实，[NTIRE 2026 报告](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Wang_NTIRE_2026_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2026_paper.pdf)） |

若测试集沿用（推断，依据是 2024–2026 年 README 使用同一份验证基线表和同一下载链接，且各年报告都以"比上届冠军高多少"来表述），历届 Track 1 冠军成绩约为 30.66 → 30.80 → 31.16 → 31.36 dB（推断，待核实）。每年增幅为 +0.14、+0.36、+0.2 dB，在收窄；2026 年的提升伴随着约 9 倍于 2025 年 Track 1 模型的参数量（95.2M）。

冠军表还有两个容易被忽略的信号。第一，**2024、2025 年的冠军都同时用了集成、TTA、扩大视差分布的视角重采样和保持视差的分块推理；2023 年的 DistgEPIT 有视角重采样、PSW 和 TTA（未见集成记录）；2026 年冠军方案未公开**。这些手段与骨干选择无关。BigEPIT 明确指出，补零和分窗会"破坏光场视差结构"，并建议用整图或更大的块推理（待核实，[BigEPIT](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chao_BigEPIT_Scaling_EPIT_for_Light_Field_Image_Super-Resolution_CVPRW_2024_paper.pdf)）。GTF 的消融里，EPSW 单项带来 +0.23 dB，比 TTA 的 +0.12 dB 还大（待核实，[arXiv:2605.04581](https://arxiv.org/abs/2605.04581)）。

第二，**2025 年的效率赛道（≤1M 参数、≤20 GFLOPs）里，OpenMeow 提交的是 T=0 的纯 Mamba 模型**（`LFTransMamba_D16_R1_K6_C32_T0`，checkpoint 名为 L2FMamba），获得第 2 名（[README](../README.md)）。按 FLOPs 估算（C=32 时一层视图内全局注意力约 4 GF，约占 20 GF 上限的 20%），Mamba 的线性复杂度很可能在这里起了作用，而且起作用的是推理 FLOPs 上限，不是训练迭代次数（推断，团队未说明理由）。但这也可能只是沿用团队已发表的纯 Mamba 模型 L²FMamba；而且 2024 年效率赛道第 2 名 LGFN 在 19.33 GFLOPs 内仍使用了全局注意力（待核实，[arXiv:2409.17759](https://arxiv.org/abs/2409.17759)），说明上限并不排除少量注意力。

Mamba 出现之前的格局同样值得参考。在 NTIRE-2023 测试集上，两个约 1.5–1.7M 参数的光场专用 Transformer 排在前列：LF-DET 29.944 dB，EPIT 29.874 dB，都高于 13.87M 参数的 HLFSR-SSR（29.830 dB）和 DistgSSR（29.637 dB）（[BasicLFSR README](https://github.com/ZhengyuLiang24/BasicLFSR)）。标准测试集上的排名在比赛分布上基本保持不变。它们的注意力都限制在光场子空间里：LF-DET 是空间/角度子空间，EPIT 是 EPI。也就是说，4D 几何是通过网络结构编码的，而不是通过损失函数。2026 年报告指出，近期方法越来越多地采用 Mamba–Transformer 或 Mamba–CNN 混合，并以 LFTransMamba 为代表（待核实，[NTIRE 2026 报告](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Wang_NTIRE_2026_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2026_paper.pdf)）。

## 多篇论文指向互补分工：语言模型中已有因果证据，光场超分中仍是类比推断

下表先列出调研中涉及的主要混合架构论文，后面各小节是对这张表的解读。表中"机理证据"一列说明该论文是否做了能解释"为什么结合有效"的消融或分析。

| 领域 | 论文 | 混合方式与注意力位置 | 主要发现 | 机理证据 |
|---|---|---|---|---|
| 光场 | LFTransMamba（CVPRW 2025） | 层间交错：视图内全局注意力（T 层）+ 跨视图 SS2D3D/SS2D4D 扫描 | NTIRE 2025 Track 1、Track 3 第一（[README](../README.md)） | 论文消融表未取得 |
| 光场 | LFTramba（CVPRW 2025）/ LFMT（期刊版） | 空间-角度 Mamba 块 + 后段 EPI Transformer 块 | 比纯 Mamba 高约 0.15 dB、比纯 Transformer 高约 0.12 dB；LFTramba 获 2025 Track 1 第 3（待核实） | 组件消融，训练设置未取得 |
| 光场 | L²FMamba（IEEE TCI 2025） | 纯 Mamba：SAI 内、SAI 间、宏像素三段扫描；LFTransMamba 的前身 | 称参数更少、推理更快、PSNR 更高（待核实） | 逐层保留消融，数字未取得 |
| 光场 | MLFSR（ACCV 2024） | Mamba 学生 + Transformer 教师蒸馏（T2M 损失） | 效率优势随输入分辨率增大（待核实） | 无 |
| 光场 | SMART（ICME 2026） | 斜率引导 Mamba + 角度细化 Transformer | 斜率引导扫描平均约 +0.5 dB（待核实） | 组件消融 |
| 超分/复原 | MambaIRv2（CVPR 2025） | 窗口 MHSA + 注意力式 SSM（ASSM）交错 | 在窗口注意力基线上再加 0.08/0.13 dB（轻量 ×2，待核实） | 单向消融，未做"去掉注意力"一臂 |
| 超分/复原 | Contrast（2025） | 卷积 + Transformer + SSM | ×4 Urban100 27.92 dB、14.1M 参数（第三方摘要，待核实） | 评审称互补性未被消融隔离 |
| 超分/复原 | MatIR（2025） | Transformer（三角窗口 + 通道全局注意力）与 4 路扫描 Mamba 交替 | 覆盖超分、去噪、去模糊（[README](https://github.com/wenjuan7275/MatIR)） | 消融表未取得 |
| 超分/复原 | T-PMambaSR（IEEE TIP 2026） | 窗口自注意力 + 渐进 Mamba + 高频细化 | 最佳 39.20 dB 对 39.13–39.16 dB（BSD100 ×2，待核实） | 只消融扫描，未隔离注意力与 Mamba |
| 超分/复原 | RestorMixer（2025） | 最高分辨率用 CNN，低分辨率用 Mamba + 多尺度窗口 Transformer | 去雨 35.47 dB 对 Restormer 34.46 dB，参数 2.8M 对 26.1M（第三方摘要，待核实） | 无 |
| 超分/复原 | TransMamba 去雨（2024） | 双分支：频谱 Transformer + 双向 Mamba，每级通道拼接融合（[README](https://github.com/sunshangquan/TransMamba)） | — | 无 |
| 超分/复原 | Transformer→SSM 蒸馏（2026） | SSM 块蒸馏模仿 Restormer 块，再搜索混合 U-Net | 骁龙 8 Elite CPU 上比 Restormer 快 1.17–3.4 倍，质量接近（待核实） | 无 |
| 视觉 | MambaVision（CVPR 2025） | 后两个 stage 先 Mamba 混合器、后自注意力 | 自注意力放在每个 stage 后半段最好（待核实） | 位置消融 |
| 视觉 | A2Mamba（2025） | SSM 隐状态与多尺度注意力图交互 | 比 MambaVision-B 高 1.2 AP^b、参数少 40%（跨模型比较，待核实） | 无受控消融 |
| 视觉 | Heracles（2024） | 前层 SSM、深层注意力 | 与 MambaVision 同为"先 SSM 后注意力"（待核实） | 无 |
| 视频 | Vamba（ICCV 2025） | Mamba-2 编码视频 token，保留文本交叉注意力 | 每训练步快近 2 倍、显存省至少 50%；精度与 Transformer 基线持平（42.1% 对 42.0%，待核实） | 无 |
| 语言 | Jamba（ICLR 2025） | Mamba + 注意力 + MoE，1:7 | 1:3 与 1:7 几乎无差别；归纳头出现在注意力层（待核实） | 比例消融、注意力可视化 |
| 语言 | NVIDIA Mamba-2-Hybrid 8B（2024） | 43% Mamba-2、7% 注意力、50% MLP | 12 项任务全部超过 Transformer，平均 +2.65 分（待核实） | 受控对比 |
| 语言 | Samba（ICLR 2025） | Mamba + 滑窗注意力 + MLP 层间交错 | 128K 提示吞吐 3.73 倍；4K 训练零样本外推到 1M（待核实） | 无 |
| 语言 | Hymba（ICLR 2025） | 每层注意力头与 SSM 头并行；仅 3 层全注意力 | 比 Llama-3.2-3B 平均高 1.32%，缓存小 11.67 倍（待核实） | 头移除分析 |
| 语言 | Zamba / Zamba2（2024） | Mamba 主干 + 共享注意力块 | 以极小参数代价获得注意力收益（待核实） | 无 |
| 语言 | Nemotron-H（2025） | Mamba-2 + 约 8% 注意力 | 精度相当，推理最多快 3 倍（待核实） | 无 |
| 语言 | Falcon-H1（2025） | 每块内注意力与 Mamba-2 并行 | 34B 称可比肩约 70B 模型（待核实） | 被缓存干预研究用作对象 |
| 语言 | MiniMax-01 / M2（2025–2026） | 每 7 层 lightning 线性注意力接 1 层 softmax；M2 报告称改回全注意力 | 混合在大规模多跳推理上出现缺陷（说法有冲突，待核实） | 无公开实验细节 |
| 语言 | Qwen3-Next / Qwen3.5（2025–2026） | Gated DeltaNet : 门控注意力 = 3:1 | 比例无公开消融（二手来源） | 被缓存干预研究用作对象 |
| 语言 | Kimi Linear（2025） | KDA : MLA = 3:1，层间交错 | KV 缓存最多省 75%，1M 上下文解码最多快 6 倍（待核实） | 比例消融（数字未取得） |
| 语言 | Jet-Nemotron（2025） | 线性块 + 搜索出的少数全注意力层 | 对检索关键的全注意力层只有 2–3 层（待核实） | 层位置搜索 |
| 语言 | Mamba in the Llama（NeurIPS 2024） | 从 Transformer 蒸馏，保留 1/4 注意力层 | 在对话基准上追平教师（待核实） | 无 |

专门分析"为什么混合有效"的研究（理论证明、合成任务、头移除、缓存干预、补丁实验）集中在语言模型领域，见下文"通用复原给出相关性证据"一节的要点列表。

### 光场超分内部的消融显示混合模型优于单一架构

对"为什么要结合"给出最直接定量回答的，是 LFTramba 的期刊扩展版 LFMT。作者认为纯 Mamba 光场方法存在"二维子空间扫描耦合"，同一像素在不同子空间里被重复建模；同时 Mamba 有限的状态空间会漏掉一部分极平面结构，所以要在后段加上作用于 EPI 的 Transformer 块来补偿。消融中，**混合模型比纯 Mamba 高约 0.15 dB，比纯 Transformer 高约 0.12 dB**。块数增加时性能先升后降，3 个 Mamba 块加 3 个 EPI-Transformer 块最好（均待核实，[arXiv:2509.04824](https://arxiv.org/abs/2509.04824)）。

LFMamba 从反方向提供了证据。把它的 Mamba 模块换成 LFT、EPIT 或 LFT_EPIT 式 Transformer，EPFL 上分别下降 0.14、0.21、0.25 dB。直接套用 VMamba 的 SS2D 效果"次优"，需要沿光场子空间（SAI、EPI）组织扫描。去掉 EPI 分支后，EPFL 降 0.27 dB、INRIA 降 0.35 dB，但 STFgantry 反而升 0.16 dB（均待核实，[arXiv:2406.12463](https://arxiv.org/abs/2406.12463)）。

MLFSR 走的是第三条路：引入 Transformer 到 Mamba 的蒸馏损失（T2M），用 Transformer 教师把注意力学到的知识迁移给 Mamba 学生。作者报告效率优势随输入分辨率增大而扩大（待核实，[ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/papers/Gao_Mamba-based_Light_Field_Super-Resolution_with_Efficient_Subspace_Scanning_ACCV_2024_paper.pdf)）。2026 年的 SMART 让 Mamba 沿 EPI 斜率（即视差）自适应扫描，并加入角度细化 Transformer。其中斜率引导扫描平均贡献约 0.5 dB（待核实，[arXiv:2607.00965](https://arxiv.org/abs/2607.00965)）。

本仓库代码显示了 LFTransMamba 的具体分工（[models/LFTransMamba.py](../models/LFTransMamba.py)）。每个复合块依次由三个子块组成，`type_layers = [VSSBlock, VLSSBlock, LFSSBlock] * num_blocks`（L2241）。**VSSBlock** 先做 T 层视图内的全局多头自注意力：4 个头（头维度 80/4 = 20），经 `F.scaled_dot_product_attention` 调用 PyTorch 的高效注意力内核（输入为 fp16/bf16 时才可能走 FlashAttention 后端），token 是单个子孔径视图的 h×w 个像素。之后再做一次视图内的 SS2D 四向扫描。**VLSSBlock** 把特征排成 SAI 阵列布局 `(u h)(v w)`，用 SS2D4D 扫描；**LFSSBlock** 把特征排成宏像素布局 `(h u)(w v)`，用 SS2D3D 扫描。这两种扫描都覆盖 25×h×w 个 token。初始卷积（`Conv3d`，核为 (1,3,3)，L2227–2235）和所有 FFN（`Conv_Mlp` 中的深度卷积）都在单视图内进行，所以**在特征提取主干中，跨视图的信息交换只发生在 SS2D3D/SS2D4D 两种扫描中**（末端上采样的 3×3 卷积作用于 SAI 拼图，只在视图边界处混合少量相邻像素）。

损失只有 L1：训练脚本在 [lfsr.py](../lfsr.py) L853 创建 `nn.L1Loss()`，并在 `optimize()`（L818）中使用；模型文件中的 `get_loss`（L2415）同样是 L1，但训练脚本没有调用它。MLFIM 在训练时用可学习的 mask token 随机替换 35% 的浅层特征 token（L2382）；论文的二手摘要写的是 25%，代码与论文可能不一致（待核实）。另一个对后文很重要的细节是：TransBlock **没有相对位置偏置，也没有显式位置编码**，位置信息来自 FFN 中的深度卷积、SSM 的衰减和逐视角嵌入 `ang_embed`。这使得视图内注意力可以直接使用高效注意力内核。

LFTransMamba 与 LFMT 的切分方式正好相反。前者把注意力放在视图内的空间维度，把 Mamba 放在跨视图的 4D 维度；后者把 Mamba 放在空间-角度维度，把注意力放在 EPI 上。前者赢得 2025 年 Track 1；后者的会议版 LFTramba 获同届 Track 1 第 3，期刊版 LFMT 报告 ×2 时比 LF-DET 高约 0.18 dB（待核实）。这与**"两种归纳偏置的混合加上按光场子空间划分 token"**的解释相符，而不指向某一种固定的切分方式（推断：仅两例，且没有等参数、等 FLOPs 的对照）。

本地也有一组等 epoch 的对照：LF_DET_0214（纯注意力，160 通道）与混合模型都用 batch 3、训练 30 epoch。在五个标准测试集上，混合模型平均高 0.10 dB（33.8803 对 33.7795），其中 real 组高 0.14 dB，synth 组高 0.05 dB（[README](../README.md)）。这组对照有混杂因素：两者学习率不同（3e-4 对 2e-4），推理分块不同（40 对 16），评估集是同分布的 val_all 且只算 5 个对角视角，参数量也没有列出。按代码粗算，LF_DET_0214（160 通道、48 个 MLP 比为 8 的视图内空间注意力块加 12 个角度注意力块）约 30M 参数、每个 5×5×32×32 样本前向约 2.4 TFLOPs，约为混合模型（约 9M、约 0.8 TFLOPs）的 3 倍。也就是说，等 epoch 不等于等算力，混合模型是以约 1/3 的每步算力胜出 0.10 dB（粗算，需 fvcore 实测核对）。

### 通用复原给出相关性证据，语言模型已用因果干预验证这种分工

在单图超分领域，最具代表性的 Mamba 模型之一 MambaIRv2 本身就是混合架构（Hi-Mamba 报告在 Urban100 上比它高 0.2–0.27 dB，待核实，[Hi-Mamba](https://arxiv.org/pdf/2410.10140)）。作者说，ASSM 只需扫描一次，省下的参数预算可以用来"加入窗口多头自注意力以加强局部交互"。轻量版 ×2 的消融中，只用 MHSA 时 Urban100/Manga109 为 32.89/39.11 dB，完整模型为 32.97/39.24 dB，即注意力 SSM 在窗口注意力基线上再加 0.08/0.13 dB（待核实，[MambaIRv2](https://arxiv.org/html/2411.15269v2)）。Contrast 给出的理由最直白：Transformer 的二次复杂度迫使它只能用窗口注意力，限制了上下文；Mamba 虽然是线性的，但其"近似的隐状态会丢失超分所需的像素级精度"（[arXiv:2501.13353](https://arxiv.org/abs/2501.13353)）。第三方评审也指出，这种互补性在 Contrast 中"没有被消融单独隔离"（待核实，[Pith](https://pith.science/paper/2501.13353)）。

NTIRE 的通用超分赛道把同一思路推到了集成层面。2025 年 ×4 赛道中 BBox 把 HAT-L 与 MambaIRv2-L 集成；2026 年至少一支队伍把 HAT 与 MambaIRv2 做无需训练的输出融合，在 DIV2K 上同时超过了 HAT 单模型和 MambaIRv2 单模型（队伍总数未核实；待核实，[arXiv:2504.14582](https://arxiv.org/pdf/2504.14582)、[arXiv:2604.11564](https://arxiv.org/pdf/2604.11564)）。这间接提示两类模型的误差可能部分不相关。不过 2026 年报告的结论仍是"预训练的 Transformer 复原模型仍是高保真超分最强的基础"（待核实，[arXiv:2604.14558](https://arxiv.org/html/2604.14558v1)）。

语言模型领域的证据最系统，而且已经从相关性推进到了因果干预。下面按分析方法列出专门解释"为什么混合有效"的研究：

- **受控对比**：NVIDIA 在 8B 规模、最多 3.5T token、相同超参下对比了纯 Mamba、Mamba-2、Transformer 和混合模型。纯 SSM 在五样本 MMLU、电话簿查找这类需要复制或上下文学习的任务上落后。由 43% Mamba-2、7% 注意力、50% MLP 组成的混合模型在 12 项标准任务上全部超过 Transformer，平均高 2.65 分（待核实，[arXiv:2406.07887](https://arxiv.org/abs/2406.07887)）。
- **理论证明**：Wen 等证明，状态为亚线性大小的 RNN（含 SSM）无法完成联想回忆等上下文检索任务；加入检索增强或一层 Transformer 后，RNN 就能在思维链下解决所有多项式时间问题。这是"少量注意力就够"的理论依据（待核实，ICLR 2025，[arXiv:2402.18510](https://arxiv.org/abs/2402.18510)）。Zoology 发现，门控卷积类模型与注意力之间困惑度差距的 82% 可以用联想回忆解释（待核实，[arXiv:2312.04927](https://arxiv.org/abs/2312.04927)）。
- **合成任务**：MambaFormer 的研究发现，Mamba 在稀疏奇偶任务上胜过 Transformer，却学不会非标准检索；混合模型两类任务都能完成，这是"失败模式互补"最直接的证据（待核实，ICML 2024，[arXiv:2402.04248](https://arxiv.org/abs/2402.04248)）。MAD 训练了 500 多个模型，交错式混合在计算最优和过训练两种扩展设置下都优于 Transformer++、Hyena 和 Mamba，且合成任务成绩与扩展后的困惑度相关（待核实，ICML 2024，[arXiv:2403.17844](https://arxiv.org/abs/2403.17844)）。
- **头移除与可视化**：Bick 等人发现，检索依赖一个"Gather-and-Aggregate"两段式电路，SSM 中这类头的注意力模式过于平滑。**预训练混合模型会把 Aggregate 这一步交给注意力层**；剪枝后的 Llama-3.1-8B 中，去掉一个这样的头，MMLU 就从 66% 跌到 25%（待核实，[arXiv:2504.18574](https://arxiv.org/abs/2504.18574)）。Hymba 的头移除分析显示，SSM 头（尤其第一层）对基本功能不可或缺，注意力头的重要性随输入而变，符合"SSM 是常驻摘要器、注意力按需做高分辨率召回"的说法（待核实，[arXiv:2411.13676](https://arxiv.org/pdf/2411.13676)）。Jamba 观察到归纳头出现在混合模型的注意力层中，而纯 Mamba 做不好上下文学习（待核实，[arXiv:2403.19887](https://arxiv.org/abs/2403.19887)）。
- **缓存级干预与补丁实验**：2026 年的缓存级干预研究在 Qwen3.5 和 Falcon-H1 上做了更直接的验证。只保留 KV cache 时，精确检索仍能达到完整精度的 64–98%；只保留循环状态时为 0；而语言和人设信息存放在循环状态里（预印本，待核实，[arXiv:2609.04434](https://arxiv.org/abs/2609.04434)）。另一项补丁研究（12 个稠密和混合模型）发现，在其中四个混合模型里，循环层负责把"键"传给注意力层，再由少数注意力头解析出答案（预印本，待核实，[arXiv:2609.32942](https://arxiv.org/abs/2609.32942)）。
- **混合方式对比**：一项研究发现串行混合在短上下文更好，并行混合在长上下文更好（待核实，[arXiv:2510.26912](https://arxiv.org/abs/2510.26912)）。2026 年一项研究发现"大窗口惰性"：滑窗越大（廉价的局部路径越强），全注意力层越晚形成检索头（待核实，[arXiv:2606.15378](https://arxiv.org/abs/2606.15378)）。

这些研究还给出了配比和位置上的经验。Jamba 发现注意力与 Mamba 的比例取 1:3 或 1:7"几乎没有性能差别"，于是选了更省算力的 1:7（待核实，[arXiv:2403.19887](https://arxiv.org/abs/2403.19887)）。Jet-Nemotron 发现对检索真正关键的全注意力层只有 2–3 层（待核实，[arXiv:2508.15884](https://arxiv.org/html/2508.15884v1)）。一项系统研究训练了 72 个模型，结论是对 HGRN-2 或 GatedDeltaNet，线性层与全注意力层的比例取 3:1 到 6:1 最好（比例对语言建模和回忆的影响不同），而且"单独表现更好的线性模型不一定是更好的混合组件"（待核实，[arXiv:2507.06457](https://arxiv.org/abs/2507.06457)）。视觉方面，MambaVision 的位置消融显示，把自注意力放在每个 stage 的后半段，ImageNet top-1 为 82.3%；放在前半段为 81.5%，交错放置为 81.4–81.6%，随机放置为 81.3%（待核实，[arXiv:2407.08083](https://arxiv.org/abs/2407.08083)）。A2Mamba 的作者认为以往混合只是简单堆叠，提出让 SSM 隐状态与多尺度注意力图交互；A2Mamba-S 在 COCO 上比 MambaVision-B 高 1.2 AP^b、参数少 40%（跨模型比较，并非受控消融；待核实，[arXiv:2507.16624](https://arxiv.org/abs/2507.16624)）。

这些研究汇总起来就是一句话：**循环负责携带和压缩信息，注意力负责精确查找**。放到光场超分里，这一分工在两类设计中的落点不同。LFMT/LFTramba 把注意力放在 EPI 上，注意力确实承担跨视图的精确匹配。LFTransMamba 的注意力严格限于单个子孔径视图内（token 为单视图的 h×w），只能做视图内的非局部自相似匹配；跨视图对应完全依赖有损的 SS2D3D/SS2D4D 扫描。按 LLM 文献，这正是固定状态最弱的环节（类比推断，未在光场数据上测量）。另外，"大窗口惰性"提示：VSSBlock 中注意力之后还接一次视图内 SS2D 和 ConvFFN，这些廉价路径可能与注意力争夺同一项工作，两者的配比值得消融（类比推断，见 E8）。

### 反面证据提醒：增益不大，而且常常没有被干净地隔离

混合架构并非处处占优。MiniMax-M2 报告称已放弃 lightning 注意力混合、改回全注意力，原因是混合模型在标准基准上与全注意力持平，但"在更大规模上，复杂多跳推理出现明显缺陷"（另有第三方来源称 M2.x 仍为混合架构，说法冲突；MiniMax 的原始工程博文未能取得；待核实，[arXiv:2605.26494](https://arxiv.org/pdf/2605.26494)）。MambaOut 证明，去掉 SSM 的门控 CNN 在 ImageNet 分类上胜过所有视觉 Mamba，但在约 4K token 的检测和分割任务上追不上（[MambaOut README](https://github.com/yuweihao/MambaOut)）。2026 年一项研究发现，不同混合方案只要训练足够长，长上下文能力最终会收敛到相近水平，高效模块主要影响这种能力出现得多快（待核实，[arXiv:2606.15378](https://arxiv.org/abs/2606.15378)）。

在超分领域，至今找不到一篇在参数和 FLOPs 匹配条件下、做了"注意力开/关 × Mamba 开/关"完整 2×2 消融的论文。MaIR 的 0.03–0.07 dB 消融差距被评审认为统计上很脆弱（待核实，[Pith 评审](https://pith.science/paper/2412.20066)）。LFMT 的 +0.12 dB 和本地的 +0.10 dB，与 NTIRE 冠军相对上一届冠军的提升幅度（0.14–0.36 dB，待核实）处于同一量级。这正是比赛中这些增益值得争取的原因；这也意味着单次实验的随机种子波动可能与之相当（本仓库没有多种子数据，需要实测）。

## Mamba 与 Transformer 的优劣：线性复杂度换来的是长序列能力，约 2K token 以上才变成真实速度

全面比较两者，可以先用两组要点概括。

**Mamba 的优势**：

- 线性时间、推理状态 O(1)；高分辨率和长序列下省显存、提速（见下文 Vim 的数字），生成吞吐量高 4–5 倍（待核实）。
- 数据依赖的衰减带来局部性和新近性先验。
- 相近计算量下获得全局感受野：MambaIR 比 SwinIR 最多高 0.36 dB（[MambaIR README](https://github.com/csguoh/MambaIR)）。
- 规模上有竞争力：Mamba-3B 优于同尺寸 Transformer，并追平两倍尺寸的模型（待核实，[Mamba](https://arxiv.org/abs/2312.00752)）。
- 部分任务能长度泛化：归纳头任务上能外推到训练长度约 4000 倍；flip-flop 等星自由语言上能长度泛化（待核实）。

**Mamba 的劣势**：

- 固定大小的状态有损，精确复制和检索能力弱。
- 因果扫描带来顺序依赖、十字形有效感受野和局部像素遗忘。
- 短序列（约 2K token 以下）墙钟时间不占优，多向扫描还受访存限制。
- 有效感受野受训练长度约束。
- 训练对精度和初始化敏感。

机制上可以归结为三个差异。SSD 理论把两者统一为带掩码的序列混合矩阵 M = L∘(CBᵀ)，SSM 对应半可分矩阵加数据依赖的累积衰减掩码（待核实，[arXiv:2405.21060](https://arxiv.org/abs/2405.21060)）。据此可归纳出三点差异（推断）：秩（softmax 注意力的混合矩阵满秩，SSM 的秩不超过状态维度 N）、尖锐度（softmax 经过归一化和指数化，选择接近 one-hot；SSM 的混合较平滑）、掩码结构（SSM 带因果的累积衰减，注意力默认没有）。"Demystify Mamba"把 Mamba 改写成带输入门、遗忘门、捷径、无归一化、单头的线性注意力，并在 ImageNet 上证明遗忘门和整体块设计是 Mamba 的两大贡献来源（待核实，[arXiv:2405.16605](https://arxiv.org/abs/2405.16605)）。下表把这些差异与光场超分对应起来。

| 维度 | Mamba（选择性 SSM） | Softmax 注意力 | 对光场超分的含义 |
|---|---|---|---|
| 渐近复杂度 | 时间 O(L)，推理状态 O(1) | 计算 O(L²)，KV cache O(L) | 单层联合的全 4D 注意力（训练 5×5×32×32 = 25,600 token）只有线性方法负担得起；但因子分解的空间/角度/EPI 注意力（LF-DET、EPIT）能以很小代价获得全 4D 感受野（推断） |
| 墙钟速度 | 融合扫描核在约 2K token 以上才快于 FlashAttention-2；生成吞吐量高 4–5 倍（待核实，[Mamba](https://arxiv.org/abs/2312.00752)）；视觉多向扫描受访存限制 | ≤2K token 时 FlashAttention 同样快或更快 | 按 MambaOut 公式粗算，Track 1 混合模型中 36 个视图内 TransBlock 约占前向 FLOPs 的 54%（二次项约 37%），是全模型最大的单项开销；墙钟占比未测量，需由 E1 用 torch.profiler 分模块计时；生成吞吐量优势与超分无关 |
| 精确检索/复制 | 固定状态有损，容量随状态维度增长（多条目联合回忆要求状态随条目数线性增长；单条 passkey 检索容量随状态维度指数增长，待核实，[HAX](https://arxiv.org/abs/2507.00449)、[Stuffed Mamba](https://arxiv.org/abs/2410.07145)）；复制任务约需多 10 倍数据（待核实，[Jelassi](https://arxiv.org/abs/2402.01032)） | 缓存无损；2 层即可复制指数长度的字符串（待核实） | 视图内非局部自相似交给注意力；跨视图对应在 LFTransMamba 中由扫描承担，在 LFMT 中由 EPI 注意力承担（推断） |
| 归纳偏置 | 数据依赖的衰减相当于局部性和新近性先验；因果，结果依赖扫描顺序 | 置换等变，需要位置信息，没有局部先验 | SAI、宏像素、EPI 的扫描顺序本身就是一种几何先验 |
| 视觉伪影 | 十字形有效感受野、局部像素遗忘、通道冗余、背景高范数伪影 token | 没有顺序伪影 | 需要卷积、多向扫描或注意力来补偿 |
| 长度外推 | 有效感受野受训练长度约束；相对训练长度而言状态维度过大时，超出训练长度会出现状态塌缩（所需最短训练长度随状态维度线性增长）（待核实，[DeciMamba](https://arxiv.org/abs/2406.14528)、[Stuffed Mamba](https://arxiv.org/abs/2410.07145)） | 受位置编码限制 | 训练用 32×32、推理用 40/48 分块时需要验证 |
| 状态追踪 | 与 Transformer 同属 TC⁰，非负特征值无法解奇偶；但对 flip-flop、星自由语言能长度泛化（待核实） | 同样无法解 S5 置换组合 | 对低层视觉可能基本无关（文献未讨论，推断） |
| 训练稳定性 | 官方 README 提示 SSM 对循环动态敏感，出现不稳定时应先以 fp32 存储主模型参数（如 AMP）；框架若把所有 Linear 偏置置零会破坏 Δ 偏置初始化（[mamba README](https://github.com/state-spaces/mamba)）；纯 Vision Mamba 监督训练放大时容易饱和甚至崩溃（待核实，[ARM](https://arxiv.org/abs/2406.07537)）；在混合精度加 PEFT 微调下比 Transformer 更稳（待核实，[arXiv:2406.00209](https://arxiv.org/abs/2406.00209)） | 训练配方成熟 | 混合模型通常更容易优化（推断） |

Mamba 在速度上的优势，取决于序列长度和工程实现。Mamba-1 论文给出的交叉点是：序列长于约 2K 时，融合扫描才比 FlashAttention-2 快。这个测量条件是 A100、batch 1、模型维度 1024、因果设置（待核实，[Mamba](https://arxiv.org/abs/2312.00752)）。Mamba-2 的 SSD 内核同样从 2K 起更快，但作者提醒，完整的 Mamba-2 模型在 2K 这种短序列上的训练效率"可能不如 Transformer"（待核实，[Mamba-2](https://arxiv.org/abs/2405.21060)、[Tri Dao 博客](https://tridao.me/blog/2024/mamba2-part4-systems/)）。Vim 的优势随分辨率增大：README 给出的数字是在 1248×1248（6,084 token）时比 DeiT 快 2.8 倍、省 86.8% 显存；低分辨率下的对比数据未能检索到（[Vim README](https://github.com/hustvl/Vim)）。VMamba 要靠 Triton 交叉扫描、fp16 输入/fp32 输出的内核、把 einsum 换成线性层等一系列工程手段，吞吐量才超过 Swin（待核实，[arXiv:2401.10166](https://arxiv.org/abs/2401.10166)）。往线性注意力里加入 Mamba 式遗忘门后，精度从 77.6% 升到 78.4%，但吞吐量从 1152 降到 743（待核实，[arXiv:2405.16605](https://arxiv.org/abs/2405.16605)）。

复原领域的论文也普遍把 Mamba 实际速度不及预期归因于"SSM 内核优化尚处早期"，以及多方向扫描带来的开销。EAMamba 用多头选择性扫描（MHSSM）聚合多条扫描序列，并配合全方位扫描，报告 FLOPs 比现有低层视觉 Mamba 方法低 31–89%（[EAMamba README](https://github.com/daidaijr/EAMamba)）。连复杂度统计本身也要小心：MambaIR 早期用 `thop` 统计参数和 MACs 时漏算了参数，改用 fvcore 后数值更大（[MambaIR README](https://github.com/csguoh/MambaIR)）。

按 MambaOut 的 FLOPs 公式（MLP 比为 4 的 Transformer 块约为 24D²L + 4DL²，当 L > 6D 时二次项占主导，[arXiv:2405.07992](https://arxiv.org/abs/2405.07992)；若按 2DL² 计注意力 FLOPs，阈值变为 12D = 960，见 [MambaOut issue #205](https://github.com/yuweihao/MambaOut/issues/205)），代入 Track 1 配置的 D = C = 80，阈值为 480 个 token。下表逐项计算了 LFTransMamba 相关序列长度下单个 Transformer 块（注意力 + MLP 比 4，与 LFTransMamba 的 TransBlock 一致）的前向 FLOPs（推断性算术）。

| 序列 | L | L/(6D) | 二次项 4DL² | 线性项 24D²L |
|---|---|---|---|---|
| HAT 式 16×16 窗口 | 256 | 0.53 | 0.021 GF | 0.039 GF |
| LFTransMamba 视图内（训练 32×32） | 1,024 | 2.1 | 0.34 GF（25 个视图合计 8.4 GF） | 0.16 GF |
| 视图内（推理 48×48） | 2,304 | 4.8 | 1.70 GF | 0.35 GF |
| 假想的全 4D 注意力（训练 5×5×32×32） | 25,600 | 53 | 210 GF | 3.9 GF |
| 假想的全 4D 注意力（推理 5×5×48×48） | 57,600 | 120 | 1,062 GF | 8.8 GF |

表中结果需要分训练和推理两种情况看。训练时（32×32，1,024 token）视图内注意力从 FLOPs 角度看已算"长序列"，但仍低于约 2K 的内核交叉点；推理用 40×40、48×48 分块时为 1,600、2,304 token，后者已略超过该点。交叉点本身是在语言模型式负载上测得的（batch 1、模型维度 1024、因果），视觉负载下"大 batch × 短序列 × 四向扫描"的交叉点目前没有公开测量。此外，Track 1 以 fp32 训练（训练命令没有 `--fp16`），而 PyTorch SDPA 的 FlashAttention 后端只支持 fp16/bf16，此时会退回 memory-efficient 或 math 内核；头维度也只有 20。因此视图内用 Mamba 替换注意力能否更快，目前无法判断，需由 E1 在 fp32 与 AMP 两种精度下、强制指定 SDPA 后端分别实测。

另一方面，按上表粗算，Track 1 混合模型中 36 个视图内 TransBlock 每样本前向约 440 GF，约占全模型（约 0.8 TFLOPs）的 54%，其中二次项约 300 GF、约占 37%，是全模型最大的单项开销（粗算，需实测）。

跨视图的全 4D 混合则不同：单层二次项高达 210 GF，没有 FlashAttention 时，4 个头的 fp16 注意力矩阵单样本单层约需 5.2 GB 显存。若要在单层内对全部 token 做联合的全局混合，线性扫描是唯一负担得起的方式；但因子分解的注意力（如 LF_DET_0214 的视图内空间注意力加宏像素角度注意力）也能以很小代价覆盖全 4D 感受野。LFTransMamba 的分工是：注意力只用在视图内（1,024–2,304 token），跨视图混合只用扫描（25,600–57,600 token），而视图内同时也做一次 SS2D 扫描。**两者结合的优势可以概括为：每一层都同时拥有"跨视图的联合扫描"和"视图内的精确匹配"；跨视图混合的代价是线性的，但视图内注意力仍是 O(L²)，并占约一半 FLOPs（推断）。** LLM 文献中"少量注意力就够"的结论（7% 注意力层、2–3 个关键层）也提示，T 的取值可能还有压缩空间。

### 二者结合的优势（小结）

1. **分工**：SSM 压缩并传播上下文，注意力做按内容寻址的精确匹配。语言模型中已有因果证据（Bick 等、arXiv:2609.04434、arXiv:2609.32942，待核实）；光场上属于类比推断，而且在 LFTransMamba 中注意力只做视图内匹配，跨视图对应由扫描承担。
2. **复杂度分配**：把二次代价的注意力放在序列短的地方（LFTransMamba 的视图内，训练时 1,024 token），把线性扫描放在序列长的地方（全 4D，25,600 token）。跨视图部分的代价是线性的，但视图内注意力仍是 O(L²)（代码事实加粗算）。
3. **少量注意力即够**：NVIDIA 混合模型只有 7% 注意力层，Jet-Nemotron 只有 2–3 个关键全注意力层（待核实）。光场上尚无对应测量，T 的取值可以消融。
4. **位置先验**：SSM 的衰减自带位置和新近性信息，注意力可以不加位置编码（Jamba 无显式位置编码，待核实）。LFTransMamba 的 TransBlock 也没有相对位置偏置，因而可以直接调用 SDPA 高效内核（代码事实）。
5. **集成多样性**：两类模型的误差可能部分不相关，异构集成收益更大（NTIRE 通用超分赛道的 HAT + MambaIRv2 融合，间接证据，待核实）。LFTransMamba 的 Track 1 提交也把纯注意力的 LF_DET_0214 与混合模型集成。

## 逐条检验用户假设：原料正确，因果链需要改写

下表把用户的推理链拆成七个环节，逐一给出判定和证据。

| 用户论断 | 判定 | 关键证据 |
|---|---|---|
| ① 测试集与训练集的光场相机参数 gap 很大 | 部分正确 | 验证集和测试集的一半是 Lytro ILLUM 实拍，与训练集中的 EPFL、INRIA 是同一相机系列；另一半是 3DS MAX 渲染、共面虚拟相机，视差范围未公开（[BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)、[NTIRE2026 README](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2026)）；合成半在 2024–2026 验证集上比实拍半低约 1 dB（DistgSSR 31.23 对 32.26，EPIT 31.53 对 32.54），但在 NTIRE-2023 测试集上两半基本持平（LF-DET 29.911 对 29.976）。半区 PSNR 差反映的是内容难度，不等于训练-测试分布差；而且标准测试集上的方法排名在 NTIRE-2023 测试集上基本不变，说明这种差异没有改变架构排序 |
| ② 因此训练时不考虑物理约束，让模型纯学超分即可 | 损失层面正确，架构和数据层面错误 | LFTransMamba 只用 L1（代码核实）；其他届未见显式物理损失（未逐一核实）；但已知的冠军方案都把 SAI/宏像素/EPI 结构写进网络，用选视角重采样扩大视差分布，并用 PSW/EPSW 做保持视差的推理；LFMamba 去掉 EPI 分支在 EPFL、INRIA 上损失 0.27–0.35 dB，但在大基线的 STFgantry 上反而 +0.16 dB（待核实） |
| ③ 纯回归需要足够多的迭代轮次 | 单图超分中基本正确，光场超分未测 | 完整配方（LAMB、大 batch、SiLU、长训）在 Manga109 ×2 上最多带来 0.44 dB，其中可归于加长训练的部分约 0.1 dB（与 SiLU 合计，待核实）；本仓库 Track 1 训练 30 个 ALL4 epoch、batch 3（ALL4 含 4 套选视角副本，实际步数需打印核实），同时使用 35% 掩码和 DropPath 0.1 这类正则 |
| ④ 只用 Transformer 时单位时间计算量过大 | 对窗口注意力不成立；对视图内注意力尚无定论（FLOPs 上二次项已占主导，墙钟未测）；对联合全 4D 注意力成立 | 窗口注意力 L/(6D) < 1；视图内训练时 1,024 token、推理 48 分块时 2,304 token，内核交叉点约 2K；全 4D 注意力单层约 210 GF。本仓库这一对模型中，纯注意力模型的每步算力确实约为混合模型的 3 倍（粗算），用户直觉在此成立，但原因是宽度、深度与 MLP 比，而非注意力本身 |
| ⑤ Mamba 的线性复杂度让单位时间内迭代轮次足够多 | 光场场景下没有直接证据，其他领域的证据不一致；"每步更快"版本在本地未被利用，"每步学得更多"版本尚未排除 | 官方命令中混合模型与 LF_DET_0214 同为 30 epoch：混合模型有每步更便宜的余量，但团队没有用它多跑迭代；没有任何光场超分的墙钟对比；视觉扫描受访存限制；其他领域的速度证据见下文 |
| ⑥ 再结合 Transformer 保留注意力优势，所以效果不减 | 结论正确，但机制不同：效果是增加而非持平；本地对照表明在相同 epoch 下也是增加，LFMT 的训练设置未核实 | LFMT 为 +0.15/+0.12 dB（待核实）；本地等 epoch 对照为 +0.10 dB（非等算力） |
| 隐含前提：比赛限制了训练时长 | 规则层面不成立，实质上部分成立 | 已读到的官方 README 规则只限制推理阶段的参数量和 FLOPs（效率赛道 ≤1M、≤20 GFLOPs，TTA 计入）以及可用数据（Track 3 允许外部数据），未见训练时长限制（[NTIRE 2025 README](https://raw.githubusercontent.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/343a275/README.md)；CodaLab 条款页未能访问；有二手摘录提到 2025 年可能另有参数/FLOPs 上限，未核实）。但开发窗口（2026 年：1 月 20 日发布训练数据，3 月 17 日截止，约 8 周）× 团队 GPU 是真实的墙钟预算，用户的"单位时间"在这个意义上存在。团队把这份预算花在了更大的模型（2026 冠军 2733.5 GFLOPs，2024 纯 Transformer 冠军 569.30 GFLOPs，均待核实）和多模型集成上，而不是用更便宜的骨干多跑迭代 |

**关于域差异与物理约束（①②）**：比赛中的差异更可能来自合成渲染器和视差分布，而不是相机类型整体不同（推断：合成半的视差范围未公开）。冠军们应对这种差异的方式是让模型接触更多几何变化，而不是更少。BigEPIT 指出现有光场数据的视差分布不均衡，大视差样本很少；2024 年报告也说明，增广采样（中心、均匀、非均匀）显式增加了大视差样本，提高了对视差变化的鲁棒性（待核实，[NTIRE 2024 报告](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf)）。本仓库的 `LFSR.ALL4` 正是这样做的：同一场景用 matlab 默认中心视角、13579、24568、12589 四套 5-of-9 方案各生成一遍。其中 13579 等于把基线和视差都放大一倍（[lfsr.py L44–106](../lfsr.py)）。`parse_mat` 里另有一段只在读入 9×9 的 .mat 时才触发的随机选视角逻辑（[lfsr.py L241–252](../lfsr.py)）；官方 LFSR.ALL4 读取的是预生成的 5×5 数据（目录名为 `SR_5x5_4x_128_32_*`），这段逻辑在官方训练中不会生效。

因此，用户观点中可以成立的部分是：**显式物理模块（视差估计、warping、角度一致性损失）在已查到的冠军方案中都没有出现，至少不是夺冠的必要条件**。SkimLFSR 的作者也认为，在具备视差感知能力的结构上，网络可以从纯回归任务中隐式学到场景深度和相机配置（待核实，[arXiv:2407.15329](https://arxiv.org/abs/2407.15329)）。但"纯回归"只在损失函数层面成立，网络结构本身是很强的光场几何先验。用户担心的"相机参数 gap"也有可以成立的一面：LFMamba 去掉 EPI 分支后，在大基线的 STFgantry 上反而 +0.16 dB（待核实），提示固定的 EPI 设计可能与大视差数据不匹配（推断）；这与 SMART、SkimLFSR 等视差自适应设计的动机相符。2025 到 2026 年的研究方向也在朝更显式的几何发展：SMART 的斜率引导扫描、GTF 的对角 EPI，以及 OpenMeow 团队自己的后续工作 RASLF 的"全景极平面表示"（待核实，[arXiv:2603.16243](https://arxiv.org/abs/2603.16243)）。

另外，Track 3 模型（K=24、加入外部数据 DLFD/SLFD、fp16、推理步长 8）在同分布的 val_all 上只比 Track 1 模型高约 0.05 dB（33.9310 对 33.8803）。这个差值混合了容量和数据两个因素，也不反映比赛的域差异，所以既不能说明外部数据对域差异无效，也不能说明有效。更相关的是二手摘要给出的测试集 31.16→31.22 dB（集成对集成，待核实）；外部数据对合成半的作用，需要在 NTIRE 验证集的实拍半和合成半上分别评估（见 E5 和评估协议）。

**关于迭代次数（③④⑤）**：Revisiting RCAN 指出，"欠拟合仍是限制模型能力的主要问题……增加训练迭代明显提升性能，而正则化一般会降低结果"（[rcan-it README](https://github.com/zudi-lin/rcan-it)）。改进配方后，RCAN 在 Manga109 ×2 上提升 0.44 dB，其中可归于 SiLU 加长训练的部分约 0.1 dB（待核实，[arXiv:2201.11279](https://arxiv.org/abs/2201.11279)）。所以在单图超分中，"需要足够多迭代"是对的；光场超分尚无公开的 PSNR–训练时长曲线。但是 SOTA 超分论文在比较架构时迭代数几乎都是对齐的：MambaIRv2 与 DAT 的 ×4 配置都是 250k 迭代，学习率里程碑完全相同（[MambaIRv2 配置](https://github.com/csguoh/MambaIR/blob/main/options/train/mambairv2/train_MambaIRv2_SR_x4.yml)、[DAT 配置](https://github.com/zhengchen1999/DAT/blob/main/options/Train/train_DAT_x4.yml)）。

等迭代下增益没有消失，只能否定"每步更快、所以能多跑迭代"这一版本；它不能否定"混合模型每步学到更多（样本效率更高），所以在有限预算内领先"这一版本。后者同样是"迭代/预算决定结果"的一种形式，而 LLM 研究恰好发现高效模块主要决定能力出现的快慢、长训后会收敛（待核实，[arXiv:2606.15378](https://arxiv.org/abs/2606.15378)）。区分两者要看 E2 的长训曲线：若 LF_DET 随 epoch 增加持续逼近并追平，差距属于样本效率；若在更低水平饱和，才是容量或偏置差异。

本仓库的官方命令同样让混合模型和 LF_DET_0214 都训练 30 epoch，而 Track 2 的小模型反而训练了 100 epoch（[README](../README.md)）。团队把算力花在了容量、第二种架构（LF_DET）和推理侧集成上，而不是用来给单个模型多跑迭代。超分 Transformer 过去训练慢，原因之一是相对位置偏置让它们用不了 FlashAttention；另一个原因是窗口划分和张量重排带来的访存开销（[EchoSR](https://arxiv.org/html/2605.17470)）。2026 年的 SST 把相对位置偏置换成 RIB 后，训练快 2.1 倍、显存降低 24.6%（待核实，[arXiv:2603.06738](https://arxiv.org/abs/2603.06738)）。LFTransMamba 的 TransBlock 本来就没有位置偏置，所以这个瓶颈在它身上并不存在。

其他领域的墙钟证据并不一致。支持的一面：Vamba 处理小时级视频 token 时每个训练步快近 2 倍、显存至少省 50%（待核实，[arXiv:2503.11579](https://arxiv.org/abs/2503.11579)）；DVMSR 报告 SwinIR 的推理时间约为它的 20 倍（待核实，[arXiv:2405.03008](https://arxiv.org/html/2405.03008v2)）；MambaOSR 发现输入变大后 SwinIR 的延迟约为 MambaIR 的 3 倍（待核实，[doi:10.3390/e27040446](https://doi.org/10.3390/e27040446)）。反对的一面：EchoSR 发现 MambaIR 虽快于 SRFormer，但仍慢于 HiT-SIR，作者归因于 SSM 内核尚不成熟（待核实，[arXiv:2605.17470](https://arxiv.org/html/2605.17470)）；一项 2025 年的研究中 MambaIRv2、SwinIR、HAT 的全基准推理时间为 50/58/57 分钟，差别不大（硬件与负载不明，待核实，[arXiv:2506.14541](https://arxiv.org/abs/2506.14541)）。这些速度优势只在序列很长或对手是未优化的窗口注意力时出现；本仓库的纯注意力对照 LF_DET_0214 本来就不做联合的全 4D 注意力，所以不能直接套用，需要用 E1 实测。

本仓库的训练配置还说明，**"迭代够不够"这个问题值得检验，而且训练配方层面就有可改之处**。`--train-lr-scheduler 2` 对应 `StepLR(step_size=25, gamma=0.5)`，并且按 epoch 更新（[lfsr.py L821–835、L855–898](../lfsr.py)）。在 30 epoch 的训练里，学习率只在第 25 个 epoch 之后减半一次，最后五个 epoch 仍保持 1.5e-4，从未衰减到接近零。不过 EMA(0.999) 每步更新，推理时使用 EMA 权重（L880、L887、L906），可能部分抵消学习率未退火的影响（推断）。另外，"30 epoch"并不等于训练很短：`LFSR.ALL4` 由 4 套选视角副本组成（L84–106），每个 epoch 遍历的样本数约为单套数据的 4 倍，30 个 ALL4 epoch 在样本量上约相当于单套数据的 120 个 epoch（作为参照，BasicLFSR 默认训练 51 epoch、batch 4，但两者的切块方式不一定相同）。实际步数等于训练文件数 ÷ 3 × 30，可在 L847 之后打印 `total_step`，或用 L180 日志中的样本数除以 3 再乘以 30 得到，再与 HAT、DAT、MambaIRv2 配置中 250k–1M 次的迭代量对照。

优化器用的是 `Adam(betas=(0.99, 0.999))`（L753）。Budgeted Training 发现，迭代预算固定时，让学习率调度与预算匹配（例如线性衰减到零）至关重要（待核实，[arXiv:1905.04753](https://arxiv.org/abs/1905.04753)）。MLFIM 每次训练都遮掉 35% 的 token，刻意让回归任务变难、损失下降变慢，这本身就与"单位时间内把回归拟合到最好"的思路相反。它更像是用拟合速度换取鲁棒性；去噪领域的掩码训练正是为了提升泛化而设计的（待核实，[arXiv:2303.13132](https://arxiv.org/abs/2303.13132)）。

把以上证据综合起来，可以把用户的假设改写成一个与证据一致的版本（推断）：**超分保真度是受欠拟合限制的回归问题（单图超分有证据，光场超分未测），所以算力确实重要；光场超分中有用的上下文是整个 4D 光场（数万个 token），在它上面做单层联合的全局注意力每一步都贵得不可接受，而 Mamba 的线性扫描让每一层都负担得起跨视图的联合混合；保留视图内注意力，则维持了按内容的精确匹配。纯注意力模型也能用因子分解的空间/角度注意力获得全 4D 感受野，所以混合模型在相同迭代次数下胜出，更可能来自"联合、有序、带衰减的扫描"与注意力这两种互补先验，或来自更高的样本效率，而不是"看到了更多上下文"；在有 FLOPs 上限的赛道里，它还能放进限制之内。**

## 八个本仓库实验可以判定真正的机制

下面的实验都可以直接在 `lfsr.py` 和 `models/LFTransMamba.py` 上完成。每个实验都写明了它检验的两种解释，以及两者各自预测的结果，以便判定。

| 编号 | 实验（检验的两种解释） | 在本仓库怎么做 | 若前一种解释成立 | 若后一种解释成立 |
|---|---|---|---|---|
| E1 | 吞吐量与显存基准（约 10 分钟）；"每步更快→多迭代" vs "架构/偏置" | 在训练循环的每个 step 前后加 `torch.cuda.synchronize()` 计时，记录 `torch.cuda.max_memory_allocated()`，并用 `torch.profiler` 分模块计时。对 LF_DET_0214、`LFTransMamba_D16_R1_K18_C80_T2`、`..._T0`，以及一个按 FLOPs 匹配的纯注意力基线（如把 SS2D3D/SS2D4D 换成 LF-DET 式宏像素角度注意力，调 C 使 FLOPs 相当）各跑 200 step；在 fp32 与 AMP 两种精度下、强制指定 SDPA 后端分别测，报告 it/s 和每步 GFLOPs。训练 patch 由预生成的 h5 数据固定为 32（`transform_train_lfsr` 不裁剪，数据集的 `patch_size = 32` 只用于 .mat，L163、L607–619），所以测 48 需要重新生成 HR 192 / LR 48 的训练数据，或直接用 5×5×48×48 的随机张量测吞吐。用 fvcore 统计参数和 FLOPs 时，检查 selective scan 是否漏算，并确认 `F.scaled_dot_product_attention` 是否被统计（仓库 `__main__` 只为 selective scan 注册了 handle，L2466–2475；未统计时可注册 handle，或把 `use_flashatten` 设为 False 走显式 matmul 路径），同时核对计数口径（一次乘加计为 1 还是 2 FLOPs） | 混合模型的 it/s 明显高于 FLOPs 匹配的纯注意力基线 | 两者 it/s 接近，或者混合模型更慢 |
| E2 | 等迭代与等墙钟对照；"迭代/预算（含样本效率）" vs "容量或偏置" | 按 E1 测得的 it/s 换算，给 LF_DET_0214（及 FLOPs 匹配基线）和混合模型相同的 GPU 小时；另外各跑 30/60/90 epoch 画出曲线。两模型各做小规模学习率扫描（如 {1e-4, 2e-4, 3e-4, 4e-4}），各取最优。在 `lr_scheduler()` 里新增选项 7（线性或余弦衰减到 0）时，要把 `scheduler.step()` 从 epoch 末尾（L898）移到 step 循环内，用 `total_step = len(train_loader) × epoch`（L847）计算衰减，且 30/60/90 epoch 各自按自身预算重设调度。EMA 统一为 0.999，同时报告 EMA 与非 EMA 权重的结果，并加一组"EMA 开/关 × StepLR/线性衰减到 0"的对照 | 等墙钟时差距明显扩大；LF_DET 加 epoch 后持续逼近并追平（追平说明差距属于样本效率） | 等迭代时混合模型仍然领先；LF_DET 加 epoch 后在更低的水平饱和 |
| E3 | 跨视图混合方式；"因子分解注意力足以承担跨视图混合" vs "联合扫描有独特价值" | 把 SS2D3D/SS2D4D 换成等 FLOPs 的宏像素角度注意力（5×5、10×10 窗口，仿 [LF_DET_0214.py](../models/LF_DET_0214.py) L408–425）或 EPI 注意力；T 取 {0,1,2}。另设一臂在 VLSSBlock/LFSSBlock 中加入（而非替换）等 FLOPs 的跨视图 softmax 注意力：若加入后提升，说明跨视图精确匹配正是扫描的短板。把 VLSSBlock/LFSSBlock 换成 VSSBlock 的"无跨视图混合"版本只作下限参照：初始卷积与所有 FFN 都是逐视图的，这样做会切断全部视图间信息流，任何解释都预测大幅掉分 | 注意力版与原模型相当 | 注意力版明显更差；T=0 掉分集中在纹理丰富的场景 |
| E4 | 机制替换；"收益来自 SSM 本身" vs "收益来自全局覆盖加门控/位置先验" | 把 SS2D3D/SS2D4D 换成带 LePE 或 CPE 的线性注意力（MLLA 式）；或把视图内全局注意力换成 8×8/16×16 窗口注意力 | 线性注意力明显差于原模型 | 线性注意力与原模型相当；换成窗口注意力后掉分，说明视图内的全局匹配有价值 |
| E5 | 几何先验消融；用户②"无需几何先验" vs "隐式几何必要" | 新建只含 `SR_5x5_4x_128_32_matlab` 的训练协议与 `LFSR.ALL4` 对比；`--processor psw++` 与朴素分块对比；推理 patch 取 32/40/48/整图。整图推理需先在 `--processor` 的 choices（L1176）中加入 `'full'`，并在 `LFSR.__init__`（L685–691）中对 `'full'` 跳过 Processor 构建；LF_DET_0214 需按整图尺寸重新实例化 `patch_size` | 去掉几何增广和保视差推理后影响很小 | ALL4 和 PSW++ 在合成半上贡献明显 |
| E6 | MLFIM 掩码率；"欠拟合是瓶颈" vs "正则换泛化" | 把 `random_masking` 的 0.35（L2382）改为 0、0.25、0.5 | 掩码率为 0 时，等迭代下更好（拟合更快） | 掩码能提升 NTIRE 验证集上的泛化，尤其是合成半 |
| E7 | 几何排序消融；用户②"纯回归即可" vs "光场结构化扫描顺序必要" | 把 VLSSBlock/LFSSBlock 换成在 `(u h)(v w)` 整幅 SAI 拼图上做普通 SS2D 四向扫描的块（保留跨视图上下文，只去掉光场结构化的扫描顺序），调整 C 使参数量和 FLOPs 匹配，在 NTIRE 验证集的实拍半和合成半上分别比较 | 掉分很小 | 掉分明显，且合成半更明显 |
| E8 | 注意力位置与配比；"位置无关紧要" vs "注意力宜集中在后段（MambaVision）/廉价路径与注意力相互竞争（大窗口惰性）" | 在 `_make_layer` 的调用循环（L2245 起）按 `i_layer` 传入不同的 `num_spa_trans`，保持注意力块总数不变，只给后一半复合块设 T>0（例如前 9 块 T=0、后 9 块 T=4），与均匀放置的 T=2 对比；另做一组减弱 VSSBlock 中视图内 SS2D 或 ConvFFN 的对照 | 各种放置的差别在种子波动以内 | 后段集中放置更好，或减弱廉价路径后注意力的收益变大 |

评估协议决定了这些实验能否得出结论。

1. **域差异必须在 NTIRE 验证集上分别衡量实拍半和合成半**，而不能用 `val_all` 的 real/synth 分组，因为后者是五个同分布测试集。`--task test --dataset LFSR.NTIRE.VAL` 只会输出 BMP 图像，不计算 PSNR（[lfsr.py L135–146、L1065–1107](../lfsr.py)），需要提交到 CodaBench 验证服务器打分，或者在有公开 GT 时本地计算（GT 是否公开待确认）。
2. 本文讨论的增益都在 0.05–0.15 dB 量级，所以每个关键配置至少要跑 2–3 个随机种子，并报告均值和极差。
3. FP16 训练（Track 3 用了 `--fp16`）时，SSM 在调用扫描内核之前始终把 A 转为 fp32，并在训练模式下（`force_fp32 = self.training`）把 x、Δ、B、C 也转为 fp32（[models/LFTransMamba.py L1126–1143、L401–413](../models/LFTransMamba.py)，3D/4D 版本相同）；推理时 x、Δ、B、C 不做转换。这条安全措施应当保留，Δ 投影的偏置也不应被零初始化（[mamba README](https://github.com/state-spaces/mamba)）。
4. Mamba 的有效感受野受训练长度约束，这提示推理分块不宜远大于训练 patch。Track 3 模型在 32/40/48 分块下分别为 33.9249/33.9310/33.9192（[README](../README.md)），与训练尺寸 32 相比没有退化。但 BigEPIT 发现 Transformer 的 PSNR 随推理块增大而上升，而这里几乎持平，也可能说明模型没有利用更大的上下文（与有效感受野受训练长度约束一致，推断）。"整图推理"在 Mamba 上是否像 BigEPIT 在 Transformer 上那样带来增益，仍需 E5 验证。如果想用 48 或整图推理，可以先生成更大分块（如 HR 192 / LR 48）的训练数据，在最后几个 epoch 用低学习率微调；NLP 中的 state-passing 后训练只需约 0.1% 的预训练计算量就能显著改善长度外推（待核实，Buitrago & Gu，[arXiv:2507.02782](https://arxiv.org/abs/2507.02782)）。这一点可并入 E5，与只在 32 上训练的模型做对照。
5. `compute_psnr_ssim` 用 `zip(range(angular), range(angular))` 只计算 5 个对角视角（[lfsr.py L988–992](../lfsr.py)），而 BasicLFSR 和 NTIRE 对全部 25 个视角取平均。本文引用的本地数字都基于这 5 个视角；做 E1–E8 之前，应先把该循环改为遍历全部 25 个视角（如 `itertools.product(range(angular), repeat=2)`），或同时报告两种口径。

## 结论

这次调研的核心推断是：**在光场超分里，线性复杂度换来的是单层内联合的跨视图混合，而不是速度**（尚未在光场数据上用等墙钟、等 FLOPs 对照验证）。在 LFTransMamba 中，Mamba 承担单层联合的跨视图 4D 混合；注意力承担视图内的精确匹配。注意力只用在视图内（1,024–2,304 token），跨视图混合只用扫描（25,600–57,600 token），而视图内同时也做一次 SS2D 扫描。纯注意力模型也能用因子分解的注意力覆盖全 4D 光场，所以联合扫描是否优于因子分解注意力，要由 E3 判定。因此，"每步更快、所以多跑迭代"这条路在骨干层面没有证据；"每步学得更多"是否成立，要看 E2 的长训曲线。在训练配方层面确实还有空间：官方 Track 1 训练 30 个 ALL4 epoch（实际步数需打印核实），学习率只在第 25 个 epoch 后减半一次，但每步更新的 EMA 可能在一定程度上替代了退火。因此预算匹配的学习率调度值得用 E2 验证（推断），但不宜断言为最可能的收益来源。

第二个可以付诸行动的结论是，"放弃物理约束"应当改写为"放弃显式物理损失，同时加强隐式几何"。比赛中的域差异更可能集中在合成渲染那一半和视差分布上（推断）。冠军们的有效做法，是让训练数据覆盖更宽的视差范围、让推理阶段保持视差结构，并让扫描或注意力顺着极线进行。2026 年的 SMART、RASLF、GTF 都在加强几何建模，而不是去掉它。混合架构的下一步增益，更可能来自"视差自适应的扫描路径"加上"少量放在正确位置的注意力"，而不是更大比例的 Transformer，也不是更多迭代。

### 比赛行动清单（按优先级）

1. **先修评估**：把 `compute_psnr_ssim` 改为 25 视角口径；每个关键配置跑 2–3 个随机种子，并在 NTIRE 验证集的实拍半和合成半上分别评分。
2. **训练配方**：先打印 Track 1 的实际步数；新增按 step 更新的余弦或线性衰减到 0 的调度，并把 Track 1 延长到 60–90 epoch 做对照（E2）。
3. **保留隐式几何先验**：ALL4 选视角重采样、PSW++/EPSW、SAI/宏像素/EPI 扫描都应保留，不要换成通用 2D 骨干（E5、E7）。
4. **集成**：保留纯注意力 LF-DET 与混合模型的异构集成，加上 TTA 和 EMA。
5. **Track 1/3 的容量**：参照 2026 年报告描述的 K48/C160/T3（待核实），先在 K、C 上画缩放曲线。
6. **Track 2**：20 GFLOPs 上限下，C=32 时每个视图内注意力层约 4 GF（按 MambaOut 公式粗算），T=0 纯 Mamba 是稳妥的默认选择；但上限并不排除一两层注意力（LGFN 先例），以 fvcore 实测为准。
7. **查阅原文消融**：LFTransMamba 论文（CVPRW 2025）自带的消融表（纯 Mamba 对比加 Transformer、MLFIM 开关、EPSW 对比 PSW）本调研未能取得，可直接对照原文。

## 参考文献

核实状态说明："已核实"指 GitHub README、官方配置或本仓库代码已直接读取；"待核实"指数字来自论文摘录、搜索引擎摘要、研究者回忆或第三方摘要，引用前请对照原文。

**光场超分与 NTIRE**

- NTIRE 2023 LFSR：NTIRE 2023 Challenge on Light Field Image Super-Resolution: Dataset, Methods and Results，Y. Wang 等，CVPRW 2023，[arXiv:2304.10415](https://arxiv.org/abs/2304.10415)（待核实）
- NTIRE 2024 LFSR：NTIRE 2024 Challenge on Light Field Image Super-Resolution: Methods and Results，Y. Wang 等，CVPRW 2024，[CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf)（待核实）
- NTIRE 2025 LFSR：NTIRE 2025 Challenge on Light Field Image Super-Resolution: Methods and Results，Y. Wang 等，CVPRW 2025，[CVF](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Wang_NTIRE_2025_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2025_paper.pdf)、[ECNU 记录](https://pure.ecnu.edu.cn/en/publications/ntire-2025-challenge-on-light-field-image-super-resolution-method/)（待核实）
- NTIRE 2026 LFSR：NTIRE 2026 Challenge on Light Field Image Super-Resolution: Methods and Results，Y. Wang 等，CVPRW 2026，[CVF](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Wang_NTIRE_2026_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2026_paper.pdf)（待核实）
- NTIRE LFSR 规则与验证基线：LAVA LF-Image-SR README，[2024](https://raw.githubusercontent.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/a473ad7/README.md)、[2025](https://raw.githubusercontent.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/343a275/README.md)、[2026](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2026)（已核实）
- BasicLFSR：[GitHub](https://github.com/ZhengyuLiang24/BasicLFSR)（已核实）
- DistgEPIT：DistgEPIT: Enhanced Disparity Learning for Light Field Image Super-Resolution，K. Jin 等，CVPRW 2023，[GitHub](https://github.com/OpenMeow/NTIRE23_LFSR_DistgEPIT)（README 已核实）
- BigEPIT：BigEPIT: Scaling EPIT for Light Field Image Super-Resolution，W. Chao 等，CVPRW 2024，[CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chao_BigEPIT_Scaling_EPIT_for_Light_Field_Image_Super-Resolution_CVPRW_2024_paper.pdf)（待核实）
- LFTransMamba：LFTransMamba: A Hybrid Mamba-Transformer Model for Light Field Image Super-Resolution，K. Jin 等，CVPRW 2025，[本仓库 README](../README.md) 与代码（代码已核实，论文消融未取得）
- L²FMamba：L²FMamba: Lightweight Light Field Image Super-Resolution With State Space Model，Z. Wei 等，IEEE TCI 2025，[arXiv:2503.19253](https://arxiv.org/abs/2503.19253)（待核实）
- LFTramba：LFTramba: Comprehensive Information Learning for Light Field Image Super-Resolution via a Hybrid Transformer-Mamba Framework，H. Liu 等，CVPRW 2025，[CVF](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/html/Liu_LFTramba_Comprehensive_Information_Learning_for_Light_Field_Image_Super-Resolution_via_CVPRW_2025_paper.html)（待核实）
- LFMT：Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution，H. Liu 等，Pattern Recognition 2026，[arXiv:2509.04824](https://arxiv.org/abs/2509.04824)（待核实）
- LFMamba：LFMamba: Light Field Image Super-Resolution with State Space Model，W. Xia 等，2024，[arXiv:2406.12463](https://arxiv.org/abs/2406.12463)（待核实）
- MLFSR：Mamba-based Light Field Super-Resolution with Efficient Subspace Scanning，R. Gao 等，ACCV 2024，[CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Gao_Mamba-based_Light_Field_Super-Resolution_with_Efficient_Subspace_Scanning_ACCV_2024_paper.pdf)（待核实）
- SMART：Slope-Guided Mamba and Angular-Refined Transformer for Light Field Super-Resolution，L. Jin 等，IEEE ICME 2026，[arXiv:2607.00965](https://arxiv.org/abs/2607.00965)（待核实）
- RASLF：RASLF: Representation-Aware State Space Model for Light Field Super-Resolution，Z. Wei 等，2026，[arXiv:2603.16243](https://arxiv.org/abs/2603.16243)（待核实）
- GTF：GTF: Omnidirectional EPI Transformer for Light Field Super-Resolution，K. Li 等，CVPRW 2026，[arXiv:2605.04581](https://arxiv.org/abs/2605.04581)（待核实）
- SkimLFSR：Less is More: Skim Transformer for Light Field Image Super-resolution，Hu 等，[arXiv:2407.15329](https://arxiv.org/abs/2407.15329)（待核实）
- LF-DET：Exploiting Spatial and Angular Correlations With Deep Efficient Transformers for Light Field Image Super-Resolution，R. Cong 等，IEEE TMM 2023（标题取自本仓库 [LF_DET_0214.py](../models/LF_DET_0214.py)）
- EPIT：Learning Non-Local Spatial-Angular Correlation for Light Field Image Super-Resolution，Z. Liang 等，ICCV 2023，[arXiv:2302.08058](https://arxiv.org/abs/2302.08058)（待核实）
- LGFN：Z. Yu 等，CVPRW 2024，[arXiv:2409.17759](https://arxiv.org/abs/2409.17759)（全名未取得，待核实）

**通用超分与复原中的混合架构**

- MambaIR：MambaIR: A Simple Baseline for Image Restoration with State-Space Model，H. Guo 等，ECCV 2024，[arXiv:2402.15648](https://arxiv.org/abs/2402.15648)、[GitHub](https://github.com/csguoh/MambaIR)（README 已核实）
- MambaIRv2：MambaIRv2: Attentive State Space Restoration，H. Guo 等，CVPR 2025，[arXiv:2411.15269](https://arxiv.org/html/2411.15269v2)（README 已核实，消融待核实）
- Contrast：Contrast: A Hybrid Architecture of Transformers and State Space Models for Low-Level Vision，A. Urumbekov、Z. Chen，2025，[arXiv:2501.13353](https://arxiv.org/abs/2501.13353)；第三方评审见 [Pith](https://pith.science/paper/2501.13353)（待核实）
- MatIR：MatIR: A Hybrid Mamba-Transformer Image Restoration Model，J. Wen 等，2025，[arXiv:2501.18401](https://arxiv.org/html/2501.18401v2)、[GitHub](https://github.com/wenjuan7275/MatIR)（README 已核实）
- T-PMambaSR：Transformer-Progressive Mamba Network for Lightweight Image Super-Resolution，S. Guo 等，IEEE TIP 2026，[arXiv:2511.03232](https://arxiv.org/html/2511.03232v1)（待核实）
- RestorMixer：An Efficient and Mixed Heterogeneous Model for Image Restoration，2025，[arXiv:2504.10967](https://arxiv.org/html/2504.10967v1)（待核实）
- TransMamba（去雨）：A Hybrid Transformer-Mamba Network for Single Image Deraining，S. Sun 等，2024，[arXiv:2409.00410](https://arxiv.org/abs/2409.00410)、[GitHub](https://github.com/sunshangquan/TransMamba)（README 已核实）
- Transformer→SSM 蒸馏：Edge-Efficient Image Restoration: Transformer Distillation into State-Space Models，S. S. Miriyala 等，2026，[arXiv:2605.02794](https://arxiv.org/abs/2605.02794)（待核实）
- MaIR：MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration，B. Li 等，CVPR 2025，[GitHub](https://github.com/XLearning-SCU/2025-CVPR-MaIR)；第三方评审见 [Pith](https://pith.science/paper/2412.20066)（README 已核实，消融差距待核实）
- EAMamba：EAMamba: Efficient All-Around Vision State Space Model for Image Restoration，Y.-C. Lin 等，ICCV 2025，[GitHub](https://github.com/daidaijr/EAMamba)（README 已核实）
- Hi-Mamba：Hi-Mamba: Hierarchical Mamba for Efficient Image Super-Resolution，J. Qiao 等，IEEE TIP 2025，[arXiv:2410.10140](https://arxiv.org/pdf/2410.10140)（待核实）
- DVMSR：DVMSR: Distillated Vision Mamba for Efficient Super-Resolution，X. Lei 等，CVPRW 2024，[arXiv:2405.03008](https://arxiv.org/html/2405.03008v2)（待核实）
- MambaOSR：Entropy 2025，[doi:10.3390/e27040446](https://doi.org/10.3390/e27040446)（待核实）
- EchoSR：2026，[arXiv:2605.17470](https://arxiv.org/html/2605.17470)（待核实）
- SR 推理时间表：2025，[arXiv:2506.14541](https://arxiv.org/abs/2506.14541)（待核实）
- NTIRE 2025 ×4：NTIRE 2025 Challenge on Image Super-Resolution (×4): Methods and Results，Z. Chen 等，CVPRW 2025，[arXiv:2504.14582](https://arxiv.org/pdf/2504.14582)（待核实）
- NTIRE 2026 ×4：The Fourth Challenge on Image Super-Resolution (×4) at NTIRE 2026，[arXiv:2604.14558](https://arxiv.org/html/2604.14558v1)（待核实）
- 无训练集成：Training-Free Model Ensemble for Single-Image Super-Resolution via Strong-Branch Compensation，2026，[arXiv:2604.11564](https://arxiv.org/pdf/2604.11564)（待核实）

**语言模型中的混合架构与机理分析**

- Jamba：Jamba: Hybrid Transformer-Mamba Language Models，Lieber 等，ICLR 2025，[arXiv:2403.19887](https://arxiv.org/abs/2403.19887)（待核实）
- NVIDIA 实证研究：An Empirical Study of Mamba-based Language Models，Waleffe 等，2024，[arXiv:2406.07887](https://arxiv.org/abs/2406.07887)（待核实）
- Samba：Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling，Ren 等，ICLR 2025，[arXiv:2406.07522](https://arxiv.org/abs/2406.07522)（待核实）
- Hymba：Dong 等，ICLR 2025，[arXiv:2411.13676](https://arxiv.org/abs/2411.13676)（待核实）
- Zamba / Zamba2：Glorioso 等，2024，[arXiv:2405.16712](https://arxiv.org/abs/2405.16712)、[arXiv:2411.15242](https://arxiv.org/abs/2411.15242)（待核实）
- Nemotron-H：NVIDIA，2025，[arXiv:2504.03624](https://arxiv.org/abs/2504.03624)（待核实）
- Falcon-H1：TII，2025，[arXiv:2507.22448](https://arxiv.org/abs/2507.22448)（待核实）
- MiniMax-01：MiniMax，2025，[arXiv:2501.08313](https://arxiv.org/abs/2501.08313)（待核实）
- MiniMax-M2：The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence，2026，[arXiv:2605.26494](https://arxiv.org/pdf/2605.26494)（待核实）
- Qwen3-Next：架构说明来自二手来源，[deeplearning.ai](https://community.deeplearning.ai/t/qwen3-next-employs-hybrid-attention-for-long-context-inputs/879461)、[Raschka](https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/)（二手来源）
- Kimi Linear：Moonshot AI，2025，[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)（待核实）
- Jet-Nemotron：Gu 等，2025，[arXiv:2508.15884](https://arxiv.org/html/2508.15884v1)（待核实）
- Mamba in the Llama：The Mamba in the Llama: Distilling and Accelerating Hybrid Models，J. Wang 等，NeurIPS 2024，[arXiv:2408.15237](https://arxiv.org/abs/2408.15237)（待核实）
- MAD：Mechanistic Design and Scaling of Hybrid Architectures，Poli 等，ICML 2024，[arXiv:2403.17844](https://arxiv.org/abs/2403.17844)（待核实）
- MambaFormer：Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks，Park 等，ICML 2024，[arXiv:2402.04248](https://arxiv.org/abs/2402.04248)（待核实）
- Wen 等：RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval，Wen、Dang、Lyu，ICLR 2025，[arXiv:2402.18510](https://arxiv.org/abs/2402.18510)（待核实）
- Zoology：Zoology: Measuring and Improving Recall in Efficient Language Models，Arora 等，ICLR 2024，[arXiv:2312.04927](https://arxiv.org/abs/2312.04927)（待核实）
- G&A：Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism，Bick、Xing、Gu，ICML 2025，[arXiv:2504.18574](https://arxiv.org/abs/2504.18574)（待核实）
- 缓存级干预：What Attention Recalls and Recurrence Controls in Hybrid Language Models，Afendulev 等，2026 预印本，[arXiv:2609.04434](https://arxiv.org/abs/2609.04434)（待核实）
- Key Handoff：The Key Handoff: Retrieval in Hybrid Language Models，Kale 等，2026 预印本，[arXiv:2609.32942](https://arxiv.org/abs/2609.32942)（待核实）
- 72 模型系统研究：A Systematic Analysis of Hybrid Linear Attention，Wang、Zhu、Eshraghian 等，2025，[arXiv:2507.06457](https://arxiv.org/abs/2507.06457)（待核实）
- 串行与并行混合：Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall and Language Modeling，Lee、Yu 等，BabyLM 2025，[arXiv:2510.26912](https://arxiv.org/abs/2510.26912)（待核实）
- 大窗口惰性：Rethinking the Role of Efficient Attention in Hybrid Architectures，Qiao 等，2026，[arXiv:2606.15378](https://arxiv.org/abs/2606.15378)（待核实）

**视觉混合骨干与视觉 Mamba**

- MambaVision：MambaVision: A Hybrid Mamba-Transformer Vision Backbone，A. Hatamizadeh、J. Kautz，CVPR 2025，[arXiv:2407.08083](https://arxiv.org/abs/2407.08083)（待核实）
- A2Mamba：Attention-augmented State Space Models for Visual Recognition，Lou、Fu、Yu，2025，[arXiv:2507.16624](https://arxiv.org/abs/2507.16624)（待核实）
- Heracles：Patro 等，2024，[arXiv:2403.18063](https://arxiv.org/abs/2403.18063)（待核实）
- Vamba：Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers，Ren 等，ICCV 2025，[arXiv:2503.11579](https://arxiv.org/abs/2503.11579)（待核实）
- MambaOut：MambaOut: Do We Really Need Mamba for Vision?，W. Yu、X. Wang，CVPR 2025，[arXiv:2405.07992](https://arxiv.org/abs/2405.07992)、[GitHub](https://github.com/yuweihao/MambaOut)、[issue #205](https://github.com/yuweihao/MambaOut/issues/205)（README 已核实）
- Vim：Vision Mamba，Zhu 等，ICML 2024，[arXiv:2401.09417](https://arxiv.org/abs/2401.09417)、[GitHub](https://github.com/hustvl/Vim)（README 已核实）
- VMamba：VMamba: Visual State Space Model，Y. Liu 等，NeurIPS 2024，[arXiv:2401.10166](https://arxiv.org/abs/2401.10166)（待核实）
- Demystify Mamba：Demystify Mamba in Vision: A Linear Attention Perspective，D. Han 等，NeurIPS 2024，[arXiv:2405.16605](https://arxiv.org/abs/2405.16605)（待核实）
- ARM：Autoregressive Pretraining with Mamba in Vision，Ren 等，ICLR 2025，[arXiv:2406.07537](https://arxiv.org/abs/2406.07537)（待核实）

**Mamba 与 Transformer 的机理对比**

- Mamba：Mamba: Linear-Time Sequence Modeling with Selective State Spaces，A. Gu、T. Dao，COLM 2024，[arXiv:2312.00752](https://arxiv.org/abs/2312.00752)（待核实）
- Mamba-2 / SSD：Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality，T. Dao、A. Gu，ICML 2024，[arXiv:2405.21060](https://arxiv.org/abs/2405.21060)、[Tri Dao 博客](https://tridao.me/blog/2024/mamba2-part4-systems/)（待核实）
- mamba 官方仓库：state-spaces/mamba，[GitHub](https://github.com/state-spaces/mamba)（README 已核实）
- Jelassi 等：Repeat After Me: Transformers are Better than State Space Models at Copying，Jelassi 等，ICML 2024，[arXiv:2402.01032](https://arxiv.org/abs/2402.01032)（待核实）
- HAX：Zhan 等，NeurIPS 2025，[arXiv:2507.00449](https://arxiv.org/abs/2507.00449)（待核实）
- Stuffed Mamba：Chen 等，2024（2026 年版改题为 Oversized States Lead to the Inability to Forget），[arXiv:2410.07145](https://arxiv.org/abs/2410.07145)（待核实）
- DeciMamba：DeciMamba: Exploring the Length Extrapolation Potential of Mamba，Ben-Kish 等，ICLR 2025，[arXiv:2406.14528](https://arxiv.org/abs/2406.14528)（待核实）
- 长度泛化：Understanding and Improving Length Generalization in Recurrent Models，Buitrago、Gu，ICML 2025，[arXiv:2507.02782](https://arxiv.org/abs/2507.02782)（待核实）
- 状态追踪：The Illusion of State in State-Space Models，Merrill、Petty、Sabharwal，ICML 2024，[arXiv:2404.08819](https://arxiv.org/abs/2404.08819)；The Expressive Capacity of State Space Models: A Formal Language Perspective，Sarrof、Veitsman、Hahn，NeurIPS 2024，[arXiv:2405.17394](https://arxiv.org/abs/2405.17394)（待核实）
- 混合精度稳定性：Mamba State-Space Models Are Lyapunov-Stable Learners，Halloran 等，TMLR 2025，[arXiv:2406.00209](https://arxiv.org/abs/2406.00209)（待核实）

**训练预算与配方**

- RCAN-it：Revisiting RCAN: Improved Training for Image Super-Resolution，Z. Lin 等，2022，[arXiv:2201.11279](https://arxiv.org/abs/2201.11279)、[GitHub](https://github.com/zudi-lin/rcan-it)（README 已核实，数字待核实）
- Budgeted Training：Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints，M. Li、E. Yumer、D. Ramanan，ICLR 2020，[arXiv:1905.04753](https://arxiv.org/abs/1905.04753)（待核实）
- SST：Rank-factorized Implicit Neural Bias: Scaling Super-Resolution Transformer with FlashAttention，D. Lee 等，2026，[arXiv:2603.06738](https://arxiv.org/abs/2603.06738)（待核实）
- 掩码训练：Masked Image Training for Generalizable Deep Image Denoising，H. Chen 等，CVPR 2023，[arXiv:2303.13132](https://arxiv.org/abs/2303.13132)（待核实）
- 训练配置：[MambaIRv2 ×4](https://github.com/csguoh/MambaIR/blob/main/options/train/mambairv2/train_MambaIRv2_SR_x4.yml)、[DAT ×4](https://github.com/zhengchen1999/DAT/blob/main/options/Train/train_DAT_x4.yml)、[HAT ×4](https://github.com/XPixelGroup/HAT/blob/main/options/train/train_HAT_SRx4_ImageNet_from_scratch.yml)、[BasicLFSR option.py](https://github.com/ZhengyuLiang24/BasicLFSR/blob/main/option.py)（已核实）
