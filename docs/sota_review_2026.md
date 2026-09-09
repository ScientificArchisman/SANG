# Audio-Driven Talking-Head Generation — SOTA Review (September 2026)

**Purpose.** A compute-aware review of the field, written to answer one question for SANG:
*what architecture can reach published-SOTA quality on 1× H100, and what cannot?*

**Scope.** One-shot / few-shot audio-driven talking head: input = arbitrary speech audio +
arbitrary reference portrait, output = a video of that person speaking that audio.

**Method.** Literature retrieved September 2026 via Consensus (220M peer-reviewed corpus),
OpenAlex/Semantic Scholar (FastTrack), and direct arXiv retrieval. Every quantitative claim below
is traceable to a fetched source; numbers taken from a secondary table (e.g. a competitor's
comparison) are marked as such. Nothing here is quoted from memory.

---

## 0. The one-paragraph verdict

The field has split into two families that win for *different reasons*. **Appearance-space
generators** (EMO, Hallo3, Sonic, OmniHuman, SoulX-FlashHead, MultiTalk, FantasyTalking) produce
the best raw fidelity, but every one of them is a fine-tune of a large **pretrained** video
diffusion prior (Wan2.1, SVD, CogVideoX) on hundreds of hours of curated data with tens of GPUs.
**Motion-space generators** (FLOAT, Ditto, Teller, FlowTalk, KDTalker, IF-MDM) predict a compact,
identity-agnostic motion code and delegate pixels to a frozen pretrained renderer — and they reach
*competitive* FID/FVD with **one to eight GPUs**. FLOAT is the decisive datapoint: **1× A100, 22
days, 11.3 hours of training data**, and it beats Hallo on HDTF FID, FVD and LSE-C. For a
single-GPU lab, the motion-space family is not a compromise — it is the only family whose
compute-optimal point is inside the budget.

---

## 1. Taxonomy: four families

| # | Family | What the network predicts | Pixels produced by | Representative work |
|---|---|---|---|---|
| A | **Warping / implicit-keypoint** | dense flow or implicit keypoints | learned warp + inpainting decoder | LivePortrait, LIA, LIA-X |
| B | **Motion-latent generative** | a compact motion code (10²–10³ dims) conditioned on audio | frozen family-A renderer | **FLOAT**, Ditto, FlowTalk, KDTalker, IF-MDM, Teller (FMLG) |
| C | **Appearance-latent diffusion / DiT** | VAE latents of the actual video | pretrained VAE decoder | EMO, Hallo/Hallo2/Hallo3, Sonic, LatentSync, MultiTalk, FantasyTalking, SoulX-FlashHead, OmniHuman |
| D | **Discrete-token AR / masked** | video tokens from a VQ/FSQ tokenizer | tokenizer decoder | **← where SANG v3 sits**; EARTalking (continuous AR), Lumos-1, VideoAR (general video) |

The critical structural difference between B and C/D is **what the model must learn**. In family B
the reference image supplies appearance and the network only models motion. In C the pretrained
diffusion prior supplies appearance and the fine-tune only *steers* it. In D — with no pretrained
prior — the network must learn appearance *and* motion from scratch. Family D is therefore the
only one that pays the full appearance bill out of its own training budget.

---

## 2. Benchmark reality check

### 2.1 HDTF — reported numbers

⚠️ **These are not directly comparable.** Different papers use different HDTF test splits, crop
conventions, frame counts, and SyncNet checkpoints. Sonic's own paper reports Sync-C 4.20 while
SoulX-FlashHead's reproduction of Sonic reports 5.17 on the same dataset name. Treat the table as
a *map of the landscape*, not a ranking, and re-measure any baseline you intend to claim against.

| Method | Family | FID ↓ | FVD ↓ | Sync-C ↑ | Sync-D ↓ | Source of numbers |
|---|---|---|---|---|---|---|
| SoulX-FlashHead (Pro) | C | **9.97** | **111.38** | 5.73 | 8.77 | own paper, Table 3 |
| SoulX-FlashHead (Lite) | C | 11.37 | 126.52 | 4.21 | 9.49 | own paper, Table 3 |
| Sonic | C | 13.53 | 113.31 | 5.17 | 8.69 | SoulX Table 3 (secondary) |
| Hallo3 | C | 15.95 | 160.94 | 3.18 | 10.72 | SoulX Table 3 (secondary) |
| EchoMimic | C | 9.00 | 155.71 | 3.56 | 10.22 | SoulX Table 3 (secondary) |
| Ditto | B | 12.35 | 199.13 | 3.57 | 10.49 | SoulX Table 3 (secondary) |
| SadTalker | B | 21.58 | 207.67 | 4.60 | 9.21 | SoulX Table 3 (secondary) |
| AniPortrait | B | 19.83 | 242.29 | 1.89 | 11.91 | SoulX Table 3 (secondary) |
| Teller | B | 21.35 | 173.46 | **7.70** | 7.54 | own paper (via search summary — **verify**) |
| AsymTalker | C | 13.72 | 116.78 | **8.11** | 7.25 | search summary — **verify before citing** |
| READ | C | 15.07 | 235.32 | 8.66 | 6.89 | search summary — **verify before citing** |

Reference point: **real video Sync-C ≈ 8.09** (Teller). Methods reporting Sync-C > 8 are at or
above the ceiling of the metric, which is itself a warning sign — see §2.3.

### 2.2 FLOAT's table (different protocol, reports LSE-C/LSE-D and CSIM)

From the FLOAT paper directly, HDTF / RAVDESS:

| Method | FID ↓ | FVD ↓ | CSIM ↑ | E-FID ↓ | LSE-D ↓ | LSE-C ↑ |
|---|---|---|---|---|---|---|
| SadTalker | 71.95 / 119.43 | 339.06 / 376.29 | 0.644 / 0.644 | 1.914 / 3.500 | 7.947 / 7.273 | 7.305 / 4.748 |
| EDTalk | 50.08 / 75.02 | 211.28 / 304.93 | 0.626 / 0.676 | 1.579 / 3.468 | 8.123 / 7.682 | 7.623 / 5.318 |
| AniTalker | 39.51 / 70.43 | 184.45 / 265.34 | 0.643 / 0.725 | 1.830 / 2.330 | 7.907 / 8.176 | 7.288 / 4.555 |
| Hallo | 25.36 / 57.65 | 197.20 / 375.56 | **0.869** / **0.860** | 1.039 / 2.492 | 7.792 / 7.613 | 7.582 / 4.795 |
| EchoMimic | 33.55 / 81.84 | 296.76 / 320.22 | 0.823 / 0.805 | 1.234 / 3.201 | 8.903 / 8.161 | 6.242 / 4.144 |
| **FLOAT** | **21.10** / **31.68** | **162.05** / **166.36** | 0.843 / 0.810 | 1.229 / **1.367** | **7.290** / **6.994** | **8.222** / **5.730** |

Note the FID discrepancy for the *same* methods between §2.1 and §2.2 (SadTalker 21.58 vs 71.95).
This is the protocol problem in one line. **SANG must build its own harness and re-run baselines.**

### 2.3 The metrics are known to be unreliable

- Sync-C/Sync-D are "unstable and sensitive to mouth cropping and head pose" and "correlate
  poorly with human preferences" (per the evaluation literature: THEval arXiv:2511.04520;
  *Temporally-Aligned Evaluation for Audio-Driven Talking Head Generation* arXiv:2606.01031).
- **TalkVid** (arXiv:2508.13618, CVPR 2026 Findings) shows aggregate metrics *hide* subgroup
  failure: "our analysis on TalkVid-Bench reveals performance disparities across subgroups that
  are obscured by traditional aggregate metrics."

Practical consequence: a credible 2026 submission reports aggregate HDTF numbers **plus** a
stratified breakdown **plus** a human study. That is also the opening SANG can exploit (§6).

---

## 3. Family C in detail — why it wins, and what it costs

### 3.1 The recipe every family-C system uses

1. Take a **pretrained** video diffusion/DiT backbone (Wan2.1, SVD-XT, CogVideoX).
2. Inject identity by **channel-concatenating the reference-image latent** with the noisy latent
   (SoulX-FlashHead) or via a parallel **ReferenceNet** (EMO, Hallo).
3. Inject audio via **cross-attention adapters**, using multi-layer wav2vec/Whisper/WavLM features
   with a ±m frame window.
4. Fine-tune **only the audio adapters** (MultiTalk keeps Wan2.1-I2V-14B entirely frozen and
   trains only the audio cross-attention layer and adapter) or the full backbone with heavy compute.
5. Optionally distill to few-step / streaming (SoulX DMD, LeapTalk, Live Avatar, REST).

### 3.2 The cost

| System | Backbone | GPUs | Steps / data |
|---|---|---|---|
| SoulX-FlashHead | Wan2.1 T2V **1.3B** pretrained | **32× H20** | 100k steps, batch 256; **782 h** curated from 10,000 h raw (VividHead, 330k clips @512²) |
| MultiTalk | Wan2.1-I2V **14B**, frozen | multi-GPU | audio cross-attn + adapter only |
| Live Avatar | **14B** diffusion | multi-GPU (TPP pipeline parallel) | two-stage distillation |
| Teller | AR + LivePortrait renderer | — | 662 h AV-Speech + 2 h VFHQ + 32 h SFT |

**Implication for SANG:** family C is reachable *only* in its frozen-backbone-plus-adapter form,
and even that means holding a 1.3B–14B model plus activations on one H100. It is not where a
single-GPU lab should spend its first six weeks.

---

## 4. Family B in detail — the single-GPU frontier

### 4.1 FLOAT (ICCV 2025, arXiv:2412.01064) — the key datapoint

- **Motion space:** uses **LIA** (Latent Image Animator) as a frozen motion autoencoder rather
  than an SD VAE. Motion latent **d = 512** with **M = 20 orthogonal directions**; latents
  decompose as `wS = wS→r + wr→S`, separating identity from motion.
- **Generator:** a Flow Matching Transformer — 8 attention heads, hidden dim 1024, attention
  window T = 2, frame-wise AdaLN conditioning.
- **Training: 22 days on a *single* NVIDIA A100**, 2,000k steps, on **11.3 hours** of HDTF
  (240 videos, 230 identities).
- **Inference: 41.37 FPS on a single V100** at 10 NFE — "125× faster than Hallo".
- **Result:** best FID (21.10), FVD (162.05), E-FID and LSE-C/LSE-D in its comparison, beating
  Hallo and EchoMimic (§2.2).

This is the existence proof that a single GPU and ~11 hours of data can produce competitive
published numbers — provided the model predicts **motion**, not appearance.

### 4.2 The rest of the family

- **Ditto** (ACM MM 2025, arXiv:2411.19509): **265-dim identity-agnostic motion space** +
  neural renderer, real-time end-to-end (every module RTF < 1). Trained on **8× A100**, batch
  1024, Adan, lr 1e-4, wd 0.02, 500 epochs.
- **Teller** (CVPR 2025, arXiv:2503.18429): first *streaming autoregressive* talking head. Facial
  Motion Latent Generation compresses LivePortrait implicit keypoints (21 expression keypoints +
  rotation + translation) into **discrete motion tokens via Residual VQ**, sliced against Whisper
  embeddings; an Efficient Temporal Module refines physical consistency (neck, earrings) with a
  region-masked loss. Sync-C 7.70 — near the real-video ceiling of 8.09.
  **This is SANG's architecture family, but with motion tokens instead of appearance tokens.**
- **FlowTalk** (ACM MM Asia 2025): "off-the-shelf motion extractor to disentangle facial
  appearance from motion" + OT-based flow matching transformer predicting identity-agnostic
  motion. >100 FPS at 32 ODE steps, ~5× faster than 500-step diffusion baselines. Explicitly
  motivated by **cross-ethnic generalization**, trained on a *balanced* mix of DH-FaceVid-1K + HDTF
  with HuBERT-CN audio features.
- **KDTalker** (IJCV 2025): unsupervised implicit 3D keypoints + spatiotemporal diffusion;
  claims SOTA on lip-sync accuracy, head-pose diversity and efficiency simultaneously — the
  argument being that implicit keypoints adapt information density better than fixed 3DMM points.
- **IF-MDM** (arXiv:2412.04000), **JAM-Flow** (arXiv:2506.23552): further motion-space variants.

### 4.3 The renderers available off the shelf

| Renderer | Paper | Notes |
|---|---|---|
| **LIA** | arXiv:2203.09043 | Linear Motion Decomposition, orthogonal motion dictionary; what FLOAT uses |
| **LIA-X** | arXiv:2508.09959 | newer; Sparse Motion Dictionary, interpretable disentangled factors |
| **LivePortrait** | arXiv:2407.03168 | trained on ~69M frames, stitching + retargeting modules, **12.8 ms/frame on RTX 4090**; what Teller uses |

### 4.4 The honest limitation of family B

Warping-based renderers cap out on large head pose, occlusion, and fine mouth interior (teeth,
tongue). That is visible in the numbers: FLOAT's FID 21.10 vs SoulX-FlashHead's 9.97. A family-B
system will win on **efficiency, lip-sync and motion naturalness**, and will *not* win on raw FID
against a 32-GPU family-C model. Any SANG paper must position on the former.

---

## 5. Family D — discrete tokens, and why SANG v3 stalled

SANG v3 predicts VidTok FSQ appearance tokens (V=32768) with a MaskGIT-style masked transformer.
The relevant general-video literature says the tokenizer is the binding constraint:

- **HART** (arXiv, 2024): "Existing AR models face limitations due to the **poor image
  reconstruction quality of their discrete tokenizers**"; adding a continuous residual raises
  reconstruction FID from 2.11 → 0.30 and generation FID 7.85 → 5.38.
- **NOVA** (*Autoregressive Video Generation without Vector Quantization*): removing VQ improves
  "data efficiency, inference speed, visual fidelity, and video fluency" at 0.6B params.
- **EARTalking** (arXiv:2603.20307, USTC/iFLYTEK): end-to-end GPT-style AR talking head that
  explicitly abandons quantization — the repo's own doc already quotes its claim that
  "quantizing continuous latents to utilise a discrete cross-entropy loss introduces quantization
  errors, which degrade the final generation quality."
- **VideoAR** (2026), **Lumos-1** (2025): AR video generation is closing the gap with diffusion,
  but Lumos-1 still needed **48 GPUs** for pretraining, and both operate on general video with
  large models.

SANG's own measurement is consistent and is the strongest evidence in the repo: VidTok-FSQ at
128 px caps at **23.09 dB PSNR**, i.e. even a *perfect* token predictor decodes to mush. Switching
to the Wan2.1 VAE (26.69 dB @128, **31.80 dB @256**) was the right diagnosis. But it fixes the
*representation* while leaving the deeper problem — a from-scratch appearance generator on one
GPU — untouched.

---

## 6. Where the opening is for SANG

Three findings converge on the same gap:

1. **TalkVid** (arXiv:2508.13618) states plainly that SOTA models "lack generalization to the full
   spectrum of human diversity in ethnicity, language, and age groups," and releases
   **TalkVid-Bench**: 500 clips stratified over age (5 bands), gender, ethnicity (Black/White/Asian)
   and **15 languages**. Its headline finding is that aggregate metrics *hide* subgroup disparities.
2. **FlowTalk** independently names cross-ethnic generalization as an unsolved trade-off and builds
   a balanced training mix to attack it.
3. SANG **already trains on TalkVid** — currently an 8,000-clip subset of a 1,244-hour, 7,729-speaker
   corpus. This is an asset, not an accident.

A single-GPU lab will not beat a 32×H20 model on aggregate HDTF FID. It *can* plausibly own:

> **a compute-efficient streaming motion-latent talking head that is SOTA on TalkVid-Bench
> subgroups and competitive on HDTF**, with the subgroup/robustness analysis as a first-class
> contribution rather than an appendix.

That framing turns the compute constraint from a weakness into the paper's method section, and it
targets a gap the field's own benchmark paper says is open.

---

## 7. Citation-integrity audit of `v3_improvement_plan.md`

The existing document cites ~18 works, several with 2026 arXiv IDs. Fabricated or mis-numbered
references sink a submission, so each was checked against arXiv directly.

| Ref | Claimed ID | Status |
|---|---|---|
| Teller | arXiv:2503.18429 | ✅ **verified** — real; note it is **CVPR 2025 / submitted March 2025**, not 2026. Authors: Zhen, Yin, Qin, Yi, Zhang, Liu, Qi, Tao |
| LeapTalk | arXiv:2608.00079 | ✅ **verified** — real; Rongxiang Zhang & Songhua Liu; code released |
| EARTalking | arXiv:2603.20307 | ✅ **verified** — real; Weng et al., USTC/iFLYTEK/Zhejiang, 19 Mar 2026 |
| AsymTalker | arXiv:2605.02948 | ✅ **verified** — real |
| REST | arXiv:2512.11229 | ✅ **verified** — real |
| Live Avatar | arXiv:2512.04677 | ✅ **verified** — real; **ECCV 2026 Spotlight**, Alibaba-Quark, code released. Note: it is a **14B** model with multi-GPU pipeline parallelism |
| UniSync | arXiv:2603.03882 | ✅ **verified** — real; Fan et al., Mango TV. ⚠️ **name collision**: a *different* UniSync exists at arXiv:2503.16357 (*A Unified Framework for Audio-Visual Synchronization*) — disambiguate when citing |
| EMO | arXiv:2402.17485 | ✅ verified (Consensus, 278 citations) |
| LatentSync | arXiv:2412.09262 | ✅ verified (Consensus, 57 citations) |
| EchoMimic | arXiv:2407.08136 | ✅ verified (OpenAlex, AAAI 2025) |
| Ditto | arXiv:2411.19509 | ✅ verified (ACM MM 2025) |
| VASA-1, Hallo, OmniHuman, StableAvatar, InfiniteTalk, X-Dub, Sonic | various | ⏳ **not yet individually verified** — check before submission |

**Also to fix in that document:** several §16–§19 numbers are quoted without a retrieval trace
(e.g. Teller "HDTF FVD 173.5 vs Hallo 174.2"). Teller's FVD 173.46 is corroborated by
SoulX-FlashHead's independent table, but the Hallo comparison value is not — and Hallo's FVD is
reported as 197.20 by FLOAT and 160.94 (Hallo3) by SoulX. Re-source every number before it enters
a paper.

---

## 8. References

Verified arXiv/DOI links, September 2026.

**Motion-space (family B)**
- Ki, Min, Chae. *FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait.* ICCV 2025. https://arxiv.org/abs/2412.01064
- Li et al. *Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis.* ACM MM 2025. https://arxiv.org/abs/2411.19509
- Zhen et al. *Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation.* CVPR 2025. https://arxiv.org/abs/2503.18429
- Deng, Guo, Shen. *FlowTalk: Real-Time Audio-Driven Talking Head Synthesis via Motion-Space Flow Matching.* ACM MM Asia 2025. https://doi.org/10.1145/3769748.3773363
- Yang et al. *Unlock Pose Diversity: … Implicit Keypoint-based Spatiotemporal Diffusion (KDTalker).* IJCV 2025. https://link.springer.com/article/10.1007/s11263-025-02695-x
- *IF-MDM: Implicit Face Motion Diffusion Model.* https://arxiv.org/abs/2412.04000
- *JAM-Flow: Joint Audio-Motion Synthesis with Flow Matching.* https://arxiv.org/abs/2506.23552

**Renderers (family A)**
- Guo et al. *LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control.* https://arxiv.org/abs/2407.03168
- Wang et al. *Latent Image Animator (LIA).* https://arxiv.org/abs/2203.09043
- *LIA-X: Interpretable Latent Portrait Animator.* https://arxiv.org/abs/2508.09959

**Appearance-space (family C)**
- *SoulX-FlashHead: Oracle-guided Generation of Infinite Real-time Streaming Talking Heads.* https://arxiv.org/abs/2602.07449
- Tian et al. *EMO: Emote Portrait Alive.* https://arxiv.org/abs/2402.17485
- Li et al. *LatentSync: … Lip Sync with SyncNet Supervision.* https://arxiv.org/abs/2412.09262
- Ji et al. *Sonic: Shifting Focus to Global Audio Perception in Portrait Animation.* CVPR 2025. https://github.com/jixiaozhong/Sonic
- *MultiTalk / Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation.* NeurIPS 2025. https://arxiv.org/abs/2505.22647
- *FantasyTalking: Realistic Talking Portrait Generation via Coherent Motion Synthesis.* ACM MM 2025. https://github.com/Fantasy-AMAP/fantasy-talking
- Chen et al. *EchoMimic.* AAAI 2025. https://arxiv.org/abs/2407.08136
- Huang et al. *Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length.* ECCV 2026 Spotlight. https://arxiv.org/abs/2512.04677
- Zhang & Liu. *LeapTalk: Breaking the Latency–Quality Trade-off in Talking Head Generation.* https://arxiv.org/abs/2608.00079
- Wang et al. *REST: Diffusion-based Real-time End-to-end Streaming Talking Head Generation.* https://arxiv.org/abs/2512.11229
- Lu et al. *AsymTalker: Identity-Consistent Long-Term Talking Head Generation via Asymmetric Distillation.* https://arxiv.org/abs/2605.02948
- Fan et al. *UniSync: Towards Generalizable and High-Fidelity Lip Synchronization for Challenging Scenarios.* https://arxiv.org/abs/2603.03882

**Token / AR video generation (family D)**
- Weng et al. *EARTalking: End-to-end GPT-style Autoregressive Talking Head Synthesis with Frame-wise Control.* https://arxiv.org/abs/2603.20307
- Tang et al. *HART: Efficient Visual Generation with Hybrid Autoregressive Transformer.* https://arxiv.org/abs/2410.10812
- Deng et al. *Autoregressive Video Generation without Vector Quantization (NOVA).* https://arxiv.org/abs/2412.14169
- Yuan et al. *Lumos-1: On Autoregressive Video Generation with Discrete Diffusion.* 2025.

**Data & evaluation**
- Chen et al. *TalkVid: A Large-Scale Diversified Dataset for Audio-Driven Talking Head Synthesis.* CVPR 2026 Findings. https://arxiv.org/abs/2508.13618 · https://github.com/FreedomIntelligence/TalkVid
- *THEval: Evaluation Framework for Talking Head Video Generation.* https://arxiv.org/abs/2511.04520
- *Temporally-Aligned Evaluation for Audio-Driven Talking Head Generation.* https://arxiv.org/abs/2606.01031
- Rakesh et al. *Advancements in talking head generation: a comprehensive review.* The Visual Computer, 2025. https://doi.org/10.1007/s00371-025-04232-w

---

*Compiled with AI-assisted literature retrieval (Claude Code + Consensus + OpenAlex), September 2026.
Secondary-source numbers are marked; verify against the primary paper before publication.*
