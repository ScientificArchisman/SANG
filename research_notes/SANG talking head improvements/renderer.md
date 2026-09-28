# Renderer-side improvements for a frozen-LivePortrait talking-head system (2024-2026)

Scope: raise the image-quality ceiling of an audio-to-motion system that currently drives a frozen LivePortrait renderer (arXiv 2407.03168). Current TalkVid measurement: GT-motion CSIM 0.906. Budget: 1 GPU. Evidence labels: **MEASURED** means the paper's own table shows a number. **SPECULATIVE** means an inference, or a transfer of a measured result to our setting.

**Caveat that applies to every number below:** FID, FVD and CSIM depend heavily on each paper's protocol: the subset, the resolution, the crop, and whether 256 or 512 output is scored. FLOAT's HDTF FID is 21.100 in its own Table 1 but 9.164 in IMTalker's Table 2. Numbers from different papers must not be compared directly. Only within-table deltas are reliable. — [FLOAT](https://arxiv.org/abs/2412.01064); [IMTalker](https://arxiv.org/abs/2511.22167)

### Ranked shortlist (1 GPU), for the report writer
1. **Fine-tune LivePortrait's warping and SPADE decoder on talking data, adding a mouth/eye facial-component perceptual loss.** Cheapest option, and the motion space stays the same. The evidence is MEASURED in FLOAT's analogous autoencoder: HDTF FID 21.061→19.803 (Table 4). The transfer to LivePortrait is SPECULATIVE.
2. **Use multiple reference frames or a reference-correction branch alongside LivePortrait-style 3D-keypoint warping.** MEASURED in SynergyWarpNet: HDTF FID 36.49 (LivePortrait) → 34.76 (R=1) → 32.34 (R=2), Table 1. This needs a new trained module. No weights were confirmed.
3. **Swap to the IMTalker renderer.** It is Apache-2.0 with weights released, 124M params, and runs at 40 FPS on a 4090. It beats LivePortrait on HDTF self-reenactment (FID 7.426 vs 9.049) and on cross-reenactment CSIM (0.824 vs 0.789), Table 1 (MEASURED). The cost is that the audio-to-motion model must be retrained in IMTalker's motion space.
4. **Add a temporal refinement module (Teller ETM style) on top of LivePortrait frames.** MEASURED only as part of the full system. I found no isolated ETM ablation, so its effect on its own is SPECULATIVE.
5. **Use a diffusion renderer (X-NeMo, HunyuanPortrait, PersonaLive, or a Motar-style distilled one).** These have the highest texture and FVD ceilings but also the highest cost. On distortion metrics (PSNR/LPIPS) against LivePortrait the tables conflict. Training or distilling one is beyond a 1-GPU budget, apart from inference use of released weights.
6. **Run generic face restoration (GFPGAN/CodeFormer) as a post-pass.** SPECULATIVE. I found no measured FID or sync impact in 2024-2026 talking-head papers.

---

## Q1. Alternative one-shot renderers that take a compact motion code and beat LivePortrait

### Takeaway
Several 2025 renderers beat LivePortrait on the same-table comparisons. Three are directly usable on 1 GPU:
- **IMTalker's renderer** (Apache-2.0, weights released, real-time) is the most practical swap.
- **LIA-X** scales a latent-flow renderer to 0.9B params and beats LivePortrait on 512 px self-reenactment.
- **The diffusion animators** (X-NeMo, HunyuanPortrait, PersonaLive) win on FVD, identity or perceptual quality in their own tables. Third-party tables do not consistently confirm those wins, and they cost 10-100x more compute.

KDTalker also shows that LivePortrait is not uniformly better than face-vid2vid.

### Cited Findings
- **IMTalker (arXiv 2511.22167), MEASURED, Table 1: HDTF video-driven, 50 videos, outputs cropped to 256².**

  | | PSNR | SSIM | LPIPS | CSIM | FID | Cross CSIM | Cross FID |
  |---|---|---|---|---|---|---|---|
  | IMTalker | 28.458 | 0.899 | 0.037 | 0.898 | 7.426 | 0.824 | 53.137 |
  | LivePortrait | 27.173 | 0.878 | 0.043 | 0.896 | 9.049 | 0.789 | 53.144 |

  IMTalker also leads on cross-reenactment AED (0.581 vs 0.627) and MAE (7.553 vs 9.841). — [IMTalker](https://arxiv.org/abs/2511.22167)
- **IMTalker architecture and cost.** Implicit motion transfer via cross-attention, with an identity-adaptive module; the paper shows a "w/o IA" vs "w IA" ablation in Fig. 5. Input 256², output 512². The renderer has 124M params and the motion generator 39M. They run at 40 FPS and 42 FPS on an RTX 4090. The renderer was trained on about 660 h of data (VFHQ + VoxCeleb2 + MultiTalk) on 4 A100s for about 4 days. — [IMTalker](https://arxiv.org/abs/2511.22167)
- **IMTalker audio-driven results, MEASURED, Table 2 (HDTF / CelebV).**
  - IMTalker: FID 9.084 / 17.921, FVD 143.623 / 200.592, Sync-C 7.711 / 7.364, CSIM 0.869 / 0.814.
  - FLOAT: FID 9.164 / 18.272, FVD 198.964 / 228.215, CSIM 0.843 / 0.745.
  - Ditto (a LivePortrait-motion-space system): FID 11.746 / 25.367, CSIM 0.886 / 0.806.
  - — [IMTalker](https://arxiv.org/abs/2511.22167)
- **IMTalker license and weights.** Apache-2.0, and pretrained renderer weights are released. `renderer/inference.py` supports standalone video-driven rendering. The generator borrows FLOAT's architecture. — [IMTalker GitHub](https://github.com/bigai-nlco/IMTalker)
- **LIA-X (arXiv 2508.09959), MEASURED, Table 1: self-reenactment at 512 px.**

  | Dataset | Model | L1 | LPIPS | SSIM | PSNR | FID |
  |---|---|---|---|---|---|---|
  | VoxCelebHQ | LIA-X | 0.040 | 0.160 | 0.75 | 24.39 | 12.50 |
  | VoxCelebHQ | LivePortrait | 0.087 | 0.264 | 0.67 | 17.45 | 12.90 |
  | TalkingHead-1KH | LIA-X | 0.035 | 0.115 | 0.80 | 26.07 | 38.93 |
  | TalkingHead-1KH | LivePortrait | 0.052 | 0.120 | 0.73 | 20.26 | 39.98 |

  — [LIA-X](https://arxiv.org/abs/2508.09959)
- **LIA-X cross-reenactment, MEASURED, Table 2.** ID-similarity distance (lower is better): LIA-X 0.206, X-Portrait 0.217, LivePortrait 0.243. Image quality: LIA-X 58.74, X-Portrait 55.41, LivePortrait 51.41. — [LIA-X](https://arxiv.org/abs/2508.09959)
- **LIA-X scale and cost.** Up to about 0.9B params, trained on 8 A100s with about 94M frames and 55k identities. Scaling from 0.3B to 0.9B gives only minor gains (Tables 3-4). The authors list fixed resolution and a CNN architecture as limitations. — [LIA-X](https://arxiv.org/abs/2508.09959)
- **X-NeMo (arXiv 2507.23143; verified; the renderer used by Motar), MEASURED, Table 1 of its own paper, at 256² from 512² training.**

  | | L1 | SSIM | LPIPS | ID-SIM | AED/APD | EMO-SIM |
  |---|---|---|---|---|---|---|
  | X-NeMo | 0.055 | 0.826 | 0.168 | 0.787 | 0.039/3.42 | 0.65 |
  | LivePortrait | 0.074 | 0.770 | 0.236 | 0.702 | 0.055/6.61 | 0.48 |

  It is an SD-UNet diffusion model with a 1D identity-agnostic motion latent injected by cross-attention, trained on 8 A100s. The abstract says code and models are "available for research". — [X-NeMo](https://arxiv.org/abs/2507.23143)
- **Conflicting evidence on X-NeMo.** PersonaLive's Table 1 (TalkingHead-1KH self-reenactment and LV100 cross-reenactment) reverses the ordering on distortion metrics and identity:

  | | L1 | SSIM | LPIPS | Cross ID-SIM | Cross FVD |
  |---|---|---|---|---|---|
  | LivePortrait | 0.043 | 0.821 | 0.137 | 0.723 | 557.2 |
  | X-NeMo | 0.077 | 0.689 | 0.267 | 0.691 | 639.1 |
  | HunyuanPortrait | 0.043 | 0.801 | 0.137 | 0.644 | 620.4 |

  — [PersonaLive](https://arxiv.org/abs/2512.11253), contradicting [X-NeMo](https://arxiv.org/abs/2507.23143)
- **HunyuanPortrait (arXiv 2503.18860), MEASURED, Table 1 at 512².**

  | | FID-VID | FVD | PSNR | SSIM | LPIPS | ID sim (x10⁻¹) |
  |---|---|---|---|---|---|---|
  | HunyuanPortrait | 75.81 | 333.48 | 32.98 | 0.81 | 0.11 | 8.87 |
  | LivePortrait | 82.71 | 483.38 | 31.41 | 0.72 | 0.22 | 8.71 |

  It is built on SVD and trained on **128 A100s for 3 days**. — [HunyuanPortrait](https://arxiv.org/abs/2503.18860)
- **PersonaLive (arXiv 2512.11253), MEASURED, Table 1.**
  - Self-reenactment LPIPS 0.129 vs LivePortrait 0.137.
  - Cross FVD 520.6 vs 557.2 and tLP 12.83 vs 13.51.
  - Cross ID-SIM 0.698 vs LivePortrait 0.723, i.e. lower than LivePortrait.
  - Runs at 15.82 FPS on an H100 with 4 denoising steps. It uses LivePortrait's implicit 3D keypoints for head pose and is initialized from X-NeMo weights. Trained on 8 H100s. It is a CVPR 2026 paper with a GitHub repo.
  - The authors note that LivePortrait "runs significantly faster ... [but] often lack[s] fine-grained details".
  - — [PersonaLive](https://arxiv.org/abs/2512.11253); [GitHub](https://github.com/GVCLab/PersonaLive)
- **FLOAT motion autoencoder (arXiv 2412.01064), MEASURED, Table 4: same-identity reconstruction on HDTF / RAVDESS / VFHQ.**
  - FLOAT AE: FID 19.803 / 23.350 / 43.992, FVD 147.089 / 100.345 / 291.560, LPIPS 0.108 / 0.062 / 0.161.
  - LIA at 256: FID 47.481 / 67.541 / 89.209.
  - There is no direct LivePortrait comparison in this table. FLOAT's audio-driven HDTF result is FID 21.100, FVD 162.052, CSIM 0.843 (Table 1).
  - — [FLOAT](https://arxiv.org/abs/2412.01064)
- **KDTalker (arXiv 2503.12963), MEASURED, Table 5 renderer ablation (same motion model, HDTF).**
  - Face-vid2vid: LSE-C 5.579, FID 8.625, CSIM 0.940.
  - LivePortrait: LSE-C 7.326, FID 9.756, CSIM 0.949.
  - So LivePortrait gives better sync but slightly worse FID. KDTalker's main result (Table 1) is FID 9.756, CSIM 0.949, 21.678 FPS on a 4090.
  - — [KDTalker](https://arxiv.org/abs/2503.12963)
- **SoulX-FlashHead (arXiv 2602.07449), MEASURED, Table 3, HDTF.**
  - Pro, non-streaming: FID 8.31, FVD 103.14, Sync-C 6.04.
  - Pro, streaming: FID 9.97, FVD 111.38.
  - Lite, streaming: FID 11.37, 96 FPS.
  - Ditto: FID 12.35, FVD 199.13.
  - It is a 1.3B holistic model trained on the 782 h VividHead set.
  - — [SoulX-FlashHead](https://arxiv.org/abs/2602.07449)

### Inferences
- IMTalker is the only verified renderer that meets all three of these conditions: (a) it beats LivePortrait in a third-party-style same-table HDTF comparison, (b) it is permissively licensed with weights, and (c) it runs in real time on one consumer GPU. Adopting it means retraining the audio-to-motion model in IMTalker's motion space. IMTalker's own generator took about 2 days on 4 A100s, so on 1 GPU expect roughly a week or more (SPECULATIVE).
- The CSIM gain from IMTalker in self-reenactment is tiny (0.898 vs 0.896). Its larger identity gain is in cross-reenactment (+0.035), which is the regime closest to audio-driven use from one photo.
- Diffusion renderers win on FVD and texture, but they conflict with LivePortrait on distortion metrics and ID in third-party tables (PersonaLive vs X-NeMo). None can be trained on 1 GPU. Inference-only use of released X-NeMo or PersonaLive weights is possible but slow: about 1.3 FPS for X-NeMo and 15.8 FPS for PersonaLive on an H100, per PersonaLive Table 1.
- KDTalker's FID of 9.76 already uses LivePortrait, so beating it on HDTF is mostly a matter of motion quality plus renderer detail (SPECULATIVE).

### Gaps
- License and weight status for LIA-X, X-NeMo (only "available for research"), HunyuanPortrait, and X-Portrait 2 were not verified in this session. FLOAT's license materials are reported as internally inconsistent. — [third-party note](https://github.com/richiejp/rust-vs-cpp-analysis/blob/main/README.md)
- I did not retrieve quantitative tables for X-Portrait 2, Follow-Your-Emoji or Portrait4D-v2 beyond their appearance as baselines above. Follow-Your-Emoji (FollowYE) in PersonaLive Table 1: L1 0.045, LPIPS 0.144, cross ID-SIM 0.773, the highest in that table.

---

## Q2. Lightweight refiners on warped frames (mouth/teeth, face restoration), and whether they hurt sync

### Takeaway
The best-evidenced "refinement" for teeth and eyes is not a post-hoc restorer. It is a **facial-component perceptual loss inside the renderer's training**, as in FLOAT. I found no 2024-2026 paper that measures the FID or sync effect of GFPGAN or CodeFormer post-processing on a warping renderer. Lip-sync and dubbing literature reports a real trade-off between lip-sync score and dental clarity.

### Cited Findings
- **FLOAT facial-component perceptual loss (L_comp-lp), MEASURED, Table 4.** Adding it to the motion autoencoder changes HDTF reconstruction FID 21.061 → 19.803, FVD 150.340 → 147.089, LPIPS 0.110 → 0.108. On RAVDESS FID goes 28.866 → 23.350. The paper says it "significantly improves the image fidelity of facial component (e.g., teeth...) and fine-grained motion (eyeball movement)" (Fig. 10). — [FLOAT](https://arxiv.org/abs/2412.01064)
- **MuseTalk (arXiv 2410.10122).** GAN-based real-time dubbing "sacrifice[s] lip-sync accuracy or dental details". MuseTalk's Dynamic Margin Sampling is designed to balance "audio-visual synchronization and dental clarity". Its OpenReview version says Distinct-Mouth Sampling "improves ... LSE-C, but compromises visual fidelity". — [MuseTalk](https://arxiv.org/abs/2410.10122); [OpenReview PDF](https://openreview.net/pdf/47210a00584b01e752037813ffa5a2e67695c89f.pdf)
- **EfficientSync (arXiv 2608.18832).** It argues that heavy GAN or diffusion decoders "hallucinat[e] intra-oral details such as teeth". Instead it transfers textures from multiple aligned references (a Dynamic Texture Mixer) at 166 FPS. The abstract claims state-of-the-art visual quality on HDTF and VFHQ; I did not retrieve the numbers. — [EfficientSync](https://arxiv.org/abs/2608.18832)
- **Teller's ETM is a one-step refiner on LivePortrait outputs.** A VAE encoder, a 3D U-Net with temporal self-attention, and a VAE decoder are trained with a region-masked reconstruction loss between LivePortrait-reconstructed frames and GT. It costs 71 ms per 200 ms chunk, of which the VAE takes 25 ms and the temporal module 21 ms. — [Teller](https://arxiv.org/abs/2503.18429)
- **CodeFormer++ (arXiv 2510.04410)** names a "trade-off between visual quality and identity fidelity" in generative-prior face restoration. — [CodeFormer++](https://arxiv.org/abs/2510.04410)

### Inferences
- SPECULATIVE: generic restorers (GFPGAN/CodeFormer) are frame-independent, so they would likely add flicker and could shift identity. That fits CodeFormer++'s identity/quality trade-off, but I found no talking-head measurement of it. If a restorer is used, blend it only inside a mouth mask, and gate its adoption on measured Sync-C, CSIM and FVD.
- SPECULATIVE but best-grounded: fine-tune LivePortrait's SPADE decoder, and optionally the warping module, with a mouth/eye-region perceptual loss plus a GAN loss on TalkVid/HDTF-like data. This keeps the motion code fixed, so the audio-to-motion model is untouched, and it mirrors FLOAT's measured gain. Cost on 1 GPU is likely a few days, since LivePortrait's decoder is small.

### Gaps
- There is no measured study of GFPGAN/CodeFormer effects on FID, Sync-C or CSIM for LivePortrait-rendered talking heads (searched; none found).
- No isolated mouth-inpainting refiner applied on top of LivePortrait with reported numbers was found.

---

## Q3. Fine-tuning LivePortrait, or distilling a video-diffusion renderer

### Takeaway
There is no paper that publishes a LivePortrait-decoder fine-tune on talking data. The closest measured analogs are FLOAT's component loss (above) and the renderer-retraining costs of competing systems: IMTalker at 4×A100 for about 4 days, LIA-X on 8×A100, and X-NeMo on 8×A100. Diffusion-renderer distillation (Motar's DMD/self-forcing, PersonaLive's few-step appearance distillation) is proven, but it starts from multi-GPU teachers. The resulting quality is capped by an SD-class backbone.

### Cited Findings
- **Motar (arXiv 2609.10317; verified, "Decoupled Self-Forcing Distillation for Streaming Talking Head Generation", 2026-09-09).**
  - It uses X-NeMo's 512-D motion encoder and diffusion decoder as the renderer. The renderer is distilled into a block-causal 4-step student with DMD/self-forcing (64-frame rolling KV cache, 8-frame blocks). Training uses 4 A100s. It runs at 15.4 FPS with 1.39 s latency on an H200. The motion generator is only 77M params, versus the 1.7B renderer.
  - — [Motar](https://arxiv.org/abs/2609.10317)
- **Motar results, MEASURED, Table 1 (MEAD / Hallo3 subsets, not HDTF).**
  - Motar: FID 45.0 / 50.5, FVD 185.9 / 424.2, CSIM 0.894 / 0.831.
  - GT-motion oracle through the teacher renderer: FID 44.2 / 50.2, FVD 185.1 / 467.6, CSIM 0.899 / 0.766.
  - The authors say visual quality is "capped" by the SD-based renderer: "A stronger renderer would lift visual fidelity without any change to the motion side."
  - — [Motar](https://arxiv.org/abs/2609.10317)
- **Motar ablation, MEASURED, Table 2.** Supervising motion through the frozen renderer with DMD (L_DMD^motion) improves motion FMD 18.21 → 17.07 and Std-R 0.590 → 0.704. — [Motar](https://arxiv.org/abs/2609.10317)
- **PersonaLive.** Few-step appearance distillation (4 steps), plus micro-chunk autoregressive streaming, plus historical keyframes. It gets a "7-22x speedup" and was trained on 8 H100s. — [PersonaLive](https://arxiv.org/abs/2512.11253)
- **IMTalker renderer cost.** About 4 days on 4 A100s at batch 16 on 660 h of data. — [IMTalker](https://arxiv.org/abs/2511.22167)
- **HunyuanPortrait cost.** 128 A100s for 3 days. — [HunyuanPortrait](https://arxiv.org/abs/2503.18860)

### Inferences
- The oracle-vs-generated comparison is the right diagnostic for us. Our TalkVid GT-motion CSIM of 0.906 is the renderer ceiling, and Motar shows its generated motion reaching the oracle. If our generated-motion metrics are already near the GT-motion ones, only renderer changes will help (SPECULATIVE by analogy).
- Motar's "supervise motion through the frozen renderer" idea (a video-space loss backpropagated through a frozen LivePortrait) is a cheap 1-GPU, motion-side lever that raises rendered quality without touching the renderer. SPECULATIVE for LivePortrait, but LivePortrait's renderer is differentiable.
- Distilling a video-diffusion renderer on 1 GPU is not realistic. Inference-time use of released X-NeMo or PersonaLive weights is the only diffusion option within budget.

### Gaps
- No published fine-tune of LivePortrait's SPADE decoder or warping module on talking-head data with reported numbers was found.
- LivePortrait's own training cost was not retrieved in this session.

---

## Q4. Temporal consistency fixes for frame-by-frame warping renderers

### Takeaway
The main documented add-on is Teller's Efficient Temporal Module: one-step temporal attention in VAE latent space, applied to LivePortrait outputs. Its standalone effect is not ablated in the paper. Diffusion renderers report better FVD and tLP than LivePortrait. Multi-reference warping (SynergyWarpNet) also claims better temporal consistency, shown only as a figure.

### Cited Findings
- **Teller (arXiv 2503.18429), MEASURED for the full system only.**
  - HDTF (Table 1): FID 21.352, FVD 173.463, Sync-C 7.696, 0.92 s per second of video.
  - Its training is heavy: the AR stage on 8×8 A800 and the ETM stage on 8×8 A800.
  - ETM's purpose is to keep neck, earrings and accessories physically consistent.
  - — [Teller](https://arxiv.org/abs/2503.18429)
- **PersonaLive, MEASURED, Table 1.** Temporal LPIPS (tLP, x10⁻³): LivePortrait 20.40 in self-reenactment and 13.51 in cross-reenactment; PersonaLive 21.31 and 12.83. So a frame-wise LivePortrait is not worse on tLP in self-reenactment. — [PersonaLive](https://arxiv.org/abs/2512.11253)
- **HunyuanPortrait, MEASURED, Table 1.** User-rated temporal smoothness: LivePortrait 4.06 vs HunyuanPortrait 4.61. FVD 483.38 vs 333.48. — [HunyuanPortrait](https://arxiv.org/abs/2503.18860)
- **FLOAT, MEASURED, Table 5.** Motion-side sampling steps change FVD: 2 NFE gives 178.831 and 10 NFE gives 162.052. The authors state that "image fidelity is determined by the auto-encoder". — [FLOAT](https://arxiv.org/abs/2412.01064)

### Inferences
- Much of the "flicker" in LivePortrait pipelines may be motion jitter rather than renderer jitter, because PersonaLive's self-reenactment tLP for LivePortrait is competitive. Smoothing the motion code (keypoint filtering) is a zero-cost first step. SPECULATIVE.
- A small Teller-style temporal refiner, trained on LivePortrait-reconstructed versus GT frames, is feasible on 1 GPU at reduced scale. There is no isolated evidence of its gain.

### Gaps
- No isolated ETM on/off ablation with FID or FVD was found in Teller.

---

## Q5. Identity preservation (CSIM) and source-image handling

### Takeaway
Adding reference frames is the one measured, renderer-local way to improve identity and quality over single-source warping. SynergyWarpNet improves over LivePortrait at R=1 and further at R=2. IMTalker's identity-adaptive module improves cross-reenactment CSIM over LivePortrait (0.824 vs 0.789). Diffusion methods trade identity against temporal stability through their reference and history mechanisms.

### Cited Findings
- **SynergyWarpNet (arXiv 2512.17331), MEASURED, Table 1 at 256.**

  | Dataset | Model | LPIPS | PSNR | FID |
  |---|---|---|---|---|
  | HDTF | LivePortrait | 0.1817 | 29.15 | 36.49 |
  | HDTF | Ours R=1 | 0.1527 | 30.68 | 34.76 |
  | HDTF | Ours R=2 | 0.1430 | 30.98 | 32.34 |
  | VFHQ | LivePortrait | 0.3953 | — | 31.39 |
  | VFHQ | Ours R=1 | 0.2798 | — | 27.42 |
  | VFHQ | Ours R=2 | 0.2429 | — | 21.60 |

  The architecture is 3D-keypoint explicit warping, plus reference-augmented cross-attention correction, plus confidence-guided fusion. Code and license status are unknown. — [SynergyWarpNet](https://arxiv.org/abs/2512.17331)
- **IMTalker identity-adaptive module.** Qualitative ablation in Fig. 5, and cross-reenactment CSIM 0.824 vs LivePortrait 0.789 in Table 1. In the audio-driven setting it has lower CSIM than Ditto (0.869 vs 0.886), which the authors attribute to Ditto's smaller head motion: "the CSIM metric is highly sensitive to pose variations". — [IMTalker](https://arxiv.org/abs/2511.22167)
- **PersonaLive, MEASURED, Appendix Table 3.** More history keyframes (τ=15) improve FVD to 510.9 but reduce ID-SIM to 0.6924. Fewer keyframes (τ=20) give ID-SIM 0.7159 but FVD 529.2. — [PersonaLive](https://arxiv.org/abs/2512.11253)
- **X-NeMo, MEASURED, Table 2 ablations.** Removing color/spatial augmentation drops ID-SIM 0.787 → 0.724. Replacing cross-attention with ControlNet-style spatial control drops it to 0.697. — [X-NeMo](https://arxiv.org/abs/2507.23143)
- **LIA-X.** Its "edit-warp-render" step aligns the source to the first driving frame before warping. The authors credit this for robustness to large source/driving pose gaps (qualitative, Fig. 7). — [LIA-X](https://arxiv.org/abs/2508.09959)

### Inferences
- For single-photo input, "multiple references" can be synthesized. One option is to generate a few pose or expression variants of the source (e.g. an open-mouth or teeth-visible frame) and feed them to a multi-reference renderer. This targets the mouth-interior weakness directly. SPECULATIVE.
- CSIM is pose-sensitive (IMTalker). Some of the gap between our GT-motion CSIM of 0.906 and KDTalker's 0.949 may come from TalkVid's larger head motion versus HDTF, not from renderer quality. Evaluate both on the same HDTF protocol before attributing the gap (SPECULATIVE).

### Gaps
- Code, weights and license for SynergyWarpNet were not found.
- No verified numbers for Portrait4D-v2 or X-Portrait 2 identity metrics against LivePortrait.
- LIA-X's weights and license were not confirmed.
