# Renderer-side image-quality upgrades for SANG-M (LivePortrait renderer)

Status: COMPLETE (2026-10-07). Scope: renderer-side ways to raise sharpness, teeth/mouth-interior quality, identity and FID/FVD without breaking LSE or the audio→42-d keypoint motion pipeline. Not repeated here (covered earlier): IMTalker, LIA-X, X-NeMo, LatentSync/MuseTalk post-hoc lip refiners, GFPGAN/CodeFormer-after-Wav2Lip, source-photo mouth state.
Evidence tags: [M] = number measured in a primary source (paper table / API); [C] = author or README claim with no number; [I] = my inference. All arXiv IDs were checked against arxiv.org abstract pages, the arXiv API, or the downloaded PDF.
Caveat that applies throughout: FID/FVD values are only comparable **within one table**. For example, LivePortrait's HDTF FID is 36.4944 in SynergyWarpNet's protocol, 25.07 in FRVD's and 9.049 in IMTalker's.

## Q1. Fine-tuning LivePortrait decoder/warping with component/GAN/mouth losses; LivePortrait's own recipe

### Takeaway
LivePortrait was **already trained** with face- and lip-region perceptual and GAN losses, a lip discriminator and a face-id loss. A fine-tune therefore improves things through domain adaptation (talking-head data, real 512-px supervision) and a stronger mouth/teeth weighting, not through a new kind of loss. I found **no paper or fork that reports FID/FVD numbers for a fine-tuned LivePortrait decoder**. The official team has not released training code ("It's challenging"). One community user fine-tuned all Stage-I modules to fix mouth artifacts and reported qualitative success only. Fine-tuning only the decoder G, with the motion extractor M, the appearance extractor F and the warping module W frozen, keeps the keypoint space and the warp field unchanged, so the DiT needs no retraining.

### Cited Findings
- [M] LivePortrait Stage-I recipe (this is our renderer). It starts from face-vid2vid. Losses: keypoint equivariance L_E, keypoint prior L_L, head pose L_H, deformation prior L_Δ, a **cascaded perceptual loss L_P,cascade and a cascaded GAN loss L_G,cascade over the global, face and lip regions** (discriminators D_global, D_face and D_lip trained from scratch; face and lip regions defined by 2D landmarks), a face-id loss and a landmark-guided loss L_guide. — [LivePortrait 2407.03168 §3.2](https://arxiv.org/abs/2407.03168)
- [M] Compute and data:
  - Stage I was trained from scratch on 8× A100 for about 10 days. Stage II (stitching and retargeting only) took about 2 days.
  - Input is aligned and cropped to 256×256, batch 104. A final PixelShuffle layer in the SPADE decoder upsamples 256→512, so the 512 output adds no new source information.
  - Adam, lr 2e-4, β=(0.5, 0.999).
  - Data: 69M frames (92M before KVQ quality filtering) from about 18.9K identities, plus 60K styled portraits. Sources include VoxCeleb, MEAD, RAVDESS, AAHQ, private 4K portrait video, 200 h of talking-head video and private LightStage data.
  — [LivePortrait 2407.03168](https://arxiv.org/abs/2407.03168)
- [M] LivePortrait Table 2 (self-reenactment):
  - TalkingHead-1KH: PSNR 32.0082, SSIM 0.8193, LPIPS 0.0664, CSIM 0.9125.
  - VFHQ: PSNR 31.5616, SSIM 0.7653, LPIPS 0.0798, CSIM 0.9121.
  - Table 3 (cross-reenactment) FID: 58.0370 on TH-1KH and 56.4165 on VFHQ. The diffusion model AniPortrait gets 47.8739 on TH-1KH, which the authors acknowledge beats them on FID there.

  — [LivePortrait 2407.03168](https://arxiv.org/abs/2407.03168)
- [C] Maintainer, issue #468 (training-code release): "It's challenging : |". Issues #117 "Training code", #478 and #553 "Train on custom dataset" are all open. No official training code exists. — [LivePortrait #468](https://github.com/KlingAIResearch/LivePortrait/issues/468)
- [C] Community fine-tune (issue #506, user ZardZen, Jan 2026):
  - Fine-tuned all Stage-I modules "to solve the artifacts of mouth" for the animal model, with paper settings (lr 2e-4, β1 0.5, β2 0.999).
  - Losses: equivariance, keypoint prior, deformation prior, landmark-guided, perceptual and GAN.
  - Data: about 10k human videos plus 1–2k animal videos. The user said "the result seems ok" and shared a demo video, but gave no metrics.
  - The user suggests that if only the decoder is trained, the keypoint losses are unnecessary: "just freeze other modules except decoder module, without these losses, I think it would work too".
  - The repo moved from KwaiVGI to **KlingAIResearch/LivePortrait**.

  — [LivePortrait #506](https://github.com/KlingAIResearch/LivePortrait/issues/506)
- [M] Licence: LivePortrait code is MIT. The LICENSE file adds that InsightFace models are "for non-commercial research purposes only" and must be replaced for commercial use. GitHub reports the licence as NOASSERTION; 19,169 stars. — [LivePortrait LICENSE](https://github.com/KlingAIResearch/LivePortrait/blob/main/LICENSE)
- [C] Issue #427 "Blur face": a user reports the output is "blurrier than the original" even when the face is about 128×128, small enough that no downsampling should occur. This suggests the softness is not only a resolution artifact. — [LivePortrait #427](https://github.com/KlingAIResearch/LivePortrait/issues/427)
- [M] Data point from a related paper: PersonaLive's few-step distillation adds a StyleGAN2 discriminator initialised from FFHQ weights. That is a cheap, proven way to add a strong pretrained face discriminator to a fine-tune. — [PersonaLive 2512.11253](https://arxiv.org/abs/2512.11253)

### Inferences
- [I] **Recommended fine-tune design.**
  - Freeze F, M, W and the stitching module. Fine-tune only the SPADE decoder G, adding W only as a second ablation.
  - Train in self-reenactment mode on 512-px talking-head crops (HDTF-train, TalkVid, CelebV-HQ), using LivePortrait's own pipeline to extract keypoints.
  - Losses: L1 + VGG perceptual (global, face, mouth crop) + GAN (global discriminator and a new mouth-crop discriminator, the latter optionally initialised from FFHQ StyleGAN2) + ArcFace face-id loss.
  - Optionally add a temporal discriminator or a frame-difference loss on short clips to target FVD.
  - Because the keypoints, and therefore the warp, are unchanged, motion following and LSE should be preserved by construction. Lip shape comes from the flow, not the decoder.
  - Risk: the decoder can learn to "re-draw" the mouth inconsistently with the warp, for example by adding teeth on a slight jaw opening. Guard against this with the TalkVid re-render ceiling test (CSIM 0.906–0.911, mouth corr 0.82–0.85) as a regression gate.
- [I] **Cost estimate** (not measured in any source): G holds a large share of the ~130M parameters, and training at batch 8–16 on 512 crops fits on one 80 GB GPU. Expect 1–3 GPU-days for 50–150k iterations on 2–10k clips, compared with the original 8 A100 × 10 days from scratch. Discriminators start from scratch (or from FFHQ StyleGAN2 weights), so budget for GAN instability; a short L1+perceptual warm-up is advisable.
- [I] Because the original already has D_lip, simply adding "a lip GAN" is not new. The levers that could add something are (a) a higher-resolution mouth crop (e.g. a 128-px crop from the 512 output), (b) supervising with sharp 512 ground truth from talking-head data instead of the original mix, and (c) negative examples that penalise teeth when the GT mouth interior shows none (the "teeth plate" issue).

### Gaps
- No paper or fork with measured FID/FVD/sharpness gains from fine-tuning LivePortrait's decoder or warping network turned up. Issue #350 ("Training of the second stage", 28 comments) was not read in full.
- No quantitative evidence on how much motion-following fidelity a decoder-only fine-tune loses. It has to be measured on our own ceiling test.
- The LivePortrait loss weights are not given in the main text. The appendix (§D, cited in #506) has the animal-model loss differences and was not read.

## Q2. Higher-resolution rendering and (video) super-resolution

### Takeaway
LivePortrait was trained only at 256 in / 512 out, and the maintainers point users to SR for 1024 (issue #317). No 512-in / 1024-out weights exist, and building them means retraining F and G at higher resolution. For post-hoc SR:
- Temporal face-video SR/restoration (KEEP, SVFR, SeedVR2) clearly beats per-frame restorers on identity similarity and temporal stability.
- Per-frame CodeFormer lowers identity similarity: IDS 0.6272 vs 0.7960 for KEEP on VFHQ.
- **No paper I found measures SyncNet/LSE before and after video SR on talking-head outputs**, so the effect on sync has to be measured by us.

### Cited Findings
- [C] Issue #317 "Is LivePortrait 512px output max?", maintainer zzzweakman (2024-08-27): "We have currently only trained the model for 512x512 resolution. If you want to obtain images with a resolution of 1024x1024, you can use some super-resolution models for further processing." Issue #351 ("pixel boost") is unanswered. — [LivePortrait #317](https://github.com/KlingAIResearch/LivePortrait/issues/317)
- [M] **KEEP** (ECCV 2024, 2408.05205; Kalman-filter feature propagation for video face SR) on VFHQ-mild. Columns: PSNR / SSIM / LPIPS / IDS / AKD / σIDS (×1e-2) / σAKD.
  - KEEP: 27.9994 / 0.8267 / 0.1619 / 0.7960 / 8.8182 / 3.6866 / 3.2538
  - GFPGAN: 26.2933 / 0.7795 / 0.2482 / 0.7437 / 10.5467 / 4.5700 / 3.6482
  - CodeFormer: 24.6597 / 0.7454 / 0.2742 / 0.6272 / 11.4983 / 6.3726 / 3.6927
  - BasicVSR++: 27.1996 / 0.8057 / 0.1958 / 0.7641 / 11.3136 / 5.2543 / 4.6425

  Repo jnjaby/KEEP: GitHub licence field "NOASSERTION", 504 stars. — [KEEP 2408.05205](https://arxiv.org/abs/2408.05205); [GitHub jnjaby/KEEP](https://github.com/jnjaby/KEEP)
- [M] **SVFR** (2501.01235; SVD-based unified video face restoration) on VFHQ-test blind face restoration. Columns: PSNR / SSIM / LPIPS / IDS / VIDD (adjacent-frame identity jitter) / FVD.
  - SVFR: 29.563 / 0.862 / 0.223 / 0.902 / 0.479 / 89.316
  - PGTFormer: 28.996 / 0.843 / 0.248 / 0.845 / 0.577 / 154.857
  - KEEP: 27.335 / 0.813 / 0.259 / 0.790 / 0.787 / 399.239
  - CodeFormer: 26.528 / 0.762 / 0.361 / 0.784 / 0.700 / 379.53
  - GPEN: 26.237 / 0.795 / 0.320 / 0.786 / 0.575 / 412.81

  Repo wangzhiyaoo/SVFR: no licence field, 872 stars. — [SVFR 2501.01235](https://arxiv.org/abs/2501.01235); [GitHub wangzhiyaoo/SVFR](https://github.com/wangzhiyaoo/SVFR)
- [M] **SeedVR2** (2506.05301, ByteDance Seed):
  - One-step diffusion video restoration via adversarial post-training; 3B and 7B variants; adaptive window attention handles 1080p in one forward step.
  - Repo ByteDance-Seed/SeedVR is **Apache-2.0** (1,387 stars).
  - Its Table 2 (relative preference vs SeedVR2): VEnhancer-50 at -82% / -86% / -94% (fidelity / quality / overall); Upscale-A-Video-50 at 0% / -26% / -26%.
  - It has no face-specific or lip-sync metric.

  — [SeedVR2 2506.05301](https://arxiv.org/abs/2506.05301); [GitHub ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR)
- [M] Verified IDs of the other requested VSR methods. None reports SyncNet/LSE or talking-head FID in what I checked: Upscale-A-Video 2312.06640, VEnhancer 2407.07667, STAR 2501.02976, SeedVR 2501.01320. — [2312.06640](https://arxiv.org/abs/2312.06640); [2407.07667](https://arxiv.org/abs/2407.07667); [2501.02976](https://arxiv.org/abs/2501.02976); [2501.01320](https://arxiv.org/abs/2501.01320)
- [M] **Hallo2** (2410.07718) reaches 4K by extending a CodeFormer-style VQGAN codebook-prediction restorer with temporal alignment; only the temporal-alignment and codebook-prediction weights are trained. HDTF (FID / FVD / Sync-C / Sync-D):
  - Hallo2: 16.616 / 239.517 / 7.379 / 7.697
  - Hallo: 16.748 / 366.066 / 7.268 / 7.714
  - Real video: Sync-C 8.377

  I found no ablation that isolates the SR stage's effect on Sync-C. — [Hallo2 2410.07718](https://arxiv.org/abs/2410.07718)
- [M] Timing from HDTR-Net's table (s/frame; hardware not extracted): Real-ESRGAN 0.0622, ESRGAN 0.0597, GFPGAN 1.2332. — [HDTR-Net 2309.07495](https://arxiv.org/abs/2309.07495)

### Inferences
- [I] **Expected effect of SR on our metrics.**
  - SyncNet (LSE-C/D) sees low-resolution mouth crops, so geometry-preserving SR should leave LSE roughly unchanged.
  - Generative restorers that redraw the mouth (CodeFormer-like codebooks) can shift lip shape. That matches the earlier finding that CodeFormer-after-Wav2Lip damaged FID (10.85→26.36), and KEEP's table shows CodeFormer also drops identity (IDS 0.6272).
  - Prefer temporal, fidelity-oriented VSR (SeedVR2-3B one-step under Apache-2.0, or KEEP for internal tests), applied to the 512 face crop **before** paste-back.
  - Measure LSE-C/D, CSIM, FID and FVD on HDTF with and without SR.
- [I] FID is computed on Inception features at 299 px. If our evaluation resizes faces to ≤512, part of SR's gain shows up as sharpening and texture rather than resolution. The FVD gain depends on temporal stability, which is exactly where per-frame restorers fail (σIDS, VIDD above).
- [I] A 512-in / 1024-out LivePortrait means retraining the appearance encoder (feature volume 32×16×64×64 from a 256 crop) and the decoder. That is roughly the original Stage-I budget minus the motion extractor and well beyond one-GPU scale. It is not recommended.

### Gaps
- No source measures SyncNet/LSE after video SR (KEEP, SVFR, SeedVR2, STAR, Upscale-A-Video, VEnhancer, RealBasicVSR) on talking-head outputs.
- SeedVR2-3B and KEEP throughput on one GPU at 512 px was not extracted. KEEP's custom licence terms were not read; NOASSERTION usually means a non-commercial S-Lab licence, which needs checking.
- RealBasicVSR (2111.12704) is outside the 2023–2026 window and was not examined.

## Q3. Motion-conditioned diffusion renderers that could take LivePortrait keypoints

### Takeaway
- Only **PersonaLive** (CVPR 2026, Apache-2.0) feeds LivePortrait 3D implicit keypoints directly, and only for head pose. Facial dynamics come from an X-NeMo-style implicit encoder applied to *driving frames*.
- Every other candidate is driven by driving video frames or face crops: HunyuanPortrait, FantasyPortrait, X-Portrait, Follow-Your-Emoji, MegActor-Σ, Wan-Animate. SkyReels-A1 is driven by explicit 3D face renders instead.
- The practical form for SANG is therefore a **two-pass hybrid**: render with LivePortrait from our motion, then use that video as the driving input to a diffusion renderer.
- The numbers are mixed. In PersonaLive's independent test, LivePortrait has better identity similarity (ID-SIM 0.723) than PersonaLive (0.698), HunyuanPortrait (0.644) or X-Portrait (0.678), and **better FVD than all diffusion renderers except PersonaLive** (557.2 vs 520.6). Only the large video-DiT models (FantasyPortrait, Wan-Animate, SkyReels-A1) clearly beat it on FID/FVD, and they run at seconds per frame.
- None of these papers reports lip-sync (LSE) for video-driven reenactment.

### Cited Findings
- [M] **Playmate** (ICML 2025, 2502.07203) is the closest published analogue of SANG-M: an audio-conditioned DiT generates motion in the face-vid2vid/LivePortrait decoupled implicit-3D space, which is then rendered.
  - HDTF Table 1 (FID / FVD / Sync-C / Sync-D / CSIM / LPIPS):
    - Playmate: 19.138 / 231.048 / 8.580 / 6.985 / 0.848 / 0.099
    - JoyVASA (also LivePortrait-space): 29.581 / 306.683 / 8.522 / 7.215 / 0.781 / 0.157
    - Hallo2: 30.768 / 288.385 / 7.754 / 7.649 / 0.822 / 0.138
    - Sonic: 29.189 / 305.867 / 9.139 / 6.549 / 0.783 / 0.149
  - Training: 4× A100 for 3 days, then 2× A100 for 2 days; 256 crop, 512 output.

  — [Playmate 2502.07203](https://arxiv.org/abs/2502.07203)
- [C] Playmate "adopt[s] and enhance[s] the decoupled facial representation proposed by face-vid2vid and LivePortrait". The text I extracted does not say whether the renderer weights were retrained, so the 10-point FID gap over JoyVASA cannot be attributed to the renderer. — [Playmate 2502.07203](https://arxiv.org/abs/2502.07203)
- [M] **PersonaLive** (CVPR 2026, 2512.11253):
  - Architecture: ReferenceNet-style diffusion renderer. Driving signals are an X-NeMo-style implicit facial representation plus **LivePortrait 3D implicit keypoints** (through a pose guider).
  - Speed: "fewer-step appearance distillation" with an FFHQ-pretrained StyleGAN2 discriminator brings sampling to 4 denoising steps; generation streams in autoregressive micro-chunks.
  - Training: VFHQ + NerSemble + DH-FaceVid-1K at 512×512 on 8× H100.
  - Repo: Apache-2.0, 3,943 stars, last push 2026-08-28.

  — [PersonaLive 2512.11253](https://arxiv.org/abs/2512.11253); [GitHub GVCLab/PersonaLive](https://github.com/GVCLab/PersonaLive)
- [M] PersonaLive Table 1 (TalkingHead-1KH test; speed on one H100). Columns: self-reenactment L1 / SSIM / LPIPS / tLP (×1e-3); cross-reenactment ID-SIM / AED / APD / FVD / tLP; then FPS / latency (s).
  - LivePortrait: 0.043 / 0.821 / 0.137 / 20.40; 0.723 / 0.729 / 0.027 / 557.2 / 13.51; FPS and latency not reported (frame-wise GAN)
  - X-Portrait: 0.049 / 0.777 / 0.173 / 25.87; 0.678 / 0.823 / 0.061 / 587.8 / 24.52; 0.851 FPS / 14.10 s
  - Follow-Your-Emoji: 0.045 / 0.803 / 0.144 / 26.92; 0.773 / 0.911 / 0.043 / 696.5 / 35.13; 1.558 FPS / 7.793 s
  - MegActor-Σ: 0.055 / 0.766 / 0.183 / 23.55; 0.606 / 0.855 / 0.079 / 585.3 / 28.86; 2.216 FPS / 6.918 s
  - X-NeMo: 0.077 / 0.689 / 0.267 / 25.11; 0.691 / 0.679 / 0.022 / 639.1 / 18.10; 1.281 FPS / 15.32 s
  - HunyuanPortrait: 0.043 / 0.801 / 0.137 / 22.33; 0.644 / 0.804 / 0.069 / 620.4 / 16.84; 1.443 FPS / 14.91 s
  - PersonaLive: 0.039 / 0.807 / 0.129 / 21.31; 0.698 / 0.703 / 0.030 / 520.6 / 12.83; 15.82 FPS / 0.253 s

  The table footnote says LivePortrait's portraits "often lack fine-grained details" [C]. — [PersonaLive 2512.11253](https://arxiv.org/abs/2512.11253)
- [M] **HunyuanPortrait** (CVPR 2025, 2503.18860):
  - Stable Video Diffusion with adapters. Motion comes from a pretrained implicit motion encoder (ResNet-50, MegaPortrait-style) applied to driving frames. Trained on 128× A100 for 3 days.
  - Its own Table 1 (LMD / FID-VID / FVD / PSNR / SSIM / LPIPS):
    - LivePortrait: 9.14 / 82.71 / 483.38 / 31.41 / 0.72 / 0.22
    - HunyuanPortrait: 2.02 / 75.81 / 333.48 / 32.98 / 0.81 / 0.11
  - **This conflicts with PersonaLive's independent run**, where HunyuanPortrait ties LivePortrait on L1 and LPIPS and has worse FVD (620.4 vs 557.2).

  — [HunyuanPortrait 2503.18860](https://arxiv.org/abs/2503.18860); contradicted by [PersonaLive 2512.11253](https://arxiv.org/abs/2512.11253)
- [M] **FantasyPortrait** (2507.12956):
  - Wan2.1-I2V-14B with implicit expression features taken from the driving video. Trained on 24× A100 for about 3 days; 30 sampling steps.
  - ExprBench self-reenactment (FID / FVD / PSNR / SSIM):
    - LivePortrait: 79.32 / 438.27 / 23.38 / 0.789
    - SkyReels-A1: 66.84 / 373.98 / 24.58 / 0.812
    - HunyuanPortrait: 74.86 / 409.14 / 24.54 / 0.783
    - X-Portrait: 83.28 / 445.25 / 22.51 / 0.739
    - Follow-Your-Emoji: 103.75 / 489.93 / 21.47 / 0.692
    - FantasyPortrait: 64.66 / 358.08 / 25.76 / 0.818

  — [FantasyPortrait 2507.12956](https://arxiv.org/abs/2507.12956)
- [M] **SkyReels-A1** (2502.10841): a CogVideoX-5B video DiT driven by explicit 3D face-expression conditions. Table 1:
  - ID similarity: LivePortrait 0.7011 / 0.7305 vs SkyReels-A1 0.7196 / 0.7314.
  - "Image Quality" (lower is better): LivePortrait 83.3168 vs SkyReels-A1 59.6884.

  — [SkyReels-A1 2502.10841](https://arxiv.org/abs/2502.10841)
- [M] **Wan-Animate** (2509.14055; Wan 14B): face expression comes from implicit latents encoded from the driving video's face crops through a Face Adapter. Portrait table (SSIM / LPIPS / FVD):
  - LivePortrait: 0.811 / 0.231 / 118.67
  - AniPortrait: 0.791 / 0.252 / 135.08
  - Follow-Your-Emoji: 0.803 / 0.244 / 127.95
  - **X-Portrait 2: 0.825 / 0.212 / 98.03** (the only numbers I found for X-Portrait 2, which has no paper)
  - SkyReels-A1: 0.821 / 0.231 / 101.45
  - Wan-Animate: 0.834 / 0.205 / 94.65

  — [Wan-Animate 2509.14055](https://arxiv.org/abs/2509.14055)
- [C] **Wan-Animate-2** (2608.06009, Aug 2026): an end-to-end DiT that "directly consumes the driving video" with no motion extractor. A Lite variant is said to reach "real-time thresholds" through Self-Forcing distillation, and base weights are "will release". The evaluation I saw is qualitative plus a user study. — [Wan-Animate-2 2608.06009](https://arxiv.org/abs/2608.06009)
- [M] **FRVD** (2507.16341, Computers & Graphics 2025) is a "warp then diffusion-correct" hybrid. It extracts implicit facial keypoints, warps the source with a warping module, then maps the warped features through a Warping Feature Mapper into SVD's latent space.
  - HDTF self-reenactment (L1 / PSNR / SSIM / LPIPS / ID):
    - FRVD: 0.0226 / 27.708 / 0.8702 / 0.0760 / 0.8570
    - LivePortrait: 0.0418 / 22.788 / 0.7746 / 0.1222 / 0.8967
  - HDTF cross-identity (ID / FID / FVD):
    - FRVD: 0.8975 / 14.73 / 140.8
    - LivePortrait: 0.9294 / 25.07 / 142.8
  - Trained on a **single A6000** (8-bit Adam, lr 1e-5). Inference: 30 DDIM steps, about **4 min per 100 frames on an RTX 4090**.

  — [FRVD 2507.16341](https://arxiv.org/abs/2507.16341)

### Inferences
- [I] **Pattern across every independent table:** diffusion renderers lower FID (FRVD 25.07→14.73 vs LivePortrait) but **lose identity similarity**. Examples: FRVD ID 0.8975 vs 0.9294; PersonaLive 0.698 vs 0.723; HunyuanPortrait 0.644 vs 0.723. Most are not better on FVD (PersonaLive's table), and none measures lip sync. For SANG, where CSIM and LSE must hold, a diffusion renderer is a **stretch option, not the first move**.
- [I] The most compatible candidate is **PersonaLive**: Apache-2.0, 15.82 FPS and 0.253 s latency on an H100, and it already consumes LivePortrait keypoints for pose. Its facial-dynamics branch needs driving frames, so it would take our LivePortrait render as the driving video. Mouth shape would then pass through a second, lossy implicit encoder, so LSE and mouth correlation must be re-measured. Expect FVD gains of the order of its 557→521 reduction and a CSIM drop.
- [I] FRVD's design (keep the LivePortrait-style warp, then diffusion-correct) is the conceptually right hybrid for "keep our motion, sharpen texture". It is about 0.4 FPS and its code status is unverified (see Gaps).

### Gaps
- No video-driven diffusion renderer paper reports LSE-C/D; sync preservation through a second implicit encoder is untested.
- FRVD: I did not confirm whether it reuses LivePortrait's motion-extractor weights, nor whether code or weights are released.
- Not extracted: MegActor-Σ and Follow-Your-Emoji-Faster (2509.16630) details, and licences for HunyuanPortrait, FantasyPortrait, SkyReels-A1 and Wan-Animate weights. PersonaLive's SD backbone licence terms were also not read.

## Q4. Teeth / mouth-interior specific methods

### Takeaway
The strongest evidence points to **reference-based texture transfer** (copying real teeth pixels or features from a teeth-visible frame) rather than hallucination:
- HDTR-Net's reference module accounts for a large share of its sharpness gain.
- EfficientSync's deformation-based reference mixing gives the best HDTF FID among lip-sync methods.
- SynergyWarpNet shows that adding a second reference frame to a LivePortrait-style warper cuts HDTF FID from 36.49 to 32.34.

None of these is conditioned on keypoint motion out of the box. A teeth-reference branch would need to be built into our decoder. Tongue synthesis (TongueReenact) exists, but it needs tongue masks from a driving video.

### Cited Findings
- [M] **HDTR-Net** (2309.07495), a plug-in teeth restorer with a reference Fine-Grained Feature Fusion module ("ref FGFF"):
  - Evaluated with no-reference sharpness metrics only; it has no FID or LSE table.
  - Table 1 (s/frame / Brenner / Laplacian / Entropy):
    - ESRGAN: 0.0597 / 2003566 / 81.38 / 4.56
    - GFPGAN: 1.2332 / 2302120 / 140.18 / 4.56
    - Real-ESRGAN: 0.0622 / 2029371 / 75.15 / 4.55
    - HDTR-Net: 0.0187 / 3121676 / 102.28 / 4.75
  - Ablation: removing ref FGFF drops Brenner from 3121676 to 2803462; removing the perceptual loss drops it to 2003462.
  - It was applied to Wav2Lip, MakeItTalk, PC-AVS and IP-LAP. It claims, without measurement, that this works "without suffering from lip synchronization and frame coherence".

  — [HDTR-Net 2309.07495](https://arxiv.org/abs/2309.07495)
- [M] HDTR repo yylgoodlucky/HDTR: **no licence** (GitHub API returns None), 149 stars, last push 2023-09-18. — [GitHub yylgoodlucky/HDTR](https://github.com/yylgoodlucky/HDTR)
- [M] **EfficientSync** (2608.18832, Aug 2026; audio-driven dubbing): a deformation-based "Dynamic Texture Mixer" selects channel-wise among aligned reference frames instead of hallucinating teeth; 166 FPS on one GPU. Columns: SSIM / LPIPS / VL sharpness / FID / FVD / Sync-conf.
  - HDTF:
    - EfficientSync: 0.979 / 0.013 / 58.691 / 1.824 / 49.144 / 9.107
    - LatentSync: 0.963 / 0.028 / 47.906 / 2.342 / 50.820 / 9.183
    - KeySync: 0.906 / 0.064 / 42.868 / 11.321 / 121.321 / 8.765
  - VFHQ: EfficientSync FID 4.133 vs LatentSync 5.782, **but Sync-conf 6.058 vs 7.765**.
  - The authors state that "deformation quality is ultimately bounded by the reference pool".

  — [EfficientSync 2608.18832](https://arxiv.org/abs/2608.18832)
- [M] **RGOR** (2609.38019, preprint): audio-driven oral refinement conditioned on enrollment frames of the same person and on **high-resolution mouth patches that bypass the VAE**. It is trained against a "paired judge" that rejects a realistic mouth belonging to someone else. Abstract claim: best or second-best on most metrics while "preserv[ing] the person's own lip and dental detail". I did not extract its numbers. Like LatentSync, it regenerates the mouth from audio. — [RGOR 2609.38019](https://arxiv.org/abs/2609.38019)
- [M] **TongueReenact** (2607.28039): latent masked diffusion for the tongue in face reenactment, with tongue masks bootstrapped from foundation models; 20 DDIM steps on one A40. VFHQ cross-identity, 9,000 pairs:
  - Tongue-region LPIPS: TongueReenact 0.2306 vs LivePortrait 0.3454 (X-NeMo 0.3937, X-Portrait 0.4209).
  - Tongue Presence: 0.7891 vs LivePortrait 0.5986.
  - Tongue IoU: 0.3582 vs about 0.16 for LivePortrait (value truncated in my extraction).

  — [TongueReenact 2607.28039](https://arxiv.org/abs/2607.28039)
- [M] **SynergyWarpNet** (2512.17331): explicit 3D-flow warping plus a reference-augmented correction module (cross-attention over 3D keypoints and textures of extra reference images) plus confidence fusion. Evaluated at 256×256; columns LPIPS / PSNR / SSIM / L1 / FID.
  - HDTF self-reenactment:
    - LivePortrait: 0.1817 / 29.1516 / 0.8954 / 0.0213 / 36.4944
    - SynergyWarpNet R=1: 0.1527 / 30.6826 / 0.9205 / 0.0203 / 34.7572
    - SynergyWarpNet R=2: 0.1430 / 30.9842 / 0.9255 / 0.0197 / 32.3417
  - VFHQ FID: LivePortrait 31.3928 → R=1 27.4209 → R=2 21.5998.

  — [SynergyWarpNet 2512.17331](https://arxiv.org/abs/2512.17331)
- [C] Playmate's Fig. 4 caption: competing methods were "prone to generate artifacts in tooth rendering". — [Playmate 2502.07203](https://arxiv.org/abs/2502.07203)

### Inferences
- [I] For SANG (one source photo), the practical version is an **optional second "teeth reference" image**. Sources: a user-supplied smiling or talking photo, a frame from the identity's real video during evaluation, or, at worst, an edited version of the source. Its appearance features would be injected into the decoder (cross-attention or SPADE modulation inside the mouth mask) during the decoder fine-tune from Q1. This is conditioned on our warp/keypoints, not on audio, so it does not discard our motion.
- [I] A cheaper diagnostic: render the TalkVid ceiling test with a teeth-visible source frame vs a closed-mouth source frame. If mouth-region FID or sharpness improves a lot, the reference route is worth building. This connects to the untested source-mouth-state question.
- [I] HDTR-Net is the closest drop-in teeth sharpener (18.7 ms/frame), but it has no licence and no FID/LSE evidence. It is suitable only as an internal A/B test.
- [I] Tongue ghosts: TongueReenact shows LivePortrait's tongue fidelity is poor (tongue LPIPS 0.3454), but the fix needs a tongue signal our 42-d motion lacks. It is not actionable without extending the motion space.

### Gaps
- No method does mouth inpainting conditioned on keypoint motion (rather than audio) with released weights.
- No standard mouth-region quality metric exists. Papers use VL, Brenner/Laplacian, tongue LPIPS and IoU, or mouth-crop FID. We should define our own: FID or LPIPS on the 96-px mouth crop, plus a Laplacian-variance ratio vs GT.
- Not verified: code or licence for SynergyWarpNet, EfficientSync, RGOR and TongueReenact.

## Q5. Alternative warping renderers 2025-2026 and switching cost; ranked recommendation

### Takeaway
- Beyond IMTalker and LIA-X (covered earlier), the 2025–2026 warping-family improvements I found are reference-augmented warpers (SynergyWarpNet) and warp-plus-diffusion hybrids (FRVD).
- I found no released-weights warping renderer that reports HDTF/VFHQ reconstruction numbers and **keeps LivePortrait's keypoint space**.
- Switching to IMTalker or LIA-X means re-extracting motion for all training data and retraining the 53M DiT. A LivePortrait-keypoints → other-latent mapping is possible as a learned regressor on paired frames, but it would be lossy.
- Ranked recommendation: (1) a decoder-only fine-tune with mouth-weighted losses and an optional teeth reference, then (2) temporal video SR on the face crop with LSE gating, then (3) a PersonaLive or FRVD-style diffusion pass as a stretch, and (4) a renderer switch only if (1)–(3) stall.

### Cited Findings
- [M] SynergyWarpNet (2512.17331) beats LivePortrait on HDTF and VFHQ self-reenactment at 256 px, even with a single reference (HDTF FID 36.4944 → 34.7572; R=2 gives 32.3417). Its inputs are a source, a driving image and extra reference images with 3D keypoints. Training: Adam, lr 2e-4, 150 epochs. — [SynergyWarpNet 2512.17331](https://arxiv.org/abs/2512.17331)
- [M] FRVD (2507.16341) keeps implicit-keypoint warping and adds SVD-based correction: HDTF cross FID 14.73 vs 25.07, FVD 140.8 vs 142.8, but ID 0.8975 vs 0.9294. The authors attribute LivePortrait's ID advantage to "a large-scale private facial video dataset exceeding 16 million frames". — [FRVD 2507.16341](https://arxiv.org/abs/2507.16341)
- [C] PortraitDirector (2604.19129): diffusion reenactment with hierarchical motion disentanglement, "512 x 512 face reenactment at 20 FPS with a end-to-end 800 ms latency on a single 5090 GPU" (abstract). — [PortraitDirector 2604.19129](https://arxiv.org/abs/2604.19129)
- [C] A one-shot Gaussian-head alternative (2509.05582) reports "90 FPS at a resolution of 512x512" with a separate reconstruction/reenactment design (abstract). It would require a different motion space. — [arXiv 2509.05582](https://arxiv.org/abs/2509.05582)
- [M] KDTalker (2503.12963) and JoyVASA (2411.09209) are other audio-to-LivePortrait-keypoint systems (verified IDs). Playmate shows the same motion space can reach HDTF FID 19.138 / FVD 231.048 (see Q3). That is evidence the keypoint space itself does not cap FID at JoyVASA's 29.581 level. — [KDTalker 2503.12963](https://arxiv.org/abs/2503.12963); [JoyVASA 2411.09209](https://arxiv.org/abs/2411.09209); [Playmate 2502.07203](https://arxiv.org/abs/2502.07203)

### Inferences
- [I] **Switching cost** (IMTalker, LIA-X or any non-LivePortrait renderer):
  - Re-extract motion for the whole training corpus with their motion encoder.
  - Retrain the 53M flow-matching DiT, with a new output dimension and normalisation statistics.
  - Re-tune the stitching and paste-back logic.
  - Re-establish the renderer ceiling and LSE baselines.

  This is about one full training cycle of the DiT plus evaluation. A **keypoint→latent mapping** (an MLP or small temporal net trained on frames where both encoders see the same image) avoids retraining the DiT, but adds a second approximation on top of our motion, exactly where mouth detail lives. It should only be tried if a candidate renderer's ceiling, re-rendering *real* motion, is clearly above LivePortrait's 0.906–0.911 CSIM and 0.82–0.85 mouth corr on TalkVid.
- [I] **Ranked recommendation: what to try first to cut FID/FVD and sharpen teeth while keeping LSE and CSIM.**
  1. **Decoder-only fine-tune of LivePortrait** (F, M, W frozen; 512-px talking-head GT; L1 + VGG + face-id + global and mouth-crop GAN, with the mouth discriminator optionally initialised from FFHQ StyleGAN2; a temporal or frame-difference term for FVD).
     - Cost: about 1–3 GPU-days on one 80 GB GPU (estimate), plus writing a training loop because no official one exists.
     - Motion space untouched, so no DiT retraining. LSE should hold because lip geometry comes from the warp.
     - Evidence: LivePortrait's own region losses, the community mouth-artifact fine-tune (#506), and diffusion renderers' FID gains showing headroom in texture rather than motion.
     - Gate: TalkVid ceiling CSIM ≥ 0.906, mouth corr ≥ 0.82, HDTF LSE-C/D unchanged ±0.1.
  2. **Temporal face-crop SR/restoration before paste-back**: SeedVR2-3B (Apache-2.0, one step) or KEEP (custom licence; for internal A/B). Never per-frame CodeFormer or GFPGAN.
     - Cost: zero training, inference only.
     - Expected: sharper skin and teeth and lower FID. FVD depends on temporal stability.
     - Must measure LSE because no paper has.
  3. **Teeth-reference conditioning** (a second teeth-visible image injected into the decoder in the mouth region), built on top of (1).
     - Evidence: SynergyWarpNet R=2 (HDTF FID 36.49→32.34), HDTR-Net ref-FGFF ablation, EfficientSync.
     - Cost: an extra 1–3 GPU-days plus architecture work. Run the cheap source-mouth-state diagnostic first.
  4. **Diffusion refinement pass** (PersonaLive at 15.82 FPS on an H100 under Apache-2.0, driven by our LivePortrait render; or an FRVD-style warp+SVD correction at about 0.4 FPS on a 4090).
     - Likely FID/FVD gains, but every independent table shows an ID drop, and lip sync is unmeasured.
     - Treat as a research branch.
  5. **Renderer switch** (IMTalker or LIA-X with a re-extracted motion space). Highest cost; do it only if (1)–(3) do not close the gap.

### Gaps
- SynergyWarpNet and FRVD weights and licences are unverified. I found no 2025–2026 warping renderer with released weights that reports HDTF/VFHQ reconstruction numbers and beats LivePortrait, beyond those listed here and the earlier-covered IMTalker and LIA-X.
- No published keypoint→latent mapping between LivePortrait and IMTalker or LIA-X; its fidelity is untested.
- All one-GPU cost figures for fine-tuning are my estimates, not measurements from any source.
