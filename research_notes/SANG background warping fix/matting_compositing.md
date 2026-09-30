# Foreground/background separation and compositing to keep the background static in one-shot talking-head video (LivePortrait-based SANG-M)

Status: COMPLETE (2026-09-29). Sources: arXiv papers + official GitHub READMEs/code/LICENSE files + Hugging Face model cards; licences checked via GitHub/HF APIs on 2026-09-29.

Evidence grading used below: **[M]** = measured/reported number in a primary source (paper table, official README); **[D]** = documented design/behaviour in a primary source (code/README/paper text, no number); **[I]** = inference by the note author (not directly stated in a source).

## Q1. Matting / segmentation tools: single source portrait vs per-frame on rendered video (hair accuracy, speed, licences)

### Takeaway
For the ONE source photo, a permissively licensed high-resolution image model is best: BiRefNet (MIT; dedicated "portrait matting" and "HR-matting" checkpoints) optionally refined by trimap-based ViTMatte (MIT) in the hair band. For PER-FRAME cut-out of the rendered 512x512 video, the most temporally stable published model is MatAnyone (dtSSD 1.18 vs RVM 1.36 on VideoMatte 512x288) but it is NTU S-Lab non-commercial; RVM is fast (100+ FPS) but GPL-3.0; SAM 2.1 (Apache-2.0) gives temporally tracked binary masks, not soft alpha. BackgroundMattingV2 is not applicable (needs a captured clean background plate).

### Cited Findings
**Licences (verified from GitHub API / LICENSE files, 2026-09-29) [D]**
- Robust Video Matting (RVM): GPL-3.0 — [GitHub PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)
- MODNet: README states "The code, models, and demos in this repository ... are released under the Apache License 2.0" — [GitHub ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)
- BiRefNet: MIT — [GitHub ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- MatAnyone and MatAnyone 2: "NTU S-Lab License 1.0" whose text permits "Redistribution and use for non-commercial purpose" — [GitHub pq-yang/MatAnyone LICENSE](https://github.com/pq-yang/MatAnyone/blob/main/LICENSE); [GitHub pq-yang/MatAnyone2](https://github.com/pq-yang/MatAnyone2)
- SAM 2 / SAM 2.1: Apache-2.0 — [GitHub facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- BackgroundMattingV2: MIT — [GitHub PeterL1n/BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)
- PP-Matting / PP-MattingV2 (PaddleSeg): Apache-2.0 — [GitHub PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
- ViTMatte: MIT — [GitHub hustvl/ViTMatte](https://github.com/hustvl/ViTMatte)

**Accuracy / temporal stability [M]**
- MatAnyone (arXiv 2501.14677, "Stable Video Matting with Consistent Memory Propagation", CVPR 2025) Table 1, VideoMatte 512x288: MAD / MSE / Grad / Conn / dtSSD = MODNet 9.41 / 4.30 / 1.89 / 0.81 / 2.23; RVM 6.08 / 1.47 / 0.88 / 0.41 / 1.36; RVM-Large 5.32 / 0.62 / 0.59 / 0.30 / 1.24; MaGGIe 5.49 / 0.60 / 0.57 / 0.31 / 1.39; MatAnyone 5.15 / 0.93 / 0.67 / 0.26 / 1.18. VideoMatte 1920x1080 MAD / dtSSD: MODNet 11.13 / 3.08; RVM 6.57 / 1.90; RVM-Large 5.81 / 1.78; MatAnyone 4.24 / 1.19 (numbers extracted from the arXiv HTML via a summarising fetch; spot-check against the PDF before quoting in a paper) — [arXiv 2501.14677](https://arxiv.org/abs/2501.14677)
- MatAnyone YouTubeMatte 512x288 MAD: MODNet 19.37, RVM 4.08, RVM-Large 3.36, MatAnyone 2.72 — [arXiv 2501.14677](https://arxiv.org/abs/2501.14677)
- MatAnyone "only requires the segmentation mask for the first frame as a target assignment"; README says the first-frame mask "could be obtained from interactive segmentation models such as SAM2" [D] — [arXiv 2501.14677](https://arxiv.org/abs/2501.14677); [MatAnyone README](https://github.com/pq-yang/MatAnyone)
- MatAnyone 2 ("Scaling Video Matting via a Learned Quality Evaluator", arXiv 2512.11782, CVPR 2026 Highlight per repo description), inference code released 2026.03 — [GitHub pq-yang/MatAnyone2](https://github.com/pq-yang/MatAnyone2) (arXiv ID taken from the README badge link; numbers not extracted)
- ViTMatte (arXiv 2305.15272; trimap-based) Composition-1k SAD / MSE / Grad / Conn: ViTMatte-S 21.46 / 3.3 / 7.24 / 16.21; ViTMatte-B 20.33 / 3.0 / 6.74 / 14.78. Distinctions-646: ViTMatte-B 17.05 / 1.5 / 7.03 / 12.95 — [ViTMatte README](https://github.com/hustvl/ViTMatte)
- BiRefNet (arXiv 2401.03407, CAAI AIR 2024): released "BiRefNet-matting" (trimap-free, Oct 2024), "BiRefNet_HR-matting" trained at 2048x2048 (Feb 2025), "BiRefNet_dynamic" 256–2304 px (Mar 2025); portrait-matting checkpoint (P3M-10k + humans) scores "0.983, 0.989" on P3M-500-P (metric names per README column) — [BiRefNet README](https://github.com/ZhengPeng7/BiRefNet)
- PP-Matting human models on PPM-AIM-195 (SAD / MSE / Grad / Conn / FPS on V100 at 512): PP-Matting-512 31.56 / 0.0022 / 31.80 / 30.13 / 28.9; PP-MattingV2-512 40.59 / 0.0038 / 33.86 / 38.90 / 98.89; MODNet-HRNet_W18 35.55 / 0.0035 / 31.73 / 34.07 / 62.6 — [PaddleSeg Matting README](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/Matting)

**Speed [M]**
- RVM throughput (tensor only, batch 1): RTX 3090 FP16 172 FPS HD / 154 FPS 4K; GTX 1080 Ti FP32 104 FPS HD / 74 FPS 4K; README warns the provided video conversion script "is expected to be much slower" — [RVM README](https://github.com/PeterL1n/RobustVideoMatting)
- BiRefNet standard model: "17 FPS with resolution==1024x1024 with 3.45GB GPU memory on a single RTX 4090" (FP16) — [BiRefNet README](https://github.com/ZhengPeng7/BiRefNet)
- SAM 2.1 video FPS (A100, torch 2.5.1, CUDA 12.4): hiera_tiny 91.2, small 84.8, base_plus 64.1, large 39.5; SA-V test J&F 76.5 / 76.6 / 78.2 / 79.5 — [SAM 2 README](https://github.com/facebookresearch/sam2)
- MatAnyone paper gives no FPS/memory figures (per the fetched HTML) — [arXiv 2501.14677](https://arxiv.org/abs/2501.14677)

### Inferences
- [I] Source photo (done once, cost irrelevant): BiRefNet portrait/HR-matting at full photo resolution → erode/dilate its alpha into a trimap → ViTMatte-B refinement for hair. All MIT. MODNet/PP-Matting are older and score worse on hair; RVM on a single image lacks its recurrent temporal context.
- [I] Per-frame on rendered video: if a per-frame matte is needed, MatAnyone (research use) seeded with the source-photo mask is the most stable choice by dtSSD; for commercial-safe use, SAM 2.1 masks + per-frame soft edge (feathering or ViTMatte in a thin band) or MODNet (Apache-2.0) are the alternatives. RVM's GPL-3.0 only matters if the pipeline is distributed.
- [I] Rendered LivePortrait frames are only 512x512, so HR models (BiRefNet_HR, 4K RVM) bring no benefit per-frame; a 512-px model at 60–170 FPS makes matting cost negligible relative to rendering.
- [I] BackgroundMattingV2 requires a separately captured clean plate of the same scene; we only have one photo, so it is ruled out (it could, however, be run AFTER we synthesise an inpainted plate, but the plate would already make it redundant).

### Gaps
- No published benchmark of matting models on synthesised/GAN-rendered talking-head frames (domain gap: LivePortrait SPADE output has soft/blurred hair); needs an internal test.
- MatAnyone 2 quantitative numbers and speed not extracted (time budget).
- MODNet: licence is Apache-2.0 per README, but its training data licences were not checked.

## Q2. Background inpainting of the region revealed behind the person (single static plate)

### Takeaway
Because the plate is ONE static image, video inpainters (ProPainter, E2FGVI, DiffuEraser) are unnecessary — there is no temporal consistency to enforce once the plate is fixed. Big-LaMa (Apache-2.0, resolution-robust, good on periodic textures) is the pragmatic default; MAT scores better FID on large masks but is CC BY-NC 4.0; diffusion fills (SD/SDXL inpainting, FLUX.1 Fill) give the most plausible large-hole content but hallucinate and FLUX.1 Fill [dev] is non-commercial. Real3DPortrait — the one talking-head paper that explicitly inpaints the background — used only a naive 1-nearest-neighbour fill and measured a small but positive effect.

### Cited Findings
**Licences [D]**
- LaMa code Apache-2.0 ([GitHub advimman/lama](https://github.com/advimman/lama)); big-lama weights on Hugging Face tagged apache-2.0 ([HF smartywu/big-lama](https://huggingface.co/smartywu/big-lama))
- MAT: Creative Commons Attribution-NonCommercial 4.0 — [GitHub fenglinglwb/MAT LICENSE](https://github.com/fenglinglwb/MAT/blob/main/LICENSE)
- ProPainter: "made available for use, reproduction, and distribution strictly for non-commercial purposes" (NTU S-Lab License 1.0) — [ProPainter README](https://github.com/sczhou/ProPainter)
- E2FGVI: CC BY-NC 4.0 — [GitHub MCG-NKU/E2FGVI LICENSE](https://github.com/MCG-NKU/E2FGVI/blob/master/LICENSE)
- DiffuEraser: Apache-2.0, BUT README: "This repository uses Propainter as the prior model. Users must comply with Propainter's license" — [DiffuEraser README](https://github.com/lixiaowen-xw/DiffuEraser)
- FLUX.1 Fill [dev]: "flux-1-dev-non-commercial-license" — [HF black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
- SD 1.5 inpainting: creativeml-openrail-m — [HF stable-diffusion-v1-5/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting); SDXL inpainting 0.1: openrail++ — [HF diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)

**Quality [M]**
- MAT paper (arXiv 2203.15270, CVPR 2022) Table 2, Places 512x512 FID (small mask / large mask): MAT† (8M imgs, 62M params) 0.78 / 1.96; CoModGAN† (109M) 1.10 / 2.92; LaMa† (Big LaMa, 4.5M imgs, 51M params) 0.99 / 2.97; P-IDS large mask MAT† 23.42 vs LaMa† 13.09. CelebA-HQ 512 large mask FID: MAT† 4.86, LaMa† 8.15 — [arXiv 2203.15270](https://arxiv.org/abs/2203.15270)
- MAT limitation (own paper): "struggles when processing objects with a variety of shapes" and fails to recover a cat and a car "due to the lack of semantic context understanding" — [arXiv 2203.15270](https://arxiv.org/abs/2203.15270)
- LaMa (arXiv 2109.07161): "generalizes surprisingly well to much higher resolutions (~2k) than it saw during training (256x256)" and handles "completion of periodic structures"; a high-resolution feature-refinement add-on exists (arXiv 2206.13644) — [LaMa README](https://github.com/advimman/lama)
- Real3DPortrait (arXiv 2401.08503, ICLR 2024) background: "a K-nearest-neighbor (KNN)-based inpainting method" with K=1; "we can also use state-of-the-art neural network-based inpainting methods ... but since that is not the focus of this paper, we simply used the naive KNN-based method." Ablation "w/o inpaint": CSIM 0.744, FID 43.95 vs full model CSIM 0.758, FID 42.37 (extracted via summarising fetch of arXiv HTML) — [arXiv 2401.08503](https://arxiv.org/abs/2401.08503)

**Cost [M] (video inpainters, for reference — not needed for a static plate)**
- ProPainter GPU memory (fp32/fp16), 50 frames: 1280x720 28G / 19G; 640x480 10G / 6G — [ProPainter README](https://github.com/sczhou/ProPainter)
- DiffuEraser (arXiv 2501.10018): 250 frames (~10 s) on an L20: 1280x720 33G, 314 s; 640x360 12G, 92 s — [DiffuEraser README](https://github.com/lixiaowen-xw/DiffuEraser)

### Inferences
- [I] The disoccluded region for a talking head is a thin band around the head/hair/shoulder silhouette (head yaw of ±20–30 deg and small translations), i.e. a "small/medium" mask in inpainting-benchmark terms, where LaMa and MAT FIDs are both below 1 on Places (small mask). A LaMa fill on a DILATED person mask (dilate by the maximum expected head displacement + hair margin) should be visually adequate; the region is mostly hidden behind the moving head anyway.
- [I] Diffusion inpainting (SDXL-inpaint / FLUX Fill) is worth it only for very large masks (big hair, wide shoulders, strong yaw) or textured scenes where LaMa blurs; it runs once per source photo, so its cost (seconds) is irrelevant. Risk: hallucinated objects/people "behind" the subject — prompt with "empty background" and use a negative prompt.
- [I] Video inpainters (ProPainter/E2FGVI/DiffuEraser) solve a different problem (moving camera/background) and carry non-commercial licences; skip them.
- [I] Real3DPortrait's positive "w/ inpaint" ablation (FID 42.37 vs 43.95) is weak evidence that even crude inpainting beats leaving the source person's pixels in the background, because the old silhouette otherwise "ghosts" when the head moves.

### Gaps
- No published comparison of inpainters specifically on portrait backgrounds behind a person (the in-distribution case for us).
- LaMa's own paper tables were not extracted (MAT's Table 2 is used as the LaMa-vs-MAT comparison).

## Q3. Compositing strategy trade-offs: (a) per-frame matting over an inpainted plate, (b) warp the source alpha with the renderer's flow, (c) LivePortrait-style soft-mask paste-back

### Takeaway
Strategy (c) as LivePortrait/Ditto implement it does NOT fix our problem: their paste-back mask covers ~88% of the 512x512 crop at full opacity, and the crop is ~2.3x the face box, so the warped background inside the crop is pasted back too; paste-back only hides the crop border and restores full-resolution context outside the crop. The fix has to decide per pixel "person vs background" — i.e. (a) or (b), or an equivalent flow-level fix. (a) per-frame matting of the rendered frame over a static inpainted plate is the simplest and is how person-specific NeRF talkers (ER-NeRF, GeneFace++, SyncTalk) and Real3DPortrait composite. (b) is attractive because LivePortrait's dense motion already contains an explicit "background = identity motion" channel, so a person mask can be enforced inside the renderer's own flow.

### Cited Findings
**How LivePortrait/Ditto paste back [D, M]**
- LivePortrait `paste_back`: `result = mask_ori * warpAffine(img_crop, M_c2o) + (1 - mask_ori) * img_ori`, where `mask_ori` is a fixed `mask_template.png` warped into original-image space — [LivePortrait src/utils/crop.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py); template path in [inference_config.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/config/inference_config.py)
- [M, own measurement of the released file] LivePortrait's `mask_template.png` is 512x512; 88.3% of pixels are >0.99 (fully pasted), 11.3% are a feathered border (0.01–0.99), 0.5% are ~0 — i.e. it is a rounded-rectangle feather over the whole crop, not a person/face mask — [mask_template.png](https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/resources/mask_template.png)
- LivePortrait source-crop config: `dsize 512`, `scale 2.3`, `vy_ratio -0.125` (crop ~2.3x the face box, shifted down) — [crop_config.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/config/crop_config.py)
- LivePortrait defaults `flag_stitching: True` ("recommend to True if head movement is small, False if head movement is large") and `flag_pasteback: True` — [argument_config.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/config/argument_config.py)
- LivePortrait paper (arXiv 2407.03168): "The stitching module pastes the animated portrait back into the original image space without pixel misalignment, such as in the shoulder region"; trained with a consistency pixel loss on the shoulder region, with a mask operator that "masks out the non-shoulder region"; limitation: "when the driving video involves significant shoulder movements, there is a certain probability of resulting in jitter" — [arXiv 2407.03168](https://arxiv.org/abs/2407.03168)
- Ditto (antgroup/ditto-talkinghead, Apache-2.0) has a `PutBack` component doing the identical blend (`mask_warped * frame_warped + (1 - mask_warped) * frame_rgb`) with either LivePortrait's template or a generated `get_mask(512, 512, 0.9, 0.9)` — [Ditto core/atomic_components/putback.py](https://github.com/antgroup/ditto-talkinghead/blob/main/core/atomic_components/putback.py). So Ditto DOES paste back, the LivePortrait way.
- SadTalker "full" mode: "our model will automatically process the croped region and paste back to the original image. Remember to use `--still` to keep the original head pose"; paste-back uses `cv2.seamlessClone(..., cv2.NORMAL_CLONE)` with an all-255 rectangular mask over the crop; optional `--enhancer gfpgan|RestoreFormer` (face) and `--background_enhancer realesrgan` (full frame) — [SadTalker best_practice.md](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md); [SadTalker src/utils/paste_pic.py](https://github.com/OpenTalker/SadTalker/blob/main/src/utils/paste_pic.py)

**LivePortrait's flow has an explicit background channel (relevant to option b) [D]**
- `create_sparse_motions` concatenates an identity grid ("adding background feature") in front of the per-keypoint motions; the dense-motion hourglass predicts a softmax mask over (1 + num_kp) channels at (d=16, h=64, w=64), and `deformation = sum(sparse_motion * mask)`; keypoint motions are pure translations ("NOTE: there lacks an one-order flow"). An occlusion map (Bx1x64x64, sigmoid) multiplies the warped 3D feature volume before decoding — [LivePortrait src/modules/dense_motion.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/modules/dense_motion.py); [warping_network.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/modules/warping_network.py)

**Composition in methods that separate layers [D, M]**
- Real3DPortrait fuses layers alpha-style: F = (F_head·M_head + F_torso·(1−M_head))·M_person + F_bg·(1−M_person); torso uses a "2D warping-based renderer" (face-vid2vid-like) driven by 68 keypoints; naive concatenation instead "leads to hollow artifacts and blurry results in the boundary region". Ablation "w/ concat": CSIM 0.737, FID 46.38, AED 0.144, APD 0.025 vs full 0.758, 42.37, 0.138, 0.022 (summarising fetch of arXiv HTML Table 4) — [arXiv 2401.08503](https://arxiv.org/abs/2401.08503)
- ER-NeRF (and the AD-NeRF/RAD-NeRF lineage) build one static background image `bc.jpg` from the training video: background pixels from parsing masks across frames, remaining holes filled by 1-nearest-neighbour (`NearestNeighbors(n_neighbors=1)`); the rendered head (+ separately trained torso) is composited over it — [ER-NeRF data_utils/process.py](https://github.com/Fictionarry/ER-NeRF/blob/main/data_utils/process.py); head+torso result in README (PSNR 26.594, LPIPS 0.0446, LMD 2.550 on Obama per README table columns) — [ER-NeRF README](https://github.com/Fictionarry/ER-NeRF)
- SyncTalk: torso training "to repair double chin", but then "you will not be able to use the '--portrait' mode" (its paste-back/portrait-sync mode) — shows the torso/neck seam problem is real even in person-specific systems — [SyncTalk README](https://github.com/ZiqiaoPeng/SyncTalk)

**Artifact prevalence [D]**
- TalkingHeadBench (arXiv 2505.24866) built its curation around a generator-specific artifact taxonomy "such as severe background warping or unnatural static hair" across Hallo, Hallo2, AniPortrait, LivePortrait, EMOPortraits (+ Hallo3, MAGI-1 test-only); detectors' attention for Hallo2 and EMOPortraits "shifts away from facial features and toward areas surrounding the head"; EMOPortraits shows "neck distortions"; no per-generator artifact rates in the main text — [arXiv 2505.24866](https://arxiv.org/abs/2505.24866)

**Halo/colour-spill tooling [D]**
- RVM predicts both a foreground colour map and alpha (`fgr, pha, *rec = model(...)`; `com = fgr * pha + bgr * (1 - pha)`), i.e. it composites with estimated foreground colour, not the raw frame — [RVM README](https://github.com/PeterL1n/RobustVideoMatting)
- PyMatting (MIT) implements "Closed Form Foreground Estimation" and "Fast Multi-Level Foreground Estimation (CPU, CUDA and OpenCL)" from an image + alpha — [PyMatting README](https://github.com/pymatting/pymatting)

### Inferences
- [I] **(c) paste-back as shipped is not a fix.** With the template mask, ~88% of the crop, including all background inside a 2.3x face-box crop, is replaced by the rendered (warped) pixels. Paste-back only removes the crop edge and padding bars (because the output then lives in original-photo coordinates). If the mask is changed to a tight face/head mask (SadTalker-style, EMOPortraits "source image body"), background and body become exactly static, but the head outline, hair and jaw come from two sources: fine for small motion (SadTalker recommends `--still`), double contours for real head turns. Verdict: useful as the OUTER frame (use original-photo coordinates), not as the person/background decision.
- [I] **(a) per-frame matting of the rendered frame over a static plate.** Pros: works for any motion; uses the person as rendered (hair and shoulders as the renderer moves them); background pixels are exactly static wherever alpha = 0. Cons: (1) matte flicker on hair/edges (dtSSD 1.18–1.36 class even for the best models); (2) halos: the rendered frame's pixels near the hair contain WARPED background, so a soft alpha mixes wobbling background into the edge. Composite with an estimated foreground colour (RVM `fgr`, PyMatting) rather than the raw frame. (3) The domain gap of GAN/SPADE-rendered frames for matting models is untested.
- [I] **(b) warp the source alpha with the renderer's own motion.** LivePortrait's deformation is a 3D field (16x64x64x3) acting on features, and the 512 output comes from a SPADE decoder, so a warped 2D alpha is only a coarse guide: 8x upsampled, and not guaranteed to match hair the decoder hallucinates. It is temporally smooth by construction (driven by smooth keypoints), so it is ideal as a prior/trimap for (a): clamp alpha to 1 in the eroded core, 0 outside a dilated band, and let a matting model decide only inside the band.
- [I] **(b′) flow-level background lock (not found in the literature; our own idea, untested).** Because LivePortrait's dense motion is a softmax over [identity/background, 21 keypoint translations], one can force the identity channel to 1 (or blend the deformation toward the identity grid) outside a dilated source-person mask projected into the 64x64x16 volume. Geometric background motion is then removed before decoding. Residual decoder texture changes stay, so it complements pixel compositing and does not replace it. Risk: breaking the learned deformation near the silhouette (tears); needs a soft transition band.
- [I] **Neck/torso continuity.** With stitching off, LivePortrait moves shoulders somewhat, and LivePortrait's own paper admits shoulder jitter. Two options: take the rendered torso inside the matte (continuous with the neck, but the wobble remains on the body), or blend to the SOURCE torso below a neck line with a vertical feather (static body, the EMOPortraits approach, risk of a seam at large yaw). Given the 2.4–22x background-motion finding, the pragmatic choice is rendered head+neck+upper shoulders, fading to source torso lower down.

### Gaps
- No published comparison of (a) vs (b) vs (c) for implicit-keypoint warping renderers; the ranking above is inference and needs an ablation on SANG-M outputs (BG-Flicker ratio, boundary-band temporal difference, FID/CSIM of the face region).
- Whether Real3DPortrait's per-frame M_person comes from warping the source person mask with its torso deformation (a published instance of option b) was not confirmed in this pass.

## Q4. How published talking-head / reenactment systems keep the background (and what they report)

### Takeaway
Three families: (1) crop-and-paste-back with a crop-shaped feathered mask (LivePortrait, Ditto, SadTalker "full") — hides the crop seam but keeps whatever the renderer did to the background inside the crop; (2) explicit layer separation with a static background (Real3DPortrait: head/torso/background with alpha fusion + inpainted background; ER-NeRF/GeneFace++/SyncTalk: a per-video static `bc.jpg`; EMOPortraits: head animated and integrated "with a source image body") — static background by construction, with seams at neck/torso and blending quality as the failure mode; (3) diffusion renderers with reference networks (Hallo, Hallo2, EchoMimic, AniPortrait, Sonic, V-Express) that regenerate the whole frame — measured background flicker ranges from 0.63x to 5.33x the real video's. No paper found that measures background stability for LivePortrait-style warping renderers specifically.

### Cited Findings
- Real3DPortrait (arXiv 2401.08503, ICLR 2024): separate head (3D), torso (2D warping renderer, face-vid2vid-like) and background (1-NN inpainted) branches fused with person/head masks; prior methods "either only model the head part ... or model the head and torso as a whole, which overlook the necessity of a natural torso and background" (numbers in Q2/Q3) — [arXiv 2401.08503](https://arxiv.org/abs/2401.08503) [D, M]
- EMOPortraits (arXiv 2404.19110, CVPR 2024) limitation: "It doesn't generate the avatar's body or shoulders, limiting some use cases. We currently integrate our output with a source image body." — [arXiv 2404.19110](https://arxiv.org/abs/2404.19110) [D]
- FG-Portrait (arXiv 2603.23381) on EMOPortraits: "its results suffer from poor identity preservation and inconsistent foreground–background blending. This stems from its design that animates only the segmented head region while keeping the background fixed." — [arXiv 2603.23381](https://arxiv.org/abs/2603.23381) [D; third-party qualitative judgement]
- SadTalker: "full" mode pastes the crop back with `cv2.seamlessClone` and recommends `--still` (keep original head pose), i.e. paste-back is only advised with little head motion (details Q3) — [SadTalker best_practice.md](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md) [D]. FluentAvatar attributes SadTalker's very low background flicker (CMLR 0.21x GT) to animating "only the facial region while keeping the background nearly static" — [arXiv 2509.12052](https://arxiv.org/abs/2509.12052) [M]
- LivePortrait (arXiv 2407.03168) and Ditto: paste-back with a crop-wide feathered template; LivePortrait's stitching module exists to avoid shoulder misalignment at paste-back; stitching recommended off for large head motion (details Q3) — [arXiv 2407.03168](https://arxiv.org/abs/2407.03168); [LivePortrait argument_config.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/config/argument_config.py); [Ditto putback.py](https://github.com/antgroup/ditto-talkinghead/blob/main/core/atomic_components/putback.py) [D]
- Person-specific NeRF talkers (ER-NeRF; same AD-NeRF-style preprocessing lineage used by GeneFace++ and SyncTalk) composite the rendered head/torso over a static background image extracted from the training video (1-NN hole filling); SyncTalk's torso model is incompatible with its `--portrait` paste-back mode — [ER-NeRF process.py](https://github.com/Fictionarry/ER-NeRF/blob/main/data_utils/process.py); [SyncTalk README](https://github.com/ZiqiaoPeng/SyncTalk) [D]. (GeneFace++ README shows separate `head_ckpt`/`torso_ckpt` — [GeneFace++ README](https://github.com/yerfor/GeneFacePlusPlus); its background extraction code was not opened.)
- Diffusion renderers, measured background flicker (FluentAvatar Table 3, ratio to GT; CMLR / HDTF): Hallo 2.84x / 3.33x; Hallo2 3.47x / 3.44x; EchoMimic 2.53x / 5.33x; AniPortrait 1.47x / 2.67x; V-Express 1.26x / 1.11x; Sonic 0.63x / 0.89x; StableAvatar 0.89x / 1.78x; "diffusion-based methods exhibit widespread pixel changes across the entire frame between consecutive frames" — [arXiv 2509.12052](https://arxiv.org/abs/2509.12052) [M]
- TalkingHeadBench (arXiv 2505.24866): "severe background warping" is one of the generator-specific artifact classes across LivePortrait, EMOPortraits, Hallo/Hallo2, AniPortrait; EMOPortraits shows "neck distortions" — [arXiv 2505.24866](https://arxiv.org/abs/2505.24866) [D]
- Licences of the systems (GitHub API, 2026-09-29): Real3DPortrait MIT; GeneFace++ MIT; ER-NeRF MIT; SyncTalk CC BY-NC 4.0; SadTalker Apache-2.0 with third-party exceptions; Ditto Apache-2.0; Hallo MIT; AniPortrait Apache-2.0; X-Portrait Apache-2.0 — [GitHub repos as linked above; e.g. Real3DPortrait](https://github.com/yerfor/Real3DPortrait) [D]

### Inferences
- [I] Our measured symptom (textured background moves a median 2.4x more than in real video, up to 22x on plain backdrops) is the same quantity FluentAvatar calls BG-Flicker ratio; SANG-M sits in the Hallo/EchoMimic band, while "background-static by construction" systems (SadTalker-style face-only animation, layer-composited systems) sit at or below 1x. A static composited plate should push SANG-M to ≤1x on the background region, with the error moving to the matte boundary instead.
- [I] Systems that keep the background static by construction all accept a weaker failure mode at the neck/shoulder boundary (EMOPortraits "inconsistent foreground–background blending", SyncTalk double chin vs paste-back, LivePortrait shoulder jitter). The compositing boundary must therefore sit where the renderer's motion is small (lower neck/shoulders move little under LivePortrait's head-pose keypoint motion) or be feathered over a band.
- [I] Diffusion-based renderers keeping backgrounds "via reference networks" do NOT guarantee a static background; the measured flicker (up to 5.33x GT) shows they also need compositing if a static background is required.

### Gaps
- VASA-1, DAWN, X-Portrait, Follow-Your-Emoji, MegaPortraits: no primary statement on background compositing or a background metric was found in this pass (not searched in depth due to time budget). VASA-1 has no public code.
- Ditto's paper (arXiv 2411.19509) was not opened; the paste-back evidence is from its code only.
- No paper reports BG-Flicker (or similar) for LivePortrait, Ditto or other implicit-keypoint warping renderers.

## Q5. Metrics for background stability / compositing quality and published values

### Takeaway
There is now one published background-specific metric for talking heads: BG-Flicker (FluentAvatar, arXiv 2509.12052), a background-only mean absolute inter-frame difference using DeepLabV3 person segmentation, reported as a ratio to the ground-truth video's own background change. It is directly usable for SANG-M (our "2.4x more background motion than real" is essentially the same ratio idea). Matting temporal quality is measured with dtSSD; general video temporal consistency with flow-based warping error (E_warp). Mainstream talking-head evaluation suites (e.g. THEval, 8 metrics over 17 models) contain no background metric.

### Cited Findings
- FluentAvatar (arXiv 2509.12052v3, 20 Apr 2026) defines "BG-Flicker: Background-Isolated Flickering Metric": BG-Flicker = 1 − (1/((T−1)·255)) Σ_t MAE(t), MAE over background pixels only, background isolated with DeepLabV3; "Since videos contain background variations, we use GT background error as the reference" — [arXiv 2509.12052](https://arxiv.org/abs/2509.12052) [D]
- FluentAvatar Table 3 "Inter-frame Flickering Evaluation. Error = (1−BG-Flicker)×10^3; Ratio = Error / GT (closer to 1 is better)" [M] (verified from the arXiv HTML source):
  - CMLR Error / Ratio: GT 1.90 / 1.00x; SadTalker 0.40 / 0.21x; V-Express 2.40 / 1.26x; AniPortrait 2.80 / 1.47x; Hallo 5.40 / 2.84x; Hallo2 6.60 / 3.47x; EchoMimic 4.80 / 2.53x; Sonic 1.20 / 0.63x; StableAvatar 1.70 / 0.89x; FluentAvatar 1.90 / 1.00x
  - HDTF Error / Ratio: GT 0.90 / 1.00x; SadTalker 1.20 / 1.33x; V-Express 1.00 / 1.11x; AniPortrait 2.40 / 2.67x; Hallo 3.00 / 3.33x; Hallo2 3.10 / 3.44x; EchoMimic 4.80 / 5.33x; Sonic 0.80 / 0.89x; StableAvatar 1.60 / 1.78x; FluentAvatar 1.20 / 1.33x
  - Authors' caveat: "SadTalker achieves low BG-Flicker error, because its generation only animates the facial region while keeping the background nearly static, which trivially minimizes background pixel variation"; "diffusion methods exhibit frequent high-amplitude spikes characteristic of stochastic denoising" — [arXiv 2509.12052](https://arxiv.org/abs/2509.12052)
  - LivePortrait/Ditto are NOT in this table.
- MatAnyone reports dtSSD (temporal matte-gradient consistency) alongside MAD/MSE/Grad/Conn; e.g. VideoMatte 512x288 dtSSD: RVM 1.36, MatAnyone 1.18 (see Q1) — [arXiv 2501.14677](https://arxiv.org/abs/2501.14677) [M]
- THEval (arXiv 2511.04520) — 8 metrics (Global Aesthetics, Mouth Quality, Face Quality, Lip Dynamics, Head Motion Dynamics, Eyebrow Dynamics, Silent Lip Stability, Lip-Sync) over 17 models; no background, flicker or warping metric (per summarising fetch) — [arXiv 2511.04520](https://arxiv.org/abs/2511.04520) [D]
- Real3DPortrait evaluates background/torso handling only indirectly via full-frame CSIM/FID/AED/APD ablations (see Q2/Q3) — [arXiv 2401.08503](https://arxiv.org/abs/2401.08503) [M]

- Lai et al., "Learning Blind Video Temporal Consistency" (ECCV 2018, arXiv 1808.00449) is the standard reference for the flow-based warping error (E_warp) used in video temporal-consistency work (title/venue verified; the formula was not re-read in this pass) — [arXiv 1808.00449](https://arxiv.org/abs/1808.00449) [D]

### Inferences
- [I] Metric suite recommended for the SANG-M ablation. (1) **BG-Flicker ratio** (FluentAvatar definition: background-only inter-frame MAE, normalised by the GT video's value; use a dilated union of person masks so the moving head never counts as background). (2) Our existing **background motion ratio** (optical-flow magnitude on textured background vs real video). (3) **Masked background PSNR/SSIM against the source plate** over time: for a truly static composite this should be ~inf/1.0 outside the dilated band, a sanity check. (4) **Boundary-band temporal difference** (mean |alpha_t − alpha_{t−1}| or E_warp restricted to a 10–20 px band around the silhouette) to catch matte flicker/halo. (5) The usual face metrics (FID, CSIM, LSE-C/D, FVD) to make sure compositing does not hurt the face.
- [I] Numbers from FluentAvatar show real videos have non-zero background change (GT Error 1.90 on CMLR, 0.90 on HDTF), so a perfectly static plate scores BELOW 1x. The target is therefore "≤1x of GT", not "equal to GT".

### Gaps
- No published BG-Flicker values for LivePortrait, Ditto or other warping renderers; no published matting-flicker numbers on synthetic talking-head frames.
- No talking-head paper was found reporting masked background PSNR/SSIM over time. The warping-error formula and any talking-head values of it were not verified in this pass.

## Q6. Recommended compositing pipeline for SANG-M

### Takeaway
Composite in ORIGINAL-photo coordinates. Put a static, LaMa-inpainted background plate (made once from a BiRefNet+ViTMatte source matte) under the rendered person. The per-frame person alpha comes from a temporally smooth prior (the source alpha carried by the renderer's keypoint motion) that is refined only in a thin boundary band by a matting model, with foreground-colour estimation to avoid halos. Optionally add a flow-level "background lock" inside LivePortrait's dense-motion softmax. LivePortrait's stock paste-back mask alone will NOT fix the problem. All per-source costs are one-off (under a few seconds); per-frame costs are small next to rendering on one GPU.

### Cited Findings
- Stock paste-back mask covers 88.3% of the crop at full opacity, and the crop is ~2.3x the face box (Q3) — [LivePortrait crop.py / mask_template.png / crop_config.py](https://github.com/KwaiVGI/LivePortrait) [M/D]
- LivePortrait dense motion = softmax over identity (background) + 21 keypoint translations on a 16x64x64 volume (`num_kp: 21`, `reshape_depth: 16`) — [LivePortrait dense_motion.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/modules/dense_motion.py); [models.yaml](https://github.com/KwaiVGI/LivePortrait/blob/main/src/config/models.yaml) [D]
- Component evidence: BiRefNet 1024^2 at 17 FPS / 3.45 GB on an RTX 4090 (MIT); ViTMatte-B Composition-1k SAD 20.33 (MIT); Big-LaMa Places-512 large-mask FID 2.97 vs MAT 1.96 (Apache vs CC BY-NC); MatAnyone dtSSD 1.18 vs RVM 1.36 (S-Lab NC vs GPL-3.0); RVM 104–172 FPS at HD; SAM 2.1 39.5–91.2 FPS (Apache) — see Q1/Q2 for sources [M]
- Layer-composited systems (Real3DPortrait, EMOPortraits, NeRF talkers) keep the background static by construction and pay at the neck/torso seam (Q4) [D]

### Inferences
**Recommended pipeline (all [I], to be validated by ablation):**
1. **Source matte (once, full resolution).** BiRefNet (portrait-matting or HR-matting checkpoint) → alpha α_s. Build a trimap by eroding/dilating α_s (e.g. 8–20 px scaled to resolution) and refine with ViTMatte-B. Keep α_s and a binary "core" mask (eroded) and a "band" mask.
2. **Static plate (once).** Dilate the person mask by the maximum expected head/hair displacement over the clip plus a margin (estimate from SANG-M's pose range: project the 3D keypoints' max yaw/pitch/translation to pixels). Inpaint with Big-LaMa at full resolution (Apache-2.0). Fall back to SDXL-inpainting (OpenRAIL++) for large masks or heavy texture, with an "empty background" prompt. Inspect once, since this is a single image. Outside the photo there is nothing to show, because output is in photo coordinates, so padding bars disappear.
3. **Render as today** (512 crop, stitching off). Optional **(b′) background lock**: in the dense-motion module, blend the deformation toward the identity grid (or set the softmax to the identity channel) outside the dilated source-person mask resampled to 64x64 (all 16 depth slices), with a soft band. Evaluate separately; if it alone brings BG motion near 1x, later steps become a safety net.
4. **Per-frame person alpha.** Prior α̂_t = source crop alpha carried by the renderer's motion. Cheapest version: warp α_s by the 2D projection of the (softmax-weighted, depth-averaged) deformation, upsampled to 512. Alternative: seed a video matting model with α_s on frame 0. Final α_t = matting-model output inside the band only, 1 in the warped eroded core, 0 outside the warped dilated band. Temporal smoothing (EMA or flow-guided) inside the band only. Model choice: MatAnyone for research use (best dtSSD); for a permissive licence, MODNet (Apache) or ViTMatte-on-trimap (MIT) per frame, or RVM if GPL is acceptable.
5. **Foreground colour** in the band via PyMatting fast multi-level foreground estimation (MIT) or RVM `fgr`, to avoid a halo of wobbling background in the hair.
6. **Composite in photo space:** I_t = warp_c2o(F_t)·warp_c2o(α_t) + plate·(1 − warp_c2o(α_t)). Below a neck line, optionally cross-fade from the rendered torso to the SOURCE torso (static body, EMOPortraits-style) if shoulder wobble is still visible. Tune on large-yaw clips for seams.
7. **Evaluate** with the metric suite in Q5 (BG-Flicker ratio ≤1x GT, background-motion ratio, boundary-band flicker, face FID/CSIM/LSE unchanged). Ablate: (c) template paste-back only vs (a) matting-only vs (a)+(b) prior vs +(b′).
- **Cost on one GPU (inference from the cited speeds):** source matte + inpaint well under 5 s per identity. Per frame at 512: warp + band matting + foreground estimation ≈ a few ms to tens of ms, which is negligible offline.
- **Licence-clean stack:** BiRefNet (MIT) + ViTMatte (MIT) + Big-LaMa (Apache-2.0) + MODNet/SAM 2.1 (Apache-2.0) + PyMatting (MIT). Research-only upgrades: MatAnyone/MatAnyone 2 (S-Lab NC), MAT (CC BY-NC), FLUX.1 Fill [dev] (non-commercial). RVM is GPL-3.0.
- **Expected residual failure modes:** hair edge flicker (mitigated by the band-only matting and smoothing), neck seam at large yaw, and inpainting artefacts in revealed regions at strong head translation (mitigated by a generous dilation margin and diffusion fallback). Hair that the renderer moves beyond the dilated band would be clipped, so the band must be sized from the actual motion range.

### Gaps
- No empirical result yet on SANG-M outputs. The pipeline ranking rests on component benchmarks plus design reasoning, not on a published end-to-end comparison for LivePortrait-style renderers.
- The feasibility and side-effects of the (b′) flow-level lock (tearing at the silhouette, SPADE decoder texture leakage) are unknown.
- Matting accuracy on SPADE-rendered, slightly blurred hair has not been benchmarked anywhere found.
