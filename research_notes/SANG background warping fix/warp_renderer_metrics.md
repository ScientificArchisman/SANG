# Warp-field and renderer-level fixes for background warping in SANG-M (LivePortrait renderer), alternative renderers, and background-stability metrics

Status: COMPLETE. Date: 2026-09-29. Scope: (a) constraining/masking the warp field, (b) renderer fine-tuning, (c) alternative renderers, (d) metrics. Paste-back/stitching and matting compositing are covered by the sibling notes (`liveportrait_internal_fixes.md`, `matting_compositing.md`), not here.
Evidence grades: **[M]** = number reported in a primary source; **[C]** = claim or description in a primary source (paper text, code or issue) with no number; **[I]** = my inference, not verified by any source.
All arXiv IDs below were verified by downloading the PDF or querying the arXiv API on 2026-09-29.

## Q1. Masking the flow (identity flow outside a foreground mask / mask-blending warped and unwarped features)

### Takeaway
No paper, fork or ComfyUI node I found masks LivePortrait's dense flow. The two most-used ComfyUI ports are functionally identical to the official warping code. LivePortrait's own architecture already contains the mechanism: slot 0 of a 22-way softmax is an identity ("background") flow. So forcing identity outside a head mask is a small inference-time code change (about 5 lines in `WarpingNetwork.forward`). The published precedents are MRAA, which zeroes its background transform at inference, and Real3D-Portrait, which alpha-blends head, torso and background features. Both suggest the approach is sound. How it behaves on LivePortrait at the mask boundary (hair, neck, shoulders) has not been measured anywhere and must be tested.

### Cited Findings
**How LivePortrait's warp field is built (read from the official code, KwaiVGI/KlingAIResearch LivePortrait, main branch):**
- [C] The dense motion network builds `1 + num_kp` candidate 3D flows on a (d=16, h=64, w=64) grid. Slot 0 is the identity grid (code comment: "adding background feature"). Slots 1..21 are pure per-keypoint translations, `identity_grid - kp_driving + kp_source`. The code comment "NOTE: there lacks an one-order flow" means there is no per-keypoint Jacobian/affine term, unlike FOMM. — [dense_motion.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/modules/dense_motion.py)
- [C] An hourglass predicts a softmax mask over the 22 slots (`mask = F.softmax(self.mask(prediction), dim=1)`, shape bs x 22 x 16 x 64 x 64), and the final deformation is `(sparse_motion * mask).sum(dim=1)`. The background therefore stays still only where the network puts its softmax weight on slot 0. — [dense_motion.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/modules/dense_motion.py)
- [C] The pipeline, step by step:
  1. The 3D feature volume f_s (Bx32x16x64x64) is grid-sampled with the deformation, using `align_corners=False`.
  2. The result is reshaped to Bx512x64x64 and passed through two convs.
  3. A 2D occlusion map (`sigmoid(conv(prediction))`, Bx1x64x64) is multiplied in: `out = out * occlusion_map`.

  The occlusion map is a multiplicative gate. It does not blend in un-warped source features. — [warping_network.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/modules/warping_network.py)
- [C] `make_coordinate_grid` builds the identity grid as `2*(x/(w-1)) - 1`, which is the align-corners=True convention, while `grid_sample` is called with `align_corners=False`. — [util.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/modules/util.py), [warping_network.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/modules/warping_network.py)
- [C] Config: 21 implicit keypoints; `estimate_occlusion_map: True`; dense-motion `reshape_depth: 16`, `compress: 4`; SPADE generator `upscale: 2` (256 -> 512); stitching MLP input 126 = (21*3)*2 and output 65 = 21*3 + (tx, ty). — [models.yaml](https://github.com/KwaiVGI/LivePortrait/blob/main/src/config/models.yaml)
- [C] The two most-used ComfyUI ports do not modify the flow:
  - kijai/ComfyUI-LivePortraitKJ's `warping_network.py` / `dense_motion.py` differ from the official files only by MPS/CPU `grid_sample` fallbacks and a dtype try/except.
  - PowerHouseMan/ComfyUI-AdvancedLivePortrait's copies differ from the official files only in comments (diffed 2026-09-29).

  — [ComfyUI-LivePortraitKJ](https://github.com/kijai/ComfyUI-LivePortraitKJ), [ComfyUI-AdvancedLivePortrait](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait)
- [C] Negative result from a limited search: GitHub code search for forks that mask `deformation` with fg/bg masks (`kp_driving kp_source deformation fg_mask` / `bg_mask`) returned 0 hits before hitting the code-search rate limit. I found no fork, ComfyUI node or paper that sets LivePortrait's dense flow to identity outside a mask.

**LivePortrait paper and issue tracker on background/shoulder behaviour:**
- [C] LivePortrait's only "static region" constraint acts through keypoints, not the dense flow:
  - The stitching module is trained in stage 2 with the extractors, warping module and decoder frozen.
  - Its loss is a pixel consistency term on the shoulder region only, L_st,const = ||(I_p,st - I_p,recon) ⊙ (1 - M^st(I_s))||_1, plus an L1 regulariser on the keypoint offset Δ_st.
  - It predicts only a keypoint offset (K x 3).

  — [LivePortrait, arXiv 2407.03168](https://arxiv.org/abs/2407.03168)
- [C] LivePortrait's limitations section says: "when the driving video involves significant shoulder movements, there is a certain probability of resulting in jitter." — [arXiv 2407.03168](https://arxiv.org/abs/2407.03168)
- [C] Open issues confirm the problem, with no maintainer answers:
  - #495 "Moving background" (2025-03-11): "the generated video counts parts of the background as the face ... both the head and part of the background are moving". 0 comments.
  - #507 (2025-04-23) reports shoulder "breaking" after paste-back.
  - #204 asks how the shoulder region is detected and got no answer.

  — [issue #495](https://github.com/KlingAIResearch/LivePortrait/issues/495), [issue #507](https://github.com/KlingAIResearch/LivePortrait/issues/507), [issue #204](https://github.com/KlingAIResearch/LivePortrait/issues/204)
- [C] Issue #28: "I managed to remove wobble of shoulders by masking driving vid so only the head is visible this had a great effect". This is a driving-side fix that does not apply to us (our driving signal is already keypoints), but it shows that shoulder motion leaks in through the keypoints. — [issue #28](https://github.com/KlingAIResearch/LivePortrait/issues/28)
- [M] Closest published precedent for a feature-level mask blend inside a renderer is Real3D-Portrait's HTB-SR module:
  - Fusion: F = (F_head·M_head + F_torso·(1 - M_head))·M_person + F_bg·(1 - M_person), where F_bg comes from shallow convs on a KNN-inpainted static background.
  - Ablation (Table 4, CSIM / FID / AED / APD): Full 0.758 / 42.37 / 0.138 / 0.022; "w/ concat" (plain channel concatenation) 0.737 / 46.38 / 0.144 / 0.025; "w/o inpaint" 0.744 / 43.95 / 0.140 / 0.022.
  - The authors say alpha-blending is "necessary to ... eliminate the artifacts" (hollow artifacts and blur appear with concat).
  - This module was trained with the masks, not added at test time.

  — [Real3D-Portrait, arXiv 2401.08503](https://arxiv.org/abs/2401.08503)
- [C] SynergyWarpNet (arXiv 2512.17331) is an example of learned fusion: an explicit 3D-flow warp is followed by "a confidence-guided fusion module [that] integrates the warped outputs with spatially-adaptive fusing". I read only the abstract and saw no background numbers. — [arXiv 2512.17331](https://arxiv.org/abs/2512.17331)
- [C] FOMM and TPSMM handle disocclusion (background revealed when the head moves) with occlusion masks, which are single-scale in FOMM and 32..256 multi-resolution in TPSMM. The masked feature regions are then inpainted by the generator. — [FOMM, arXiv 2003.00196](https://arxiv.org/abs/2003.00196); [TPSMM, arXiv 2203.14367](https://arxiv.org/abs/2203.14367)

### Inferences
- [I] **There are three hooks, and the flow-level one is preferred.** A head mask H (1 = head/hair/neck, 0 = elsewhere) can be injected at:
  1. The softmax logits: add a large bias to slot 0 where H = 0.
  2. The deformation: `deformation = H*deformation + (1-H)*identity_grid`.
  3. The warped features: `out = H*warped + (1-H)*unwarped`.

  Hooks 1 and 2 give a soft band of intermediate displacement, i.e. a stretch. Hook 3 gives a cross-fade, i.e. ghosting or a double edge. Stretching is usually less visible than ghosting in thin structures such as hair, so hook 2 should be the primary option and hook 3 an ablation.
- [I] **Use LivePortrait's own slot-0 grid, not a mathematically exact identity.** Because of the align_corners mismatch, the network's "identity" is a ~1/63 (1.6%) scale-about-centre resampling with zero padding at the outer feature ring. The decoder was trained on this convention. It is constant over time, so it is harmless for stability.
- [I] **The mask must cover where the head was and where it goes, or you get ghosts.** The deformation is a backward map. An output pixel set to identity samples the source at the same location. If the head has moved away from that location, identity samples old head or hair pixels and leaves a ghost. A safe mask is H_t = dilate(H_src ∪ warp(H_src, deformation_t), r), with the feather r ≈ 1–2 feature pixels (8–16 px at 512). Inside H_t, LivePortrait's normal flow and occlusion/inpainting handle disocclusion as usual. A simpler version is a static dilation of H_src by the maximum head displacement in the clip; for talking heads this is probably 10–30 px at 512 (my guess, not measured).
- [I] **Also clamp the occlusion map outside H_t.** Set it to its value in the source self-reconstruction (x_d = x_s) or to 1. The occlusion map is predicted per frame from the same hourglass, so even with an identity flow it can make background features "breathe" in brightness.
- [I] **Some motion near the mask edge will remain.** The SPADE decoder is convolutional with a finite receptive field, so a band a few feature pixels wide around the edge will still change with head motion. Pixels farther away should become frame-invariant, because their neighbourhood of input features is identical in every frame.
- [I] **Mask choice sets the torso behaviour.**
  - Head-only mask: the torso becomes rigid.
  - Person mask: torso wobble stays.

  Global head rotation R and translation t move all 21 keypoints, and the per-keypoint flows are pure translations. A keypoint that sits on the shoulders therefore drags them with head pose, which fits the torso/border leakage we measure. For a talking-head demo a rigid torso with a soft neck band is likely less objectionable than a swimming torso. The neck seam is the main risk to check visually.

### Gaps
- No quantitative or qualitative report exists (paper, issue or fork) of identity-flow masking on LivePortrait, so how visible the boundary artifacts are (hair halo, neck stretch) is unknown until we test it.
- I did not verify, by inspecting the softmax masks, which of LivePortrait's 21 implicit keypoints carry weight on the torso or background. That would test the "shoulder keypoint dragged by head pose" mechanism directly: log `out_dict['mask']` per slot on one clip.

## Q2. Explicit background-motion models (FOMM, MRAA, TPSMM, DaGAN, face-vid2vid) and their ablations

### Takeaway
Every unsupervised keypoint model since FOMM has an identity or background slot. MRAA and TPSMM add a learned global affine background transform because, without one, keypoints get "hijacked" to explain camera motion. That is the same leakage mechanism we see.

The measured gains, however, are whole-frame L1 on Tai-Chi:
- MRAA: No-bg 0.059 vs Full 0.048, with the PCA change also contributing.
- TPSMM: +L_bg leaves L1 unchanged at 0.046.

No paper reports a background-stability metric. The most transferable idea is MRAA's inference trick: estimate background motion in training, then set it to zero at test time.

### Cited Findings
- [C] **FOMM (NeurIPS 2019):**
  - Flow: T_S<-D(z) = M_0 z + Σ_k M_k (T_S<-R(p_k) + J_k(z - T_D<-R(p_k))). The authors say "the term M_0 z is considered in order to model non-moving parts such as background".
  - Occlusion: ξ' = O_S<-D ⊙ f_w(ξ, T_S<-D), the same gating as LivePortrait.
  - On Tai-Chi, "a light green keypoint is constantly located in the bottom left corner in order to model background or camera motion".

  — [FOMM, arXiv 2003.00196](https://arxiv.org/abs/2003.00196)
- [C] **MRAA (CVPR 2021):**
  - Diagnosis: prior models "assume static backgrounds ... leading to leakage of background motion information into one or several of the detected keypoints"; "The model automatically adapts by assigning several of the available keypoints to model background".
  - Fix: an encoder predicts a global affine A0 (six values) from (S, D) as an extra flow slot.

  — [MRAA, arXiv 2104.11280](https://arxiv.org/abs/2104.11280)
- [M] **MRAA ablation, Table 4** (TaiChiHD 256, K = 10; L1, (AKD, MKR), AED):

  | Variant | L1 | (AKD, MKR) | AED |
  |---|---|---|---|
  | No pca or bg model | 0.060 | (6.14, 0.033) | 0.163 |
  | No pca | 0.056 | (9.58, 0.034) | 0.206 |
  | No bg model | 0.059 | (5.55, 0.026) | 0.165 |
  | Full | 0.048 | (5.59, 0.027) | 0.152 |

  The authors: "Background motion modeling significantly lowers L1 error ... Since background constitutes a large portion of the image". AKD/MKR "are not improved by background modelling". — [arXiv 2104.11280](https://arxiv.org/abs/2104.11280)
- [C] MRAA appendix D: "though we estimate background motion, we set it to zero during animation". — [arXiv 2104.11280](https://arxiv.org/abs/2104.11280)
- [C] **TPSMM (CVPR 2022):**
  - Adds a BG Motion Predictor (affine A_bg) because "camera motion in videos will cause the predicted keypoints to appear in the background area".
  - L_bg = |[A'_bg;001][A_bg;001] - I|, a cycle-consistency loss where A'_bg comes from reversed inputs.
  - L_warp = Σ_i |T(E_i(S)) - E_i(D)|, a feature-level warp loss.
  - Uses multi-resolution occlusion masks.

  — [TPSMM, arXiv 2203.14367](https://arxiv.org/abs/2203.14367)
- [M] **TPSMM ablation, Table 4** (TaiChiHD; L1 / (AKD, MKR) / AED):

  | Variant | L1 | (AKD, MKR) | AED |
  |---|---|---|---|
  | MRAA | 0.048 | (5.41, 0.025) | 0.149 |
  | TPS | 0.048 | (4.96, 0.020) | 0.153 |
  | +Dropout | 0.048 | (4.66, 0.018) | 0.156 |
  | +Multi-Masks | 0.046 | (4.73, 0.018) | 0.150 |
  | +L_bg | 0.046 | (4.64, 0.020) | 0.151 |
  | +L_warp | 0.045 | (4.57, 0.018) | 0.151 |

  — [arXiv 2203.14367](https://arxiv.org/abs/2203.14367)
- [C] **face-vid2vid** (the architecture LivePortrait inherits):
  - K softmax masks combine K warping flows, plus a 2D occlusion mask. The paper describes no explicit background affine.
  - When comparing with bi-layer, "we subtract the background when doing quantitative analyses".
  - There is no background ablation.

  — [face-vid2vid, arXiv 2011.15126](https://arxiv.org/abs/2011.15126)
- [C] **DaGAN (CVPR 2022):** depth-aware attention on the motion field, because "The motion field may contain noisy information from the cluttered background". The authors claim that, unlike the no-depth variant, it does not attend to cluttered backgrounds. This is qualitative only; there is no background metric. — [DaGAN, arXiv 2203.06605](https://arxiv.org/abs/2203.06605)

### Inferences
- [I] **An MRAA/TPSMM-style affine background slot adds little for our static-camera case.** LivePortrait already has slot 0 = identity. The affine slot exists to absorb camera motion during training so that keypoints stop modelling background. Our leakage is the reverse problem: head-pose motion bleeds into background pixels at inference.
- [I] **Why LivePortrait's net may have learned to move the background with the head** (speculative): its training data (69M in-the-wild frames) probably contains hand-held and tracking-camera footage, and without a background slot some of that background motion would be encoded through the keypoints.
- [I] **What transfers is MRAA's recipe.** Give the network somewhere to put background motion during training, then force that slot to identity at test time. For LivePortrait this is the fine-tuning option in Q3, which adds a slot trained on camera motion and zeroes it at inference. The TPSMM numbers suggest the loss itself barely moves reconstruction metrics.

### Gaps
- None of FOMM/MRAA/TPSMM/DaGAN/face-vid2vid reports a background-only stability number (masked PSNR, background flow, or flicker), so their ablations only indirectly support background fixes.

## Q3. Renderer fine-tuning (warping module / SPADE decoder, static-background loss) and papers that fine-tuned LivePortrait

### Takeaway
No paper has fine-tuned LivePortrait's warping module or decoder for background stability.
- Playmate fine-tuned the LivePortrait latent space for pose/expression disentanglement, using a VASA-1-style cross-transfer loss.
- LivePortrait's own authors fine-tuned it for animals.
- A community user reports fine-tuning stage 1 at lr 2e-4, with no metrics.

Official training code is not released ("It's challenging"). A static-background fine-tune therefore means building on the community re-implementation plus a new masked loss. It is feasible on one GPU but costs roughly 1 to 2 weeks in total; the breakdown is in the Inferences below.

### Cited Findings
- [C] **LivePortrait training recipe:**
  - Stage 1 trains F, M, W and G from scratch on ~69M frames (92M before filtering) from ~18.9K identities, plus 60K static styled portraits.
  - Resolution 256 in / 512 out; batch 104; Adam lr 2e-4, betas (0.5, 0.999).
  - Losses include perceptual and GAN terms (global and local), plus keypoint, pose and deformation-prior terms.
  - Stage 2 freezes F, M, W and G.

  — [LivePortrait, arXiv 2407.03168](https://arxiv.org/abs/2407.03168)
- [C] LivePortrait's own animal fine-tune used "a small dataset of animal portraits combined with the original data", dropping the head-pose, lip-GAN and face-id losses. — [arXiv 2407.03168](https://arxiv.org/abs/2407.03168)
- [C] **Playmate (arXiv 2502.07203):**
  - Motivation: "the original LivePortrait also suffered from poor disentanglement between facial dynamics and head pose".
  - Method: adds L_p = ||V(Î_i,j^pose) - V(Î_j,i^exp)||_2, rendered through G(W(f, x_s, x_d)), then "extract[s] motions by utilizing the frozen M".
  - Limitations: "texture sticking due to neural rendering, and minor inconsistencies in complex backgrounds".
  - Playmate also notes that JoyVASA built its latent space on LivePortrait.

  — [Playmate, arXiv 2502.07203](https://arxiv.org/abs/2502.07203)
- [C] Closest precedents for a masked consistency loss on a warping renderer:
  - TPSMM's L_warp: feature L1 between the warped source and the driving encoder features.
  - LivePortrait's L_st,const: a masked shoulder pixel L1.

  — [TPSMM, arXiv 2203.14367](https://arxiv.org/abs/2203.14367); [LivePortrait, arXiv 2407.03168](https://arxiv.org/abs/2407.03168)
- [C] Training code is not released. Maintainers answered "It's challenging : |" on #117 (2024-07-12) and #468 (2025-02-02). A community re-implementation exists: liutaocode/LivePortrait-Train ("Unoffical LivePortrait Training Script [Under Construction]", 41 stars when checked). — [issue #117](https://github.com/KlingAIResearch/LivePortrait/issues/117), [issue #468](https://github.com/KlingAIResearch/LivePortrait/issues/468), [LivePortrait-Train](https://github.com/liutaocode/LivePortrait-Train)
- [C] Community anecdote (issue #506, user ZardZen, 2026-01):
  - Fine-tuned the stage-1 generator, then "all modules", on human + animal videos.
  - Settings: "basically followed the training settings of stage 1 in the paper, with a lr of 2e-4", using 2 frames of one video as source and driving.
  - Says the keypoint losses are unnecessary if only the generator is tuned.
  - Reports no metrics.

  — [issue #506](https://github.com/KlingAIResearch/LivePortrait/issues/506)
- [M] IMTalker renderer cost, for scale (trained from scratch):
  - Data: 660 h / 35,000 clips (VFHQ + VoxCeleb2 + MultiTalk).
  - Compute: 4x A100, batch 16, ~4 days.
  - Size and speed: 124M params, 40 FPS at 512 on an RTX 4090.

  — [IMTalker, arXiv 2511.22167](https://arxiv.org/abs/2511.22167)

### Inferences
- [I] **Proposed fine-tune (option F1): W only.** Freeze F, M and G, and train the dense-motion hourglass (plus its mask and occlusion heads) on our own static-camera talking clips.
  - **Losses:**
    - (i) L1 + VGG reconstruction against the target frame, as in stage 1.
    - (ii) Masked static-background L1: ||(Î_t - I_src_recon) ⊙ (1 - H_t)||_1, where I_src_recon is the frozen model's self-reconstruction of the source. This mirrors L_st,const.
    - (iii) Flow regulariser: ||(deformation - identity) ⊙ (1 - H_t)||_1, or cross-entropy pushing softmax slot 0 to 1 outside H_t.
    - (iv) Optionally, distil from the inference-time masked-flow output (Q1), so the network learns the mask itself and no matting model is needed at inference.
  - Because G is frozen, no GAN loss is needed. That lowers the risk of losing lip or teeth detail.
  - **Cost (my estimate, unverified):** 3–5 days of engineering on the community training code, plus ~1–3 GPU-days on one 80 GB GPU (batch 8–16 at 256, 20–50k steps).
- [I] **Larger fine-tune (option F2): W + G with a GAN loss.** Only worth it if F1 or the Q1 inference mask leaves visible boundary artifacts that need the decoder to learn inpainting. Estimated 3–7 GPU-days, with a higher risk of degrading mouth detail. Re-run SyncNet and LPIPS on the face crop to check.
- [I] **The DiT does not need retraining under F1 or F2.** The 42-d keypoint motion interface is unchanged, since M is frozen and keypoints feed W exactly as before.

### Gaps
- No published numbers exist for any LivePortrait W/G fine-tune (steps, data size, before/after metrics). The only evidence is a GitHub-issue anecdote.
- I did not verify how complete the community training code is (losses implemented, discriminator, released configs).

## Q4. Alternative renderers and their background behaviour; switching cost

### Takeaway
No alternative renderer offers a measured background-stability advantage over LivePortrait, with one exception: Real3D-Portrait's separate static-background branch, where stability holds by construction.
- **Diffusion reference-net methods** report worse flicker than some warp methods on HDTF:
  - Flicker: Hallo 2.33, FantasyTalking 1.87, Sonic 0.688 vs SadTalker 0.545.
  - They are much slower than LivePortrait (the LivePortrait authors' observation).
  - The LivePortrait authors say they are qualitatively less temporally consistent.
- **Flow-free or other-latent renderers** (IMTalker, LIA-X) beat LivePortrait on whole-frame PSNR but do not address the background.

Every alternative except JoyVASA/Ditto-style LivePortrait-based systems uses a different motion space. Switching means re-extracting motion and retraining our 53M DiT. It is the most expensive option and the least supported by evidence.

### Cited Findings
- [C] **IMTalker (arXiv 2511.22167):**
  - Replaces flow + warping with cross-attention "implicit motion transfer": full attention at 64^2 and a top-k Guided Sparse Resampler at 128^2/256^2.
  - Uses its own motion latent.
  - Renderer 124M params plus a 39M flow-matching generator.
  - No background-specific design or claim in the text I read.

  — [arXiv 2511.22167](https://arxiv.org/abs/2511.22167)
- [M] IMTalker Table 1 (HDTF, video-driven self-reenactment, full frame), PSNR / SSIM / LPIPS / FID where given:

  | Method | PSNR | SSIM | LPIPS | FID |
  |---|---|---|---|---|
  | LivePortrait | 27.173 | 0.878 | 0.043 | 9.049 |
  | IMTalker | 28.458 | 0.899 | 0.037 | 7.426 |
  | MCNet | 27.974 | 0.886 | – | – |
  | LIA | 25.991 | 0.839 | – | – |
  | X-Portrait | 25.590 | 0.841 | – | – |

  — [arXiv 2511.22167](https://arxiv.org/abs/2511.22167)
- [M] LIA-X Table 1 (512x512 self-reenactment):
  - VoxCelebHQ PSNR: LivePortrait 17.45, X-Portrait 16.99, LIA 22.14, LIA-X 24.39.
  - VoxCelebHQ L1: LivePortrait 0.087, LIA-X 0.040.
  - TalkingHead-1KH PSNR: LivePortrait 20.26, LIA 23.37, LIA-X 26.07.
  - LIA-X is a flow "warp-render" model with a Sparse Motion Dictionary, scaled to ~1B params.

  — [LIA-X, arXiv 2508.09959](https://arxiv.org/abs/2508.09959)
- [M] X-NeMo (ICLR 2025, arXiv 2507.23143) is a diffusion model whose reference network "extracts reference features of identity appearance and background which are then cross-queried by the UNet self-attention". Self-reenactment: L1 0.055 / SSIM 0.826 vs LivePortrait 0.074 / 0.770. — [arXiv 2507.23143](https://arxiv.org/abs/2507.23143)
- [M] HunyuanPortrait (arXiv 2503.18860, SVD-based):
  - Aims at "strong consistency of the characters and background in the reference image".
  - Uses an ArcFace + DINOv2 appearance extractor for identity and "background details".
  - Says early GAN methods "often produce results with background jitter".
  - Its table: LivePortrait PSNR 31.41 / SSIM 0.72 / FVD 483.38 vs AniPortrait 30.54 / 0.67 / 430.24. No background metric.

  — [arXiv 2503.18860](https://arxiv.org/abs/2503.18860)
- [C] MegActor (arXiv 2405.20851) segments the reference background and encodes it with CLIP "thereby ensuring the stability of the background". The LivePortrait paper nonetheless shows "the red banner disappears in some frames of MegActor". — [MegActor, arXiv 2405.20851](https://arxiv.org/abs/2405.20851); [LivePortrait, arXiv 2407.03168](https://arxiv.org/abs/2407.03168)
- [C] Real3D-Portrait (arXiv 2401.08503) has a separate KNN-inpainted static background branch alpha-blended with head and torso features, so the background is static by design. Its motion input is 3DMM-based (PNCC), not LivePortrait keypoints. — [arXiv 2401.08503](https://arxiv.org/abs/2401.08503)
- [M] ConsistTalk (arXiv 2511.06833) Table 1 on HDTF; Flicker is lower-better, VBench is motion-smoothness / background-consistency, higher-better:

  | Method | Flicker | VBench |
  |---|---|---|
  | SadTalker | 0.5445 | 99.5 / 97.77 |
  | AniTalker | 0.7519 | 99.5 / 98.37 |
  | Hallo | 2.3274 | 99.37 / 97.18 |
  | Hallo-v2 | 0.5914 | 99.4 / 96.95 |
  | EchoMimic | 0.6552 | 99.34 / 97.13 |
  | Sonic | 0.688 | 99.61 / 97.47 |
  | FantasyTalking | 1.874 | 99.63 / 96.56 |
  | ConsistTalk | 0.4218 | 99.68 / 98.28 |

  Qualitatively, ConsistTalk cites "background warping (e.g., AniTalker, SadTalker)". — [arXiv 2511.06833](https://arxiv.org/abs/2511.06833)
- [C] SkyReels-A1 (arXiv 2502.10841) lists "background instability" among the failures of existing portrait-animation methods. Hallo3 (arXiv 2412.00733) targets dynamic backgrounds ("generating immersive, realistic backgrounds"), i.e. it moves backgrounds on purpose. — [arXiv 2502.10841](https://arxiv.org/abs/2502.10841); [arXiv 2412.00733](https://arxiv.org/abs/2412.00733)
- [C] The LivePortrait authors write: "the temporal consistency of the foreground and background is not as good compared to non-diffusion-based methods, due to the high variability of the diffusion models". Examples: a statue vanishing in FADM, "pedestrian-like unnatural background movements" in AniPortrait. — [arXiv 2407.03168](https://arxiv.org/abs/2407.03168)
- [C] Verified IDs (renderers named in the brief, not read in full): Follow-Your-Emoji 2406.01900; X-Portrait 2403.15931 (built on a pre-trained face-vid2vid, per IMTalker); Hallo2 2410.07718; Hallo3 2412.00733; EchoMimic 2407.08136; Sonic 2411.16331; Loopy 2409.02634; LIA 2203.09043; MegActor-Σ 2408.14975; JoyVASA 2411.09209; Ditto 2411.19509. — [arXiv API](http://export.arxiv.org/api/query?id_list=2406.01900,2403.15931,2410.07718,2412.00733,2407.08136,2411.16331,2409.02634,2203.09043,2408.14975,2411.09209,2411.19509)

### Inferences
- [I] **Whole-frame PSNR/L1 gaps do not show better background stability.** LIA-X's large margin over LivePortrait (e.g. 17.45 vs 24.39 dB on VoxCelebHQ) is likely inflated by LivePortrait being run in its crop/paste pipeline or background misalignment. These are reconstruction metrics, not stability metrics.
- [I] **Switching cost by family:**
  - **IMTalker / LIA / LIA-X:** new motion latent, so re-extract motion for all training clips and retrain the DiT. That is days of GPU time plus a new audio-to-motion interface. Background behaviour is unknown.
  - **Real3D-Portrait:** static background, but 3DMM motion. We would need a keypoints->3DMM mapping or DiT retraining, it is lower resolution (my recollection, not re-checked), and its head/torso seams are a known risk.
  - **Diffusion reference-net methods** (X-NeMo, HunyuanPortrait, FYE, X-Portrait, Hallo2/3, EchoMimic, Sonic, Loopy):
    - The video-driven ones would need a driving video, e.g. LivePortrait renders of our keypoints, i.e. a two-renderer stack.
    - Audio-driven ones would replace SANG-M entirely.
    - All are far slower than real time on one GPU (my inference from their multi-step sampling; not measured here).
    - The flicker numbers above do not show better background stability.
- [I] **Ranking of this family:** last, behind any warp-level fix.

### Gaps
- I did not find any paper measuring masked background motion for IMTalker, LIA-X, X-NeMo, HunyuanPortrait, FYE or X-Portrait.
- I did not verify code or weight availability for IMTalker or LIA-X, or their licence terms.

## Q5. Metrics for background stability, and a precise metric for SANG demos; ranked options

### Takeaway
Published metrics come in three kinds:
- **Whole-frame reconstruction** (PSNR/SSIM/L1/LPIPS): dominated by the background, but measuring reconstruction, not stability.
- **VBench Background Consistency** (CLIP similarity): coarse; 96.5–98.4 across 8 talking-head methods.
- **Flicker / flow warping error:** Flicker is a mean absolute frame difference on static content. Flow warping error does not penalise a background that moves coherently.

None is masked to the static background as ours is. I found no paper that reports masked-background PSNR or RAFT background flow.

Our current metric is sound as a jitter measure. Recommended additions:
1. A person mask instead of a fixed head box.
2. A drift term anchored to a reference frame, which catches slow swim that frame differences miss.
3. A RAFT background-flow term in pixels, which avoids the ratio blow-up on studio backdrops (22.6x vs 7.75x with the eps floor).
4. Separate zones for background, torso and border.

### Cited Findings
- [C] **VBench (arXiv 2311.17982) definitions:**
  - Background Consistency: "CLIP feature similarity across frames".
  - Temporal Flickering: "We take static frames and compute the mean absolute difference across frames".
  - Dynamic Degree: uses RAFT.
  - Caveat from the paper: "a completely static video can score well in the aforementioned temporal quality dimensions".

  — [VBench, arXiv 2311.17982](https://arxiv.org/abs/2311.17982)
- [C] **Flow warping error (Lai et al., ECCV 2018):**
  - E_warp(V_t, V_t+1) = (1/Σ_i M_t^(i)) Σ_i M_t^(i) ||V_t^(i) - V̂_t+1^(i)||_2^2, where V̂_t+1 is the next frame back-warped by optical flow and M_t is a non-occlusion mask.
  - The video score is the mean over t.

  — [Lai et al., arXiv 1808.00449](https://arxiv.org/abs/1808.00449)
- [M] ConsistTalk reports VBench background consistency of 96.56–98.37 and Flicker of 0.4218–2.3274 across 8 methods on HDTF (full table in Q4). — [arXiv 2511.06833](https://arxiv.org/abs/2511.06833)
- [C] Whole-frame metrics are background-dominated:
  - MRAA: "background constitutes a large portion of the image, and L1 treats all pixels equally".
  - face-vid2vid "subtract[s] the background" for one comparison.

  — [MRAA, arXiv 2104.11280](https://arxiv.org/abs/2104.11280); [face-vid2vid, arXiv 2011.15126](https://arxiv.org/abs/2011.15126)
- [C] **SANG's current metric** (`sang/bench.py::background_motion`, `scripts/bg_warp.py`):
  - Mean |frame-to-frame difference| in gray levels.
  - Pixels: outside a fixed central head box (rows 10–80%, cols 20–80%), among the quietest 40% of real-video pixels, and above the 70th gradient percentile.
  - The ratio gen/real is floored at eps 0.05.
  - First measurement: median 2.35x, range 1.42–11.97x over 6 clips; studio backdrop 7.75x with the floor, 22.6x without it.

  — local code in this repo (`scripts/bg_warp.py` docstring)
- [C] Negative result, not exhaustive (searched arXiv abstracts and a full-text research index): no portrait-animation paper I found reports masked-background PSNR/SSIM against a reference over time, or RAFT flow restricted to a background mask, as a named metric.

### Inferences
- [I] **Gaps in the current metric:**
  - (a) Frame differences measure velocity. A slow coherent swim (e.g. 0.1 px/frame, drifting 10 px over 100 frames) barely registers but is visible.
  - (b) The fixed head box leaves the torso and lower border in the "background" set. The quiet-pixel filter only removes pixels that move in the real video.
  - (c) Ratios blow up when the real background is almost perfectly still.
  - (d) Flow warping error cannot catch coherent swim; VBench CLIP consistency is too coarse.
- [I] **Proposed SANG background-stability protocol (BG-3):** fixed settings, run on real R and generated G (both 512, frame-aligned) and on the self-reconstruction G_ref (the renderer run with x_d = x_s).
  - **Masks**
    - P = dilate(∪_t seg(R_t) ∪ ∪_t seg(G_t), 15 px), where seg is a person-segmentation model frozen for the whole project (e.g. RVM, BiRefNet or MediaPipe selfie).
    - B = ¬P.
    - B* = B ∩ {pixels with real temporal MAD ≤ 1.0 gray level} ∩ {Sobel |∇R_0| > 70th percentile within B}. This keeps our texture filter.
    - Zones reported separately: BG = B*; TORSO = P minus dilate(face-parse head+hair, 15 px), restricted to real-static textured pixels; BORDER = outer 32-px ring ∩ B*.
  - **Metrics** (Y = luma, 0–255)
    1. **Jitter** J = mean_{t, p∈zone} |Y(G_{t+1}) − Y(G_t)|, also computed on R. Report J_gen, J_real and J_gen / max(J_real, 0.05). This is the current metric with better masks.
    2. **Drift** D = mean_t mean_{p∈zone} |Y(G_t) − Y(G_ref)|, plus BG-PSNR_t = PSNR(G_t, G_ref on zone), reported as mean and minimum over t. Anchoring to G_ref (not R) removes the renderer's static reconstruction error, so only motion-induced change is counted. For real video use R_0 as the anchor.
    3. **Flow** F = median_t P95_{p∈zone} ||RAFT(G_ref → G_t)(p)||_2, in pixels at 512 (torchvision `raft_large`), keeping only forward-backward-consistent pixels. Compute the same on R for a noise floor.
  - **Headline and targets:** F_gen(BG) in px, J ratio, and D.
    - F_gen(BG) ≤ max(F_real, 0.3 px).
    - J ratio ≤ 1.2.
    - BG-PSNR_min ≥ 40 dB.

    These thresholds are my proposal and should be calibrated on real clips. Absolute pixel flow stays finite on studio backdrops, where the ratio exploded.
- [I] **Ranked warp/renderer-level options for SANG-M** (single GPU; costs are my estimates unless cited):
  1. **Inference-time masked flow (hook 2 in Q1). Do first.**
     - Recipe: deformation = H_t·def + (1−H_t)·slot-0 grid, with H_t = dilate(H_src ∪ warp(H_src, def_t)) and an 8–16 px feather. Clamp the occlusion map outside H_t to its G_ref value. Use a head+hair+neck mask from segmenting the source once.
     - Cost: ~1 day of engineering; about 1 GPU-hour to re-render and score the 6 validation clips; negligible runtime overhead. No DiT change.
     - Expected: background pixels outside the mask plus the decoder's receptive-field band become frame-invariant by construction.
     - Risks: neck seam, hair halo, rigid torso.
     - Support: MRAA's zero-background trick [C] and Real3D's masked fusion [M]; no LivePortrait-specific data.
  2. **Feature-level blend (hook 3), as an ablation of option 1.** Same cost; more ghosting risk. Keep it if hook 2 produces visible stretching.
  3. **Keypoint diagnosis and pinning.** Find the implicit keypoints whose softmax slot mass lies on the torso or border and hold them at x_s. Cost: hours. May overlap with the sibling "LivePortrait internal fixes" note, and it cannot fix the background near the head.
  4. **Fine-tune W only (F1)** with masked static-background L1 + an identity-flow regulariser, optionally distilled from option 1.
     - Cost: 3–5 days of engineering (no official training code) + 1–3 GPU-days.
     - Gain over option 1: no matting at inference, learned boundaries.
     - Precedents: L_st,const and TPSMM's L_warp.
  5. **Fine-tune W + G with GAN (F2).** 3–7 GPU-days; only if option 1 or 4 leaves boundary artifacts. There is risk to lip detail.
  6. **Explicit affine background slot à la MRAA/TPSMM + retraining.** Low value: LivePortrait already has an identity slot, and TPSMM's L_bg left L1 unchanged (0.046 → 0.046) [M].
  7. **Switch renderer** (IMTalker, LIA-X, Real3D-Portrait, diffusion). Weeks of work, including re-extracting motion and retraining the DiT. No measured background advantage except Real3D's by-construction static background; diffusion flicker is equal or worse [M].

### Gaps
- There are no published background-stability numbers for LivePortrait itself, so no external baseline for our 2.35x median.
- The RAFT noise floor on static, lightly textured, compressed backgrounds at 512 px is unmeasured. The 0.3 px target must be calibrated on real clips.
- There is no human-perception threshold for background swim in talking heads (how many px or gray levels become noticeable). Thresholds above are engineering guesses.
