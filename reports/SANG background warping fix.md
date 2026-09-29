# Predict the missing motion, then pin the background

SANG's background moves mainly because of what its 42-d target throws away, not because the frozen LivePortrait renderer can't hold a scene still. On real motion, SANG's own 2×2 ablation shows the renderer driven by the full 70-d keypoint motion with stitching off reproduces the real clip at **25.08 dB PSNR (+7.73 dB over a frozen frame)**. Squeezing the same real motion through the 42-d target (scale, translation and 8 keypoints held from frame 0) drops that to **17.57 dB (+0.23 dB)** and makes the background move about **1.6x** as much as the real video. Generated demos move it a median **2.35x** (up to 12x). So the cheapest fixes come first: hold the 8 non-driven keypoints in LivePortrait's own camera-frame convention, then decompose which discarded components (scale, translation, the 8 "shape" keypoints' per-frame offsets) the renderer needs, then widen the target and retrain, which is cheap (the cache already holds the 70-d motion; a run is under 20 minutes of step time). Masking the warp field outside a person mask, compositing onto an inpainted plate, and renderer fine-tuning are guarantees and fallbacks for whatever residual motion is left, not the first fix.

## Update (2026-09-29, jobs 173579–173584): the decomposition changes the ranking

The queued decomposition has been run: 20 clips, 19 rendered, stitching off, real motion.

| Render of real motion | PSNR | Face PSNR | Mouth corr | Eyes corr | CSIM | Background ratio (median / mean) |
|---|---|---|---|---|---|---|
| 70-d | 24.69 | 25.66 | 0.848 | 0.878 | 0.900 | **1.25 / 1.51** |
| 42-d, held=head | 17.26 | 16.82 | 0.809 | 0.820 | 0.911 | 1.50 / 2.32 |
| 42-d, held=camera | 17.27 | 16.81 | 0.809 | 0.821 | 0.911 | 1.53 / 2.28 |
| 42-d camera + real s, t | 21.83 | 24.50 | 0.836 | 0.840 | 0.901 | 1.89 / 2.42 |
| 42-d camera + real held keypoints | 17.17 | 16.82 | 0.828 | 0.874 | 0.903 | 2.11 / 3.11 |

The generated demos agree. With held=camera (`results/extras/demo_held`), the background ratio is **2.41x**, against **2.36x** for the same clips and seed under the old convention.

Three conclusions revise the ladder below:

1. **The held-keypoint convention does not matter.** Fix (a) changes nothing measurable. It is kept as the default because it matches upstream and costs nothing, but it is not a background fix.
2. **The 7.5 dB gap is mostly head placement, not background.** Real scale and translation recover most of it (face PSNR 16.8 → 24.5). But they *raise* background motion (1.89x). Every extra moving component (translation, scale, held keypoints) leaks more motion into the background. Widening the target (b) would improve fidelity and worsen the background, so it is **not a background fix**. It stays a separate fidelity question.
3. **The renderer leaks any keypoint motion into the background.** Even perfect 70-d motion gives 1.25x. The one configuration measured still is **stitching** (70-d + stitching: 0.95x, job 173138). Its measured cost on the *42-d* target is nil: PSNR 17.54 vs 17.57, mouth correlation 0.822 vs 0.811 without it, CSIM 0.912 both (jobs 173140/1). Its large cost appeared only on 70-d motion, where it fights the real head's translation. Stitching therefore becomes the first rung.

The revised ladder:

1. Stitching on the 42-d output, measured with `demo_motion.py --stitch` and M0 `--target 42` with stitching.
2. If a residual remains, the slot-0 background lock (c).
3. Compositing onto an inpainted plate (d), as the guarantee.
4. Renderer fine-tuning (e), last.

Fix (b) moves to a separate fidelity track: predicting scale and translation for head placement, with stitching or the lock holding the background.

## The frozen renderer reproduces real motion; the 42-d round trip loses 7.5 dB

SANG's strongest evidence is its own, and it moves the diagnosis away from the renderer. The 2×2 real-motion ablation extracts real motion from a clip and renders it from the clip's frame 0 with the frozen renderer. It then compares the render with the real crop at 256 px (jobs 173138–173141, 2026-09-29, [scripts/motion_ceiling.py](../scripts/motion_ceiling.py)). Results are paired over the 20 clips that all four runs finished. Both "42" rows used the **old** convention, which rotates the 8 held keypoints' δ with the head.

| Render of REAL motion | PSNR vs real | PSNR gain over a frozen frame | Mouth corr | CSIM |
|---|---|---|---|---|
| 70-d, stitching OFF | **25.08** | **+7.73 dB** | 0.847 | 0.9006 |
| 70-d, stitching ON | 18.27 | +0.93 | 0.832 | 0.8942 |
| 42-d (old held convention), stitching ON | 17.54 | +0.20 | 0.822 | 0.9120 |
| 42-d (old held convention), stitching OFF | 17.57 | +0.23 | 0.811 | 0.9119 |

Driven by the full 70-d motion with stitching off, the renderer gains **+7.7 dB over simply freezing frame 0**, so it tracks the real video closely. Two operations each destroy almost all of that gain. The first is the 42-d round trip: `to_target` followed by `from_target`, holding scale, translation and the 8 non-driven keypoints at frame 0. The second is stitching, which costs 6.8 dB even on untouched 70-d motion. **The dominant cause is therefore what the 42-d target discards and how the held parts are filled back in, not the renderer itself.** This is also why SANG's own design gate did not fire. The 2026-09-23 design doc said to restore t_x, t_y if 42-d and 70-d differ by more than 0.5 dB PSNR or 0.01 CSIM ([docs/bidirectional_design_2026-09-23.md](../docs/bidirectional_design_2026-09-23.md)). The true gap is **7.5 dB**, but earlier checks ran with stitching ON, where both configurations sit near 18 dB and the gap is hidden. The slightly *higher* CSIM of the 42-d rows (0.912 vs 0.901) is not an identity gain. It is the known pose bias of CSIM, which IMTalker's authors describe as "highly sensitive to pose variations" ([IMTalker](https://arxiv.org/abs/2511.22167)): less head motion scores better.

Three cautions keep this reading honest. First, PSNR against the real clip also punishes *placement*. Holding frame 0's scale and translation leaves the head where it started rather than where the real head went, so part of the 7.5 dB is a fidelity loss, not a stability loss. Second, the background evidence is thinner than the PSNR evidence. Only two runs finished their background summaries (29 clips): **70-d with stitching ON gave a median background ratio of 0.95** (mean 1.38), and **42-d old convention with stitching OFF gave 1.60** (mean 2.20). These two runs differ in both target and stitching, so the comparison confounds them. The 70-d stitching-OFF ratio timed out, and it is the single most important missing number. The +7.7 dB PSNR makes it likely to sit near 1, but that is inference. Third, the 42-d round trip also costs lip fidelity (mouth correlation 0.847 → 0.811) even though the lip keypoints themselves pass through exactly. The re-extracted mouth evidently depends on how the rest of the keypoint cloud is placed around it.

The generated demos (runs/motion_12k_anneal: 6 unseen speakers, 512 px, stitching off, no paste-back, old held convention) are measured by `sang.bench.background_motion` ([sang/bench.py](../sang/bench.py), [scripts/bg_warp.py](../scripts/bg_warp.py)). The metric is the mean absolute frame difference in gray levels on textured pixels that are among the quietest 40% in the real video and lie outside a fixed central head box. It is reported as a generated/real ratio with a 0.05 floor. The demos move the background a **median 2.35x** (per clip 1.42, 1.51, 1.80, 2.89, 7.75 and 11.97x). The ratio is the same with or without sampling constraints and for both start modes, so the sampler is not the cause. Motion maps show thin bright edges on every background object, the shoulders and the padding bars: a small global displacement every frame. The real-motion 42-d floor of 1.60x is what the generator inherits even with perfect motion. That the demos are worse still is plausibly because generated poses depart further from the source pose than real frame-0-anchored motion does, which enlarges the old convention's rotation error (inference).

Three things are implemented and unit-tested but **not yet run on the cluster**. `from_target(..., held="camera")` is now the default, keeping the 8 non-driven keypoints at the source's camera-frame δ as upstream LivePortrait does ([sang/motion.py](../sang/motion.py)). `motion_ceiling.py --real s t held` takes scale, translation and/or the held keypoints' per-frame δ from the real motion instead of frame 0; with all three it reproduces the 70-d render exactly (verified). And a five-run decomposition is queued, all with `--no-stitch`: 70; 42 held=head; 42 held=camera; 42 camera + real s,t; 42 camera + real held. The sixth, sanity row (42 camera + real s,t,held) was not queued, because a unit check already shows it reproduces the 70-d motion exactly. That decomposition is the experiment the rest of this plan hangs on.

## Every keypoint tugs on the background through a 22-way softmax

LivePortrait's warp explains why small keypoint errors show up as edges on distant objects. The dense-motion network builds 22 candidate flows on a 16×64×64 grid. Slot 0 is an identity grid, which the code labels "adding background feature". Slots 1–21 are pure per-keypoint translations x_s,k − x_d,k, with no Jacobian term ("NOTE: there lacks an one-order flow"). An hourglass predicts a per-voxel softmax over the 22 slots, and the deformation is their weighted sum ([dense_motion.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/modules/dense_motion.py)). **A background voxel therefore stays still only if the network puts all of its weight on slot 0.** Any residual weight on keypoint k moves it by that weight times the keypoint's displacement. The warped features are then gated by a 64×64 occlusion map and decoded by a SPADE generator from 256 to 512 px ([warping_network.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/modules/warping_network.py), [models.yaml](https://github.com/KwaiVGI/LivePortrait/blob/main/src/config/models.yaml)). **One flow cell spans 8 output pixels**, so sub-cell leakage lands as the thin displaced edges SANG sees. All 21 keypoints are driven through one transform, x = s(x_c R + δ) + t, so the global rotation, scale and translation reach every keypoint ([live_portrait_wrapper.py](https://github.com/KwaiVGI/LivePortrait/blob/main/src/live_portrait_wrapper.py)). Users who plotted the keypoints report that they all sit "close to [the] face region", and the maintainers call them a latent code with no fixed semantics ([#29](https://github.com/KlingAIResearch/LivePortrait/issues/29), [#499](https://github.com/KlingAIResearch/LivePortrait/issues/499)). SANG's exact symptom, "both the head and part of the background are moving", is open issue #495, which has no reply ([#495](https://github.com/KlingAIResearch/LivePortrait/issues/495)).

Older keypoint models show how such networks spend capacity on the background. In FOMM, "a light green keypoint is constantly located in the bottom left corner in order to model background or camera motion" ([FOMM](https://arxiv.org/abs/2003.00196)). MRAA's authors found that "the model automatically adapts by assigning several of the available keypoints to model background" ([MRAA](https://arxiv.org/abs/2104.11280)). This supports SANG's central hypothesis, though it does not prove it. LivePortrait's motion extractor is trained to reconstruct whole frames, so on a real clip the per-frame scale, translation and all 63 δ coordinates together form a configuration under which the static background reconstructs in place. Replace 27 of those numbers with frame-0 constants while the head rotates, and the compensation breaks. **This is inference.** The measured support is the 7.5 dB round-trip gap and the confounded 1.60x vs 0.95x background ratios. The queued decomposition is the direct test.

### What the 42-d target throws away, and what everyone else keeps

The 70-d motion is scale (1), three angles (3), translation (3) and 63 expression offsets δ. SANG's target keeps the three angles and 39 head-frame δ coordinates of keypoints 1, 2, 6 and 11–20. It discards **27 effective dimensions**: scale, t_x, t_y, and the 24 δ coordinates of keypoints 0, 3, 4, 5, 7, 8, 9 and 10. (t_z is zeroed at render anyway.) Other LivePortrait-based audio generators, from their code, fill the same slots as follows.

| Component | SANG (42-d) | Upstream absolute mode | KDTalker / Xemo-Talker | Ditto | AVTR-1 |
|---|---|---|---|---|---|
| Scale | source | source | source | source | source |
| t_x, t_y | **source** | driving | **generated** | generated, relative to frame 0 | **source** |
| 8 non-driven keypoints' δ | source, rotated with head (old) / camera frame (new) | camera-frame source, plus 6 driven scalars on kp 3, 4, 5, 8, 9 | **generated** (all 63) | camera-frame source (also brows 1, 2) | source, rotated with head |
| Stitching / paste-back | off / off | on / on | off / off (KDTalker); on / off (Xemo) | on / on | on / on + MODNet matte |

Sources: [upstream pipeline](https://github.com/KwaiVGI/LivePortrait/blob/main/src/live_portrait_pipeline.py), [KDTalker inference.py](https://github.com/chaolongy/KDTalker/blob/main/inference.py), [Xemo-Talker inference.py](https://github.com/chaolongy/Xemo-Talker/blob/main/inference.py), [Ditto motion_stitch.py](https://github.com/antgroup/ditto-talkinghead/blob/main/core/atomic_components/motion_stitch.py), [AVTR-1 motion_stitch.py](https://github.com/avaturn-live/avtr-1/blob/main/src/avtr1_renderer/components/liveportrait/motion_stitch.py).

Two facts stand out. First, **only SANG and AVTR-1 hold translation fixed**, and AVTR-1 hides the consequences behind stitching, paste-back and a MODNet background matte ([AVTR-1 §2.1, §4.2](https://arxiv.org/html/2609.22913)). SANG copied AVTR-1's target but not its safeguards. Second, SANG's old convention rotated the held δ with the head: δ_new = δ_src R_srcᵀ R_new. LivePortrait's authors explicitly rejected that form. They wrote that the scale-orthographic x = s((x_c + δ)R) + t "leads to overly flexible learned expressions δ, causing texture flickering", which is why the renderer was trained with δ added after rotation ([LivePortrait §3.2](https://arxiv.org/html/2407.03168v1)). Per held keypoint, the old convention's error is s·δ_src,k(R_srcᵀR_new − I). That is zero at the source pose, roughly s·|δ_k|·θ for a small rotation θ away from it, and different in every frame. It is a small, per-frame, global displacement of exactly the keypoints most likely to own jaw, hair, neck and nearby background: a match for the motion maps (inference). The typical |δ| of those keypoints is unpublished. It can be computed from SANG's cache to size the error in pixels. Upstream absolute mode also drives six scalars that SANG holds: the y of keypoints 3 and 4, z of 5 and 8, and y and z of 9. So the minimal "upstream-faithful" target has 45 expression coordinates, 48-d with the angles, not 42-d.

The camera-frame fix is exact at the source pose and leaves the generator untouched. It cannot restore the per-frame translation, scale or shape-keypoint motion that real clips carry. If the decomposition shows those components matter, the fix has to happen in what SANG predicts.

## Fixes, cheapest first

The fixes form a ladder. The first two act on the cause, the keypoints the renderer receives. The next two guarantee a still background whatever the keypoints do. The last two change the renderer. Each rung has a gate, and the next rung is climbed only if the gate fails.

| # | Fix | Where in SANG | Retrain? | Evidence | Main risk | Cost (1 GPU) |
|---|---|---|---|---|---|---|
| a | Hold the 8 non-driven keypoints in the camera frame | `sang/motion.py` `from_target(held="camera")` (done, default) | No | Upstream convention; LivePortrait §3.2. Untested on SANG | None for the lips: driven keypoints are unchanged (tested) | Done; ~1 GPU-h to evaluate |
| b | Predict what the renderer needs: scale, t_x, t_y and/or the 8 keypoints' δ | `sang/motion.py`, `sang/motion_model.py`, `scripts/train_motion.py`, `sang/naturalness.py` | Yes, one run | Measured: 7.5 dB round-trip gap; KDTalker and Xemo-Talker generate all 63 δ plus translation | Predicted translation drifts or jitters; lip sync must not regress | ~1 day engineering; one run (<20 min step time) |
| c | Background lock: force slot-0 identity motion outside a person mask | Wrapper around LivePortrait's warping in `MotionCodec.render` | No | **Untested anywhere** | Neck seam, hair halo, ghosting | ~1 day; ~1 GPU-h |
| d | Composite the rendered person onto an inpainted static plate | New step after `MotionCodec.render` | No | Real3D-Portrait ablation; component benchmarks | Hair halo, matte flicker | 2–3 days; seconds per clip |
| e | Fine-tune the warping module with a static-background loss | Training code for LivePortrait (not released) | Renderer | No published LivePortrait fine-tune metrics | Lip detail if the decoder is touched | 3–5 days + 1–3 GPU-days |
| f | Switch renderer | Everything downstream of the DiT | DiT and renderer | None shows a background advantage | Weeks of work | Weeks |

**(a) Camera-frame held keypoints.** This is implemented and is now the default. It removes the old convention's per-frame rotation error, s·δ_src,k(R_srcᵀR_new − I), for the 8 held keypoints (inference). The 13 driven keypoints come out identical under both conventions, which a unit test checks, so lip sync cannot move. Upstream also drives six extra scalars on keypoints 3, 4, 5, 8 and 9. They come from the driving video, so SANG can only use them by predicting them, which belongs under (b). **Gate:** in the decomposition, "42 camera" should close a clear share of the 7.5 dB gap, and `scripts/bg_warp.py` on the generated demo (`results/extras/demo_held` vs `demo_fix/*start-source*`) should fall from 2.35x toward 1.

**(b) Widen the target and retrain.** The decomposition picks the components:

- If `--real s t` recovers most of the gap, add scale, t_x and t_y.
- If `--real held` recovers it, add the 24 camera-frame δ of keypoints 0, 3, 4, 5, 7, 8, 9 and 10.
- If both are needed, predict the full motion: 3 angles, 63 δ, scale, t_x and t_y (69-d; t_z is zeroed at render).

The minimal upstream-faithful choice is 45 expression coordinates, 48-d with the angles: SANG's 39 plus upstream's six extra scalars. No re-cache is needed, because the cache stores the full 70-d motion per frame and the 42-d target is derived at load time. A run costs ~15k steps at ~15 it/s, under 20 minutes of step time (slurm log 172696).

The code changes are contained but touch several files:

- **`sang/motion.py`:** `T_DIM`, `REGIONS` and `to_target`/`from_target` generalise to a target spec. The region-balanced loss gains a "global" region (scale, translation) and a "shape" region (the 8 keypoints), so they get a loss share of their own instead of diluting the mouth's.
- **`sang/motion_model.py`:** `REF_DIM = 63 + T_DIM`, and the input and output projections follow `T_DIM`.
- **`scripts/train_motion.py`:** refit the normaliser into a new file (`norm42.pt` becomes `norm{T}.pt`), and add the new regions to the std/velocity ratios.
- **`sang/naturalness.py`:** the lip and eye readouts are linear maps on the 42-d mouth and eye columns. Their column indices shift, so rerun `scripts/calibrate_openness.py` (~10 min), and the constraint vectors follow through `Norm`.

Keep the 13 driven keypoints in the head frame, which keeps mouth shape independent of pose. Keep the 8 added keypoints in the camera frame, as the extractor produces them. Scale and translation should be predicted relative to the reference frame's values, so that the photo's own framing anchors the clip. KDTalker and Xemo-Talker, which generate translation, smooth it with a Kalman filter; SANG may need the same (inference).

The risk is that the model now has to learn background-compensating keypoint motion from audio and a reference. That motion is mostly a function of pose, so it is learnable in principle, but untested. **Gate:**

- Render the new model's demos: background ratio ≤ 1.2.
- `audio_gain` no lower than 1.50, and re-extracted mouth correlation no lower than today's.
- No new jitter from `scripts/video_jitter.py`.

**(c) Background lock in the warp field.** LivePortrait already has a "no motion" option: slot 0 of its 22-way softmax. Forcing it outside a person mask makes the background provably still at the flow level. The idea is untested in any paper or fork the research found. MRAA sets its background motion to zero at test time, and Real3D-Portrait blends head, torso and background features with masks (plain concatenation instead: FID 46.38 vs 42.37) ([MRAA](https://arxiv.org/abs/2104.11280), [Real3D-Portrait](https://arxiv.org/abs/2401.08503)).

The implementation is a wrapper, not an edit to `third_party`:

1. Run LivePortrait's dense-motion network as usual.
2. Blend its `deformation` with the slot-0 grid under a mask M: deformation′ = M·deformation + (1 − M)·grid₀, broadcast over the 16 depth slices.
3. Clamp the occlusion map to its unwarped value outside M.
4. Reuse LivePortrait's own slot-0 grid rather than an exact identity. The grid is built with the align-corners=True formula but sampled with `align_corners=False`, so the network's "identity" is a ~1.6% scale about the centre.

M must cover the head both where it is in the source and where it is now, dilated and feathered by 8–16 px. Otherwise, when the head moves away, identity flow shows the source photo's hair as a ghost. The source part comes from a matte of the photo; the current part comes from the projected driving keypoints. The softmax weights are computed inside `DenseMotionNetwork` but not returned, so a per-keypoint diagnosis needs a hook there.

The risks are:
- a seam at the neck, where the head keeps moving but the torso no longer follows;
- a halo in hair at the mask edge;
- a rigid-looking torso.

**Gate:** background ratio ≤ 1.2 with unchanged mouth correlation and CSIM, and a visual check on the largest-pose clips.

**(d) Composite onto an inpainted static plate.** This is the only option that makes the background still *by construction*. LivePortrait's paste-back is not it: its `mask_template.png` is 1.0 over **88.3%** of the 512×512 crop, with only a 16-px feathered border, so it pastes the warped background straight back ([LivePortrait code](https://github.com/KwaiVGI/LivePortrait)). The permissively licensed pipeline:

1. **Matte the source photo once.** BiRefNet (MIT) for the person, with ViTMatte-B (MIT; Composition-1k SAD 20.33) refining the hair band.
2. **Build the plate.** Dilate the person region by the expected head movement and fill it with Big-LaMa (Apache-2.0).
3. **Get each frame's alpha** from the source alpha carried by the renderer's own deformation. Let a video matting model decide only inside a thin edge band. MODNet and SAM 2.1 are Apache-2.0. MatAnyone is the most temporally stable (dtSSD **1.18 vs 1.36** for RVM) but non-commercial, and RVM is GPL-3.0 ([MatAnyone](https://arxiv.org/abs/2501.14677)).
4. **Estimate foreground colour in the band** with PyMatting (MIT), so moving background colour does not halo into hair.
5. **Composite in the original photo's coordinates.** This also removes the crop's padding bars.

The closest measured support is Real3D-Portrait's ablation: without background inpainting, FID rises from **42.37 to 43.95** and CSIM falls from **0.758 to 0.744**, with only a 1-nearest-neighbour fill. **Gate:** background ratio ≈ 1 outside the alpha, low flicker in the edge band, and unchanged face metrics.

**(e) Fine-tune the warping module.** A masked static-background L1 between output and source, plus an identity-flow penalty outside the person, with the SPADE decoder frozen, targets the problem directly. The evidence is thin:
- No paper has fine-tuned LivePortrait's warping for background stability.
- The official training code is unreleased; the maintainers call it "challenging", and the community re-implementation is unfinished.
- Explicit background models barely moved whole-frame reconstruction elsewhere. TPSMM's background loss leaves Taichi L1 at **0.046**, and MRAA's background model moves L1 from **0.059 to 0.048** together with a PCA change.

This rung is for residual artifacts only.

**(f) Switch renderer.** Not now:
- IMTalker beats LivePortrait on whole-frame HDTF PSNR (**28.458 vs 27.173**), and LIA-X beats it on VoxCelebHQ (**24.39 vs 17.45**), but neither targets the background.
- Diffusion talking heads flicker as much or more. On HDTF, BG-Flicker is **3.44x** for Hallo2 and **5.33x** for EchoMimic ([FluentAvatar](https://arxiv.org/abs/2509.12052)).
- Every alternative uses a different motion space, so the DiT would need new motion extraction and retraining.

## Measure the background properly

SANG's current metric is a sound jitter measure with four known gaps:
- Its fixed head box leaves the torso and the lower border counted as "background".
- Its ratio explodes on still studio backdrops (22.6x before the 0.05 floor).
- It has no pixel units.
- Being frame-to-frame, it misses slow drift.

Four changes close them, and all fit in `sang.bench.background_motion` and `scripts/bg_warp.py`:

1. **Person mask instead of the head box.** Segment the *real* clip (BiRefNet, or DeepLabV3 as FluentAvatar does), dilate by 15 px, and report background, torso and border bands separately.
2. **Keep the jitter ratio,** which is the same family as FluentAvatar's BG-Flicker ratio. That makes SANG comparable with the one published table:

| Method | BG-Flicker ratio, CMLR | BG-Flicker ratio, HDTF |
|---|---|---|
| SadTalker | 0.21x | 1.33x |
| Sonic | 0.63x | 0.89x |
| EchoMimic | 2.53x | 5.33x |
| Hallo2 | 3.47x | 3.44x |

   LivePortrait-style renderers are not in it. SANG's demos, at 2.35x on its own metric, sit among the diffusion models.
3. **Add drift.** Background PSNR of each frame against frame 0 inside the background mask, reported as the minimum over the clip, catches slow swim that frame differences miss.
4. **Add pixel units.** The 95th-percentile RAFT flow magnitude in the background, in pixels at 512.

The acceptance thresholds are proposals that need calibrating on real clips first:
- jitter ratio ≤ 1.2;
- background flow ≤ max(real, 0.3 px);
- minimum background PSNR ≥ 40 dB.

Every background fix must also pass the face-side checks:
- re-extracted mouth correlation and `audio_gain` no lower;
- no new start-up jitter;
- CSIM reported with its pose bias in mind: holding the head stiller raises it.

## A phased plan

| Phase | Work | What it decides | Cost | Gate to continue |
|---|---|---|---|---|
| 0 (queued) | Decomposition (70; 42 head; 42 camera; 42 camera + real s, t; 42 camera + real held; all `--no-stitch`) and the generated demo with `--held camera` | Whether (a) is enough, and which components (b) must predict | ~3–4 h wall-clock on the cluster | Numbers in hand |
| 0′ | Metric upgrade: person mask, bands, drift, RAFT flow | Trustworthy gates for every later phase | ½ day | Real clips calibrated |
| 1 | If (a) is not enough: widen the target per Phase 0, retrain, rerun `calibrate_openness.py` and `naturalness.py` | Whether the cause is fixed at its source | ~1 day engineering; one run; evals | Demo background ratio ≤ 1.2; `audio_gain` ≥ 1.50; mouth correlation not lower |
| 2 | Residual motion: background lock (c) first, compositing (d) if the lock leaves seams or halos, or when a guaranteed static plate is wanted | A still background by construction | 1–3 days | Background ratio ≈ 1; no visible seams on large-pose clips |
| 3 | Only if phases 1–2 leave artifacts: warping-module fine-tune (e) | Renderer-level repair | 3–5 days + 1–3 GPU-days | As phase 2 |

Phase 0 is cheap and decisive. If "42 camera" already lands near the 70-d row, SANG keeps its 42-d model, and phases 1–3 become optional polish. If only the "+ real" rows close the gap, Phase 1 is the fix. Its training cost is small, because the full motion is already cached.

## Conclusion

The background wobble is a measurement lesson more than a rendering one. SANG simplified the renderer's input from 70 numbers to 42, and planned to check that simplification with a 0.5 dB gate. The gate ran only with stitching on. Stitching pulls every render toward the source and flattens both configurations to about 18 dB, so the true 7.5 dB gap stayed invisible until the safeguards were switched off. The general rule is to check every simplification of a frozen component's input against that component with its own safeguards off, before building on it. That applies again to anything SANG changes next: the target's frames, smoothing, or a new motion space.

The decision rule follows from the ladder. Fix the cause before hiding the symptom, and measure every fix on both sides: the background metric and the face metrics. The render-time convention fix is already in place and costs nothing. Whether SANG must also predict scale, translation or the shape keypoints is a question Phase 0 answers in a few hours. The masking and compositing rungs then remain what they should be: guarantees for the residue, not substitutes for giving the renderer the motion it was trained on.
