# Composite the occluder, calibrate the guidance, then measure lip sync properly

With stitching on, SANG's background now moves **0.56×** as much as the real video. Two defects remain, and they have different causes.

**Occluders.** Objects in front of the face in the source photo, such as a podcast microphone, are encoded into LivePortrait's appearance volume and warped with the jaw and cheek keypoints, so they follow the head. No warping-based portrait animator handles this. Every working fix elsewhere is compositing: segment the occluder once, inpaint it out of the source *before* rendering, render the clean face, and paste the original occluder back on top.

**Lip sync.** The generated mouth opens **1.52×** as widely as real speech, clamps harder at /p b m/ (minimum openness 0.005 vs 0.039), and correlates with the real lip trajectory at only **0.65**. The literature points to audio classifier-free guidance at γ = 2 as the leading cause of the amplitude error. It gives no motion-space study of the correlation deficit, which the audio front end most plausibly limits.

**Measurement.** SANG has never obtained a SyncNet score, and the cause is now identified. Since April 2026, `syncnet_python` writes its results to stderr, while SANG's wrapper reads stdout.

Order of work:

1. Fix the SyncNet harness and add amplitude-aware metrics.
2. Sweep and restructure guidance (inference only).
3. Build the occluder layer and neck blend (inference only).
4. Change the audio front end (one retrain) to raise correlation.
5. Only then consider sync losses, DPO or a mouth refiner.

"Measured" marks numbers from SANG's runs or from a paper's table; "inference" marks reasoning not yet tested on SANG.

## Where SANG stands

| Quantity (unseen TalkVid speakers) | Value | Source |
|---|---|---|
| Background motion, generated demos, stitching on | **0.56×** real (was 2.41× without) | `scripts/bg_warp.py`, job 173682 |
| Real motion through the 42-d target, stitching on | background 0.73×, mouth corr 0.824, CSIM 0.911 | M0, job 173683 |
| Lip-opening std, generated / real | **1.52** | `naturalness.py`, job 173057, 300 clips, 42 speakers |
| Correlation, generated vs real lip opening | **0.65** | same |
| Lip concordance (CCC, correlation and amplitude together) | **0.597** | computed from the two rows above |
| Minimum lip openness at /p b m/ | **0.005** vs real 0.039 | same |
| Blink rate | 31.8 vs 31.5 per min | same (no blink fix needed) |
| Training-time mouth std / velocity ratio | 1.06 / 0.975 (normalised 42-d coordinates) | job 172696 |
| Renderer ceiling on real motion, re-extracted mouth corr | 0.82–0.85 | M0 |

A correction to one research note: SANG's training-time evaluation also samples with γ = 2.0 (`scripts/train_motion.py`, `evaluate`). So the 1.06 vs 1.52 difference is *not* guidance on versus off. It compares the per-coordinate spread of all 18 mouth coordinates with the spread along the lip-opening direction specifically. Guidance may still inflate the opening direction the most, but only the running γ sweep can show it.

A second point from the concordance arithmetic: fixing amplitude alone would raise CCC only from about 0.60 to about 0.65 (inference). The larger deficit is correlation, 0.65 against a renderer ceiling of 0.82–0.85. Guidance cannot fix that; the audio representation can.

## Occluders: inpaint out, render, composite back

**Why the microphone follows the head.** LivePortrait builds its appearance volume f_s from the whole 256-px source crop. A microphone overlapping the lower face becomes part of that volume, and the dense-motion softmax assigns its voxels to the nearest keypoints, the jaw and cheek. When the head turns, those keypoints move, and the microphone moves and smears with them. Stitching cannot help: it pins the shoulder region, not objects inside the face.

No keypoint-warping portrait animator handles foreground occluders. A search of LivePortrait's issues for "microphone", "glasses", "hand in front" and "occluded" returns nothing, and in these renderers "occlusion" means regions the source photo does not show, not objects in front of the face. The maintainers acknowledge related limits: hats and hair go "half moving and half still" (#122), and "non-diffusion based methods are hard to simulate the hair swaying" (#29). Full-frame diffusion is no escape either: Hallo3 lists a held "microphone" among its limitations ([Hallo3](https://arxiv.org/abs/2412.00733)).

**Every working fix elsewhere composites.**

| System (task) | Occluder handling | Measured result | Licence |
|---|---|---|---|
| FaceFusion (face swap) | DeepFaceLab XSeg masks plus a face parser; occluder kept from the target | none published | XSeg weights GPL-3.0; FaceFusion OpenRAIL-AS |
| KeySync (lip sync, [2505.00497](https://arxiv.org/abs/2505.00497)) | SAM 2 per frame; occluder removed from the edit region | none given | — |
| EdiDub (lip sync, [2505.23406](https://arxiv.org/abs/2505.23406)) | edits instead of inpainting, keeping occluders | occluded-lip set: identity error **0.033 vs 0.052**, sync opinion score **8.50 vs 8.37**, both vs LatentSync | — |
| FaceMat ([2508.03055](https://arxiv.org/abs/2508.03055), ACM MM '25) | mattes the occluder, completes the face, transforms it, composites the occluder back | real occlusions: matte IoU **0.7121 vs 0.4099** for RVM | CC BY-NC-ND 4.0: design reference only |

FaceMat is the closest analogue, and its order of operations matters for SANG (inference). If the microphone is only pasted back on top without first inpainting it out of the source, the warped copy inside f_s still moves underneath. The result is a ghost band as wide as the head's motion, visible around the pasted microphone. The occluder therefore has to leave the source before rendering.

**Design for SANG** (untested; all components permissively licensed):

1. **Segment the occluder once, on the source photo.**
   - Use SAM 2 or Grounded-SAM-2 (Apache-2.0) with a text prompt such as "microphone" or "hand", plus a click fallback.
   - Refine the edge into a soft alpha with ViTMatte (MIT).
   - Exclude head-attached items (glasses, earrings, headsets), which should move with the head.
2. **Inpaint the dilated occluder mask in the source with LaMa** (Apache-2.0; a CelebA-HQ checkpoint exists). The renderer then animates a face that is complete behind the microphone.
3. **Render from the cleaned source** with stitching on, exactly as today.
4. **Composite the original occluder pixels on top of every frame** with the soft alpha, in source-crop coordinates, or in the original photo's coordinates if the full frame is output.

In code, this is a new `sang/composite.py`, called from `MotionCodec.source_state` (the cleaned crop) and after `MotionCodec.render` (the paste), behind an `--occluders "microphone"` flag in `scripts/infer_motion.py` and `scripts/demo_motion.py`.

The risks are specific:
- LaMa's face completion may shift identity. This needs a CSIM A/B against the uncleaned source.
- An occluder that covers the lips hides lip errors. Lip metrics must be computed on the render *before* compositing.
- The design assumes the occluder is static. A hand that moves in the real video is out of scope.

**Gate:**
- Occluder integrity, measured as the difference from the source inside the occluder mask: near zero.
- Ghost-band flicker around the occluder: no higher than the real video's.
- Mouth correlation 0.824, CSIM 0.911 and background 0.56× unchanged.

## Neck and hair seams under pinned shoulders

Stitching was trained only with an L1 loss on the shoulder region. It has no numeric ablation, and how its shoulder mask was built is undocumented (#204, unanswered). Pinning the shoulders while the head rotates shears the neck. Without stitching, a user found "the head is detached from the body"; the maintainer's reply was that stitching "opposites pasting-back" (#157).

Published fixes are small, whole-frame, and have no seam-specific metric:

| Method | Fix | Measured result |
|---|---|---|
| Real3D-Portrait ([2401.08503](https://arxiv.org/abs/2401.08503), MIT) | alpha-blend head over torso over background | FID **42.37** vs 46.38 with plain concatenation |
| SyncTalk | fill the neck gap with the mean neck colour; merge into the original frame | PSNR 37.40 vs 35.35 (SyncTalk++ Table II) |
| SyncTalk++ | a small per-subject network inpaints the head–torso junction | BRISQUE 25.41 vs 33.29 (no-reference only) |
| M2DAO-Talker | a loss over the whole portrait so face and torso move together | PSNR 34.475 vs 34.123 |

**Design** (untested; the numbers are starting guesses, not sourced values):

- A three-layer composite:
  - an inpainted static background plate;
  - the source torso, held still;
  - the rendered head, hair and upper neck.
- The head alpha is the source head mask carried through the renderer's own deformation, ramped over 24–40 px across the neck.
- This reuses the compositing pipeline proposed in `reports/SANG background warping fix.md`: BiRefNet and ViTMatte for the matte, LaMa for the plate, PyMatting for foreground colour at hair edges.
- **Fallback, if neck shear remains:** scale yaw and pitch amplitude by about 0.8. Ditto exposes such gains but never evaluated them.
- **Heavier options, only if artifacts persist:** a junction-inpainting network in the style of SyncTalk++, or the slot-0 background lock applied to the torso.

**Gate:** neck-band and hair-band flicker, each as a ratio to the real video, no higher than 1.2, with the face metrics unchanged.

## Lip amplitude: guidance first

**The evidence that guidance inflates motion** is consistent but indirect. No paper plots lip amplitude against guidance for a keypoint or motion-space model.

| Evidence | Measured | Model type |
|---|---|---|
| DiffPoseTalk Tab. 1 | guidance lowers lip-vertex error (9.58 → 8.94 mm) but **worsens mouth-opening difference vs GT (1.56 → 1.62 mm)** | 3DMM motion |
| LeapTalk Tab. 7 | audio guidance 1 → 7: head-pose std **1.655 → 6.323**; beat alignment 0.723 → 0.650 | pixel (Wan2.1 latent video) |
| FantasyTalking2 | SyncNet confidence rewards exaggerated lips; agrees with human lip-sync preference only **72.34%** of the time | pixel |
| DreamTalk | a generic lip expert on 3DMM mouth vertices raises sync confidence 2.63 → 4.51 but **worsens mouth-landmark error 3.07 → 3.42** | 3DMM motion |

The mechanism is that guidance scales the audio-dependent part of the velocity by roughly γ. The direction audio drives most, lip opening, is inflated most, and pushing past closure explains the 0.005 minimum openness at /p b m/ (inference).

**Inference-only fixes,** in order of cost:

1. **Sweep γ ∈ {1.0, 1.5, 2.0, 2.5}.** Running now: `naturalness.py --modes none --cfg 1.0 1.5 2.0 2.5`. Pick the value by lip CCC while holding `audio_gain`, rather than by amplitude alone.
2. **Per-region guidance.** Use γ ≈ 1.25–1.5 on the 18 mouth coordinates and 1.0 on eyes and brows, by masking the guidance residual by region. AVTR-1, on the same 42-d LivePortrait target, uses per-region guidance, but its weights are unpublished and nothing is ablated.
3. **CFG-rescale** ([2305.08891](https://arxiv.org/abs/2305.08891), blend 0.7) matches the guided prediction's std to the conditional one. **APG** ([2410.02416](https://arxiv.org/abs/2410.02416)) removes the component of the guidance update parallel to the conditional prediction. **Guidance interval** ([2404.07724](https://arxiv.org/abs/2404.07724); ImageNet-512 FID 1.81 → 1.40) guides only at mid noise. None has been tested in motion space.
4. **Noise truncation (±1.2, as in AVTR-1) plus a clamp of lip coordinates to the training range.**

**Correlation needs a retrain.** Guidance cannot raise the 0.65 correlation. Two retrains could:
- **A learned mix of WavLM layers.** Probing speech models against measured tongue and lip movement finds accuracy peaks mid-network and "a few layers before the last" (arXiv 2210.11723; the title was not verified). No talking-head paper ablates WavLM layers.
- **Whisper.** It is the only encoder with a measured win on LivePortrait keypoints: Teller Sync-C 7.696 against 4.286 for a codec encoder.

A wider audio window has weak support. The region-balanced and velocity losses are *contradicted* as causes, because training-time mouth std (1.06) and velocity (0.975) ratios are near 1.

**Fine-tunes, only after the above:**
- **A keypoint sync expert,** first as an evaluator and then as a style-aware loss with an amplitude guard. Supporting numbers:
  - UniSync on 3DMM geometry: 91.40% accuracy vs 93.25% on RGB.
  - THUNDER's mesh-to-speech loss: lip PCC 0.568 → 0.639, CCC 0.359 → 0.426.
- **DPO with ground-truth winners against over-guided losers.** Avatar Forcing's DPO with synthetic losers *raised* motion variance (1.408 → 1.734), the wrong direction for SANG unless the losers are the over-guided samples.

## The mouth on the rendering side

LivePortrait cannot repair the mouth.
- Its lip-retargeting module is a 4-layer MLP trained only to reach a landmark lip-open ratio while leaving non-lip pixels unchanged. Nothing in its training targets teeth or the mouth interior.
- `flag_normalize_lip` (threshold 0.03) runs upstream only in relative mode, and `flag_lip_retargeting` is marked "not recommend, WIP".
- **One SANG-specific fact:** `MotionCodec.source_motion` normalises the lips only in the motion used as the DiT's reference and as `from_target`'s base. The renderer's source keypoints x_s, built by `source_state`, keep the photo's real mouth state. So an open-mouthed photo is still what gets warped (verified in `sang/motion.py`).
- Teeth issues (#176, #160, #353, #565, #375, #82) have no maintainer fixes: a fixed "plate" of upper teeth, teeth on any jaw opening, and tongue ghosts under closed lips.
- None of Ditto, KDTalker, JoyVASA, Playmate, AVTR-1 or Xemo-Talker uses a mouth refiner or reports a mouth-quality metric.

**Post-hoc refiners regenerate the mouth from audio, discarding SANG's motion,** and are a last resort:

| Refiner | Sync (HDTF) | Speed | Licence |
|---|---|---|---|
| LatentSync 1.6 | 8.9 in its own paper; 7.90 in MuseTalk's rerun; 4.99 in ComplexSync's; ground truth 7.73 | ~2–6 FPS, ≥ 18 GB | Apache-2.0 |
| MuseTalk | LSE-C 6.53 | 30 FPS at 256 px (V100) | custom (not read) |

RGOR (arXiv 2609.38019) finds these refiners render an "average mouth", hidden in their released code by using the edited frame as its own reference. Face restoration hurts rather than helps: after Wav2Lip, CodeFormer raised FID from 10.85 to 26.36 on one dataset and lowered LSE-C from 8.85 to 7.94 on another.

**The source photo's mouth state is untested anywhere.** An A/B is cheap and worth running:
- Photos: closed mouth, slightly open with incisors showing, and an open smile.
- Each driven by real and by generated motion.
- With and without normalising the renderer's x_s too.

## Measuring lip sync properly

**The nan cause.** joonson/syncnet_python merged PR #78 on 2026-04-17, replacing `print` with `logging`. "AV offset", "Min dist" and "Confidence" now go to stderr. `sang/bench.py::lse` parses only stdout (verified), so it returns nan even when face detection and SyncNet both work.

Other silent failure modes:
- `run_pipeline.py` refuses an existing output directory without `--overwrite`.
- `ffmpeg` must be on PATH.
- A face track needs more than 100 frames (over 4 s at 25 fps).
- Faces must be at least 100 px; S3FD runs at 0.25 scale with confidence 0.9.

The fix is either to parse both streams, or to switch to LatentSync's `eval/eval_sync_conf.py` (Apache-2.0, same `syncnet_v2` weights, `min_track=50`, raises on no face). The sanity check is the README clip: offset 3, min dist 5.353, confidence 10.021.

**How to read LSE once it works.** Always report three rows through one pipeline: real video, real motion re-rendered (the renderer ceiling), and SANG. There are four reasons:
- The same method on the same dataset varies widely across papers (LatentSync HDTF: 8.9, 7.90, 4.99).
- THEval's 3,519 human ratings correlate with LSE-C at only ρ = −0.164.
- Changing only the audio codec shifts LSE by 0.4 on average (up to 1.2).
- Generated video often "beats" ground truth. On RGOR's benchmark, ground truth scores 4.02, LatentSync-1.6 5.12 and HeyGen 5.37. One system there opens the mouth 1.42× more than ground truth and still scores LSE-C 3.90, close to ground truth.

So SANG's 1.52× over-articulation may not lower LSE-C at all, and could raise it. LSE above the real-video row is a warning, not a win.

**Add amplitude-aware metrics beside LSE:**
- lip-opening std ratio (target 1.0), correlation and CCC with real;
- /p b m/ closure depth;
- mouth-opening difference vs ground truth, as DiffPoseTalk reports it;
- the AV-offset histogram;
- silence stability, a LipLeak-style test (lips should rest during silence);
- mouth-region image quality (MUSIQ).

A lip-reading score (LipScore or AV-HuBERT) is a secondary check.

## Plan

| Tier | Work | Retrain? | Cost | Gate |
|---|---|---|---|---|
| 0 | Parse SyncNet's stderr (or adopt LatentSync's evaluator); check the README clip; add CCC, closure depth, opening difference and AV offset to evaluation | No | hours | README clip reproduces 3 / 5.353 / 10.021 |
| 0 | γ sweep (running); then per-region guidance and CFG-rescale | No | hours | lip std ratio 0.9–1.1, CCC up, `audio_gain` ≥ 1.50 |
| 1 | Occluder layer (SAM 2 → ViTMatte → LaMa → render → paste back) | No | 2–3 days | occluder integrity; ghost-band flicker ≤ real; CSIM, mouth corr unchanged |
| 1 | Three-layer composite with neck ramp; pose gain 0.8 fallback | No | 1–2 days | neck- and hair-band flicker ≤ 1.2× real |
| 1 | Source-mouth-state A/B, including normalising x_s | No | hours | better openness metrics without CSIM loss |
| 2 | Audio front end: learned WavLM layer mix, then Whisper | One run each (<20 min step time) | ~1 day | lip correlation > 0.65 toward the 0.82–0.85 ceiling |
| 3 | Keypoint sync expert (evaluator, then loss); DPO against over-guided losers | Fine-tune | days | CCC up, no amplitude inflation, human A/B |
| 4 | Mouth refiner (LatentSync, photo as reference) | No | days | openness metrics and blind A/B improve with no CSIM or FID loss |

## Conclusion

Both remaining defects come from SANG giving a component something it was never built for. The renderer receives a microphone as part of the face, so it animates it. The sampler receives a guidance scale tuned for pixel generators, so it amplifies the direction audio drives hardest, lip opening. Neither needs a new model: the occluder is removed from the renderer's input and put back afterwards, and the guidance is restructured per region. What does need a retrain is the correlation deficit, and that points at the audio front end, not the objective.

The measurement lesson repeats from the earlier reports. SANG has never had a lip-sync number because the harness silently read the wrong stream, and the fix is one line. With LSE working, it must be read beside the real-video row and an amplitude check, because the metric rewards exactly the over-articulation SANG currently has.
