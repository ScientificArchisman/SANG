# Occluders and head/neck/torso seams in one-shot warping talking heads (SANG-M: LivePortrait renderer, stitching ON)

Tags: **[M]** measured number in the cited source; **[C]** claim or design statement by the source's authors (or code behaviour read from the repo); **[I]** my inference. Researched 2026-09-30. All arXiv IDs below were opened on arxiv.org during this session (2407.03168, 2401.08503, 2505.00497, 1912.13457, 2311.17590, 2411.19509, 2508.03055, 2412.00733, 2505.23406, 2506.14742, 2507.08307, 2309.05095); the others (2201.08425, 2409.03605, 2602.00639, 2403.15931) were confirmed only as search results. Licences were read from each repository's LICENSE file on 2026-09-30.
Scope note: this builds on `reports/SANG background warping fix.md`, which already covers paste-back (88.3% mask), the untested slot-0 "background lock", and plate compositing (BiRefNet + ViTMatte + Big-LaMa + PyMatting). None of that is re-researched here.

## Q1. Which portrait-animation / reenactment / face-swap / lip-sync works handle foreground occluders (mics, hands, glasses, hair strands), and how?

### Takeaway
No keypoint-warping portrait animator handles foreground occluders. That includes LivePortrait, Ditto and the LivePortrait-based audio drivers. Upstream LivePortrait has no issue or fix for microphones, hands or glasses. The "occlusion map" in these renderers models disocclusion (source regions not visible in the driving pose), not objects in front of the face. Every working occluder solution found lives in the neighbouring fields: face swapping, lip-sync and face filters. They all use the same recipe: segment the occluder, run the face model on the face only, and composite the occluder's original pixels back on top. The closest published match to SANG's need is FaceMat (ACM MM '25), with four stages: matte, complete the face, transform, re-composite. Its code is non-commercial. Full-frame diffusion models (Hallo3) explicitly list a held microphone as an unsolved limitation.

### Cited Findings
- **No upstream discussion in LivePortrait [M/negative].** A GitHub issue search of KlingAIResearch/LivePortrait for "microphone", "glasses", "hand in front" and "occluded" returned zero issues on 2026-09-30. "occlusion" matched only an unrelated face-tracking issue (#562). — [LivePortrait issues](https://github.com/KlingAIResearch/LivePortrait/issues?q=microphone)
- **Accessories that should move with the head [C].** Hats, headwear and hair "will be distorted, with half moving and half still". The maintainer replied: "Yes, it's a known issue! One of the possible reasons is that the training data does not include, or has very few, adornments such as hats." — [#122](https://github.com/KlingAIResearch/LivePortrait/issues/122). The same symptom (a helmet not moving with the face) prompted an open user request for mask support, which has no maintainer reply. — [#46](https://github.com/KlingAIResearch/LivePortrait/issues/46)
- **LivePortrait's own limitations [C].** The paper's limitations mention only large-pose cross-reenactment and "significant shoulder movements ... jitter". Occluders are not mentioned. Training data: 69M frames from ~18.9K identities, plus 60K stylised portraits. — [LivePortrait, arXiv 2407.03168](https://arxiv.org/html/2407.03168v1)
- **"Occlusion" in reenactment renderers means disocclusion [C].** MaskRenderer (arXiv 2309.05095) is an "occlusion-robust" 3DMM-based reenactor. Its "multi-scale occlusion" is described as "improving inpainting and restoring missing areas", i.e. filling regions the source does not show, not preserving foreground objects. — [MaskRenderer](https://arxiv.org/abs/2309.05095)
- **Face swapping, per-frame occluder mask [C, code].** FaceFusion composites the swapped face through several masks: a box mask, a landmark "area" mask, a BiSeNet-ResNet18 face-parser "region" mask (tagged MIT) and an "occlusion" mask from DeepFaceLab XSeg models (xseg_1/2/3 at 256², tagged GPL-3.0). In "many" mode it takes the element-wise minimum of the three XSeg masks, then applies GaussianBlur σ=5 and remaps via clip to [0.5, 1], ×2 − 1. That erodes and hardens the mask, so a thin band next to the occluder always keeps the target frame's own pixels. FaceFusion's code licence is OpenRAIL-AS. — [face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py); [LICENSE.md](https://github.com/facefusion/facefusion/blob/master/LICENSE.md)
- **Face swapping, learned refinement [C].** FaceShifter (arXiv 1912.13457, CVPR 2020 oral) adds a second-stage HEAR-Net "to address the challenging facial occlusions". It is "trained to recover anomaly regions in a self-supervised way without any manual annotations". No official code. — [FaceShifter](https://arxiv.org/abs/1912.13457)
- **Lip-sync, occluders excluded from the edit region [C].** KeySync (arXiv 2505.00497) runs SAM 2 per frame to get an occluder mask M_obj, then shrinks the inpainting mask to M = M ∩ ¬M_obj, so occluder pixels come from the input video. Fig. 6 shows smaller MAE spikes in occluded frames, but no numeric occlusion benchmark is given. The authors say the trick works "out of the box" for some baselines but not for fixed-mask LatentSync. Licence not stated. — [KeySync](https://arxiv.org/html/2505.00497)
- **Lip-sync by editing instead of inpainting [M].** EdiDub (arXiv 2505.23406) argues that mask inpainting "discard[s] valuable information ... such as occlusions". Results on its Vox2-Occluded set, EdiDub vs LatentSync:

  | Metric | EdiDub | LatentSync |
  |---|---|---|
  | ID-P | 0.033 | 0.052 |
  | LSE-D (see note) | 0.523 | 0.546 |
  | MOS-Sync | 8.50±0.33 | 8.37 |
  | MOS-Nat | 9.42±0.16 | 9.14 |

  The LSE-D values are on a scale unlike the usual ~6–9, so they are presumably normalised. — [EdiDub](https://arxiv.org/html/2505.23406)
- **Face transformation with occluders: FaceMat [C].** FaceMat (arXiv 2508.03055, ACM MM '25) pipeline, quoted: "(1) Occlusion matting ... (2) Face completion, where an inpainting module can optionally reconstruct occluded facial areas to obtain a clean face; (3) Face transformation ... (4) Compositing, where the transformed face is blended with the original occlusion using the predicted alpha matte". It targets "hands, hair, transparent objects, and even semitransparent elements such as smoke or fire". — [FaceMat](https://arxiv.org/html/2508.03055)
- **FaceMat numbers and licence [M].**
  - Real occlusions (RealOcc), FaceMat vs RVM: IoU 0.7121 vs 0.4099; accuracy 0.9197 vs 0.8432; recall 0.9084 vs 0.4876.
  - Synthetic CelebAMat test set (716 images): MSE 0.0182, SAD 8.43, IoU 0.8408.
  - Licence: **CC BY-NC-ND 4.0**. — [FaceMat](https://arxiv.org/html/2508.03055)
- **Full-frame diffusion names the microphone as unsolved [C/M].** Hallo3 (arXiv 2412.00733) limitation: "accounting for significant accessories, such as holding a smartphone, microphone, or wearing closely fitted objects, presents challenges in generating realistic motion for the associated objects". It reports Subject Dynamic 13.286 and Background Dynamic 4.481, where higher means more motion. — [Hallo3](https://arxiv.org/html/2412.00733)
- **Other diffusion portrait methods [C, search-snippet level].** Diff-PC (arXiv 2602.00639) reports "incorrect facial occlusions ... for items such as masks and fans". X-Portrait (arXiv 2403.15931) notes corrupted output when face detection fails "due to face occlusion". — [Diff-PC](https://arxiv.org/pdf/2602.00639), [X-Portrait](https://arxiv.org/html/2403.15931v4)
- **Ditto's paste-back is not an occluder mask [C, code].** Ditto (LivePortrait-style renderer, Apache-2.0) builds its paste-back mask with `get_mask(512,512,0.9,0.9)`: 1.0 over the central 90%×90% with linear ramps over the outer ~26 px. Like LivePortrait's template, it is a crop-border feather with no person or occluder awareness. — [putback.py](https://github.com/antgroup/ditto-talkinghead/blob/main/core/atomic_components/putback.py), [get_mask.py](https://github.com/antgroup/ditto-talkinghead/blob/main/core/utils/get_mask.py)

### Inferences
- **[I] Mechanism of SANG's mic ghosting.** The mic pixels are part of the source crop, so they are encoded into f_s. Voxels covering the lower-right face and the mic knob are assigned by the per-voxel softmax to the nearby jaw and cheek keypoints (slots 1–21), not to slot 0. They therefore translate with the head, and the face texture behind them in the driving pose is whatever the warp brings in. The 64×64 occlusion map cannot help: it was trained to flag disocclusion, never "this object is in front". This is why bleeding and edge smear show up exactly where the mic overlaps the face.
- **[I] Every working occluder solution is a compositing solution.** Face swap, lip-sync and FaceMat all keep the occluder pixels from a reference and run the face model underneath. None changes the warp itself. In face swap and lip-sync the reference is the per-frame target video. SANG has no target video, so its reference must be the source photo: one occluder layer, segmented once and held static. That is physically right for a desk or boom mic, which stays put while the head moves.
- **[I] Head-attached vs world-attached occluders.** Glasses, headsets, earrings and hats should move with the head. They belong to the warped layer, and LivePortrait's quality on them depends on its training data (#122). Only world-attached occluders (desk mic, mic arm, pop filter, a resting hand, a table edge) go in the static top layer.

### Gaps
- I found no paper or fork that adds occluder handling to a LivePortrait-family (implicit-keypoint warping) renderer. The occluder-layer idea for this renderer is untested.
- EchoMimicV2, MegActor, Follow-Your-Emoji and AniTalker were not read in full. The searches surfaced no occluder-handling statement from them, but I cannot confirm their papers are silent.
- No source reports a numeric "mic ghosting" measurement for any talking-head method.

## Q2. How to segment occluders, fill the region behind them, and what the simplest robust SANG pipeline is (with failure modes)

### Takeaway
Segment the occluder once in the source photo with a permissively licensed promptable segmenter: SAM 2 with a click or box, or Grounded-SAM-2 with a text prompt such as "microphone". Refine the edge with a matting model. The critical design point is that the occluder must also be removed from the renderer's input, by inpainting the source crop behind it before f_s is extracted. Keeping the occluder on top alone is not enough: the warp would still carry mic texture outside the static mic's outline, a ghost as wide as the head's displacement. After rendering the clean head, composite the static occluder layer on top with its alpha. The revealed band that inpainting must fill is only as wide as the head moves, so inpainting quality matters little. Real3D-Portrait's "w/o inpaint" ablation costs only 1.58 FID.

### Cited Findings
- **Promptable segmenters, all permissively licensed [C].**
  - SAM 2 is Apache-2.0. — [SAM 2 LICENSE](https://github.com/facebookresearch/sam2/blob/main/LICENSE)
  - Grounded-SAM-2 is Apache-2.0 and supports "Ground and Segment Anything" and video tracking from "specific text prompts". Its detectors are Grounding DINO, Grounding DINO 1.5/1.6, Florence-2 and DINO-X. — [Grounded-SAM-2 README](https://github.com/IDEA-Research/Grounded-SAM-2), [LICENSE](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/LICENSE)
  - Grounding DINO is Apache-2.0. — [GroundingDINO LICENSE](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE)
  - Per the README's 2025-04-20 note, Grounding DINO 1.5/1.6 and DINO-X are reached through the `dds-cloudapi-sdk` cloud API. Locally runnable options are the original Grounding DINO or Florence-2.
- **KeySync runs SAM 2 per frame [C]** because its occluders move in the input video. — [KeySync](https://arxiv.org/html/2505.00497)
- **Face-specific occluder segmenters [C].**
  - DeepFaceLab XSeg models, tagged GPL-3.0 by FaceFusion. — [face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
  - FaceOcc (arXiv 2201.08425): manually labelled occlusion masks covering "sunglasses, spectacles, hands, masks, scarfs, and microphones", used to train a face-minus-occluder segmenter. — [FaceOcc](https://arxiv.org/abs/2201.08425), [code](https://github.com/face3d0725/FaceExtraction)
  - FaceMat's soft occlusion matte (CC BY-NC-ND 4.0). — [FaceMat](https://arxiv.org/html/2508.03055)
- **Face parsing to delimit head, hair and neck [C].**
  - SegFace code is MIT. — [SegFace LICENSE](https://github.com/Kartik-3004/SegFace/blob/main/LICENSE)
  - FaceFusion's BiSeNet parser is tagged MIT. — [face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
- **Filling behind the occluder [C].**
  - LaMa is Apache-2.0. Its README distributes Big-LaMa (Places2, "The best model") plus "All models (Places & CelebA-HQ)", so a face-domain LaMa checkpoint exists. — [LaMa README](https://github.com/advimman/lama), [LICENSE](https://github.com/advimman/lama/blob/main/LICENSE)
  - FaceMat makes face completion an optional stage between matting and transformation. — [FaceMat](https://arxiv.org/html/2508.03055)
- **Fill quality matters little [M].** In Real3D-Portrait, background inpainting with only a 1-nearest-neighbour fill moves FID 43.95 → 42.37 and CSIM 0.744 → 0.758 (w/o inpaint → full). — [Real3D-Portrait Table 4](https://arxiv.org/html/2401.08503)
- **Occluder over the lips [M].** On partially lip-occluded clips (Vox2-Occluded), the method that preserves occluder pixels (EdiDub) beats the one that repaints them (LatentSync). ID-P is 0.033 vs 0.052; MOS-Sync is similar at 8.50 vs 8.37, so hiding part of the lips behind a preserved occluder did not cost perceived sync there. — [EdiDub](https://arxiv.org/html/2505.23406)
- **Temporal consistency is open [C].** FaceMat: "ensuring temporal consistency in facial inpainting remains an open challenge". — [FaceMat](https://arxiv.org/html/2508.03055)

### Inferences
- **[I] Simplest robust pipeline for SANG**, all in the source's 512-px crop frame, done once per source photo:
  1. Get the occluder mask O. Click or prompt with SAM 2 or Grounded-SAM-2 ("microphone . microphone arm . pop filter . hand"). Keep only components that overlap the person or head region from BiRefNet or SegFace, minus the head-attached classes (glasses, headset, earrings, hat).
  2. Get a soft occluder alpha α_O by running ViTMatte (MIT, per the earlier report) on a trimap of O dilated/eroded by ~4–8 px. Mic grilles and cables are thin structures where a hard mask shows stair-steps.
  3. Build the clean source I_s′: inpaint O ⊕ dilation(k) with LaMa (CelebA-HQ or Big-LaMa checkpoint). Dilate so that no mic-edge colour remains. k of about 6–10 px at 512 is a guess to tune.
  4. Extract f_s and keypoints from I_s′ only.
  5. Render every frame from I_s′ with SANG motion and stitching as today.
  6. Composite: out = α_O·I_s + (1 − α_O)·render. The static mic is then pixel-identical to the source in every frame, and the head passes behind it.
- **[I] Why step 3 is necessary.** If the source still contains the mic, the warp carries mic texture wherever the lower-right face moves. The static layer covers the mic's original footprint only. Every frame with head displacement d then shows a ghost band of width ≈ d on the side the head moved toward. That band is today's smear, just narrower.
- **[I] Failure modes.**
  - **Occluder moves in the real video** (hand gestures, a speaker grabbing the mic). A static layer is then wrong by construction. Restrict the layer to rigid, world-anchored objects. SANG has no driving video, so a moving occluder cannot be reproduced anyway; best to remove it: inpaint it out and do not composite it back.
  - **Occluder touching the lips.** The static layer hides the covered part of the mouth, which is physically correct. SyncNet-style scores on the composite may drop because fewer mouth pixels are visible. Report lip metrics on the un-composited render as well as the composite.
  - **Wrong depth order.** If the head leans forward past the mic plane, the composite still draws the mic on top, which is still correct for a mic in front of the face.
  - **Mic shadow or reflection on the cheek** stays in the source and is carried by the warp. It is minor, but it is the one residual ghost this pipeline does not remove.
  - **Keypoint detection on the occluded source.** LivePortrait's motion extractor sees I_s′ rather than the occluded photo, which should if anything help. Untested.
  - **Inpainting fidelity near the mouth corner.** If the inpainted lip or cheek is wrong, it is visible only in the revealed band, so the error is bounded by head displacement.

### Gaps
- No measurement of how LivePortrait's appearance encoder and renderer behave on an inpainted source: CSIM, mouth correlation, or texture blur. It needs a direct A/B on SANG's real-motion benchmark (42-d + stitching: mouth correlation 0.824, CSIM 0.911).
- No per-checkpoint licence check of face-parser weights (BiSeNet/SegFace), which are typically trained on CelebAMask-HQ-style data released for non-commercial research. A permissive code licence does not settle the weight licence.
- Florence-2's licence was not verified from its model card in this session.

## Q3. Neck/torso seams when the head rotates while stitching pins the shoulders

### Takeaway
LivePortrait's stitching trades one artifact for another. With stitching off, "the head is detached from the body". With it on, the shoulders are forced to the source while the head moves, so the neck and hair outline absorb the difference. Stitching has only a shoulder-region L1 loss, no numeric ablation, and the maintainers offer no seam fix. The systems that do address the junction use one of three fixes:
- alpha-composite the head over a separately handled torso (Real3D-Portrait);
- fill the junction band, with average neck colour (SyncTalk) or a small inpainting U-Net (SyncTalk++);
- supervise the whole portrait so face and torso deform together (M2DAO-Talker).

All report only small whole-frame gains and no seam-localised metric. Limiting head-rotation amplitude is an exposed knob in Ditto's code, but it is unevaluated.

### Cited Findings
- **Stitching in LivePortrait [C].**
  - Loss: L_st = ‖(I_p,st − I_p,recon) ⊙ (1 − M^st(I_s))‖₁ + w_reg^st‖Δ_st‖₁, where "M^st is a mask operator that masks out the non-shoulder region" and Δ_st ∈ R^{K×3} offsets the driving keypoints.
  - Its ablation is qualitative only (Fig. 7: "the shoulder of the animated person is force aligned with the cropped source portrait").
  - Stated limitation: "significant shoulder movements ... jitter". — [LivePortrait §3.3, limitations](https://arxiv.org/html/2407.03168v1)
- **How the shoulder mask is made is undocumented [C].** Issue #204 ("How to detect the shoulder region") has no maintainer answer. — [#204](https://github.com/KlingAIResearch/LivePortrait/issues/204)
- **Stitching vs detachment [C].** A user reports "--no_flag_stitching, the head is detached from the body". The maintainer replies "stitching opposites pasting-back". Context: forward/back head motion makes the head "enlarge and shrink ... very strange when the human body remains unchanged". — [#157](https://github.com/KlingAIResearch/LivePortrait/issues/157)
- **Torso is out of scope [C].** Maintainer: "LivePortrait focus on head, I think you can combine body-driven and head-driven techs". — [#16](https://github.com/KlingAIResearch/LivePortrait/issues/16)
- **Real3D-Portrait, head/torso/background [C/M].** The torso is a separate 2D-warping renderer driven by 3DMM-vertex keypoints. Layers are fused with F = (F_head·M_head + F_torso·(1−M_head))·M_person + F_bg·(1−M_person), which the authors say avoids "hollow artifacts and blurry results in the boundary region". Table 4 ablation:

  | Variant | FID | CSIM |
  |---|---|---|
  | Full | 42.37 | 0.758 |
  | Concatenation instead of alpha blending | 46.38 | 0.737 |
  | Unsupervised torso keypoints | 44.86 | 0.746 |

  Code is MIT. — [Real3D-Portrait](https://arxiv.org/html/2401.08503), [LICENSE](https://github.com/yerfor/Real3DPortrait/blob/main/LICENSE)
- **SyncTalk [C/M].** At the head/torso junction "a dark gap area might appear ... We fill these areas with the average neck color". Its Portrait-Sync Generator merges the rendered face into the original frame "to enhance hair detail fidelity". — [SyncTalk](https://arxiv.org/html/2311.17590). Numbers from SyncTalk++ Table II, without → with Portrait-Sync:

  | Metric | Without | With |
  |---|---|---|
  | PSNR | 35.3542 | 37.4016 |
  | LPIPS | 0.0235 | 0.0113 |
  | FID | 3.9247 | 2.7070 |

  — [SyncTalk++](https://arxiv.org/html/2506.14742)
- **SyncTalk++ Torso-Inpainting Restorer [M].** A per-subject lightweight U-Net trained to fill the face mask expanded by 10–30 px with random rotations. Table VII, without → with:

  | Metric | Without | With |
  |---|---|---|
  | NIQE | 15.2476 | 14.3012 |
  | BRISQUE | 33.2916 | 25.4147 |
  | HyperIQA | 62.0307 | 66.1958 |

  These are no-reference image-quality scores only. Removing the Head-Sync Stabilizer: PSNR 39.093 → 29.193, LPIPS 0.0110 → 0.0749. — [SyncTalk++](https://arxiv.org/html/2506.14742)
- **M2DAO-Talker [M].** A Motion Consistency Constraint composites the face over the full portrait with the face alpha and supervises the whole portrait "to couple facial motion with torso dynamics". Table 6: PSNR 34.475 with the constraint vs 34.123 without; Sync-C 7.757 vs 7.688. — [M2DAO-Talker](https://arxiv.org/html/2507.08307)
- **AD-NeRF junction artifacts [C].** SegTalker observes that AD-NeRF "consistently exhibits artifacts in the connection between the head and neck" (search snippet). — [SegTalker](https://arxiv.org/html/2409.03605v1)
- **Ditto's motion controls [C, code].**
  - Stitching is on by default (`flag_stitching=True`), with motion relative to the first generated frame (`relative_d=True`: x_d = x_s + (v − d0)·gain).
  - Per-key gains (`use_d_keys`, `_set_scale_ratio` on exp/pitch/yaw/roll) and `alpha_pitch/yaw/roll` / `delta_*` pose controls exist.
  - The paper claims "seamless stitching of the generated head with the original body" but reports no torso or background metric. — [motion_stitch.py](https://github.com/antgroup/ditto-talkinghead/blob/main/core/atomic_components/motion_stitch.py), [Ditto](https://arxiv.org/html/2411.19509)
- **SadTalker's paste-back [C, code].** It uses `cv2.seamlessClone(..., cv2.NORMAL_CLONE)` with an all-ones mask over the whole crop. This Poisson blending fixes colour steps at the crop border only. — [paste_pic.py](https://github.com/OpenTalker/SadTalker/blob/main/src/utils/paste_pic.py)

### Inferences
- **[I] What SANG's motion maps show.** Stitching drives the shoulder keypoints toward the source, while the head keypoints follow the audio-driven rotation. Voxels between them (neck, hair outline) take mixed softmax weights and stretch or shear. SANG's measured 0.56× background ratio says the torso is now quiet, so the bright head, hair and neck outlines are mostly the head moving (legitimate) plus a stretched neck band (artifact).
- **[I] Most direct fix: a three-layer composite.** It extends the earlier report's plate composite:
  - background = inpainted static plate;
  - torso = the source torso, static (stitching pins it anyway);
  - head, hair and upper neck = the render.
  - The alpha is the source's head+hair+neck mask carried through LivePortrait's own deformation field, as the earlier report proposed for the person alpha. Real3D-Portrait's M_head/M_person fusion is the closest measured analogue: FID 46.38 → 42.37 over concatenation.
- **[I] Neck band.** Across the neck, let the head alpha ramp from 1 at the jawline to 0 at the collar (for example over 24–40 px at 512). The stretch is then spread over a soft band instead of appearing as a crease. Where the chin rises and reveals neck that the source does not show, fill from the plate or source neck. SyncTalk's "average neck colour" is the crude version; SyncTalk++'s per-subject U-Net is the heavy one.
- **[I] Amplitude limiting is a cheap complementary knob.** Scale generated yaw/pitch deviations from the source pose by 0.7–0.9, as Ditto's gains allow. It trades expressiveness for a smaller neck shear. No source measures this trade-off.

### Gaps
- No paper reports a seam-localised metric (neck-band flicker, seam PSNR) for any method. All evidence is whole-frame PSNR/LPIPS/FID or no-reference image quality.
- No numeric ablation of LivePortrait stitching exists, and how its shoulder mask M^st is computed is unpublished.
- No evidence was found on whether SANG's neck artifacts come from stitching's keypoint offsets, from the renderer's softmax mixing, or from the unheld keypoints. Separating them needs SANG's own stitching-on/off plus neck-band measurement.

## Q4. Hair boundary: soft alpha, halo avoidance, temporal consistency

### Takeaway
Hair is where warping renderers are weakest; the LivePortrait maintainer says so directly. Keypoints sit near the face, and accessories or hair can end up "half moving and half still". Published fixes are compositing with a soft alpha: SyncTalk blurs the rendered face mask and merges it with the original frame for hair detail. FaceMat treats hair as a matte layer. For temporal stability, carry one source alpha through the renderer's warp rather than matting each frame independently. No talking-head paper reports a halo metric.

### Cited Findings
- **Maintainer on hair [C].** "non-diffusion based methods are hard to simulate the hair swaying". A user who plotted the implicit keypoints found "all keypoints are close to face region". — [#29](https://github.com/KlingAIResearch/LivePortrait/issues/29)
- **Hats and hair [C].** Hats, headwear and hair "distorted, with half moving and half still"; confirmed as a known issue. — [#122](https://github.com/KlingAIResearch/LivePortrait/issues/122)
- **3DGS and hair [C].** SyncTalk++ says Gaussian splatting "struggles with high-frequency details such as individual hair strands", which motivates its Portrait-Sync fusion with the original frame. — [SyncTalk++](https://arxiv.org/html/2506.14742); [SyncTalk](https://arxiv.org/html/2311.17590)
- **FaceMat matting quality [C/M].** FaceMat mattes hair as an occluding layer with a continuous alpha. On real occlusions its IoU is 0.7121 vs RVM's 0.4099. — [FaceMat](https://arxiv.org/html/2508.03055)
- **Temporal stability of video matting [M, from the earlier report's source].** MatAnyone's dtSSD is 1.18 vs RVM's 1.36; MatAnyone is non-commercial. — [MatAnyone, arXiv 2501.14677](https://arxiv.org/abs/2501.14677)

### Inferences
- **[I] Hair has two roles in SANG's composite.**
  1. Hair that overlaps the background: the plate composite with a ViTMatte alpha, carried by the deformation field. Estimate foreground colour in the band with PyMatting so the plate's colour does not halo into strands.
  2. Hair that overlaps the torso (long hair on shoulders): it must follow the head, but the torso layer is static. Put it in the head layer and let the neck ramp handle the transition. Expect some shear on long hair, the case the maintainer calls hard for non-diffusion methods.
- **[I] Halo avoidance.** Never mix a per-frame matte with a source-space plate at a hard edge. Use a band of ±8–12 px around the warped source alpha where a matting model may refine, and use foreground-colour estimation inside that band. This mirrors the earlier report's step 3–4 recipe.

### Gaps
- No talking-head or portrait-animation paper found reports a hair-halo or hair-band temporal metric.
- No evidence on how LivePortrait's deformation field behaves at fine hair (sub-8-px strands, below one 64-grid flow cell). That is inferred from the 8-px cell size in the earlier report, not measured.

## Q5. Evaluation of occluder integrity and seam quality, and the recommended design for SANG

### Takeaway
No talking-head paper defines an occluder-integrity or seam metric. The closest are:
- KeySync's per-frame MAE curve during occluded frames;
- EdiDub's occluded-lip benchmark (ID-P, LSE-D, MOS);
- FaceMat's matte IoU;
- whole-frame FID/LPIPS deltas for torso-blending ablations (Real3D-Portrait, SyncTalk/++, M2DAO);
- FluentAvatar's BG-Flicker ratio, from the earlier report.

SANG should add region-masked metrics that follow directly from its composite, and adopt the occluder layer (segment once, inpaint source, render, composite on top) plus a neck-band head/torso composite.

### Cited Findings
- **KeySync [C].** Evaluates occlusion handling as MAE over time in occluded segments (Fig. 6), with no summary number. — [KeySync](https://arxiv.org/html/2505.00497)
- **EdiDub [M].** On Vox2-Occluded: ID-P 0.033 vs 0.052 and MOS-Sync 8.50 vs 8.37 (vs LatentSync). — [EdiDub](https://arxiv.org/html/2505.23406)
- **FaceMat [M].** Matte IoU / accuracy / recall on RealOcc: 0.7121 / 0.9197 / 0.9084. — [FaceMat](https://arxiv.org/html/2508.03055)
- **Real3D-Portrait [M].** Seam-related ablations are scored with FID/CSIM only (FID 42.37 full vs 46.38 with concatenation). — [Real3D-Portrait](https://arxiv.org/html/2401.08503)
- **SyncTalk++ [M].** The junction restorer is scored with no-reference IQA only (BRISQUE 33.2916 → 25.4147). — [SyncTalk++](https://arxiv.org/html/2506.14742)

### Inferences
- **[I] Proposed SANG metrics.** Compute on real-motion and demo renders, in the source crop's coordinates.
  1. **Occluder integrity.** Mean |out − I_s| inside α_O > 0.5 over all frames. It should be ≈ 0 by construction after compositing. Today it measures how much the mic is dragged.
  2. **Ghost band.** Frame-to-frame flicker, and |out − render_clean| in a ring O ⊕ 12 px \ O around the occluder. This catches residual mic texture carried by the warp, i.e. whether the step-3 inpaint was wide enough.
  3. **Neck-band flicker.** A band from the jawline to the collar, taken from a face-parser neck class dilated a few px. Mean abs frame difference in gray levels, reported as a ratio to the same band in the real video, the same family as SANG's background ratio. Also report the minimum PSNR vs frame 0 below the collar, which should be ≈ ∞ if the torso is static.
  4. **Hair-band flicker.** The ±8 px ring around the head+hair alpha edge, as a ratio to the real video.
  5. **Guard metrics** must stay unchanged within noise: mouth correlation (0.824 today on real motion), CSIM (0.911), background ratio (0.56×). Also compute lip metrics on the un-composited render whenever an occluder covers the mouth.
- **[I] Recommended design for SANG (ordered; each step gated by the metrics above).**
  1. **Occluder layer (fixes problem 1).**
     - Once per source: SAM 2 or Grounded-SAM-2 mask of world-anchored occluders overlapping the crop (Apache-2.0), then a ViTMatte alpha (MIT).
     - LaMa-inpaint the dilated occluder region in the source (Apache-2.0), then extract f_s from the clean source.
     - Render as today with stitching on, and composite α_O·I_s + (1−α_O)·render.
     - Exclude head-attached items (glasses, headsets).
     - Cost: minutes per source; no training.
     - Gate: occluder integrity ≈ 0, ghost-band flicker ≈ real video's, CSIM and mouth correlation unchanged.
  2. **Three-layer head/torso/background composite (fixes problem 2).**
     - Layers: static inpainted plate; static source torso; rendered head+hair+upper-neck.
     - Head alpha = the source head mask carried through the renderer's deformation field, with a 24–40 px ramp across the neck.
     - Fill revealed neck from the source or plate. This reuses the earlier report's plate step (d) and adds only the torso layer and the neck ramp.
     - Gate: neck-band and hair-band flicker ratios ≤ ~1× real, no visible crease.
  3. **Amplitude limiting (cheap fallback).** Scale generated yaw/pitch deviations by 0.8 (Ditto-style gains) if neck shear remains after step 2, and check the effect on the real-motion metrics.
  4. **Only if residuals remain**, borrow SyncTalk++'s idea: a small per-source junction-inpainting net trained on the source photo with randomly expanded head masks. Or try the earlier report's slot-0 lock inside the torso region instead of the composite.
  5. **Not recommended now.**
     - Full-frame diffusion renderers: Hallo3 names the microphone as a limitation, and diffusion background flicker is 3.44–5.33× per the earlier report.
     - FaceMat or MatAnyone in production: non-commercial licences.
     - XSeg: GPL-3.0 weights.
- **[I] Expected residuals.**
  - A mic shadow on the cheek still moves with the head.
  - Long hair over the shoulders shears at the neck ramp.
  - A speaker who touches or moves the mic in the real video cannot be matched by a static layer (SANG has no driving video, so this is a design limit, not a bug).

### Gaps
- None of the recommended steps has been measured on a LivePortrait renderer. The evidence is by analogy from face swap, lip-sync, FaceMat and NeRF/3DGS head–torso compositing.
- No published seam or occluder metric exists to compare SANG against. The proposed metrics are SANG-internal.
- The 0.7–0.9 amplitude gain, the 24–40 px neck ramp and the 6–10 px inpaint dilation are my starting guesses, not sourced values.
