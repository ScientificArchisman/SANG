# Motion generator, training recipe and data for SANG-M (LivePortrait-motion, audio-driven, one-shot)

Status: COMPLETE (2026-10-07). Full texts were read from arxiv.org/html. Each arXiv ID below was opened and its title checked: TalkLikeYou 2610.06658, AVTR-1 2609.22913, TalkVerse 2512.14938, IMTalker 2511.22167, TalkVid 2508.13618, PC-Talk 2503.14295, GAIA 2311.15230, UniTalker 2408.00762, MultiTalk 2406.14272, Livatar-1 2507.18649 and AsymTalker 2605.02948. Ditto's code was read from GitHub, and the WavLM-large training data from its Hugging Face model card.

**Evidence tags.**
- [M] measured: numbers from the cited source's tables.
- [C] claimed, described, or used but not ablated.
- [I] my inference for SANG.

**Spaces.** "Motion-space" means the generator outputs keypoints, 3DMM coefficients or motion latents, and a separate renderer makes the pixels. "Pixel-space" means a video diffusion model generates the frames directly.

**Already covered in earlier SANG notes** (repeated only where I found new numbers):
- logit-normal t;
- REPA;
- minibatch OT;
- consistency, shortcut and MeanFlow distillation (ARMFlow);
- a Motar-style sequence discriminator;
- AVTR-1's outlier mask, region-balanced loss and smoothness regularisers;
- SyncNet offset filtering;
- TalkVid's filters;
- MimicTalk in-context personalisation;
- Kimodo and HY-Motion human-motion scaling;
- KDTalker's Kalman smoothing.

---

## Data: scaling, filtering, mixing

### Takeaway
Only one talking-head paper (GAIA) has a controlled hours × parameters table for a speech→motion generator:
- 10× data (0.1 K → 1 K h) improved Sync-D by 0.23;
- ~3× parameters at 1 K h improved it by a further 0.31.
- Gains were small after ~600M parameters.

Mixing data helps small or rare domains but can hurt under-represented languages unless language is modelled. In MultiTalk, a learnable language embedding plus a multilingual speech encoder roughly halved non-English lip error. SANG's audio encoder, WavLM-large, was pre-trained on English only, yet about 30% of TalkVid's hours are non-English. That mismatch is a cheap thing to fix.

### Cited Findings

**Scaling (hours and parameters)**
- [M] **GAIA** is the only controlled hours × parameters table for a speech→motion generator (motion-latent space, Conformer diffusion).
  - Setup (Table 5): a 700M VAE is held fixed; the metric is test Sync-D (lower is better).
  - 180M @ 0.1 K h: **9.145**.
  - 180M @ 1 K h: **8.913** (10× data: −0.232).
  - 600M @ 1 K h: **8.603** (3.3× params: −0.310).
  - 1.2B @ 1 K h: **8.528** (a further 2× params: −0.075).
  - Full corpus: 8.2 K h raw, 16.9 K speakers.
  - Filters: a frontal-orientation band (angle of the eye corners about the nose tip); a threshold on adjacent-frame face-box and keypoint displacement ("no rapid shakes"); centre crop; masked or non-speaking frames dropped. The appearance VAE uses looser thresholds than the speech→motion model.
  - Source: [GAIA, arXiv 2311.15230](https://arxiv.org/abs/2311.15230).
- [M, pixel-space] **TalkVid**, Table 4. V-Express was trained on three corpora and tested on TalkVid-Bench: HDTF (15.8 h), Hallo3 data (70 h) and TalkVid-Core (160 h).

  | Training data | Sync-C English | Sync-C Chinese | Sync-C Polish |
  |---|---|---|---|
  | HDTF (15.8 h) | 4.000 | 3.285 | 2.654 |
  | Hallo3 (70 h) | **4.753** | 4.005 | 3.424 |
  | TalkVid-Core (160 h) | 4.567 | **4.041** | **3.695** |

  - "Hallo3 consistently achieves the lowest Sync-D, but the gaps are small."
  - TalkVid's hours: English 867.1 h, Chinese 248.9 h, i.e. about 90% in two languages. Mean clip length is 17.93 s.
  - Source: [TalkVid, arXiv 2508.13618](https://arxiv.org/abs/2508.13618).
- [C] **TalkVerse** (pixel-space, Wan2.2-5B). The 6.3 k h / 2.3 M clips were curated from more than 60 k h. Its 5B model trails Wan-S2V-14B on EMTD Sync-C (5.47 vs 6.41). It reports no curve of hours against quality. — [TalkVerse, arXiv 2512.14938](https://arxiv.org/abs/2512.14938)

**Filtering**
- [C] **TalkVerse SyncNet filter.** It keeps a clip only if |offset| ≤ 3 frames and confidence > 1.6. It also keeps only single-visible-person clips of 5–50 s at 25 fps. The authors "observed that even a small amount of out-of-sync data degrades lip-sync learning in audio-driven video generation, so we adopt these strict thresholds". This is an observation, not a table. — [TalkVerse, arXiv 2512.14938](https://arxiv.org/abs/2512.14938)
- [C] **AVTR-1 data cascade** (motion-space, SANG's exact 42-d target). No stage is ablated.
  - 34,219 candidate videos → 1,472 kept by human annotators (1,370.13 h) → 926 h after scene, face, speaker-separation and outlier stages.
  - Face crops are **smoothed over 25 frames before LivePortrait motion extraction**.
  - Windows are dropped by a finite-difference outlier mask: score > 0.15, dilated ±5 frames, 25-frame windows.
  - Face-detection gate: mean confidence of the 68 landmarks > 0.3, segments ≥ 5 s.
  - Source: [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913).

**Mixing datasets and languages**
- [M] **UniTalker** (3D vertex motion). Pooling 8 datasets into 18.5 h (934 speakers, multilingual speech plus songs):
  - lowers lip vertex error (LVE) on the small sets: BIWI (0.33 h) 4.279 → 3.859 (−9.8%); VOCASET (0.56 h) −9.3%;
  - **raises** LVE on the largest and most distinct sets: 3D-ETF-HDTF (5.49 h) +6.5%; Chinese speech (1.24 h) +9.7%, "because the proportion of Chinese speeches in A2F-Bench is small";
  - fine-tuning the pooled model back on each set improves all 8 (−0.3% to −12%).
  - Source: [UniTalker, arXiv 2408.00762](https://arxiv.org/abs/2408.00762).
- [M] **MultiTalk** (3D, 420 h, 20 languages), Table 5, LVE (×1e-4):

  | Configuration | En | It | Fr | El |
  |---|---|---|---|---|
  | No language embedding + multilingual encoder | 1.78 | 2.07 | 2.45 | 1.82 |
  | Language embedding + English-only encoder | 1.56 | 1.34 | 1.91 | 1.37 |
  | Language embedding + multilingual encoder (53 languages) | **1.16** | **1.06** | **1.39** | **1.26** |

  Source: [MultiTalk, arXiv 2406.14272](https://arxiv.org/abs/2406.14272).
- [C] **WavLM-large** was pre-trained on 60,000 h Libri-Light, 10,000 h GigaSpeech and 24,000 h VoxPopuli. The model card says: "The model was pre-trained in English and should therefore perform well only in English." This contradicts an earlier SANG note that listed WavLM as 23-language. — [microsoft/wavlm-large model card](https://huggingface.co/microsoft/wavlm-large)

**Data sizes of comparable motion-space generators**
- [M] **IMTalker**: a 39M generator trained on VoxCeleb2 only (~350 h, 160k clips) with batch 256, ~2 days on 4×A100. HDTF / CelebV Sync-C 7.711 / 7.364. — [IMTalker, arXiv 2511.22167](https://arxiv.org/abs/2511.22167)
- [M] **TalkLikeYou** (LivePortrait).
  - Training: HDTF (380 identities) for 48 h, then CelebV-HQ for 72 h, both on one RTX 3090.
  - HDTF Sync-C / Sync-D 8.73 / 6.82, against FLOAT 7.04 / 8.16 and Ditto 5.23 / 9.94 in the same table.
  - CelebV-HQ 7.97 / 7.25, against FLOAT 6.44 / 8.64.
  - Source: [TalkLikeYou, arXiv 2610.06658](https://arxiv.org/abs/2610.06658).

### Inferences
- [I] **Expected gain from more data.** SANG's 12.6k clips × ~18 s ≈ 63 h, far below GAIA's 0.1 K h point. GAIA's slope suggests a few tenths of Sync-D from moving to ~250–500 h at fixed size. SANG's own 3× experiment fixed overfitting, consistent with being data-bound.
- [I] **Practical blocker: the CPU motion-extraction cost** (90 s per clip).
  - Going to 250 h means about 50k clips, roughly 4.5 M CPU-seconds.
  - LivePortrait's motion extractor is a small CNN on 256² crops. Batching it on the GPU, with crops pre-computed and smoothed as AVTR-1 does, should cut this by an order of magnitude or more.
  - This is unmeasured here; benchmark it before buying hours.
- [I] **Language.** TalkVid's non-English ~30% goes through an English-only encoder, and the same split governs whether minority-language lip sync suffers (UniTalker's Chinese +9.7%). Two cheap fixes have measured support in 3D:
  - a language embedding, using TalkVid's language labels;
  - a multilingual SSL encoder (mHuBERT-147, XLS-R or Whisper) alongside or instead of WavLM-large.
- [I] **Clean, frontal data (HDTF-like).** TalkVid's Hallo3-vs-TalkVid result and TalkLikeYou's HDTF stage-1 suggest cleaner frontal data helps English lip precision more than raw hours do. A small HDTF/MEAD-neutral fine-tune or over-sampling stage is plausible. UniTalker's pooled→fine-tune pattern (−0.3% to −12% on every set) is the measured analogue. Check licences first (MEAD is research-only per earlier notes).

### Gaps
- I found no motion-space ablation of SyncNet-confidence or offset filtering, face-size, occlusion or multi-person filters. TalkVerse's claim is an observation only; LatentSync's evidence is pixel-space (earlier notes).
- I could not re-verify KDTalker's "4,282 pairs" or Ditto's "~50 h" in this pass. Earlier notes hold the KDTalker and Ditto data descriptions.
- There is no published data-scaling curve in LivePortrait motion space. GAIA's motion latent comes from its own VAE.

---

## Model scale and architecture

### Takeaway
At SANG's data size, evidence does not favour a 3× larger generator. GAIA's +3× parameter gain was measured at 1 K h. Comparable open motion generators are 39M (IMTalker), 43M (KDTalker) and 153M (AVTR-1, 926 h). AVTR-1 is only at parity with Ditto and FLOAT on lip sync.

Architecture evidence that does exist:
- Frame-aligned additive audio injection (which SANG's AdaLN already is) beat audio cross-attention by ~1.3 Sync-D in GAIA.
- A convolution module (Conformer) beat a plain Transformer on jitter and sync.
- The largest measured motion-space lip-sync gains come from **audio features pre-trained for audio-visual sync**: PC-Talk +0.94 LSE-C. TalkLikeYou, KDTalker and PC-Talk all use such a Wav2Lip/SyncNet-type audio encoder.

### Cited Findings
- [M] **GAIA**, conditioning and backbone ablation (Table 9; 180M diffusion; Sync-D↓ / motion-smoothness MSI↑ / FID↓):

  | Variant | Sync-D | MSI | FID |
  |---|---|---|---|
  | Default: speech **added** in every Conformer block, reference via cross-attention | 8.913 | 1.132 | 24.242 |
  | Speech added, reference added | 8.989 | 1.125 | 28.189 |
  | Speech cross-attn, reference cross-attn | 10.231 | 1.139 | 28.718 |
  | Speech cross-attn, reference added | 10.180 | 1.015 | 24.021 |
  | Conformer → plain Transformer | 9.312 | **0.623** | 25.385 |

  - The authors say the plain Transformer causes "significant motion jittering".
  - Removing head-pose prediction gives Sync-D 9.134 (Table 6).
  - Scale at 1 K h: 180M 8.913 → 600M 8.603 → 1.2B 8.528.
  - Source: [GAIA, arXiv 2311.15230](https://arxiv.org/abs/2311.15230).
- [M] **PC-Talk** (LivePortrait implicit keypoints, HDTF + MEAD-neutral, autoregressive Transformer), Table 3 lip-sync ablation, LSE-C:
  - no AV encoder: 6.23;
  - plus an audio encoder "pretrain[ed] … on 2D audio-visual synchronization tasks" instead of ASR features: **7.17**;
  - plus L_sync: **8.92**;
  - plus L_kp: **9.37**.
  - HDTF image-input results: LSE-C 9.37 / LSE-D 6.44, against Sonic 8.64 / 6.77 and SadTalker 7.15 / 7.93.
  - Audio enters through cross-attention. The style embedding is added to the input.
  - Source: [PC-Talk, arXiv 2503.14295](https://arxiv.org/abs/2503.14295).
- [C] **AVTR-1** (153M; 18 layers; width 512; FFN 512; 8 heads; RoPE base 1e4; dropout 0; HuBERT audio). No ablations reported.
  - Audio enters by **cross-attention**, through separate paths for near history, far history, self audio and other audio, plus an MLP for the reference. In the authors' words: "prevents audio tokens from competing with motion and reference inputs in the same attention distribution".
  - Per-head learned QK-RMSNorm.
  - Four RMS-normalised regional output heads, so classifier-free guidance can be applied per region.
  - Source: [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913).
- [M] **AVTR-1's dyadic test set** (LSE-D↓ / LSE-C↑; FID / FVD):

  | Model | LSE-D | LSE-C | FID | FVD |
  |---|---|---|---|---|
  | AVTR-1 | 7.08 | 3.28 | 14.3 | 76.8 |
  | Ditto | 7.12 | 3.11 | 16.0 | 113.6 |
  | FLOAT | 6.99 | 2.93 | 11.7 | 83.2 |
  | SoulX Lite | 6.74 | 3.38 | – | – |
  | DyStream | 6.57 | 3.21 | – | – |

  Source: [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913).
- [M] **IMTalker**: a 39M DiT of self-attention plus frame-wise AdaLN. Audio (wav2vec2), 6-D pose and gaze are summed with the timestep into one condition, the same design family as SANG. Renderer 124M; generator 42 FPS on an RTX 4090. — [IMTalker, arXiv 2511.22167](https://arxiv.org/abs/2511.22167)
- [C] **TalkLikeYou**: a 4-block DiT (self-attention → audio cross-attention → habit injection → temporal attention) over **18-d lip-motion features**. Audio comes from the **Wav2Lip audio encoder** "for its audio-lip aligned features". It runs at 32 FPS end-to-end with one step. — [TalkLikeYou, arXiv 2610.06658](https://arxiv.org/abs/2610.06658)

### Inferences
- [I] **Keep frame-wise AdaLN audio injection.** GAIA's measured result favours additive, frame-aligned speech over cross-attention. The cross-attention systems (AVTR-1, PC-Talk, TalkLikeYou) never ablate the choice. If SANG wants local audio context, add a *windowed* (±6–8 frame) cross-attention or a depthwise temporal conv *alongside* AdaLN, not instead of it.
- [I] **Add a depthwise conv module** (Conformer-style, kernel ~7–15 frames) to each block. GAIA's Transformer→Conformer gap (Sync-D 9.312 → 8.913, MSI 0.623 → 1.132) is the only measured backbone ablation in motion space. It adds about 1–2% parameters.
- [I] **Change audio features before parameter count.** Concatenate WavLM-large with an AV-sync-trained audio embedding (Wav2Lip/SyncNet audio tower, or LatentSync's StableSyncNet audio branch) and/or a multilingual SSL encoder. PC-Talk's +0.94 LSE-C from the encoder swap is the largest single architectural number found for LivePortrait keypoints.
- [I] **Defer model scale.** Grow to ~100–150M (e.g. 12–16 layers × 512) only after data passes ~250 h. GAIA's parameter gain was measured at 1 K h, and SANG overfit at lower data.

### Gaps
- I found no motion-space ablation of RoPE vs learned or sinusoidal positions, attention window length, or multi-scale audio (several WavLM layers or temporal pyramids).
- Motar (77M) is still unlocated. Earlier notes found arXiv 2609.10317 under a different title, and I did not re-search it.
- FLOAT's parameter count and ablations were not re-checked in this pass.

---

## Training recipe for small flow-matching motion models

### Takeaway
New measured or semi-measured recipe evidence for small motion flow models:
1. A **SyncNet-type lip-sync loss on frames rendered through the frozen LivePortrait renderer** gave +1.75 LSE-C in PC-Talk. It is the largest recipe number found, though possibly inflated by SyncNet-on-SyncNet evaluation.
2. **Clean-motion (x0) parameterisation** instead of velocity reduced jitter and enabled 1-step sampling in TalkLikeYou (figure-level only).
3. **Fewer Euler steps can sync better.** IMTalker: 5 steps Sync-C 7.43 vs 50 steps 7.24.
4. **Per-region, per-condition CFG** (AVTR-1), used but not ablated.

### Cited Findings
- [M] **PC-Talk loss.** L_LAC = L_sync + λ_kp L_kp + λ_reg L_reg + λ_vel L_vel + λ_style L_style.
  - L_sync is the negative cosine between a SyncNet visual embedding of 5 rendered frames and the audio embedding, "adapted from Wav2Lip". Rendering goes through the frozen keypoint renderer.
  - L_kp is L1 on implicit keypoints.
  - Ablation: adding L_sync took LSE-C 7.17 → 8.92; adding L_kp took it to 9.37.
  - HDTF ground-truth LSE-C is 8.243 in KDTalker's table (earlier notes), so 9.37 is above real video.
  - Source: [PC-Talk, arXiv 2503.14295](https://arxiv.org/abs/2503.14295).
- [M, figure only] **TalkLikeYou parameterisation.** "Directly predicting the velocity field can introduce noticeable jitter in generated motions", so the model predicts clean motion.
  - Fig. 6: "Removing reparameterization degrades both Flow Matching and DDIM". Flow matching "maintains strong lip-sync performance across sampling steps".
  - Losses: reconstruction on x0, a first-frame boundary loss, and 1st- and 2nd-order derivative losses.
  - Source: [TalkLikeYou, arXiv 2610.06658](https://arxiv.org/abs/2610.06658).
- [M] **IMTalker step count** (Table 3, Sync-C / Sync-D, no pose or gaze conditioning):
  - 5 steps: 7.43 / 7.65;
  - 10 steps: 7.36 / 7.83;
  - 50 steps: 7.24 / 7.98;
  - FID 17.97 → 17.55 → 17.49 over the same steps.
  - Source: [IMTalker, arXiv 2511.22167](https://arxiv.org/abs/2511.22167).
- [C] **AVTR-1 guidance training.**
  - Self audio, other audio and reference are kept together for 45% of samples, all three dropped for 15%, and one or two kept for the rest. Past history is retained 75% of the time.
  - At inference the past-only prediction is the baseline. Each condition adds a residual **scaled per motion region**.
  - Optimiser: Adan with 3 weight-decay groups. Training: 200k steps, about 35 h on one GH200.
  - Source: [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913).

### Inferences
- [I] **Sync loss through the renderer** is the most direct attack on SANG's lip-correlation gap (0.64 vs a renderer ceiling of 0.82–0.85) because it supervises what the renderer actually shows.
  - Implementation: decode the one-step clean estimate, render only the mouth crop for a 5-frame window on a subset of the batch at t above a threshold, and add a cosine SyncNet loss.
  - Risks:
    - over-articulation and metric gaming (PC-Talk exceeds GT LSE-C);
    - extra training cost from running the warp and decoder in the loop.
  - Judge it on lip-trajectory correlation and audio_gain (keep near 1.0–1.5), not on LSE-C. A different expert for loss and metric is preferable.
  - A keypoint-space lip expert (earlier notes) is the cheaper fallback.
- [I] **x0 parameterisation** is a near-zero-cost A/B, since the network output is reinterpreted. It may help SANG's velocity losses, which then act on a direct x0 estimate. Measure lip correlation and jitter at SANG's current NFE and at 1–4 steps.
- [I] **Sweep NFE (2/4/8/16) and the per-region CFG scale** before any distillation. IMTalker suggests more steps do not improve lip sync for small motion models.

### Gaps
- I found no motion-space measurement of self-conditioning, EMA decay, longer training or prediction-target choice beyond TalkLikeYou's figure.
- PC-Talk's supplementary (exact λ values, L_vel and L_reg forms) was not retrieved.

---

## Predicting scale, translation and shape keypoints

### Takeaway
Systems split two ways:
- AVTR-1 (SANG's exact 42-d space) and TalkLikeYou hold scale, translation and canonical keypoints from the photo.
- Ditto (265-d) and KDTalker (70-d) predict translation, and Ditto also predicts scale. Both smooth these before rendering: Ditto with a 3-frame moving average on scale, pose and t; KDTalker with a Kalman smoother and t_z zeroed.

No system predicts the 8 shape/canonical keypoints. Head placement drives most of the pixel-metric gap: IMTalker FID 17.55 → 9.73 with ground-truth pose, and SANG's own face PSNR 16.8 → 24.5 dB with real scale and translation.

### Cited Findings
- [C] **AVTR-1**: "We exclude the scale, the translation, the canonical keypoints, and the 24 remaining expression coordinates from the training target. At render time they are taken from the source image and held fixed." — [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- [C] **TalkLikeYou** predicts only rotation and expression. It also neutralises the source lip configuration and mouth opening across identities, "reducing the effects of imperfect disentanglement in implicit keypoints", and keeps the neutral mouth slightly open so the teeth are preserved. — [TalkLikeYou, arXiv 2610.06658](https://arxiv.org/abs/2610.06658)
- [C, code] **Ditto** predicts 265-d:
  - scale (stored as scale−1);
  - pitch, yaw and roll as **66-bin logits each**;
  - t (3);
  - exp (63).
  - A 3-frame moving average is applied only to dims 0–201 (scale, pose bins, t). An optional per-dimension clamp (`v_min_max_for_clip`) is applied at render time.
  - Source: [ditto-talkinghead audio2motion.py](https://github.com/antgroup/ditto-talkinghead/blob/main/core/atomic_components/audio2motion.py).
- [C, code, from earlier SANG notes] **KDTalker** predicts pose, t and 63-d exp, holds scale from the source, sets t_z = 0, and Kalman-smooths pose, t and exp (transition covariance 0.05·I, observation 0.001·I). — [KDTalker inference.py](https://github.com/chaolongy/KDTalker/blob/main/inference.py)
- [M] **IMTalker**: conditioning on ground-truth 6-D pose moves FID 17.55 → 9.73 and FVD 200.96 → 118.59, with Sync-C 7.36 → 7.17. — [IMTalker, arXiv 2511.22167](https://arxiv.org/abs/2511.22167)
- [M] **GAIA**: removing head-pose prediction worsens Sync-D 8.913 → 9.134. — [GAIA, arXiv 2311.15230](https://arxiv.org/abs/2311.15230)
- [C] **AVTR-1** smooths the face-crop boxes over 25 frames before running the LivePortrait motion extractor. Crop jitter otherwise appears as jitter in the extracted scale and translation. — [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913)

### Inferences
- [I] **Predict translation and scale as residuals.** Add Δt_x, Δt_y and Δlog-scale relative to the source (3 dims) and keep t_z = 0, as KDTalker does.
  - Give them a separate region in the region-balanced loss and stronger smoothness weights. Low-pass them at render: a 1€ filter or a Kalman filter like KDTalker's.
  - Keep stitching on, since stitching already holds the background at 0.56× real.
  - Evaluate face PSNR against real (target: close the 16.8 → 24.5 dB gap) together with background displacement.
- [I] **Check the cache before training.** If SANG's cached t and scale came from unsmoothed per-frame crops, the targets carry crop jitter. Re-deriving them with AVTR-1-style 25-frame crop smoothing, or smoothing the targets, should come first.
- [I] **Shape keypoints**: no evidence supports predicting them. Keep holding them from the photo.

### Gaps
- No paper reports a measured jitter or fidelity trade-off for predicted translation. Ditto's and KDTalker's smoothing constants are code defaults without ablation.
- Xemo-Talker's handling of translation was not re-checked in this pass. Earlier notes record that it "follows KDTalker".

---

## Long-video consistency

### Takeaway
I found no motion-space paper reporting drift numbers over 1–5 minute clips. Practice falls into four patterns:
- short conditioning plus crossfade: Ditto (1-frame condition, 10-frame linear crossfade, optional periodic reset to the source keypoints);
- long history plus self-rollout training: AVTR-1 (75-frame history, curriculum that replaces ground-truth history with the model's own estimates);
- boundary losses: TalkLikeYou (first-frame plus derivative losses);
- drift-aware distillation: AsymTalker (pixel-space).

SANG should build its own drift measurement and adopt the cheap training-side fixes.

### Cited Findings
- [C, code] **Ditto**:
  - 80-frame windows;
  - `overlap_v2=10` frames re-generated and linearly crossfaded (alpha ramps 0→1);
  - the next window is conditioned on one keypoint frame (`kp_cond = res_kp_seq[:, idx-1]`);
  - a 3-frame moving average on scale, pose and t only;
  - optional `fix_kp_cond=N` resets the condition to the source keypoints every N windows (default 0, off);
  - 50 sampling steps by default.
  - Sources: [audio2motion.py](https://github.com/antgroup/ditto-talkinghead/blob/main/core/atomic_components/audio2motion.py); [stream_pipeline_offline.py](https://github.com/antgroup/ditto-talkinghead/blob/main/stream_pipeline_offline.py).
- [C] **AVTR-1 rollout curriculum.**
  - Each 25-frame target is split into five 5-frame chunks. Epoch 1 uses ground-truth history.
  - Each later epoch replaces one more chunk, working backwards, with the model's own one-step estimate.
  - History tokens carry a learned embedding of the timestep they were estimated at (t=1 for ground truth). History length is 75 frames.
  - No ablation reported.
  - Source: [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913).
- [C] **TalkLikeYou**: a first-frame loss ties each window's first frame to the previous window's last frame, plus derivative losses across the boundary. "This constraint helps extend generation to arbitrary-length sequences" (no number). — [TalkLikeYou, arXiv 2610.06658](https://arxiv.org/abs/2610.06658)
- [C] **Livatar-1** names "long-term pose drift, where cumulative errors cause the head's pose and shape to deviate over time" as a key failure. The method is withheld. HDTF-100 Sync-C 8.501 (ground truth 7.614). — [Livatar-1, arXiv 2507.18649](https://arxiv.org/abs/2507.18649)
- [C, pixel-space] **AsymTalker**: "cascading identity drift propagated through self-generated continuity references". The fix is a teacher anchored on ground-truth continuity frames and a student trained only on self-generated references; continuity window τ = 3. — [AsymTalker, arXiv 2605.02948](https://arxiv.org/abs/2605.02948)

### Inferences
- [I] **Train with self-generated prefixes.** SANG trains its 10-frame clean prefix on ground truth, but at inference the prefix is self-generated. A cheap fix in the AVTR-1 spirit: in a fraction of batches (ramping to ~30–50%), replace the prefix with the model's own one-step x0 estimate from a forward pass under no_grad. A prefix-noise-level embedding is optional.
- [I] **Measure drift** on 1–5 minute TalkVid or HDTF clips. Track per-minute head-angle mean and std against the first minute, the mouth-opening distribution, and lip correlation per minute. Ditto's periodic source reset (`fix_kp_cond`) serves as the baseline fix.

### Gaps
- There are no measured prefix-length or overlap ablations, and no drift curves, for any motion-space talking-head model.

---

## Personalisation / test-time adaptation

### Takeaway
The newest motion-space result is TalkLikeYou (2026-10-05). A reference-motion "habit" encoder over 100 frames of the target's motion conditions a LivePortrait-space flow generator. It is trained first with one-hot identity codes and then with same-person reference/target pairs. It imitates per-person lip habits without per-person training and keeps lip sync high: HDTF Sync-C 8.73.

Fine-tuning a pooled model per domain consistently helps (UniTalker −0.3% to −12% LVE). MimicTalk's numbers are in earlier notes.

### Cited Findings
- [M] **TalkLikeYou.**
  - Habit encoder: a Transformer over a 100-frame reference motion sequence, injected into the generator.
  - Stage 1: HDTF, one one-hot code per identity (380), plus an alignment loss from the reference features to the one-hot embedding.
  - Stage 2: CelebV-HQ with two sequences of the same person; one-hot codes reused every 10th step.
  - Without the two-stage strategy, "direct training tends to produce less mouth opening and more uniform motions" (figure-level ablation).
  - Habit diversity (pairwise PLAD Jensen–Shannon distance): 71.14, against Ditto 33.99, FLOAT 27.19 and Sonic 25.47.
  - User study, habit imitation: 4.6, against PC-Talk 3.3 and StyleTalk 1.5.
  - PLAD needs at least 500 frames (20 s) per sequence.
  - Source: [TalkLikeYou, arXiv 2610.06658](https://arxiv.org/abs/2610.06658).
- [C] **PC-Talk** uses the same style-space design (a one-hot preset or a Transformer-encoded reference, randomly alternated in training), plus an amplitude-based lip-scale factor. — [PC-Talk, arXiv 2503.14295](https://arxiv.org/abs/2503.14295)
- [M] **UniTalker**: fine-tuning the pooled model on each dataset improves all 8 sets over single-set training (−0.3% to −12% LVE). — [UniTalker, arXiv 2408.00762](https://arxiv.org/abs/2408.00762)

### Inferences
- [I] **Speaker-reference condition.** SANG's hash split preserves speaker IDs, so same-speaker (reference clip, target clip) pairs are free.
  - Add a small Transformer reference encoder over 100–250 frames of the speaker's *other* clip's motion (mouth region only, or all 42-d).
  - Inject it like the existing reference vector, with dropout so the one-shot path still works.
  - This may raise lip correlation for speakers with known footage, because the model can stop averaging articulation styles. It does nothing for a pure photo.
- [I] **Per-speaker LoRA** on the 53M model is feasible on one GPU in minutes, given SANG's <20-minute full retrain. No measured motion-space LoRA result was found.

### Gaps
- I found no measured few-shot LoRA result for a talking-head motion generator. TalkLikeYou's ablation numbers are only in figures.

---

## Ranked changes for SANG

Ranked by expected impact on SANG's two bottlenecks (lip correlation 0.64 against a 0.82–0.85 ceiling; head-placement fidelity) per unit of one-GPU cost.

1. **Add AV-sync-pretrained and multilingual audio features** next to WavLM-large.
   - Evidence: [M] PC-Talk, AV-pretrained encoder +0.94 LSE-C on LivePortrait keypoints. [M] MultiTalk, multilingual encoder LVE 1.56 → 1.16 (En) and 1.91 → 1.39 (Fr). [C] WavLM-large is English-only, while ~30% of TalkVid hours are non-English.
   - Cost: audio feature re-extraction only (GPU, minutes to hours) plus 20-minute retrains. Motion caches are untouched.
   - Expected effect [I]: +0.03–0.08 lip correlation, larger on non-English test speakers.
2. **Language embedding** from TalkVid labels; also stratify the val set by language.
   - Evidence: [M] MultiTalk, LVE 1.78 → 1.16 (En) and 2.07 → 1.06 (It) with the embedding. [M] UniTalker, a minority language degrades when pooled (+9.7% LVE).
   - Cost: trivial.
   - Expected effect [I]: mainly non-English lip accuracy.
3. **Lip-sync loss through the frozen LivePortrait renderer** (mouth crops, 5 frames, applied to the one-step x0 estimate).
   - Evidence: [M] PC-Talk, +1.75 LSE-C (7.17 → 8.92).
   - Caveats: LSE-C rose above ground truth, so there is a risk of gaming the metric. Judge it on lip correlation and audio_gain using a different expert.
   - Cost: 1–2 days to implement; training becomes several times slower but stays within hours.
   - Expected effect [I]: the largest available lever on lip correlation. Watch for over-articulation.
4. **Predict Δt_xy and Δlog-scale (3 extra dims), smoothed; first re-derive targets from crop-smoothed extraction.**
   - Evidence: [M] SANG's own face PSNR 16.8 → 24.5 dB with real scale and translation. [M] IMTalker, ground-truth pose takes FID 17.55 → 9.73. [C] Ditto and KDTalker both predict t and smooth it (3-frame moving average / Kalman). [C] AVTR-1 smooths crops over 25 frames before extraction.
   - Cost: low if the full LivePortrait output is cached; otherwise it rides on the next re-cache.
   - Expected effect [I]: most of the head-placement PSNR gap. Keep stitching on to protect the background.
5. **Move LivePortrait motion extraction to batched GPU, then scale data to 250–500 h with language-balanced sampling.**
   - Evidence: [M] GAIA, 10× hours gave −0.23 Sync-D at 180M. [M] SANG's own 3× data fixed overfitting. [M] TalkVid's 160 h beat HDTF everywhere and beat 70 h Hallo3 on non-English sync.
   - Cost: engineering work on the extractor. The 90 s per clip on CPU is the real bottleneck, and the GPU speed-up is unmeasured.
   - Expected effect [I]: steady sync and generalisation gains. It is a prerequisite for item 8.
6. **Cheap recipe A/Bs**, each a single 20-minute retrain:
   - x0 parameterisation ([M, figure] TalkLikeYou);
   - NFE sweep 2–16 ([M] IMTalker: 5 steps better sync than 50);
   - per-region, per-condition CFG scales ([C] AVTR-1);
   - a depthwise temporal conv module per block ([M] GAIA, Conformer vs Transformer Sync-D 8.913 vs 9.312, MSI 1.132 vs 0.623).
   - Expected effect [I]: small sync gains and less jitter.
7. **Self-generated prefix training plus a boundary loss plus a drift test suite** (1–5 minute clips).
   - Evidence: [C] AVTR-1 rollout curriculum; [C] TalkLikeYou first-frame loss; [C] Ditto periodic source reset. No measured drift numbers exist.
   - Cost: low.
   - Expected effect [I]: protects long-clip stability. It is unlikely to move short-clip lip correlation.
8. **Grow the generator to ~100–150M only after data reaches ~250 h or more.**
   - Evidence: [M] GAIA, 180M → 600M gave −0.31 Sync-D, but at 1 K h. [M] AVTR-1 (153M, 926 h) is only at parity with Ditto and FLOAT on LSE.
   - Cost: still minutes to an hour per run.
   - Expected effect [I]: near zero at today's ~63 h.
9. **Speaker-reference "habit" condition** (same-speaker clip pairs from TalkVid) and an optional per-speaker LoRA.
   - Evidence: [M] TalkLikeYou, habit diversity 71.14 vs 27–34 for FLOAT and Ditto, with high sync. [M] UniTalker, fine-tuning gains of −0.3% to −12% LVE.
   - Cost: moderate (a new encoder plus pair sampling).
   - Expected effect [I]: helps only when footage of the target speaker exists.
