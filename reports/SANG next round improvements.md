# SANG's next gains come from timing, not size: clean the pairs, sharpen the audio, supervise the lips, then fix the renderer

SANG-M already uses the field's best-measured conditioning design: frame-wise adaLN-Zero audio, SwiGLU blocks, a region-balanced loss and a velocity loss. It also trains in about 20 minutes. Its remaining deficits have specific causes, and none of them needs more parameters.

1. **Lip timing.** Generated lip opening correlates with the real trajectory at **0.64**. Re-rendering real motion reaches **0.82–0.85**, so most of the lip gap is in the motion model.
   - The motion model hears speech through the *last layer* of an *English-only* encoder (WavLM-Large, whose model card says so) through a ±80 ms window.
   - It trains on in-the-wild clips whose audio–video offsets were never corrected.
   - No loss looks at lip timing specifically.
2. **Data.** SANG trains on about 63–70 h of TalkVid's 1,244 h. Validation flow loss has plateaued at **1.146**, and tripling the data removed overfitting.
3. **Pixels.** FID, FVD, teeth and sharpness are set by LivePortrait's decoder and by head placement, not by the motion model.

**The plan, ranked by expected gain per GPU-day:**

1. **Diagnose before changing anything** (hours, inference only). Decide whether the 0.64 comes from mouth randomness, bias, or audio–video offsets in the data.
2. **Sampling-only upgrades** from image and audio generation, on the current checkpoint: autoguidance, decomposed mouth guidance, a guidance interval, and a non-uniform step schedule.
3. **Clean the training pairs:** per-clip audio–video offset correction, a sync-confidence filter, and outlier masking.
4. **A batch of free objective changes,** each a 20-minute retrain: logit-normal time sampling, contrastive flow matching, a band-limited spectral loss, and EMA selection.
5. **A better audio stream:** a learned WavLM layer mix or a multilingual encoder, plus a second, sync-trained audio stream, injected per frame.
6. **Lip supervision without a renderer in the loop:** a frozen, audio-aware lip expert in motion space, used as a loss (THUNDER-style), and/or REPA alignment to a lip-reading encoder.
7. **Scale the data:** batch LivePortrait extraction on the GPU, pretrain on 300–1,000 h with noisy clips used audio-free, then fine-tune on the clean subset.
8. **A separate renderer track** for FID, FVD and teeth: fine-tune only LivePortrait's decoder, add temporal super-resolution of the face crop, and condition on a teeth reference image.

A note on evidence. Almost every technique below was measured in another field: image generation, video-to-audio, TTS, 3D face meshes, robotics or forecasting. **No paper has tested any of them on a 42-d LivePortrait-keypoint flow model.** "Measured" below means a number from SANG's runs or from a paper's table. "Inference" means a transfer estimate. Every expected gain on SANG is an inference.

## Where SANG stands, and what the research notes got wrong about it

| Quantity | Value | Source |
|---|---|---|
| Lip-opening correlation, generated vs real | **0.64–0.65** | `naturalness.py`, job 173057, 300 clips |
| Same, real motion re-rendered through the 42-d target | 0.82–0.85 | M0, job 173683 |
| Mouth amplitude / /p b m/ closure | fixed by mouth-only guidance γ = 1.25 | `lse_m125`, naturalness sweep |
| Validation flow loss | 1.146 at 12k steps, plateaued (cosine anneal −0.7 %) | job 172696 |
| Head velocity vs real | 0.92× | same |
| audio_gain (mouth error with wrong audio ÷ right audio) | 1.50 | same |
| Training data | 12,584 clips ≈ 63–70 h of TalkVid's 1,244 h | `cache/motion_lp` |
| Retrain cost | ~15 it/s, 15k steps ≈ 20 min | run table |
| Motion extraction cost | 90–125 s per clip, CPU-bound | cache notes |

What `sang/motion_model.py` and `configs/train_motion.yaml` show. Several suggestions in the notes are already in the code; others are genuinely open.

| Item | SANG today | Open? |
|---|---|---|
| Audio injection | frame-wise adaLN-Zero, centred Conv1d k = 5 (±80 ms) over WavLM-Large **last layer** | widen the window; mix layers |
| Block | LayerNorm + MHA + **SwiGLU**, zero-init adaLN | RMSNorm, qk-norm, ConvMLP |
| Positions | **learned absolute** `nn.Embedding(512)` | RoPE |
| Time sampling | `t_dist: uniform` (a logit-normal option already exists in `sample_t`) | flip the flag, add a shift |
| Losses | region-balanced flow MSE + velocity loss on x̂0 = x_t − t·v | spectral, contrastive, lip-specific |
| Normalisation | per-dimension z-score (`Norm.fit`) | already done; lip dims are not under-weighted |
| Crops | one fixed box per clip (`clip_box`) | AVTR-1's crop smoothing is unnecessary |
| AV offset | **none**. `filter_clips.py --sync` screens a margin and does not shift audio | correct offsets |
| Dropout / EMA | 0.1 (added after memorisation) / 0.995 (half-life ≈ 138 steps) | EMA sweep |
| Guidance | audio CFG, mouth-only γ, optional rescale and projection | autoguidance, APG, interval |

Two notes use the opposite time convention, writing x̂1 = x_t + (1−t)·v̂. In SANG, **t = 0 is clean** and x̂0 = x_t − t·v. Below, "low noise" means small t, and a positive logit-normal shift means more noise.

## 1. Diagnose first: why is lip correlation 0.64?

Three causes predict different fixes, and three cheap measurements separate them.

**Randomness or bias.** Model the real trajectory as y = μ(a) + η and a sample as s = μ̂(a) + ε, with ε independent of η. Let r_ss be the correlation between two samples for the same audio (the pending `--self-corr` job), and r_sy the correlation between a sample and the real trajectory.
- **r_ss ≈ r_sy ≈ 0.64:** the sampler is calibrated and the gap is aleatoric. The best possible single-sample correlation is about √r_ss ≈ 0.80, so only less mouth randomness can raise it: averaging, a lower mouth temperature, or a deterministic mouth anchor.
- **r_sy > r_ss:** the sampler is over-dispersed. Lower temperature or averaging helps strongly.
- **r_ss ≫ 0.64** (for example ≥ 0.85): samples agree with each other but not with reality, so the limit is bias. Change the audio front end, the data and the supervision.

Evidence for the first case: a probabilistic mesh model's single samples had 37 % higher lip error than a deterministic model (10.8 vs 7.9), yet the *mean* of its samples matched it exactly at 7.9 ([Yang et al.](https://arxiv.org/abs/2311.18168)). SubtleTalk measured lip vertex error (LVE) of 14.60 for flow matching alone, 11.35 for a deterministic prior and 12.46 for the prior plus a residual flow ([SubtleTalk](https://arxiv.org/abs/2608.06408)). Extend the `--self-corr` job with **K-seed averaging of the 18 mouth coordinates** (K = 2, 4, 8, 16), plotting correlation, CCC and amplitude ratio against K. That curve shows how much a deterministic mouth anchor (section 6) could buy before any code is written.

**Offset in the data.** Compute the cross-correlation between generated and real lip opening over lags of −4…+4 frames for each clip. A peak consistently away from zero means the model learned a shifted timing, which points straight at uncorrected audio–video offsets in TalkVid. A broad peak means blurred timing. Either way, also report the lag-tolerant correlation next to the frame-locked one. A Soft-DTW evaluation paper argues frame-locked scores punish harmless phase shifts ([2606.01031](https://arxiv.org/abs/2606.01031)). This costs minutes.

**Language.** Split lip correlation and CCC by clip language. WavLM-Large's model card says it "was pre-trained in English and should therefore perform well only in English" ([model card](https://huggingface.co/microsoft/wavlm-large)), and about 30 % of TalkVid's hours are not English. If English clips clearly lead at matched clip counts, the encoder change in section 5 moves to the front. In MultiTalk, a 53-language encoder plus a language embedding cut non-English LVE from 1.91 to 1.39 (French) and from 1.34 to 1.06 (Italian), against an English-only encoder with the same embedding ([MultiTalk Tab. 5](https://arxiv.org/abs/2406.14272)).

**Reference shortcut (one more cheap check).** The reference vector includes the reference frame's own 42-d target, mouth state included. LatentSync reports that generators "shortcut" from visual context instead of using audio ([LatentSync](https://arxiv.org/abs/2412.09262)). Measure lip correlation with each clip's reference swapped for another frame of the same speaker. A large drop means the reference is doing work the audio should do.

## 2. Sampling-only upgrades (no retraining; hours)

These run on `runs/motion_12k_anneal/best.pt` with `naturalness.py` and `eval_lse.py`. Judge every arm on lip correlation, CCC, amplitude ratio, /p b m/ closure and LSE together. Guidance raises LSE-C by exaggerating the mouth, so LSE alone is gameable: THEval found LSE-C correlates *negatively* with human preference, ρ = −0.164 ([THEval](https://arxiv.org/abs/2511.04520)).

**2a. Autoguidance: guide with a weaker copy of SANG, not with "no audio".** Autoguidance computes v = v_weak + w·(v_main − v_weak), where both models get the same audio and reference. On ImageNet-512 it took FID from 2.56 to **1.34**, against 2.23 for CFG and 1.68 for CFG with a guidance interval. A guide of the *same size* with less training kept most of the gain (1.51), and "a majority of the improvement comes from reduced training of the guiding model" ([Autoguidance, Karras et al.](https://arxiv.org/abs/2406.02507)).
- Why it may suit SANG (inference): CFG's difference (with audio − without audio) is dominated by the average mouth shape and amplitude that audio implies, which is why it inflated amplitude to 1.52×. The main-vs-weak difference amplifies what the model learned *late*, plausibly the fine timing.
- How: train a guide with the same config for about 1/16 of the schedule (~750 steps, under a minute), sweep w ∈ {1.5, 2, 2.5}, and apply it mouth-only like the current guidance.
- Guides built from synthetic degradation (dropout, input noise) "did not work at all".
- Main and guide need their own EMA (1.34 with separate EMAs vs 1.53 with the same).
- Parameter cost: a same-size guide doubles inference compute. A small guide (4 layers × 256, a few M parameters) keeps SANG near 53 M; the paper's XS guide cost only +3.6 % training.

**2b. Separate the amplitude knob from the timing knob (APG).** APG splits the guidance update into a part parallel to the conditional prediction, which acts as a gain and causes over-saturation, and an orthogonal part, which carries quality ([APG](https://arxiv.org/abs/2410.02416); table numbers not retrieved).
- How: on the mouth dims, in x̂0 space per clip, compute Δ = x̂0_c − x̂0_u and split it into Δ∥ (along x̂0_c) and Δ⊥. Weight Δ∥ by η (amplitude) and Δ⊥ by w (shape and timing) separately.
- Why: today γ = 1.25 is a compromise between amplitude and sync. APG could make amplitude its own control.

**2c. Guidance interval.** Skip guidance at the highest noise levels (t near 1 in SANG's convention), where the update is mostly a mean shift. In REPA, guiding only on part of the path took FID from 1.80 to 1.42 and allowed a larger scale ([REPA Tab. 10](https://arxiv.org/abs/2410.06940)). Sweep guidance on t ∈ [0.1, 0.7] and [0, 0.8]. Test it on its own first, because it reduced autoguidance's benefit.

**2d. Step schedule and count.**
- **Sway Sampling,** a non-uniform schedule that spends more steps near noise, cut F5-TTS word error rate from 2.84 to 2.41 at the same 32 NFE ([F5-TTS](https://arxiv.org/abs/2410.06885)). The F5 authors attribute this to early flow steps deciding how closely the output follows the condition. Sweep s ∈ {0, −0.4, −0.8} at 10 steps.
- **NFE sweep {4, 6, 10, 16}.** In IMTalker, a 39 M motion DiT synced *better* with fewer steps: Sync-C 7.43 at 5 steps vs 7.24 at 50 ([IMTalker Tab. 3](https://arxiv.org/abs/2511.22167)).

**2e. Joining windows for long clips.** Do not average overlapping independently sampled windows. For flow-matching robot policies, averaging "performs poorly across the board" and caused oscillations severe enough to trigger a robot's protective stop. Freezing the first frames and guiding the rest of the overlap with a decaying weight (Real-Time Chunking) beat it, and soft masks beat hard ones ([RTC](https://arxiv.org/abs/2506.07339)). This is a seam and naturalness fix, not a lip fix. SANG's current hard 10-frame prefix has no measured seam spike (0.45–0.53×), so it is low priority until longer clips are tested.

Expected effect of section 2 (inference): +0.00 to +0.04 lip correlation, and a cleaner trade-off between amplitude and sync. Its main value is telling how much of the gap is a sampler problem.

## 3. Clean the training pairs (preprocessing; GPU-hours)

**3a. Per-clip audio–video offset correction and a sync-confidence filter.** This has the strongest cross-field support of any preprocessing step, and SANG has not done it.
- LatentSync shifts each clip's audio by the SyncNet-estimated offset and drops clips with confidence below 3. Without the offset correction, "the model's convergence is significantly impaired" (training curves) ([LatentSync](https://arxiv.org/abs/2412.09262)).
- TalkVerse keeps only clips with |offset| ≤ 3 frames and confidence > 1.6, because "even a small amount of out-of-sync data degrades lip-sync learning" ([TalkVerse](https://arxiv.org/abs/2512.14938)).
- In video-to-audio generation, filtering training data by audio-visual match took sync error from 60 to 49 ms with 2.5× less training. Over-filtering hurt: 71 ms when about 75 % of the data was dropped ([V-AURA Tab. II](https://arxiv.org/abs/2409.13689)).

Why it matters for SANG (inference): a 1–2 frame offset in part of the data teaches the model blurred or shifted lip timing. That caps trajectory correlation directly, and no amount of model capacity recovers it.

How:
1. Run the existing `syncnet_python` harness, which reports an AV offset and a confidence per clip, on the real TalkVid clips.
2. Shift each clip's cached WavLM ticks by 2 × offset (WavLM runs at 50 Hz and video at 25 fps).
3. Sweep the confidence cut so that it removes 10 / 20 / 30 % of clips.
4. Keep the low-confidence clips for audio-dropped training (3c) instead of discarding them.

Cost: one SyncNet pass over 12.6k clips (a few hours) and no re-extraction of motion. Use the same SyncNet only for preprocessing, and evaluate with LSE plus non-SyncNet metrics.

**3b. Mask outlier frames and smooth the slow regions lightly.**
- Give frames with velocity jumps above k·σ, or face-detection failures, zero loss weight. AVTR-1 uses a finite-difference outlier mask (score > 0.15, dilated ±5 frames) ([AVTR-1](https://arxiv.org/abs/2609.22913)).
- Apply Savitzky–Golay smoothing (window 5, order 2) to eyes, brows and head, but not to lips, so plosive closures survive.
- StableFace measured that smoothing extracted coefficients raised a lip-stability index from 0.470 to 0.504 with Sync-C essentially unchanged (5.77 vs 5.80) ([StableFace](https://arxiv.org/abs/2208.13717)).
- The validation flow loss is measured on noisy targets, so cleaning them can move it either way. Judge the change on lip correlation and LSE.

**3c. Check the 42-d target for redundancy.**
- MARDM found that removing redundant, derived dimensions from a diffusion target was its largest single lever: FID 2.196 → 0.116 on HumanML3D ([MARDM](https://arxiv.org/abs/2411.16575)).
- SANG's 39 expression coordinates are 13 implicit keypoints × 3. Some z-coordinates or rigidly co-moving points may be near-duplicates.
- Compute the correlation matrix and PCA spectrum of the normalised target on the training set. If a handful of components carry almost no variance, train in a whitened PCA basis with those components dropped; UniTalker uses PCA output heads to balance targets ([UniTalker](https://arxiv.org/abs/2408.00762)).
- The renderer needs the inverse transform to be exact, so gate the change on reconstruction lip correlation ≥ 0.99.

**3d. Predict head placement too (3 more output dims).** SANG holds scale and translation from the photo.
- With real scale and translation, SANG's face PSNR rises from 16.8 to 24.5 dB.
- In IMTalker, ground-truth pose moved HDTF FID from 17.55 to 9.73 and FVD from 200.96 to 118.59 ([IMTalker](https://arxiv.org/abs/2511.22167)). This is the largest FID lever on the motion side.
- Ditto predicts scale and translation and smooths them with a 3-frame moving average; KDTalker uses a Kalman filter with t_z = 0 ([Ditto code](https://github.com/antgroup/ditto-talkinghead/blob/main/core/atomic_components/audio2motion.py), [KDTalker](https://arxiv.org/abs/2503.12963)).
- The 70-d cache already stores scale and translation, so no re-extraction is needed. Add Δt_x, Δt_y and Δlog-scale relative to the source (45-d output, t_z held) as a fifth region with its own loss share. Low-pass them at render time, and keep stitching on.
- Gate on background motion staying near 0.56×.

## 4. Free objective changes (one 20-minute retrain each)

With 20-minute retrains, SANG can run about 20 arms per GPU-day. The cost here is evaluation, not training. Run these as a small factorial on the cleaned data from section 3, one group at a time. LightningDiT found tricks interact: gradient clipping stopped helping once logit-normal sampling and a direction loss were on ([LightningDiT](https://arxiv.org/abs/2501.01423)).

**4a. Logit-normal time sampling.** The flag exists (`t_dist: logit_normal`). Add a shift m and sweep m ∈ {0, +0.3, +0.5} with s = 1.
- Measured elsewhere: LightningDiT FID 16.61 → 13.99; SD3's global rank 5.67 → 1.54 for logit-normal(0, 1) over uniform t ([SD3](https://arxiv.org/abs/2403.03206)).
- SD3 also shows a *narrow* distribution (s = 0.6) ranks first at 50 steps but 8.50th at 5 steps, so keep s = 1 at SANG's 10 steps.
- Changing the distribution re-weights the loss, so uniform-t validation loss is not comparable across arms.

**4b. Contrastive flow matching (ΔFM).** L = ‖v̂ − v‖² − λ‖v̂ − ṽ‖², where ṽ is the target of another clip in the batch. It needs no extra forward pass and no parameters.
- Image models: SiT-B/2 FID 42.28 → 33.39 at 130 M parameters, the scale closest to SANG. It stacked with REPA: 27.33 → 20.52 ([ΔFM](https://arxiv.org/abs/2506.05350)).
- The optimum is sharp: λ = 0.05 gave 7.29, 0.15 gave 19.21. Sweep {0.02, 0.05, 0.08}.
- In gesture generation, SemConFlow reports that replacing it with plain flow matching worsens FGD, beat consistency and diversity (no numbers retrieved) ([SemConFlow](https://arxiv.org/abs/2603.26553)).

Untested variant for SANG: apply it to the mouth dims, with negatives drawn from the *same speaker* in a different audio window. The repulsion then targets audio content, not identity, which builds into training the audio-specificity that mouth guidance supplies at inference. Expect higher amplitude, so re-tune γ.

**4c. A band-limited spectral loss on x̂0.**
- Add an L1 loss on the rFFT coefficients along time, per region, restricted to below about 10 Hz. At 64 frames and 25 fps the bins are 0.39 Hz.
- Syllable-rate lip motion is about 3–8 Hz. Above 10 Hz, extractor jitter dominates, and matching it would teach jitter.
- Weight it towards low noise (small t) and keep the total weight around 0.1–0.3 of the flow loss, because a deterministic loss on a stochastic generator pulls towards the mean.
- Measured elsewhere: FreDF cut forecasting MSE by 2.7–9.6 %. With 30 % of the data it matched the time-domain model trained on all of it ([FreDF](https://arxiv.org/abs/2402.02399)). SANG is data-limited.

**4d. A lip-aperture CCC loss at low noise** (untested anywhere).
- Map x̂0's mouth coordinates to lip opening with the existing linear `Readout` (`sang/naturalness.py`). Add 1 − CCC(opening(x̂0), opening(x0)) per window for t ≤ 0.3.
- CCC penalises timing and amplitude errors together, so unlike a pure sync loss it guards against exaggeration.
- It optimises the evaluation metric, so add a held-out metric (LSE, or the lip expert in section 6) to avoid Goodhart's law.

**4e. EMA selection.**
- EMA 0.995 has a half-life of about 138 steps.
- In EDM2, FID varied about 4× across EMA lengths (2.56 at the best length, 5.6–9.9 at others), and the optimum moved with guidance strength ([EDM2](https://arxiv.org/abs/2312.02696)).
- Keep three or four EMA copies (0.995 / 0.999 / 0.9995; ≈ 214 MB each) and pick by lip correlation under the guidance actually used. Re-pick whenever guidance changes.

**4f. Masked-span training.** With some probability, replace the fixed 10-frame clean prefix with a random clean span or random clean frames, and generate the rest.
- EMAGE's "masked hints" took gesture FGD from 6.833 to 5.423 ([EMAGE Tab. 6](https://arxiv.org/abs/2401.00374)).
- It generalises SANG's prefix and teaches continuation from any context. It also acts as a regulariser in the data-limited regime.

**4g. Modernise the block at fixed size.** These are architecture changes, not objective changes, but they need the same 20-minute retrain.
- Replace the learned absolute positions with **RoPE plus qk-norm**.
- Replace LayerNorm with **RMSNorm** (keeping adaLN modulation).
- Make the first FFN projection a **kernel-3 temporal convolution (ConvMLP)**, shrinking the hidden size to stay at 53.4 M.

Measured elsewhere:
- At fixed size in image DiTs, RoPE gave −2.12 FID and RMSNorm −0.85; SANG already has SwiGLU, worth −2.42 ([LightningDiT](https://arxiv.org/abs/2501.01423)). In JiT, RoPE plus qk-norm gave −0.79 ([JiT](https://arxiv.org/abs/2511.13720)).
- In video-to-audio generation, ConvMLP cut sync error from 0.533 to 0.483 s ([MMAudio Tab. 7](https://arxiv.org/abs/2412.15322)).
- GAIA's only motion-space backbone ablation found that a Conformer's convolution module, against a plain Transformer, improved Sync-D (9.312 → 8.913) and halved jitter (smoothness MSI 0.623 → 1.132) ([GAIA](https://arxiv.org/abs/2311.15230)).

**Not worth trying here:**
- **x0-prediction:** SANG's 42-d tokens sit far below its 512 width, the regime where all prediction targets are within about 0.2 FID ([JiT Tab. 2b](https://arxiv.org/abs/2511.13720)).
- **Dispersive loss:** a regulariser whose gains grow with model size.
- **MeanFlow, shortcut and consistency distillation:** they reduce latency, not error.
- **CFG-Zero\*:** helps only under-trained models.

Expected effect of section 4 (inference): +0.02 to +0.06 lip correlation from the group, mostly from 4b–4d, with smaller general-quality gains from 4a, 4e and 4g. The gains will not add.

## 5. A better audio stream (re-cache audio only; a few GPU-hours per arm)

FLOAT measured frame-wise adaLN beating cross-attention (LSE-D 7.290 vs 7.757), and GAIA measured additive speech beating cross-attention by about 1.3 Sync-D. In MMAudio, feeding sync features as attention tokens was no better than the per-token adaLN path (0.490 vs 0.483). SANG's injection is already right. The lever is *what* goes into it.

**5a. Use more than the last WavLM layer, or a multilingual encoder.**
- **Layer mix:** cache four WavLM-Large layers (for example 12, 16, 20, 24; about 20 GB each for 12.6k clips in fp16) and learn a softmax-weighted sum. The last layer is the most specific to WavLM's pre-training task.
- **Multilingual encoder:** swap in the Whisper-large-v3 encoder, XLS-R, or mHuBERT-147, plus a language embedding from TalkVid's labels.
  - UniTalker measured that swapping the audio encoder alone moves lip error by 10–20 %, with the 53-language XLSR-53 best on 5 of 8 test sets ([UniTalker Tab. 6](https://arxiv.org/abs/2408.00762)).
  - Fine-tuning the audio encoder alone raised THUNDER's lip correlation from 0.568 to 0.623 ([THUNDER](https://arxiv.org/abs/2504.13386)). A LoRA on WavLM's top layers is the parameter-light version, at the cost of running WavLM in the training loop.

**5b. Add a sync-trained second audio stream.** This is the largest single motion-space audio number found.
- In PC-Talk, on LivePortrait keypoints, replacing ASR-style features with an audio encoder "pretrain[ed] … on 2D audio-visual synchronization" raised LSE-C from 6.23 to **7.17** ([PC-Talk Tab. 3](https://arxiv.org/abs/2503.14295)).
- In person-specific renderers, a Wav2Lip-SyncNet audio encoder gave +10 % LSE-C over HuBERT, against +4.6 % for AV-HuBERT ([PASE](https://arxiv.org/abs/2504.05803)).
- In video-to-audio, sync-trained features injected per token halved sync error (0.973 → 0.483 s) ([MMAudio](https://arxiv.org/abs/2412.15322)).

Two ways to get the stream:
- **Ready-made:** the audio tower of the StableSyncNet already in `sang/syncnet.py` (LatentSync's), run on the mel spectrogram at frame rate.
- **Purpose-built and novel:** train a *motion-space* SyncNet on TalkVid that contrasts audio windows with 5-frame windows of the GT 42-d lip coordinates, with negatives at ±2–15 frame shifts within the same clip. Learn2Talk trained a similar mesh-space SyncNet in 12–15 h on one RTX 4090 ([Learn2Talk](https://arxiv.org/abs/2404.12888)).

Freeze the audio tower and add its frame features to the per-frame condition through a small projection (+0.5–1 M parameters). Use a different SyncNet for evaluation.

**5c. Give each frame a wider, refined audio view.** Replace the single k = 5 conv with a 2–4 block depthwise ConvNeXt or dilated stack covering ±8–12 frames (±320–480 ms), still inside the adaLN path.
- ARTalk's window sweep cut LVE from 11.73 to 9.34 as context grew from 8 to 100 frames ([ARTalk](https://arxiv.org/abs/2502.20323)).
- A phonetic-context lip-sync study found an optimum of about ±13 frames, or 1.2 s in total ([CALS](https://arxiv.org/abs/2305.19556)).
- F5-TTS's ConvNeXt refinement of the condition stream halved word error rate against an otherwise equal model (4.17 vs 9.63). A pure adaLN DiT without it "failed to learn alignment" ([F5-TTS](https://arxiv.org/abs/2410.06885)).
- Cost is about 1–2 M parameters, recovered by trimming the FFN.

**Not worth trying:** phoneme forced alignment across 15 languages. PD-GS's gated phoneme features gave only −2.6 % landmark distance ([PD-GS](https://arxiv.org/abs/2608.05218)). An explicit loudness/onset channel is near-free and can ride along with 5c.

Expected effect of section 5 (inference): +0.03 to +0.08 lip correlation, larger on non-English speakers if the language split in section 1 shows a gap. Run 5a and 5b one at a time so their effects can be told apart.

## 6. Lip supervision without rendering

**6a. A frozen, audio-aware lip expert in motion space** (the strongest measured lip-correlation lever). THUNDER trains a motion-to-speech regressor on pseudo-ground-truth face motion, freezes it, and during diffusion training asks it to recover the input speech representation from the *generated* mouth motion.
- With a frozen audio encoder, lip PCC rose from **0.568 to 0.639** and CCC from 0.359 to 0.426.
- As a plug-in it helped three other generators: 0.531 → 0.609, 0.612 → 0.653 and 0.57 → 0.614.
- Its cost was diversity: upper-face diversity fell from 0.0419 to 0.0322 ([THUNDER](https://arxiv.org/abs/2504.13386)).
- In 3D mesh models, lip-reading consistency losses cut lip error by about 9–14 % (SelfTalk 13.6 %, AV Guidance about 9 %) ([SelfTalk](https://arxiv.org/abs/2306.10799), [AV Guidance](https://arxiv.org/abs/2407.01034)).
- The expert must see **audio as well as lips** and be trained on large real data. AV Guidance's visual-only and from-scratch experts lost most of the gain.

For SANG:
1. Train a small temporal network (2–5 M parameters, outside the generator) on all cached TalkVid GT mouth coordinates to regress WavLM features or log-mel from 42-d lip and jaw motion.
2. As a feasibility probe, check its held-out R²: do 18 implicit lip coordinates carry enough speech information?
3. Freeze it, and add its loss on x̂0's mouth dims for small t only.

Expected (inference): +0.04 to +0.07 lip correlation, THUNDER's measured range. About 0.5 GPU-day for the expert, then ordinary retrains. The generator never sees the expert at inference, so SANG stays at 53.4 M.

Do not use a sync-style contrastive loss alone. In Learn2Talk, a motion-space SyncNet loss improved LSE but worsened LVE by 6–7 %: it learned to open "as quickly as possible", not correctly. A regression-to-speech loss avoids that.

**6b. Sync-REPA: align SANG's early features with a lip-reading encoder.** REPA aligns an early DiT block to a frozen encoder's features of the clean target, and it is the most replicated fixed-size gain of 2024–26.
- SiT-B/2 FID went from 33.0 to 24.4 ([REPA](https://arxiv.org/abs/2410.06940)).
- iREPA showed the gain comes from the *token-to-token similarity structure* of the target, and improved it with a conv projector and mean subtraction (SiT-B 49.50 → 43.37) ([iREPA](https://arxiv.org/abs/2512.10794)).
- For SANG the tokens are frames, and frame-to-frame structure is exactly what lip correlation measures.
- The closest cross-modal evidence: a video-to-audio flow DiT with REPA on frame-level audio features improved sync error from 0.79 to 0.75 ([HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930)).

How:
1. Extract frame-level features of a frozen lip-reading visual encoder (AV-HuBERT or Auto-AVSR front end) from the GT mouth crops once, offline.
2. Subtract each clip's temporal mean.
3. Align block 2 or 3 of 8 through a 1D-conv projector with per-frame negative cosine, λ = 0.5, on the 64 generated frames only.
4. Use one target encoder: combining two was worse than none in HunyuanVideo-Foley.

The projector is dropped at inference. No talking-head or motion paper has published this, so it is a novel contribution if it works. Risk: REPA's speed-up shrinks at small scale (about 2× at 130 M vs 17.5× at 675 M). Cost: about half a GPU-day of feature extraction.

**6c. A deterministic mouth anchor plus a residual flow** (only if section 1 shows a randomness limit).
- A small regressor (SubtleTalk used 4 transformer blocks at d = 256, about 3 M parameters) predicts the 18 mouth coordinates from audio with MSE plus velocity loss. The flow then models the residual.
- SubtleTalk measured this at LVE 14.60 (flow only) → 12.46 with the anchor, while improving diversity metrics. Its full model reaches 11.96 ([SubtleTalk Tab. 5](https://arxiv.org/abs/2608.06408)). GoHD's deterministic stage was worth 7.8 % mouth landmark distance ([GoHD](https://arxiv.org/abs/2412.09296)).
- ECHO (ACM MM 2026) published the same decomposition for 3D dyadic motion ([ECHO](https://arxiv.org/abs/2609.05506)), so the idea is no longer novel by itself. It is still the right fix if r_ss ≈ r_sy.

**Later, and only if needed: a sync loss through the renderer.** PC-Talk's SyncNet loss on frames rendered through the frozen keypoint renderer raised LSE-C from 7.17 to 8.92 ([PC-Talk](https://arxiv.org/abs/2503.14295)). That is the largest recipe number found, but it is above real video (8.24 in KDTalker's table) and evaluated with a related SyncNet, so part of it is likely metric gaming. It also puts the warp and decoder in the training loop. Try it only after 6a, with a different expert for the loss and for evaluation.

## 7. Scale the data (the lever that moves the plateau)

The 1.146 plateau, the −0.7 % from annealing, and the fact that 3× data removed overfitting all say SANG is data-bound at 53 M.
- GAIA, the only controlled hours × parameters table for speech → motion, gained Sync-D 9.145 → 8.913 from 10× data at 180 M parameters ([GAIA](https://arxiv.org/abs/2311.15230)).
- HY-Motion's broad pretraining (3,000 h of noisier motion) followed by a 400 h curated fine-tune beat curated-only training (3.20 vs 3.05) ([HY-Motion](https://arxiv.org/abs/2512.23464)).
- In gesture generation, adding motion-only data without speech still improved an audio-conditioned model (FGD 5.319 → 5.174) ([EMAGE Tab. 7](https://arxiv.org/abs/2401.00374)).
- The warning: adding a noisily labelled dataset was the one case where UniTalker's joint training *hurt*, by +6.5 % LVE.

The recipe, at fixed 53.4 M:
1. **Remove the extraction bottleneck.** The 90–125 s per clip is landmarks plus decoding on CPU. Batch LivePortrait's motion extractor (a small CNN on 256² crops) on the GPU, with decoding in parallel workers. Measure throughput before buying hours; an order-of-magnitude speed-up is plausible but unmeasured.
2. **Extract 300–1,000 h of TalkVid**, language-balanced, with the same filters and the offset correction from section 3.
3. **Pretrain broad:** every clip is used, but low-sync-confidence clips are trained *audio-dropped* (unconditional), so they teach motion statistics rather than wrong audio–motion pairs.
4. **Fine-tune** with full conditioning on the offset-corrected, sync-filtered subset.

Keep the model at about 53 M as requested. GAIA's parameter gain (180 M → 600 M: −0.31 Sync-D) was measured at 1 K h, and the 39–43 M motion generators (IMTalker, KDTalker) are competitive. This is the most expensive item, at days of engineering and extraction. It is also the most likely to move the validation loss, so start the extraction engineering in parallel with sections 2–6.

## 8. The renderer track: FID, FVD, teeth

None of the motion changes above can fix blur or teeth. LivePortrait's 512-px output comes from a PixelShuffle upsample of a 256-px input, and users report it is "blurrier than the original" even for small faces ([issue #427](https://github.com/KlingAIResearch/LivePortrait/issues/427)).

**8a. Fine-tune only the decoder.**
- Freeze the appearance encoder, motion extractor, warping network and stitching. Train the SPADE decoder in self-reenactment on 512-px talking-head crops with these losses:
  - L1;
  - VGG perceptual (global, face, mouth crop);
  - ArcFace identity;
  - a global GAN and a higher-resolution mouth-crop GAN, the latter optionally initialised from FFHQ StyleGAN2, as PersonaLive did ([PersonaLive](https://arxiv.org/abs/2512.11253));
  - a frame-difference or temporal term for FVD.
- Keypoints and warp are untouched, so lip shape, LSE and the DiT are unaffected by construction.
- No official training code exists; the maintainer says it is "challenging" ([#468](https://github.com/KlingAIResearch/LivePortrait/issues/468)). One community fine-tune fixed mouth artifacts with only qualitative evidence ([#506](https://github.com/KlingAIResearch/LivePortrait/issues/506)).
- No paper reports FID after such a fine-tune, so the gain must be measured. Cost: about 1–3 GPU-days (estimate) plus writing the training loop.
- Gate: the TalkVid ceiling stays at CSIM ≥ 0.906 and mouth correlation ≥ 0.82, and HDTF LSE moves by at most ±0.1.

**8b. Temporal super-resolution of the face crop before paste-back.**
- Use SeedVR2-3B (one step, Apache-2.0) ([SeedVR2](https://arxiv.org/abs/2506.05301)), or KEEP for internal tests. Never use per-frame CodeFormer or GFPGAN.
- On VFHQ, CodeFormer's identity similarity is 0.6272 against 0.7960 for the temporal KEEP ([KEEP](https://arxiv.org/abs/2408.05205)).
- No paper measures LSE after video SR, so measure it. No training is needed.

**8c. A teeth reference image.** Copying texture beats hallucinating it.
- Adding a second reference frame to a LivePortrait-style warper cut HDTF FID from 36.49 to 32.34 ([SynergyWarpNet](https://arxiv.org/abs/2512.17331)).
- A reference-mixing dubbing model reached the best HDTF FID among lip-sync methods, and its authors note quality "is ultimately bounded by the reference pool" ([EfficientSync](https://arxiv.org/abs/2608.18832)).
- Cheap test first: re-render the TalkVid ceiling set from a teeth-visible source frame and from a closed-mouth source frame, and compare mouth-crop FID and sharpness. If the gap is large, inject a second teeth-visible image into the fine-tuned decoder inside the mouth mask.

**A diffusion renderer is a research branch, not the next step.** Every independent table shows a lower FID but an identity drop:
- FRVD's warp-then-diffusion hybrid: HDTF FID 25.07 → 14.73, identity 0.9294 → 0.8975, about 4 minutes per 100 frames ([FRVD](https://arxiv.org/abs/2507.16341));
- PersonaLive's identity score is 0.698 against LivePortrait's 0.723.

None reports lip sync. Switching the renderer itself (IMTalker, LIA-X) means re-extracting all motion and retraining the DiT, so it is worth it only if 8a–8c stall.

## Parameter budget

Changes that stay inside the deployed model, against the user's "about the same parameters" constraint:

| Change | Deployed parameters | Training-only |
|---|---|---|
| Sections 2–4, 3a–3c, 4g (matched FFN) | 0 | – |
| 3d head placement (3 output dims) | ≈ 0 | – |
| 5a layer mix / language embedding | < 0.1 M | – |
| 5b sync audio stream (projection + frozen tower) | ≈ 0.5–1 M + tower | motion-SyncNet training |
| 5c audio refinement stack | ≈ 1–2 M (offset by FFN trim) | – |
| 2a autoguidance, small guide | ≈ 2–4 M | guide training (+3–5 %) |
| 6a lip expert, 6b REPA projector | 0 | expert 2–5 M, projector < 1 M |
| 6c deterministic mouth anchor | ≈ 3 M | – |

Everything fits in about 53–60 M, and the FFN can be trimmed to stay at 53.4 M.

## Which of these are publishable

The evaluation research found sync-reward post-training (ReFree, Ditto, Hallo4, FantasyTalking2) and cross-lingual transfer (MuEx) already crowded in 2026. The methods above that no talking-head or motion paper has published:

1. **Sync-REPA** for audio-to-motion generation (6b).
2. **Autoguidance** and **APG-decomposed** amplitude-vs-timing guidance in motion space (2a, 2b).
3. **Contrastive flow matching with same-speaker negatives** for lip specificity (4b).
4. **A motion-space, audio-aware lip expert on 2D implicit keypoints** (THUNDER is 3D FLAME) (6a).
5. **Broad audio-dropped pretraining on noisy clips, then a sync-filtered fine-tune**, reported as a data curve (7). No scaling curve for talking heads was found.

The strongest paper combines these with the diagnosis in section 1. SANG's decomposition can attribute the lip gap to randomness, audio representation, data offsets or the renderer, which end-to-end pixel models cannot do. That attribution, with the renderer-ceiling row and a pairwise human study (the evaluation notes recommend at least 30 raters and 1,000 judgments), is what reviewers will not have seen.

## Plan

Each step is gated on lip correlation, CCC, amplitude ratio, /p b m/ closure and audio_gain on unseen TalkVid speakers, plus HDTF LSE-C/D once the HDTF run exists. Keep an arm only if correlation or CCC rises without amplitude above 1.3× or audio_gain below 1.5.

| # | Step | Cost | Evidence (field) | Expected on SANG (inference) |
|---|---|---|---|---|
| 1 | Diagnose: r_ss and K-seed mouth averaging; lag histogram; lip corr by language; reference swap | hours, inference | Yang et al. (mesh); Soft-DTW eval; WavLM card | decides 6c vs 5 first, and the urgency of 3a |
| 2 | Sampling: autoguidance (small guide), APG split, guidance interval, Sway Sampling, NFE sweep | hours + one 1-minute guide | ImageNet, TTS, motion DiT | +0.00–0.04; amplitude and sync decoupled |
| 3 | Data: AV offset correction + sync filter; outlier masks; light non-lip smoothing; redundancy check; head-placement dims | hours + 1 retrain | lip-sync, V2A, talking face, text-to-motion | +0.01–0.04 lip corr; FID/PSNR from head placement |
| 4 | Objective grid: logit-normal t, ΔFM, spectral loss, lip CCC, EMA sweep, masked spans, RoPE/RMSNorm/ConvMLP | ~10 retrains × 20 min | ImageNet, forecasting, gesture, V2A | +0.02–0.06 |
| 5 | Audio: layer mix or multilingual encoder + language embedding; sync-trained second stream; wider refined window | re-cache audio (GPU-hours) + 1 retrain per arm | 3D face, LivePortrait keypoints (PC-Talk), V2A, TTS | +0.03–0.08 |
| 6 | Lip supervision: motion-space audio-aware expert loss; Sync-REPA; mouth anchor if step 1 says so | ~0.5–1 GPU-day + retrains | 3D face (THUNDER), ImageNet/V2A (REPA) | +0.04–0.07 |
| 7 | Data scale: GPU extraction, 300–1,000 h, audio-dropped pretraining, filtered fine-tune | days | text-to-motion, gesture, GAIA | moves the 1.146 plateau |
| 8 | Renderer: decoder fine-tune; temporal SR; teeth reference | 1–3 GPU-days + engineering | LivePortrait, video restoration, warping renderers | FID/FVD/teeth; CSIM must hold |

Steps 1–4 fit in one week on one GPU because retrains take 20 minutes. Step 5 is a week. Steps 6–8 run in parallel tracks over the following weeks. The gains will not add up. A plausible combined outcome for steps 2–6 is lip correlation rising from 0.64 to around 0.72–0.78, against a 0.82–0.85 ceiling (inference).

**On "breaking all SOTA".** Published HDTF numbers do not compare across papers: FLOAT's FID is 21.10 in its own table and 9.164 in IMTalker's, and LSE moves by 0.4 with the audio codec alone (THEval).
- **LSE-C/D:** steps 2–6 are what can put SANG ahead of the same-renderer systems (KDTalker 7.326 / 7.548; IMTalker LSE-C 7.711).
- **FID/FVD:** these need step 3d and step 8.
- **CSIM:** largely fixed by LivePortrait.

A claim to beat SOTA on all metrics needs SANG and the open baselines (KDTalker, Ditto, IMTalker, FLOAT where available) run through one protocol: `eval_hdtf.py` with a shared audio codec.

## Conclusion

SANG's lip deficit is mostly a timing problem, and timing is set by four things the model never had:
- clean audio–video alignment in its training pairs;
- an audio stream trained for synchrony and in the speaker's language;
- a loss that reads speech back from the generated lips;
- enough data to get past a 70-hour plateau.

All four are fixable at 53 M parameters. The ideas come from video-to-audio, TTS, image diffusion and 3D face animation, where comparable fixes halved sync error or added 0.07 lip correlation. First, a few hours of diagnosis will say whether the 0.64 is randomness or bias. That decides whether the mouth anchor or the audio stream comes first. Image quality is a separate problem with a separate fix in the decoder, and it can proceed in parallel.

## Update 2026-10-08: what the diagnosis and Phase A measured

The diagnosis this report asked for (section 1) has run, and the first fixes have been measured. Details and job IDs are in `docs/research_log.md` §3.5–3.7.

**The main cause was the data, not the model.**
- 30 % of TalkVid clips have audio and video off by 2 or more frames. The offset is constant within a source video (97 % of videos agree within one frame across clips) and goes both ways.
- The audio-feature probes found little to gain from other WavLM layers, Whisper or a wider window (+0.02 at most for a linear map). The audio front-end changes in section 5 move down.

**Offset correction (section 3a), measured on the same aligned val set.**
- Retraining on SyncNet-aligned audio raised lip correlation 0.687 → 0.722 and CCC 0.627 → 0.660. Timing became sharper: a 2-frame audio shift now costs +17.5 % mouth loss, against +10.6 % before.
- Scoring against aligned ground truth also lifts the reported number for any model by about 0.045.

**Sampler options (section 2), measured.**
- *Reducing mouth randomness* recovers half the remaining randomness gap at no cost. Averaging 4 samples' mouths, or starting the mouth from smaller noise (`tau=0.5`), each gives 0.759 correlation (from 0.722). `tau=0.5` also brings amplitude to 1.07.
- *Fewer steps:* 6 steps give +0.011 correlation, eye amplitude 1.09 instead of 1.22, and 40 % less time.
- *No gain:* Sway sampling and the guidance interval.
- *Worse:* autoguidance in every form tried. It inflates mouth amplitude (1.24–1.64) and lowers correlation; dropped.
- Lower mouth guidance loses more correlation than it gains in amplitude, so it stays at 1.25.

**Revised order.**
1. Combine the sampler winners (randomness removal can reach at most r_inf = 0.781).
2. The training-recipe batch (section 4) on aligned data.
3. The lip-expert loss (section 6a) and data scaling (section 7) for the remaining consistent error (about 0.22).
4. The deterministic mouth anchor (section 6c) now matters less: averaging and temperature already capture most of what it targets.
