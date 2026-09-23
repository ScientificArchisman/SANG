# SANG Recovery Plan — from below-copy-baseline to SOTA-class numbers on one GPU

**Date:** 2026-09-16 · **Status:** research + plan, diagnostic job 165645 attached (§2.6)
**Scope:** why job 165551 (resume of 163007) is stalled below the copy baseline, what the field
does that we do not, and a concrete, costed path to (a) reproduce a published baseline with our
own pipeline and (b) reach SOTA-class numbers on the standard talking-head protocol with 1× H100.
**Companion docs:** `docs/sota_review_2026.md` (landscape, verified citations),
`docs/v3_improvement_plan.md` (history of the token/flow tracks, defects D0–D13).

---

## 0. Verdict in one page

1. **The current run is not "slow"; it is at its ceiling.** `val_psnr` is flat at 15.0 dB from
   step 16k to 22k (best 15.08 @16k), 1.2 dB *below* freezing the context frame (16.25 dB) and
   the val-set latent MSE (0.27–0.29) is 3× the teacher-forced training MSE at the same
   noise levels (0.095 for t ≥ 0.5). The training loss keeps a slow downward drift, the
   validation metric does not. (§1)
2. **Three defects explain it, and none is a hyper-parameter.** (i) The SyncNet term sits at
   0.20, *below* the 0.42 that real video scores under the same mel and crop: a frozen
   discriminator driven through a differentiable decoder with full-backbone gradient is being
   satisfied by texture, not by lips. (ii) The prev-slice bridge is teacher-forced in training
   and self-fed at inference — classic exposure bias, now a named, solved problem in AR video
   diffusion (Self Forcing, Resampling Forcing). (iii) The evaluation metric (PSNR of one flow
   *sample* against ground truth, whole crop, free pose) cannot show progress even for a
   perfect model, and there is no FID/FVD/LSE harness to compare with any published number.
   (§2)
3. **The deeper problem is task choice.** A 96M-parameter transformer learning appearance
   *and* motion from scratch in Wan-VAE latent space is family C/D in the taxonomy of
   `sota_review_2026.md` without the pretrained prior that every family-C system relies on.
   The single-GPU frontier is family B: predict a compact motion code, render with a frozen
   pretrained renderer. The renderer alone gives 32.0 dB PSNR / 0.913 CSIM at 256 px in
   self-reenactment (LivePortrait, Tab. 2 of arXiv:2407.03168) — 17 dB above our model's
   output with *ground-truth* motion supplied. (§3)
4. **"Replicate baseline" has a concrete, one-GPU target.** KDTalker (IJCV 2025,
   arXiv:2503.12963, code + training code released): LivePortrait implicit-keypoint motion
   (70-d per frame) + a 43M-parameter diffusion model, trained on 4,282 VoxCeleb clips on
   **one RTX 4090**, reaches LSE-C 7.33 / LSE-D 7.55 on HDTF (real video: 8.24 / 6.93),
   FID 9.76, CSIM 0.949, 21.7 FPS. FLOAT (ICCV 2025) with the same design philosophy reaches
   FID 21.1 / FVD 162 / LSE-C 8.22 on HDTF on 1× A100. These are the numbers our pipeline
   must reproduce first. (§4)
5. **The plan (§5) keeps what SANG is about** — a *streaming*, block-causal, audio-conditioned
   generator with a MAR-style flow head — and moves it from appearance latents (1024 tokens ×
   16 ch per slice) to LivePortrait motion (1 token × 70 dims per frame). Everything that
   currently hurts disappears (no VAE decode in the loop, no pixel SyncNet, 100× shorter
   sequences), and the two things we can *own* in a paper become tractable on one GPU:
   exposure-bias-corrected streaming motion generation (Resampling Forcing in motion space —
   not done by Teller/FLOAT/KDTalker) and TalkVid-scale training with TalkVid-Bench subgroup
   analysis (1,244 h vs FLOAT's 11 h; KDTalker's ~4k clips).
6. **Cost.** Motion cache for all 25,873 local clips: ~10 GPU-hours. Training the motion
   model: hours, not days (KDTalker's 43M model; sequence of 80 tokens). Baseline
   reproduction gate (M1) is reachable inside one week of wall-clock including engineering;
   the full paper protocol (M3) inside four. (§6)
7. **Recommendation on job 165551:** stop it. Its best checkpoint is at step 20k
   (`runs/face256_wan_165551/best.pt`, val MSE 0.2732), 22k is worse, and the remaining 28k
   steps cost ~3.3 days of the H100 for a curve that has been flat for 6k steps. This is the
   user's call; nothing in this document cancels it.

---

## 1. State of the current run (facts)

Config: `configs/train.yaml` as committed (continuous Wan-VAE track, 256 px face crop,
17 frames = 5 content slices of 32×32 latents, `motion_ctx`, `bridge_init: prev`,
`syncnet_weight 0.3`, `pixel_weight 1.0`, batch 8 × accum 8, lr 3e-4 cosine to 3e-5).

### 1.1 Validation trajectory (jobs 163007 → 165551, same run resumed at 16k)

| step | val latent MSE | val_psnr (dB) | note |
|---|---|---|---|
| 2,000 | 0.4334 | 12.91 | 2.4 dB above the wrong-clip floor (10.5) |
| 4,000 | 0.3825 | 13.69 | |
| 6,000 | 0.3489 | 14.00 | |
| 8,000 | 0.3458 | 13.87 | |
| 10,000 | 0.3077 | 14.37 | |
| 12,000 | 0.3060 | 14.50 | |
| 14,000 | 0.3034 | 14.56 | |
| 16,000 | 0.2854 | **15.08** | end of 163007; resumed as 165551 |
| 18,000 | 0.2824 | 15.07 | |
| 20,000 | 0.2732 | 15.07 | best MSE |
| 22,000 | 0.2885 | 14.94 | worse |

Yardsticks measured on the same split (memory `sang-psnr-baselines`, 2026-09-11):
freeze-ctx **16.25 dB / MSE 0.300**, freeze-ref 16.19 dB / 0.278, zero-latent MSE 0.526,
wrong-clip floor 10.48 dB.

So after 22k steps (≈ 2.6 GPU-days on the H100 at ~10 s/step) the model's *sample* is
1.2 dB worse than copying a frame from elsewhere in the clip, and its latent MSE only just
matches freeze-ref. PSNR has not moved since step 16k.

### 1.2 Training signals at the plateau (steps 16k–22k, log `results/train/165551.txt`)

| term | value | what it means |
|---|---|---|
| `diff` (flow v-MSE, all t) | 0.27 → 0.26 | slow drift, no break |
| `diff_lo` (t < 0.5) | 0.44 | dominated by irreducible noise-prediction error |
| `diff_hi` (t ≥ 0.5) | **0.095** | ≈ teacher-forced conditional variance of z0 given GT prev slice + audio |
| `pixel` (face/lip-weighted L1, one decoded example) | 0.086 | flat |
| `syncnet` (cosine, one decoded example) | **0.20** | *below* the ground-truth in-sync value 0.42 (§2.2) |
| gnorm | 0.3–0.8 | healthy |
| memory | 44.6 GB | as budgeted |

Key contrast: **train-time x0 error at high noise ≈ 0.095, sampled val error ≈ 0.27–0.29.**
The model can predict the next slice well *when handed the true previous slice*; it cannot
when handed its own previous prediction. That is the exposure-bias signature (§2.3), not
under-fitting.

---

## 2. Diagnosis

### 2.1 The metric cannot show success, only failure

`evaluate_continuous` reports PSNR between **one stochastic flow sample** and ground truth on
the whole 256 px crop under free head pose. For a perfect generative model with conditional
variance σ² per latent, E‖x − x'‖² between an independent sample and the truth is 2σ², i.e.
*worse* than the conditional mean by construction. The literature therefore reports PSNR only
under pose-given / inpainting protocols (memory `psnr-anchors-talking-head`, Protocol A/C);
free-pose one-shot systems report FID, FVD, CSIM, LSE-C/LSE-D. We have none of those
implemented (`sang/metrics.py` has PSNR/SSIM only; the `eval_metrics.py` of
`docs/idea.md` Change 12 was never written).

Consequence: "we do not beat baseline" is currently a statement about a metric that a correct
model would also fail. It is still true that 15.0 dB < 16.25 dB is bad — the *mean* of the flow
should beat copying — but the number cannot be used to steer, and it cannot be put in a paper.

### 2.2 The SyncNet term is being gamed

Evidence:
- Ground truth, same mel cut (MEL_VERSION 3, +60 ms), same 1.3× zoom and lower-half crop:
  **in-sync 0.42, shuffled 0.90** on 28 windows (memory `syncnet-offset-and-floor`).
- Training `syncnet` term: **0.20** from step ~16k on. The term is computed on the one-step
  x0 estimate averaged over t ~ U(0,1); at small t that estimate is ≈ ground truth (≈ 0.42),
  so the model's own predictions at larger t must score *well below* 0.20 — better than real
  video by a wide margin.

A frozen network cannot be beaten by real data unless the generator has found inputs the
network likes that are not lips: adversarial texture in the mouth band. This is exactly the
failure the lip-sync literature warns about — Wav2Lip keeps its lip-sync expert frozen and
*never* fine-tunes it on generated frames because a discriminator exposed to generator output
starts scoring artifacts instead of sync; StableSync (ECCV 2024, arXiv:2307.09368) is a paper
about the *instability of the lip-sync loss* itself; and LatentSync (arXiv:2412.09262 §3.2)
only applies its SyncNet loss in a **second stage in which the U-Net is frozen except the
temporal and audio layers**, at λ₂ alongside LPIPS and TREPA. We instead back-propagate
`0.3 × SyncNet` through the Wan decoder into the *entire* 96M backbone and head from step 1.
The face-weighted L1 (`pixel_weight 1.0`, one decoded example per micro-batch) is too weak to
stop it.

Job 165645 (§2.6) measures this directly: SyncNet on generated video with the matched mel vs a
shuffled mel. If both are low, the loss is non-discriminative on our outputs and has been
gamed.

### 2.3 Exposure bias across slices

Training (`forward_continuous`): every content slice's input embedding is the **ground-truth
previous slice** (`bridge_init: prev`, teacher forcing). Inference (`generate_continuous`):
the previous slice is the model's own sample. With 5 slices per window and windows chained at
demo time, errors compound. This is the standard failure of teacher-forced AR video diffusion
and 2025–26 has produced the fixes:

- **Self Forcing** (arXiv:2506.08009): roll out with KV cache during training and supervise
  on self-generated context (needs a bidirectional teacher / distillation).
- **Resampling Forcing** (arXiv:2512.15702, ByteDance/CUHK): *teacher-free, from scratch*.
  Warm up with teacher forcing; then, per step, corrupt the history frames to
  t_s ~ LogitNormal(0,1) with time-shift s = 0.6, re-denoise them **with the online model,
  1 Euler step, no gradient**, and train the per-frame flow loss conditioned on that degraded
  history with clean targets. Reported: quality comparable to distilled models and better
  long-horizon stability than Self Forcing. This is a ~30-line change to `forward_continuous`
  and carries over unchanged to the motion-space model (§5.4).
- Our own `context_corrupt` (random token replacement) is the discrete-track analogue and is
  off on this track (`p_corrupt` unused in `forward_continuous`).

### 2.4 The task is mis-specified for the budget

From `docs/sota_review_2026.md` §1: family C (appearance latents) wins only as a fine-tune of a
pretrained video prior on 32 GPUs; family D (appearance tokens from scratch) is where v3
stalled; family B (motion code + frozen renderer) is the single-GPU frontier. v4 fixed the
*tokenizer* (Wan VAE 31.8 dB ceiling) but kept the from-scratch appearance generator. The
run is now demonstrating, at 2.6 GPU-days, that 96M parameters and 11k clips are not enough
to learn appearance + motion + audio-visual correlation at 256 px.

What the renderer route gives away for free (LivePortrait, arXiv:2407.03168 Tab. 2,
self-reenactment at 256 px, TalkingHead-1KH): PSNR **32.01**, SSIM 0.819, LPIPS 0.066,
CSIM 0.913 — with ground-truth motion. The audio→motion model then only has to reproduce a
70-dimensional trajectory. KDTalker's ablation (arXiv:2503.12963 Tab. 5) shows that this
renderer, driven by predicted keypoints, still yields LSE-C 7.33 vs Face-vid2vid's 5.58, i.e.
the renderer's lip fidelity is *not* the bottleneck.

### 2.5 Secondary issues (still worth fixing on any track)

- **Window length.** 17 frames = 0.64 s. FLOAT trains on 50 + 10 preceding frames (2.4 s),
  KDTalker on 64 frames, Teller on 200 ms chunks *with* unbounded AR context. Sub-second
  windows cannot learn head-motion rhythm or co-articulation beyond ±80 ms. On the
  appearance track this was forced by memory (7,168 tokens per window); in motion space it
  costs nothing to go to 100 frames.
- **Audio features.** WavLM-large final layer. Teller found Whisper-encoder features vs a TTS
  codec changed Sync-C from 4.29 to 7.70; FLOAT uses wav2vec2; KDTalker the Wav2Lip mel
  encoder. WavLM is a reasonable ASR-grade choice; keep it, but make the layer a config knob
  (LatentSync/Hallo use multi-layer Whisper features).
- **Data filtering** exists (`scripts/filter_clips.py`, 16,961 of 25,873 clips) but the run
  uses the unfiltered glob capped at `max_clips: 12000`. Every comparable system trains on
  filtered data (Teller: Sync-C/Sync-D screen + <50 % face motion; TalkVid's own pipeline).
- **The 40 ms container start-time** offset on ~9 % of TalkVid clips (memory note) skews the
  WavLM window on those clips. Fix in `_audio_window` when the cache is rebuilt.

### 2.6 Diagnostic job 165645 (`scripts/diagnose_flow.py`, new)

Runs on 24 unseen-speaker val windows with `runs/face256_wan_165551/best.pt` (step 20k):
SyncNet on GT / generated+matched mel / generated+shuffled mel / freeze-ctx / one-step x0 at
t = 0.25, 0.5, 0.75; latent MSE of the sequential sample vs teacher-forced x0 at four t;
PSNR vs freeze-ctx; inter-slice latent motion and mouth-band pixel motion, generated vs GT.
Results are appended in §9 (Progress log) when the job finishes; the reading guide is in the
script docstring.

---

## 3. What "baseline" should mean from now on

Three tiers, all measurable with one harness (§5.6):

| tier | baseline | what it proves | numbers to match |
|---|---|---|---|
| B0 | copy baselines (freeze-ctx / freeze-ref) | the model does *something* | PSNR > 16.3 dB on our split — necessary, not sufficient |
| B1 | **KDTalker recipe reproduced with our pipeline** (LivePortrait motion + diffusion/flow, non-causal, 64 frames) on our data, evaluated on HDTF-test | our data + code + eval reach a published single-GPU result | HDTF: LSE-C ≈ 7.3, LSE-D ≈ 7.5, CSIM ≈ 0.95, FID ≈ 10 (KDTalker Tab. 1; protocol: first 8 s of 349 HDTF videos, first frame as reference, 256 px) |
| B2 | released checkpoints run through *our* harness: KDTalker, FLOAT (inference-only, CC BY-NC-ND), SadTalker | the harness itself is calibrated | FLOAT own-paper HDTF: FID 21.10, FVD 162.05, CSIM 0.843, LSE-D 7.29, LSE-C 8.22; Teller HDTF: FID 21.35, FVD 173.46, Sync-C 7.70, Sync-D 7.54; real video Sync-C 8.09 / Sync-D 6.98 |
| SOTA claim | SANG-M (streaming + resampling forcing + TalkVid) | contribution | ≥ FLOAT/Teller on LSE-C/D and FVD on HDTF; best on TalkVid-Bench subgroups; real-time causal |

Protocol caveats (from `sota_review_2026.md` §2): FID for the *same* method varies 3× between
papers (SadTalker 21.6 vs 72.0) because of test split, crop and frame count. Every number in
the paper must come from our harness with baselines re-run, plus the published numbers quoted
as such.

---

## 4. Literature that decides the design (verified 2026-09-16)

| work | what we take | evidence |
|---|---|---|
| **LivePortrait** (arXiv:2407.03168; code MIT, weights `KlingTeam/LivePortrait` on HF; InsightFace detector non-commercial) | frozen motion extractor + renderer. Motion of a frame = `{scale [1], pitch/yaw/roll [1 each, degrees], t [3], exp [21×3]}`; driving kp `x_d = s·(x_c R + δ) + t`; `warp_decode(f_s, x_s, x_d)` → 256 px frame; stitching MLP for paste-back; 12.8 ms/frame on a 4090; trained on 69M frames | wrapper API read from `src/live_portrait_wrapper.py`; crop config `dsize 512, scale 2.3, vy_ratio −0.125` (source) / `2.2, −0.1` (driving) |
| **KDTalker** (IJCV 2025, arXiv:2503.12963; code+train CC BY-NC 4.0) | the 70-d motion vector `[scale, yaw, pitch, roll, t(3), exp(63)]`, per-dim normalisation, relative-motion retargeting at inference (`R_new = R_d R_d0ᵀ R_s`, `δ_new = δ_s + (δ_d − δ_d0)`, …), 64-frame windows, batch 256, DDIM 50 → 21.7 FPS; 43M params, one 4090 | paper §4.1, `inference.py`, `dataset_process/extract_motion_dataset.py` |
| **FLOAT** (ICCV 2025, arXiv:2412.01064; inference code only, CC BY-NC-ND) | flow matching in motion space; frame-wise AdaLN transformer (h = 1024, 8 heads, window T = 2); 50 + 10 preceding frames; audio CFG γ_a = 2; ~10 NFE; dropout 0.1 on conditions and **0.5 on the preceding-window context**; velocity (smoothness) loss λ_vel | paper §4.2, §5.1, App. B |
| **Teller** (CVPR 2025, arXiv:2503.18429) | proof that a *streaming AR* model over LivePortrait motion (25×3 latent, RVQ 4 frames→32 tokens) reaches Sync-C 7.70 / FVD 173 — but with a 4B AR LM on 64 A800s and 662 h. We keep the streaming design, replace RVQ+CE with a continuous flow head (our MAR-style head), and add exposure-bias correction | paper §3, Tab. 1 |
| **Resampling Forcing** (arXiv:2512.15702) | teacher-free exposure-bias fix for AR flow models: warm-up, then self-resampled history (t_s ~ LogitNormal, shift 0.6, 1 Euler step, detached) | paper §3.2, Alg. 1 |
| **LatentSync** (arXiv:2412.09262) | how SyncNet supervision is *supposed* to be applied (stage 2 only, frozen backbone, +LPIPS +TREPA); StableSyncNet convergence conditions (batch 1024, 16 frames, offset-corrected data). Explains §2.2 | paper §3.1–3.3, §4 |
| **TalkVid** (arXiv:2508.13618) | TalkVid-Bench: 500 five-second held-out clips stratified over language / ethnicity / gender / age (folder `TalkVid-bench` in `FreedomIntelligence/TalkVid` on HF). Local metadata already carries Ethnicity / Age Group / Gender / Language per clip (188,127 clips, 7,730 speakers, 1,244 h), so a stratified *dev* split can be built today. V-Express fine-tuned on TalkVid-Core: HDTF-100 FID 21.8 / FVD 175 / Sync-C 3.71 / Sync-D 9.97 (3 days × 4 A100) | paper §4, Tab. 5–6; local `metadata/filtered_video_clips.json` |
| **HDTF** (github MRzzm/HDTF, CC BY 4.0) | standard test set; distributed as YouTube URLs + crop boxes; KDTalker protocol: first 8 s of 349 videos, 256 px | repo README |
| **syncnet_python** (joonson) | LSE-C = "Confidence", LSE-D = "Min dist" from `run_pipeline.py` + `run_syncnet.py`; the numbers every paper above reports | repo README |

Not applicable at our scale (kept for the related-work section): SoulX-FlashHead (32× H20),
MultiTalk/Live Avatar (14B), Hallo3/Sonic/EchoMimic (SD/Wan fine-tunes), EARTalking, Lumos-1.
FLOAT's *training* code is not released and its licence is NoDerivatives, so its motion
autoencoder is usable only as a black-box baseline, not as our renderer — hence LivePortrait.

---

## 5. The plan: SANG-M — streaming motion-latent flow matching with a frozen renderer

### 5.1 Design in one figure

```
                 reference image ─► LivePortrait crop (512, s=2.3) ─► M(·) ─► m_0 (70-d)   x_s, f_s (frozen)
                                                                                 │
 audio 16 kHz ─► WavLM-large (50 Hz) ─► proj ─┐                                  ▼
                                              ▼                    ┌──────────────────────────────┐
 previous chunk(s) of motion  m_{<k} ──► block-causal transformer  │ chunk k = 4 frames = 4 tokens │
 (self-generated at inference,          (self-attn: causal over    │ each token: 70-d motion       │
  resampled at training §5.4)            chunks, bidirectional     │ + adaLN flow head (MAR-style) │
                                         inside a chunk;           └──────────────┬───────────────┘
                                         cross-attn: audio ticks                  ▼
                                         up to chunk end + lookahead)   m̂_k  ──► relative retarget ──► x_d
                                                                                              │
                                                                       warp_decode(f_s, x_s, x_d) ─► frame
```

The generator is the existing `StreamingBlockTransformer` + `DiffusionHead` with three
substitutions: a token is one *frame's* 70-d motion vector instead of one 16-d latent cell,
a slice is a chunk of 4 frames (`r = 4`) instead of a 32×32 grid (`r = 1024`), and the
conditioning slices are `[m_0 of the reference, previous-chunk context]` as today. Everything
downstream of the head (VAE decode, pixel losses, SyncNet) is deleted from the training loop.

### 5.2 Representation and cache

- Per clip (25 fps): LivePortrait crop from a **smoothed per-clip face box** (their
  `crop_image` with `dsize 512, scale 2.3, vy_ratio −0.125`, landmarks from `landmark.onnx`
  after the InsightFace detector; one box per second, linearly interpolated, so the crop is
  stable and identical to the source-image crop at inference), resized to 256, `/255`,
  through `motion_extractor` → `kp_info`. Store `m_t = [scale, yaw, pitch, roll, t_x, t_y,
  t_z, exp(63)]` as fp16 `[T, 70]`, plus `kp` (canonical, `[21,3]`) of frame 0 for
  rendering checks, plus the WavLM features `[Ta, 1024]` at 50 Hz for the whole clip, plus
  the crop boxes. ~1 KB/frame; all 25,873 clips ≈ 12 GB.
- **Relative motion**, as KDTalker at inference and as a training target: predict
  `Δm_t = m_t − m_0` (per-dim standardised with dataset mean/std, `cal_norm`), where `m_0` is
  the reference frame's motion. This makes the target identity-agnostic and makes the
  reference frame a natural "rest pose" condition.
- Cost: detector+landmark ~15 ms/frame at 1 fps of boxes, motion extractor ~2 ms/frame
  batched → ~1.5 ms/frame amortised → 11.6 M frames ≈ **5–10 GPU-hours** for everything.
  Start with `data/clips_filtered_all.txt` (16,961 clips).
- Renderer-ceiling check (M0 gate): re-render 50 val clips from *their own* motion with the
  first frame as source; expect PSNR ≈ 30 dB, LSE-C within 0.5 of the real clip. This is the
  motion-space analogue of `scripts/ceiling.py` and bounds everything that follows.

### 5.3 Model

| | v4 (now) | SANG-M |
|---|---|---|
| token | 16-d Wan latent cell | 70-d LivePortrait motion (standardised Δm) |
| tokens per slice `r` | 1024 (32×32) | 4 (frames per chunk) |
| slices per window | 2 cond + 5 content (0.64 s) | 2 cond + 16–25 content (2.6–4 s) |
| sequence length | 7,168 | 72–104 |
| positional emb. | slice + row + col | slice (chunk) + intra-chunk frame index |
| backbone | 8 × 512-d, 8 heads | same class; start 8 × 512 (≈ 30 M) |
| head | MAR adaLN MLP, hidden 1024 × 6 | same, `z_ch = 70` |
| audio | WavLM-large, lookahead 4 ticks (80 ms) | same; lookahead 5 ticks (100 ms) |
| batch | 8 × accum 8 | 256, no accum (memory is trivial) |
| params | 96 M | ~35 M |

`StreamingBlockTransformer` needs one generalisation: `r` is no longer a square grid, so the
positional embedding becomes `slice_emb(slice) + frame_emb(intra)` (see §8.3). `block_masks`
and `slice_ticks` are unchanged (they only use `tv`, `r`, `ta`).

### 5.4 Training objective

1. **Flow matching** on standardised `Δm` with the existing v-prediction head
   (`sang/diffusion.py`). Add FLOAT's velocity term
   `λ_vel · ‖(x̂_0[t+1] − x̂_0[t]) − (x_0[t+1] − x_0[t])‖²` on the one-step x0 estimate
   (λ_vel = 1) — this is a *motion-space* temporal loss and replaces TREPA.
2. **Condition dropout for CFG**: audio 0.1 (keep `audio_dropout`), reference motion 0.1,
   preceding-chunk context 0.5 (FLOAT's number; it is what makes the first window of a
   stream well-behaved).
3. **Exposure-bias correction — Resampling Forcing** (`arXiv:2512.15702` Alg. 1), after a
   teacher-forcing warm-up of ~10 % of the steps: for each training window, run the model
   *without gradient* chunk by chunk to produce a degraded history
   `m̃_k = Euler_1step( (1−t_s)·m_k + t_s·ε ; context m̃_{<k} )`, `t_s ~ LogitNormal(0,1)`
   shifted by `s = 0.6`, and compute the flow loss on the clean `m_k` conditioned on
   `m̃_{<k}`. Two extra no-grad forwards per step at 100 tokens is negligible. This is the
   piece none of FLOAT / KDTalker / Teller has and is the first contribution claim (§7).
4. **No pixel-space loss and no SyncNet loss.** FLOAT and KDTalker reach real-video-level
   LSE-C without one; the audio-visual correlation is learned because the target is 63
   expression dims dominated by the mouth, not 16k latent values dominated by skin. If lip
   precision needs a push later, the safe option is a *motion-space* contrastive audio↔`exp`
   loss trained on our own cache (DEMO, arXiv:2510.10650, does this with InfoNCE inside the
   motion autoencoder), never a pixel discriminator through a decoder.
5. Optimiser: AdamW, lr 3e-4 → 3e-5 cosine, wd 0.1, betas (0.9, 0.95), grad-clip 1.0,
   bf16 — i.e. the current recipe, batch 256, 200k steps (≈ 6–8 h on the H100 at 100 tokens).

### 5.5 Inference

Sequential over chunks exactly as `generate_continuous`, Euler with 10–12 NFE, audio CFG
γ_a = 2, previous chunk = own sample, KV cache optional (100 tokens; not needed for real
time). Then per frame: de-standardise, `m_t = m_0 + Δm_t`, KDTalker's relative retargeting
against the source `kp_info`, `x_d = s·(x_c R + δ) + t`, `stitching`, `warp_decode`,
paste-back with `M_c2o`. Expected throughput: renderer 12.8 ms/frame on a 4090, generator
< 1 ms/frame → real-time with ≥ 160 ms audio lookahead.

### 5.6 Evaluation harness (`scripts/eval_full.py`, new — the missing Change 12)

- Inputs: a directory of generated mp4s + the matching ground-truth mp4s, 25 fps, 256 px
  face crops.
- **FID** (pytorch-fid, Inception-v3, all frames), **FVD-16** (I3D features, 16-frame
  sliding windows, as FLOAT), **CSIM** (ArcFace cosine between source frame and every
  generated frame; InsightFace `buffalo_l`, already installed for LivePortrait), **LSE-C /
  LSE-D** (syncnet_python `run_pipeline.py` → `run_syncnet.py`; Confidence and Min dist),
  E-FID optional later.
- **Test sets:** (a) HDTF, KDTalker protocol (349 videos, first 8 s, first frame as source;
  download via the HDTF repo lists, CC BY 4.0); (b) TalkVid-Bench (500 × 5 s, from the HF
  `TalkVid-bench` folder), reported per subgroup; (c) our stratified TalkVid *dev* split
  built from local metadata for daily use.
- **Baselines through the same harness:** KDTalker (released weights), FLOAT (released
  `float.pth`, inference only), SadTalker, plus real video for the LSE ceiling. Each is a few
  GPU-hours.
- Keep `val_psnr` only as a smoke test against B0; add the *mean* prediction (x0 at
  t = 1 with the ODE, or NFE = 1) so the copy baseline can actually be beaten in-run.

### 5.7 What stays from the current codebase

`sang/streaming_transformer.py` (backbone, masks, generate loop), `sang/diffusion.py` (head,
flow loss, sampler), `sang/codec.py` (WavLM), `sang/data.py` (window geometry, audio cut,
pooled cache build), `scripts/train.py` skeleton, `scripts/filter_clips.py`. The Wan-VAE
track, `sang/syncnet.py`, `sang/losses.py`, `sang/face.py`-based crops stay as the
*appearance refiner* option already sketched in `docs/v3_improvement_plan.md` — not on the
critical path.

---

## 6. Milestones, gates, cost

| # | deliverable | GPU cost | gate (must pass to continue) |
|---|---|---|---|
| M0 (days 1–3) | LivePortrait installed in `avcodec` (or a sibling env), `sang/motion.py`, motion cache for 2k clips, renderer-ceiling script | 2 h | re-rendered GT motion: PSNR ≥ 29 dB @256, LSE-C ≥ real − 0.5 on 50 val clips |
| M1 (days 4–8) | **B1 reproduction**: non-causal 64-frame flow model (KDTalker/FLOAT recipe) on filtered TalkVid; `eval_full.py`; HDTF-349 downloaded | cache 10 h + train 8 h + eval 2 h | HDTF: LSE-C ≥ 6.5, LSE-D ≤ 8.0, CSIM ≥ 0.90, FID ≤ 15 (KDTalker: 7.33 / 7.55 / 0.949 / 9.76). Anything below LSE-C 5 means a data/crop/alignment bug, not a model problem |
| M2 (days 9–14) | SANG-M streaming: block-causal chunks + Resampling Forcing; KV-cached real-time demo | 2 × 8 h | causal model within 0.3 LSE-C and 10 % FVD of M1; drift-free 60 s rollouts (LSE-C in the last 10 s ≥ first 10 s − 0.3) |
| M3 (weeks 3–4) | full TalkVid (all 25.9k clips, filtered), TalkVid-Bench subgroup table, baselines B2 through the harness, ablations (resampling forcing on/off, context dropout, lookahead, chunk size, audio layer) | ~60 h total | ≥ FLOAT/Teller on LSE-C/D and FVD on HDTF; best or tied on TalkVid-Bench subgroups |
| M4 (optional) | appearance refiner on top of the rendered frames (the Wan-VAE track as a *second stage*: mouth-interior + teeth), only if FID vs family-C is the reviewers' objection | days | FID −20 % without LSE-C loss |

Wall-clock is dominated by engineering and the HDTF/TalkVid-Bench downloads, not by GPU
time: the whole M0–M3 GPU budget (~100 h) is less than the current run's planned 5.8 days.

---

## 7. Paper positioning, risks

**Contribution claims that are defensible on one GPU**
1. *Streaming motion-latent flow matching with exposure-bias correction.* Teller is streaming
   but discrete (RVQ + 4B LM, teacher-forced); FLOAT/KDTalker/DEMO are non-causal windows.
   Resampling Forcing in motion space, with a drift analysis over minute-long rollouts, is
   new in this sub-field.
2. *TalkVid-scale training and TalkVid-Bench subgroup results.* 1,244 h / 7.7k speakers vs
   FLOAT's 11.3 h / 230 identities and KDTalker's 4.3k VoxCeleb clips; the benchmark paper's
   own finding is that aggregate metrics hide subgroup failures (their Tab. 4).
3. *A quantified negative result on pixel-space SyncNet supervision for from-scratch latent
   generators* (§2.2 + job 165645), if reviewers want it; otherwise an appendix.

**Risks and mitigations**
- *Renderer ceiling.* Warping renderers cap FID (~20 vs ~10 for 32-GPU family-C) and mouth
  interior. Position on lip-sync, motion naturalness, efficiency, subgroup robustness; M4
  refiner if needed.
- *Crop/alignment mismatch* between our TalkVid cache and LivePortrait's expectations is the
  most likely way to fail M1. Use LivePortrait's own cropper end-to-end; verify with M0.
- *Licensing.* LivePortrait code MIT; InsightFace detector non-commercial (fine for a paper;
  swap for MediaPipe boxes if ever needed — we already have `sang/face.py`). KDTalker code
  CC BY-NC (read, do not copy). FLOAT CC BY-NC-ND: inference-only baseline. HDTF CC BY 4.0.
  TalkVid released as URLs under a research licence.
- *Data hygiene.* Speaker-disjoint splits as now; check that HDTF identities do not appear in
  TalkVid (both are YouTube-sourced) by video-ID intersection before reporting.

---

## 8. Code

Everything below is written against the repo as of commit d46b32f plus the uncommitted
working tree. Names of LivePortrait symbols follow `src/live_portrait_wrapper.py` and
`src/utils/cropper.py` of the upstream repo (read 2026-09-16); adjust on install if their
API moved.

### 8.1 Install (login node has GitHub + HF access; `avcodec` has torch 2.4.1+cu121)

```bash
cd /beegfs/work/achakraborti/SANG
git clone https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait
source ~/miniconda3/etc/profile.d/conda.sh && conda activate avcodec
pip install onnxruntime-gpu insightface tyro pykalman scikit-image imageio[ffmpeg]   # LivePortrait extras
huggingface-cli download KlingTeam/LivePortrait --local-dir third_party/LivePortrait/pretrained_weights \
    --exclude "*animal*"
# eval tooling
git clone https://github.com/joonson/syncnet_python third_party/syncnet_python && (cd third_party/syncnet_python && sh download_model.sh)
pip install pytorch-fid
```

### 8.2 `sang/motion.py` — frozen LivePortrait motion codec (new)

```python
"""LivePortrait as a frozen motion codec: frames -> 70-d motion vectors -> frames.

m = [scale, yaw, pitch, roll, t_x, t_y, t_z, exp_1..exp_63]   (KDTalker's layout)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
LP = REPO / "third_party" / "LivePortrait"
sys.path.insert(0, str(LP))
from src.config.inference_config import InferenceConfig            # noqa: E402
from src.config.crop_config import CropConfig                      # noqa: E402
from src.live_portrait_wrapper import LivePortraitWrapper          # noqa: E402
from src.utils.cropper import Cropper                              # noqa: E402
from src.utils.camera import get_rotation_matrix                   # noqa: E402

M_DIM = 70
KEYS = ("scale", "yaw", "pitch", "roll", "t", "exp")


def kp_info_to_vec(kp: dict) -> torch.Tensor:
    """kp_info (refined: angles in degrees) -> [B, 70]."""
    return torch.cat([kp["scale"].reshape(-1, 1), kp["yaw"].reshape(-1, 1), kp["pitch"].reshape(-1, 1),
                      kp["roll"].reshape(-1, 1), kp["t"].reshape(-1, 3), kp["exp"].reshape(-1, 63)], dim=1)


def vec_to_kp_info(m: torch.Tensor) -> dict:
    return {"scale": m[:, 0:1], "yaw": m[:, 1:2], "pitch": m[:, 2:3], "roll": m[:, 3:4],
            "t": m[:, 4:7], "exp": m[:, 7:70].reshape(-1, 21, 3)}


class MotionCodec:
    """Frozen LivePortrait. extract(): video frames -> motion; render(): source image + motion -> frames."""

    def __init__(self, device: str = "cuda", weights: Path = LP / "pretrained_weights"):
        cfg = InferenceConfig()
        cfg.checkpoint_F = str(weights / "liveportrait/base_models/appearance_feature_extractor.pth")
        cfg.checkpoint_M = str(weights / "liveportrait/base_models/motion_extractor.pth")
        cfg.checkpoint_G = str(weights / "liveportrait/base_models/spade_generator.pth")
        cfg.checkpoint_W = str(weights / "liveportrait/base_models/warping_module.pth")
        cfg.checkpoint_S = str(weights / "liveportrait/retargeting_models/stitching_retargeting_module.pth")
        cfg.flag_use_half_precision = False
        self.lp = LivePortraitWrapper(inference_cfg=cfg)
        crop = CropConfig()
        crop.insightface_root = str(weights / "insightface")
        crop.landmark_ckpt_path = str(weights / "liveportrait/landmark.onnx")
        self.cropper = Cropper(crop_cfg=crop, device_id=0 if device == "cuda" else -1)
        self.device = device

    # ---------------------------------------------------------------- extraction
    @torch.no_grad()
    def crop_boxes(self, frames: np.ndarray, every: int = 25):
        """One LivePortrait crop (M_c2o affine) per `every` frames, linearly interpolated in between.
        Uses the source-image convention (dsize 512, scale 2.3, vy -0.125) so cached motion and
        inference-time reference crops are computed in the same frame."""
        idx = list(range(0, len(frames), every)) + [len(frames) - 1]
        crops = [self.cropper.crop_source_image(frames[i], self.cropper.crop_cfg) for i in idx]
        Ms = np.stack([c["M_c2o"] for c in crops])                       # [K, 2, 3]
        out = np.empty((len(frames),) + Ms.shape[1:], dtype=np.float32)
        for a, b, Ma, Mb in zip(idx[:-1], idx[1:], Ms[:-1], Ms[1:]):
            for j in range(a, b + 1):
                w = (j - a) / max(1, b - a)
                out[j] = (1 - w) * Ma + w * Mb
        return out

    @torch.no_grad()
    def extract(self, frames: np.ndarray, batch: int = 64) -> dict:
        """frames [T,H,W,3] uint8 RGB (25 fps) -> {"m": [T,70] fp16, "kp0": [21,3], "M_c2o": [T,3,3]}."""
        from src.utils.crop import _transform_img
        M = self.crop_boxes(frames)
        def o2c(m):                                                     # crop_source_image returns M_c2o (2x3)
            return np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2]
        crops = np.stack([cv2.resize(_transform_img(f, o2c(m), dsize=512), (256, 256), interpolation=cv2.INTER_AREA)
                          for f, m in zip(frames, M)])                   # [T,256,256,3], same path as img_crop_256x256
        x = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div_(255).to(self.device)
        ms, kp0 = [], None
        for i in range(0, len(x), batch):
            kp = self.lp.get_kp_info(x[i:i + batch])                     # refined: degrees, [B,21,3]
            ms.append(kp_info_to_vec(kp).cpu())
            if kp0 is None:
                kp0 = kp["kp"][0].cpu()
        return {"m": torch.cat(ms).half(), "kp0": kp0, "M_c2o": torch.from_numpy(M)}

    # ---------------------------------------------------------------- rendering
    @torch.no_grad()
    def render(self, source: np.ndarray, m: torch.Tensor, relative: bool = True, stitch: bool = True):
        """source [H,W,3] uint8 + motion [T,70] (absolute kp_info units) -> frames [T,256,256,3] uint8.
        relative=True applies KDTalker/LivePortrait relative retargeting w.r.t. the first motion row."""
        c = self.cropper.crop_source_image(source, self.cropper.crop_cfg)
        src = self.lp.prepare_source(c["img_crop_256x256"])              # [1,3,256,256] in [0,1]
        s_info = self.lp.get_kp_info(src)
        R_s = get_rotation_matrix(s_info["pitch"], s_info["yaw"], s_info["roll"])
        f_s = self.lp.extract_feature_3d(src)
        x_s = self.lp.transform_keypoint(s_info)
        x_c = s_info["kp"]
        d0 = vec_to_kp_info(m[:1].float().to(self.device))
        R_d0 = get_rotation_matrix(d0["pitch"], d0["yaw"], d0["roll"])
        out = []
        for i in range(m.shape[0]):
            d = vec_to_kp_info(m[i:i + 1].float().to(self.device))
            R_d = get_rotation_matrix(d["pitch"], d["yaw"], d["roll"])
            if relative:
                R_new = (R_d @ R_d0.permute(0, 2, 1)) @ R_s
                delta = s_info["exp"] + (d["exp"] - d0["exp"])
                scale = s_info["scale"] * (d["scale"] / d0["scale"])
                t = s_info["t"] + (d["t"] - d0["t"])
            else:
                R_new, delta, scale, t = R_d, d["exp"], d["scale"], d["t"]
            t = t.clone(); t[:, 2] = 0                                   # LivePortrait zeroes tz
            x_d = scale * (x_c @ R_new + delta) + t
            if stitch:
                x_d = self.lp.stitching(x_s, x_d)
            out.append(self.lp.parse_output(self.lp.warp_decode(f_s, x_s, x_d)["out"])[0])
        return np.stack(out)
```

### 8.3 Backbone: tokens-per-slice instead of a square grid (`sang/streaming_transformer.py`)

```python
# StreamingBlockTransformer.__init__: replace the square-grid assertion + row/col embeddings
        self.spatial = int(round(r ** 0.5)) if r ** 0.5 == int(r ** 0.5) else None
        if self.spatial:                          # appearance tracks: 2-D grid
            self.row_emb = nn.Embedding(self.spatial, dim)
            self.col_emb = nn.Embedding(self.spatial, dim)
        else:                                     # motion track: r frames per chunk
            self.intra_emb = nn.Embedding(r, dim)

    def _pos(self, device):
        tv, r = self.tv, self.r
        slc = torch.arange(tv, device=device).repeat_interleave(r)
        intra = torch.arange(r, device=device).repeat(tv)
        if self.spatial:
            return self.slice_emb(slc) + self.row_emb(intra // self.spatial) + self.col_emb(intra % self.spatial)
        return self.slice_emb(slc) + self.intra_emb(intra)
```

`StreamingTalkingHead` is built with `spatial=None, r=chunk_frames, z_ch=70, continuous=True`;
`_embed_grid`/`_content_latents`/`generate_continuous` already index by `r` only. The
grid tensor for the motion track is `[B, tv, 70, chunk_frames, 1]` so that the existing
`reshape(B, tv, z_ch, r)` path works unchanged (`build_talking_head` gets a `motion: true`
branch that sets these).

### 8.4 Resampling Forcing in `forward_continuous` (`sang/streaming_transformer.py`)

```python
    def resample_history(self, video_lat, audio, s_shift: float = 0.6):
        """Resampling Forcing (arXiv:2512.15702 Alg.1): degrade the teacher-forced context with the
        model's own 1-step re-denoising so training sees inference-like history. No gradient."""
        from sang.diffusion import add_noise
        B, tv, z_ch, h, w = video_lat.shape
        r, n_cond = self.r, self.ref_slices
        with torch.no_grad():
            ts = torch.sigmoid(torch.randn(B, device=video_lat.device))          # LogitNormal(0,1)
            ts = s_shift * ts / (1 + (s_shift - 1) * ts)                          # time shift, s < 1
            grid = video_lat.clone()
            known = torch.zeros(B, tv * r, dtype=torch.bool, device=grid.device)
            known[:, : n_cond * r] = True
            for si in range(n_cond, tv - 1):                                      # last slice needs no resample
                hs = self.hidden_states(grid, audio, known)
                h_si = self.norm(hs[:, si])                                       # [B, r, dim]
                z0 = grid[:, si].reshape(B, z_ch, r).permute(0, 2, 1)             # [B, r, z_ch]
                z_t, _ = add_noise(z0, ts)
                v = self.head(z_t, ts, h_si)
                z_hat = z_t - ts.view(B, 1, 1) * v                                 # 1 Euler step to t=0
                grid[:, si] = z_hat.permute(0, 2, 1).reshape(B, z_ch, h, w)
                known[:, si * r:(si + 1) * r] = True
        return grid                                                                # degraded history, clean targets kept by caller

    def forward_continuous(self, video_lat, audio, struct=None, cond_drop=0.0, resample: bool = False):
        ...
        ctx_lat = self.resample_history(video_lat, audio) if (resample and self.training) else video_lat
        hs = self.hidden_states(ctx_lat, audio, known, struct=struct, audio_drop=audio_drop)   # context: degraded
        h_content = self.norm(hs[:, self.ref_slices:].reshape(B, -1, hs.shape[-1]))
        loss, parts, z0_hat = flow_loss(self.head, h_content, self._content_latents(video_lat))  # targets: clean
        ...
```

Train loop: `resample = step > cfg["rf_warmup_steps"]` (10 % of `max_steps`). The slice-`si`
prior seen by slice `si+1` is then the model's own re-denoised sample, exactly as at
inference. Cost: (tv − n_cond − 1) extra no-grad backbone passes per step — negligible at
100 tokens; on the appearance track it would be 4 × 7,168-token passes, another reason the
fix belongs to the motion track.

### 8.5 Velocity loss (`sang/diffusion.py`, in `flow_loss`)

```python
def flow_loss(head, h, z0, chunk: int | None = None, lam_vel: float = 0.0):
    ...
    loss = F.mse_loss(v_pred.float(), v_target.float())
    z0_hat = x0_from_v(z_t, t, v_pred)
    if lam_vel > 0 and chunk:                       # motion track: tokens are frames, slice-major
        d_hat = z0_hat[:, 1:] - z0_hat[:, :-1]
        d_gt = z0[:, 1:] - z0[:, :-1]
        vel = F.mse_loss(d_hat.float(), d_gt.float())
        loss = loss + lam_vel * vel
        parts["vel"] = vel.detach()
```

### 8.6 `configs/train_motion.yaml`

```yaml
# SANG-M: LivePortrait motion (70-d/frame) + streaming block-causal flow head.
data_glob: data/clips_filtered_all.txt
cache_dir: cache/motion_lp            # from scripts/cache_motion.py
out_dir: runs/motion
windows_per_clip: 0                   # 0 = every non-overlapping window of the clip
val_frac: 0.05
seed: 0
workers: 8
fps: 25
frames: 64                            # 2.56 s content; chunk_frames 4 -> 16 content slices
chunk_frames: 4
motion: true                          # token = 70-d motion, r = chunk_frames
z_ch: 70
motion_ctx: true                      # [reference motion m_0, previous chunk, content...]
bridge_init: prev
audio_encoder: wavlm-large
audio_lookahead: 5
audio_dropout: 0.1
ref_dropout: 0.1
ctx_dropout: 0.5                      # FLOAT: preceding-window dropout 0.5
lam_vel: 1.0
resampling_forcing: true
rf_warmup_steps: 20000
rf_shift: 0.6
dim: 512
layers: 8
heads: 8
dropout: 0.1
diff_hidden: 1024
diff_depth: 6
bf16: true
lr: 3.0e-4
min_lr: 3.0e-5
weight_decay: 0.1
betas: [0.9, 0.95]
warmup_steps: 2000
max_steps: 200000
grad_clip: 1.0
batch_size: 256
grad_accum: 1
eval_every: 5000
decode_steps: 10
audio_cfg: 2.0
# pixel_weight / syncnet_loss / face_crop / continuous(Wan): not used on this track
```

### 8.7 `scripts/cache_motion.py` (new) — shape only

```
for clip in manifest (pooled CPU decode as in sang.data.pooled; GPU side in the main process):
    frames = decode at 25 fps, full clip, max 1080p (sang.video.open_video)
    d = codec.extract(frames)                       # m [T,70], kp0, M_c2o
    wav = AudioReader(clip.m4a, 16 kHz)             # whole clip
    a  = wavlm.encode(wav)                          # [Ta,1024] @ 50 Hz, whole clip
    save cache/motion_lp/<md5>.pt = {m, kp0, M_c2o, audio: a.half(), n_frames, fps}
# dataset: windows of `frames` frames at stride `frames`; audio ticks = 2 * frame index (50 Hz vs 25 fps),
# same window_starts() spread as today; per-dim mean/std computed once -> cache/motion_lp/norm.pt
```

### 8.8 `scripts/eval_full.py` (new) — shape only

```
inputs : --gen DIR --gt DIR --src DIR (first frames)  # 25 fps mp4s, same names
FID    : pytorch_fid on all frames (256 px)  ->  fid
FVD    : I3D logits over 16-frame windows (stride 8), Fréchet distance  ->  fvd16
CSIM   : insightface buffalo_l ArcFace; cos(src frame, gen frame_t) averaged  ->  csim
LSE    : for each gen mp4: syncnet_python run_pipeline.py + run_syncnet.py -> (offset, min_dist, conf)
         report LSE-D = mean(min_dist), LSE-C = mean(conf); also on --gt for the ceiling
strata : optional --meta json (TalkVid) -> per-subgroup tables (language / ethnicity / gender / age)
```

---

## 9. Progress log

- **2026-09-16** — Audit of job 165551 (§1): flat at 15.0 dB, below freeze-ctx; SyncNet term
  0.20 vs GT 0.42; train/inference MSE gap 3×. Literature pass (§4): KDTalker identified as
  the reproducible single-GPU baseline; Resampling Forcing as the exposure-bias fix;
  LivePortrait as the renderer (MIT, weights on HF). Wrote `scripts/diagnose_flow.py`;
  submitted job 165645 on gpu06 (a100 40 GB). Decision proposed: pivot to SANG-M (§5),
  keep the appearance track as an optional refiner (M4).
- *(next entries: job 165645 numbers → §2.6; M0 install + ceiling check; M1 numbers.)*

---

## 10. References (all verified this session; arXiv IDs checked)

- Guo, J. et al. *LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control.* arXiv:2407.03168. Code: github.com/KwaiVGI/LivePortrait (MIT; InsightFace models non-commercial). Weights: huggingface.co/KlingTeam/LivePortrait.
- Yang, C. et al. *Unlock Pose Diversity: Accurate and Efficient Implicit Keypoint-based Spatiotemporal Diffusion for Audio-driven Talking Portrait (KDTalker).* IJCV 2025, arXiv:2503.12963. Code + training: github.com/chaolongy/KDTalker (CC BY-NC 4.0).
- Ki, T., Min, D., Chae, G. *FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait.* ICCV 2025, arXiv:2412.01064. Inference code: github.com/deepbrainai-research/float (CC BY-NC-ND 4.0).
- Zhen, D. et al. *Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation.* CVPR 2025, arXiv:2503.18429.
- Guo, Y. et al. *End-to-End Training for Autoregressive Video Diffusion via Self-Resampling (Resampling Forcing).* arXiv:2512.15702.
- Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E. *Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion.* NeurIPS 2025, arXiv:2506.08009.
- Li, C. et al. *LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision.* arXiv:2412.09262.
- Yaman, D., Eyiokur, F. I., Bärmann, L., Ekenel, H. K., Waibel, A. *Audio-driven Talking Face Generation with Stabilized Synchronization Loss.* ECCV 2024, arXiv:2307.09368.
- Chen, S. et al. *TalkVid: A Large-Scale Diversified Dataset for Audio-Driven Talking Head Synthesis.* arXiv:2508.13618. Data: huggingface.co/datasets/FreedomIntelligence/TalkVid (folder `TalkVid-bench`).
- Chen, P. et al. *DEMO: Disentangled Motion Latent Flow Matching for Fine-Grained Controllable Talking Portrait Synthesis.* arXiv:2510.10650.
- Zhang, Z. et al. *Flow-guided One-shot Talking Face Generation with a High-resolution Audio-visual Dataset (HDTF).* CVPR 2021. Data: github.com/MRzzm/HDTF (CC BY 4.0).
- Chung, J. S., Zisserman, A. *Out of time: automated lip sync in the wild.* ACCV-W 2016. Code: github.com/joonson/syncnet_python.
- Prajwal, K. R. et al. *A Lip Sync Expert Is All You Need (Wav2Lip).* ACM MM 2020.
- Li, T. et al. *Autoregressive Image Generation without Vector Quantization (MAR).* NeurIPS 2024 — the diffusion-head design already in `sang/diffusion.py`.
- See `docs/sota_review_2026.md` §8 for the wider, previously verified list (SoulX-FlashHead, Hallo3, Sonic, Ditto, FlowTalk, IF-MDM, JAM-Flow, EARTalking, HART, NOVA, THEval).

*Compiled with AI-assisted retrieval (Claude Code; Firecrawl research index; direct GitHub/arXiv reads), 2026-09-16. Numbers quoted from papers are marked with their source table; nothing is from memory.*
