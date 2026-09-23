# SANG-M, bidirectional: design decisions and the evidence behind them

**Date:** 2026-09-23 · **Mode:** ARS deep-research, `lit-review` (clear question, evidence-graded decisions)
**Decision taken by the user, not reopened here:** drop streaming. The audio→motion generator is
non-causal and generates a whole window at once, over LivePortrait motion, rendered by the frozen
LivePortrait renderer.
**Question:** given that, what should the generator and its objective look like to reach
published single-GPU numbers on HDTF (LSE-C / LSE-D / FID / CSIM)?
**Code:** `sang/motion.py`, `sang/motion_model.py`, `scripts/{motion_ceiling,cache_motion,train_motion,infer_motion}.py`,
`configs/train_motion.yaml`, `tests/test_motion_model.py` (18/18 pass on CPU).

---

## 0. The one finding that changes the implementation

Removing the causal mask from `StreamingBlockTransformer` is **not** a correct bidirectional model.

`sang.diffusion.DiffusionHead` is a per-token MLP: every layer is a `Linear` over the last dimension,
so nothing mixes across positions. Given the backbone's hidden states, each frame is denoised
independently. Autoregressive or masked multi-step generation makes that acceptable, because each
chunk conditions on the previous chunk's *sample* (MAR and NOVA get the same effect from
multi-step masked generation). A one-pass window loses that: `p(m_1..m_T | audio)` factorises into
`∏ p(m_i | h_i)` — a mean trajectory plus independent per-frame noise.

- **Grade:** analytical, from the code, and **checked**: `tests/test_motion_model.py::test_frames_are_coupled_unlike_the_per_token_head`
  measures `∂v_3/∂x_j` for j ≠ 3. It is exactly 0 for `DiffusionHead` and non-zero for the new model.
- **Consistent with:** Motar (arXiv:2609.10317, Tab. 2), whose one-step, per-token-head variant
  collapsed to near-static motion (VC-R 0.435 vs ideal 1.0).
- **What the field does instead:** FLOAT, KDTalker and Ditto all make the noisy trajectory itself the
  sequence model's input, so attention couples noise across frames. `MotionFlowTransformer` does this.
- **Hypothesis, not measured:** the v4 appearance track samples all 1,024 latent cells of a slice in a
  single head call — one MAR step for a whole frame group, which gives spatially independent noise.
  That may be part of why v4 plateaued at 15.0 dB. Not tested; recorded here so it isn't lost.

---

## 1. Decisions, ranked by expected impact on HDTF

| # | Decision | Metric it moves | Evidence | Confidence |
|---|---|---|---|---|
| 1 | Generator = DiT-style velocity field over the **whole noisy trajectory** (not backbone + per-token head) | FVD, LSE-C, visible jitter | §0; FLOAT §4.2; KDTalker §3.4; Ditto §3.2 | high |
| 2 | Condition on the **canonical keypoints x_c** (+ the reference frame's motion) | CSIM, LSE-C | Ditto §3.2.1 (identity-adaptive c_ref, ablation confirms); KDTalker §3.3 (x_c prior "important for lip sync"); AVTR-1 §2.1 | high — 3 of 3 LivePortrait-based generators |
| 3 | **Frame-wise AdaLN** audio conditioning, not cross-attention | LSE-D / LSE-C | FLOAT Tab. 3, HDTF, same ±2 mask: LSE-D **7.290 vs 7.757**, FID 21.100 vs 21.873 | high (one controlled ablation) |
| 4 | **42-d target**: head rotation + the 39 brow/eye/mouth coords, in the **head frame**; scale, translation and the other 24 coords from the source | CSIM, LSE-C | AVTR-1 §2.1; upstream LivePortrait code (below) | medium-high |
| 5 | **Region-balanced loss** (rotation / brow / eyes / mouth one share each) + velocity term | LSE-C, pose naturalness | AVTR-1 eq. 5; FLOAT eq. 13 (λ_vel = 1) | medium |
| 6 | Window **64** new frames + **10**-frame clean prefix, prefix dropped p = 0.5 | LSE-C, long-clip continuity | KDTalker Tab. 7 (LSE-C 6.875 → 7.326, 8 → 64 frames); FLOAT §4.3, §5.2 | medium-high |
| 7 | **Lip-normalised source** at inference | LSE-C on open-mouth sources | LivePortrait `flag_normalize_lip` (upstream code) | medium |
| 8 | CFG γ_audio = 2, 10 Euler steps, EMA 0.995 | small | FLOAT Tab. 5–6; KDTalker Tab. 6; AVTR-1 §3.2 | medium |

**Found while doing #2 — a real defect in the previous plan.** The recovery plan's streaming design
put `m_0` in slice 0 and predicted `Δm = m − m_0`, standardised. Filled either way, slice 0 never
carried `x_c`, the identity-bearing quantity; filled with the reference's own `Δm` it is a
constant `−mean/std` for every clip, so the model would have had no information about whose face it
was animating. The new model conditions on `[x_c, reference target]` (`REF_DIM = 105`).

---

## 2. LivePortrait, read closely (arXiv:2407.03168 + upstream source, read 2026-09-23)

What transfers, and how it is used:

1. **Motion transform `x = s·(x_c R + δ) + t`**, not `s·((x_c + δ) R) + t`. The paper reports the
   scale-orthographic form "leads to overly flexible learned expressions δ, causing texture
   flickering when driving across different identities." `sang/motion.py` uses the paper's form.
2. **Rotation convention:** `R = (R_z R_y R_x)ᵀ` from degrees (upstream `src/utils/camera.py`).
   Ported to pure torch so training never imports the renderer; a test checks it against a literal
   transcription of the upstream function.
3. **Which keypoints do what** (upstream `src/live_portrait_pipeline.py`, `animation_region` branches):
   lip `[6, 12, 14, 17, 19, 20]`, eyes `[11, 13, 15, 16, 18]`; LivePortrait's own expression
   transfer drives `[1, 2, 6, 11–20]` in full, so brow = `[1, 2]`. The remaining 8 keypoints are
   never driven by LivePortrait itself — they hold face shape. The paper does not list these
   indices; this is from the code, and it is why decision #4 is sound: the target drops exactly the
   coordinates the renderer's authors treat as identity.
4. **`L_guide`**: the first two dims of the implicit keypoints are Wing-loss-aligned to 10 eye/lip
   2D landmarks during LivePortrait's training — which is *why* those keypoints are semantically
   stable enough to weight by region (#5).
5. **Lip normalisation** (`flag_normalize_lip`): if the source's lip ratio exceeds a threshold,
   `retarget_lip(x_s, 0)` closes it before animation. In expression space the offset is `d / s`
   (from `x = s(x_c R + δ) + t`); `MotionCodec.source_motion` applies it.
6. **Stitching MLP** (`[126, 128, 128, 64, 65]`) is kept at render time for paste-back.
7. **Discipline:** stage II freezes the base model and trains only small MLPs. Same stance here —
   the renderer never trains.
8. **Data:** 69M frames, ~18.9K identities, clips < 30 s, one person per clip via tracking +
   recognition, KVQ quality filter. For the related-work section; the renderer is theirs precisely
   because we cannot match that scale.

---

## 3. Evidence table (primary sources read this session)

| Source | Venue / tier | What was used |
|---|---|---|
| FLOAT, Ki et al., arXiv:2412.01064 | ICCV 2025 · peer-reviewed | frame-wise AdaLN (eq. 10), Tab. 3 ablation, L = 50 + L′ = 10, dropout 0.1 / 0.5, γ_a = 2, NFE 10, L1 norm, λ_OT = λ_vel = 1, 1× A100 22 days |
| KDTalker, Yang et al., arXiv:2503.12963 | IJCV 2025 · peer-reviewed | x_c reference prior, 64-frame window (Tab. 7), DDIM step ablation (Tab. 6), LivePortrait renderer ablation (Tab. 5) |
| Ditto, Li et al., arXiv:2411.19509 | ACM MM 2025 · peer-reviewed | canonical-keypoint identity conditioning, L = 80, overlap + centre-weighted fusion for long clips, 10 vs 50 steps equivalent |
| LivePortrait, Guo et al., arXiv:2407.03168 + github.com/KwaiVGI/LivePortrait | arXiv; code is primary evidence | §2 above |
| AVTR-1, arXiv:2609.22913 | technical report, Sep 2026 · **not peer-reviewed** | 42-d target, head-frame expression, region-balanced loss, EMA 0.995, lr schedule |
| Motar, An et al., arXiv:2609.10317 | preprint, Sep 2026 · **not peer-reviewed** | Std-R / VC-R rollout diagnostics (now in `train_motion.py`'s eval); per-token-head collapse |

**Not read, therefore not used for any decision:** DEMO (2510.10650), JAM-Flow (2506.23552),
IF-MDM (2412.04000), AniTalker, Hallo-family audio windows. Abstract-level knowledge only.
**Consensus** was unavailable (monthly quota exhausted); retrieval used the Firecrawl research index
and direct reads of upstream source.

---

## 4. Devil's advocate: where this could be wrong

- **#4 leans on AVTR-1 alone for the target choice**, and AVTR-1 is a streaming, dyadic, non-peer-reviewed
  report. The upstream code corroborates *which* coordinates are expression, but not that dropping
  translation and scale is free. **Measured before committing:** `motion_ceiling.py --target 42`
  vs `--target 70` on our own clips. If the gap exceeds ~0.5 dB PSNR or ~0.01 CSIM, restore t_x, t_y.
- **Full vs local attention is untested in anything read here.** FLOAT's ±2 window was held fixed
  across its AdaLN ablation; KDTalker's window-length gain is about *context length*, not the mask.
  Default is full (`attn_window: 0`); run ±2 as an ablation.
- **Reference choice.** Training draws the reference from a random frame of the same clip (Ditto);
  AVTR-1 uses per-track medians. Random frame matches inference (one source image); medians are
  more robust to a mid-blink reference. Ablation, not a certainty.
- **Uniform vs logit-normal t:** FLOAT uniform, AVTR-1 logit-normal. Default follows FLOAT because
  FLOAT is the number being reproduced.
- **The metrics.** Sync-C/LSE-C are crop- and pose-sensitive (THEval, arXiv:2511.04520). Every
  published number must be re-run through `sang/bench.py` before it is compared with ours.

---

## 5. Run order (each step gates the next)

| step | command | cost | gate |
|---|---|---|---|
| 0 | `bash bash_scripts/install_motion.sh` (login node) | — | ends with "LivePortrait API ok" |
| M0a | `sbatch bash_scripts/job.sh scripts/motion_ceiling.py --n 50` | ~1 GPU-h | PSNR ≥ 29 dB, CSIM ≥ 0.95 |
| M0b | `… motion_ceiling.py --n 50 --target 42` | ~1 GPU-h | within 0.5 dB / 0.01 CSIM of M0a |
| M0c | `… motion_ceiling.py --n 20 --lse` | ~1 GPU-h | LSE-C within 0.5 of real video |
| cache | `sbatch --array=0-7 bash_scripts/job.sh scripts/cache_motion.py --nshards 8` | ~10 GPU-h total | ≥ 90% of clips cached |
| M1 | `sbatch bash_scripts/train_motion.sh` | ~6–8 h | eval: `audio_gain` well above 1, `std_r_*` and `vc_r_*` in 0.8–1.2 |
| eval | `scripts/infer_motion.py` over HDTF-349 (KDTalker protocol) + `sang/bench.py` | a few GPU-h | LSE-C ≥ 6.5, CSIM ≥ 0.90, FID ≤ 15 (KDTalker: 7.33 / 0.949 / 9.76) |

**Still to write:** the HDTF batch-evaluation wrapper (loop `infer_motion.py` over the 349 clips,
then `sang/bench.py`) and FVD in `sang/bench.py`. Neither is needed until M1 produces a checkpoint.

---

*AI-assisted: literature retrieval, reading and code were done with Claude Code (Firecrawl research
index, upstream-source reads). Every number above was read from the cited table or file in this
session; nothing is quoted from memory.*
