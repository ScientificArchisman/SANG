# Lip-sync (LSE-C / LSE-D) interventions for a motion-space talking-head generator

Scope: things that raise SyncNet LSE-C / lower LSE-D for generators that predict compact motion (LivePortrait implicit keypoints, 3DMM, motion latents) and use a frozen renderer. Target system: 53M bidirectional flow-matching transformer, 42 LivePortrait numbers per frame, 64-frame windows at 25 fps, WavLM-large last layer, frame-wise AdaLN after Conv1d k=5, audio CFG 2.0, 10 Euler steps.

Evidence grades: **MEASURED** = number quoted from the paper's own table (table cited). **SPECULATIVE** = my inference or the authors' untested claim.

All arXiv IDs below were retrieved in this session through the Firecrawl research index (full text read unless noted "abstract only").

Reference numbers (HDTF, same SyncNet protocol, but test splits differ between papers):
- KDTalker (arXiv 2503.12963) Tab. 1: Real video LSE-C 8.243 / LSE-D 6.929; KDTalker 7.326 / 7.548.
- Teller (arXiv 2503.18429) Tab. 1: Real video Sync-C 8.094 / Sync-D 6.976; Teller 7.696 / 7.536.
- FLOAT (arXiv 2412.01064) Tab. 3: FLOAT LSE-D 7.290 on HDTF.

---

## Q1. Audio encoder choice and layer choice

### Takeaway
MEASURED: the audio representation can move Sync-C by 3 or more points. ASR-trained (Whisper), audio-visual (AV-HuBERT) and phoneme-aware encoders beat codec and acoustic-only features. In one controlled comparison, HuBERT clearly beat wav2vec2 and Whisper. None of the papers found isolates **WavLM last layer vs WavLM intermediate layers** for talking heads. The speech-probing literature suggests WavLM's phonetic content peaks in the upper-middle layers rather than at the very top, so choosing a layer, or a learned weighted sum of layers, is a cheap and plausible win, but it is untested for this task.

### Cited Findings
- **MEASURED, Teller Tab. 3 (HDTF)**: Whisper audio condition gives Sync-C 7.696 / Sync-D 7.536. The funcodec (TTS codec) condition gives Sync-C 4.286 / Sync-D 10.373. That is a 3.41-point Sync-C gap from the encoder alone. Teller is a motion-token (LivePortrait-keypoint RVQ) autoregressive model, so the result is directly relevant. — [Teller, arXiv 2503.18429](https://arxiv.org/abs/2503.18429)
- **MEASURED, Teller Tab. 4**: a single-head model gets Sync-C 7.790 / Sync-D 7.474. The multi-head model gets 7.696 / 7.536. — [Teller](https://arxiv.org/abs/2503.18429)
- **MEASURED, PASE Table (NeRF renderer, SyncTalk-style pipeline; relative to HuBERT)**:

  | Features | LSE-C | LSE-D |
  |---|---|---|
  | HuBERT | 7.6599 | 7.1402 |
  | DeepSpeech | 7.4268 | 7.5418 |
  | wav2vec 2.0 | 3.3782 | 10.200 |
  | Whisper | 3.3416 | 10.202 |
  | AV-HuBERT | 8.0136 | 6.8022 |
  | PASE (phoneme-aware encoder) | 8.7098 | 6.1255 |
  | GT | 8.8302 | 6.0570 |

  On the 3DGS renderer: HuBERT 5.0660, AV-HuBERT 5.3618, PASE 8.3487. Caveat: this is person-specific NeRF/3DGS, and the very poor wav2vec2/Whisper numbers may reflect the pipeline's tuning to HuBERT. — [PASE, arXiv 2504.05803](https://arxiv.org/abs/2504.05803)
- **MEASURED, SyncDiff Tab. 2 (LRS2)**: DeepSpeech features LSE-C 6.11 / LSE-D 7.93; AV-HuBERT features 7.05 / 7.11. — [SyncDiff, arXiv 2503.13371](https://arxiv.org/abs/2503.13371)
- **MEASURED, SyncDiff Tab. 6 (how many of the 12 AV-HuBERT layers are frozen while the rest are fine-tuned)**:

  | Frozen layers | LSE-C | LSE-D |
  |---|---|---|
  | 3 | 7.56 | 6.75 |
  | 6 | 7.64 | 6.67 |
  | 9 | 7.80 | 6.58 |
  | 12 | 7.46 | 6.87 |

  The curve is U-shaped. Fine-tuning only the top 3 layers is best, and fully frozen is worst. — [SyncDiff](https://arxiv.org/abs/2503.13371)
- **Multi-layer Whisper in practice (design, not ablated)**: Sonic uses Whisper-Tiny, concatenates "the features from the last layers of five stages" for multi-scale understanding, and gives each video frame 0.2 s of audio features. There is no single-vs-multi-layer ablation. — [Sonic, arXiv 2411.16331](https://arxiv.org/abs/2411.16331)
- **Layer-wise phonetic content (probing, not talking heads)**: Pasad et al. report two patterns. In wav2vec2 and XLSR, phonetic and word content peaks in intermediate layers and drops at the top. In HuBERT, WavLM and AV-HuBERT, which predict discrete units learned from an intermediate layer, "the phonetic and word information appears to be concentrated toward higher layers". For ASR/phone recognition, the best single layer "is always lower than at least the top two layers", and a single intermediate layer often matches or beats a learned weighted sum of all layers. — [Pasad et al., arXiv 2211.03929](https://arxiv.org/abs/2211.03929)
- A 2024 probing study found HuBERT and WavLM learn representations similar to wav2vec2, "differing mainly in later layer performance". — [arXiv 2408.13678 (abstract only)](https://arxiv.org/abs/2408.13678)
- **3D-animation-specific**: Wav2Sem argues that self-supervised features couple near-homophone syllables with different lip shapes, which averages lip motion, and adds sentence-level semantic features to decorrelate them. — [Wav2Sem, arXiv 2505.23290 (abstract only)](https://arxiv.org/abs/2505.23290). A 2026 study found that "encoding phonetic classes is beneficial for accurate facial animation" across SSL and ASR-label representations. — [arXiv 2606.13630 (abstract only)](https://arxiv.org/abs/2606.13630)

### Inferences
- SPECULATIVE: WavLM-large's last layer is probably not catastrophically bad for phonetics, since WavLM keeps phonetic content high up. But the last one or two layers are reliably not the best single layer for phone tasks (Pasad). Cheapest test: cache layers {6, 12, 18, 21, 24} of WavLM-large, or train a learned softmax-weighted sum of all 25 hidden states (SUPERB-style). Retraining cost: about one training run per variant; feature caching is roughly 2 to 5x the disk of the current cache.
- SPECULATIVE: an ASR-supervised encoder such as the Whisper encoder (Teller, LatentSync and Sonic all use it) is the best-supported swap for a motion generator. Teller is the closest analogue (LivePortrait-keypoint motion tokens). Whisper-large-v3 encoder features are 1280-d at 50 Hz, a drop-in for the 2 features per frame at 25 fps. Cost: re-extract features, then retrain.
- SPECULATIVE: AV-HuBERT audio-stream features are the strongest measured "lip-aware" audio representation (PASE table, SyncDiff Tab. 2). Partially fine-tuning the top layers (SyncDiff Tab. 6) gave another +0.34 LSE-C over a frozen encoder. That is feasible on one GPU for a 53M model if features are not cached. A cheaper alternative is to train a LoRA or adapter on the top 3 layers.

### Gaps
- No paper found that ablates WavLM (any layer) against Whisper or HuBERT in an audio-to-motion talking-head model with LSE numbers.
- FLOAT's wav2vec2 layer choice was not verified in the text read.
- The "AVTR-1 uses 75 past / 5 present / 5 future feature steps" claim in the brief could not be located or verified. No arXiv ID was found for "AVTR-1".

---

## Q2. Audio context window size / future lookahead

### Takeaway
There is little direct, numeric evidence on the ±m audio window for lip sync. The best measured proxy is temporal context in the generator itself: KDTalker's window-length ablation on LivePortrait keypoints (64 frames best, +0.45 LSE-C over 8 frames). Published designs use about ±2 frames (FLOAT, Sonic's 0.2 s per frame). The current ±80 ms (±2 frames at 25 fps, k=5 over 50 Hz, which is actually ±2 feature steps = ±40 ms) may be narrower than the ±2 frames (±80 ms) that others use.

### Cited Findings
- **MEASURED, KDTalker Tab. 7 (HDTF, LivePortrait keypoints, diffusion; window = frames per pass)**:

  | Frames per pass | LSE-C | LSE-D |
  |---|---|---|
  | 8 | 6.875 | 7.928 |
  | 16 | 6.912 | 7.904 |
  | 32 | 7.256 | 7.636 |
  | 64 | 7.326 | 7.548 |

  — [KDTalker, arXiv 2503.12963](https://arxiv.org/abs/2503.12963)
- **MEASURED, KDTalker Tab. 3**: removing RoPE gives 7.225 / 7.663, against the full model's 7.326 / 7.548. Removing spatiotemporal attention collapses to 1.360 / 11.777. Removing the reference keypoints x_c gives 6.656 / 8.203. — [KDTalker](https://arxiv.org/abs/2503.12963)
- FLOAT's frame-wise AdaLN attends to 2T neighbouring frames with T=2, i.e. [l−2 … l+2]. It generates L=50 frames with L′=10 preceding frames. — [FLOAT, arXiv 2412.01064](https://arxiv.org/abs/2412.01064)
- Sonic: "For each video frame, a duration of 0.2s audio feature is used frame-wise to give rich context" (±0.1 s). Its temporal audio cross-attention ablation (Tab. 3, CelebV-HQ): without temporal-audio attention, Sync-C 2.610 / Sync-D 10.310; full model 2.689 / 10.194. — [Sonic, arXiv 2411.16331](https://arxiv.org/abs/2411.16331)
- **MEASURED, Hallo4 Tab. 6**: temporal handling of audio features matters. Subsampling audio 4x to align with compressed latents gives Sync-C 2.769; 4x compression with 2x channel expansion gives 4.244; with 4x channel expansion (no information dropped) 5.689. Lesson: never drop audio frames; fold them into channels. — [Hallo4, arXiv 2505.23525](https://arxiv.org/abs/2505.23525)
- The LatentSync authors note that "the audio window inherently encapsulates rich temporal information" and that better use of audio improved temporal consistency. Their SyncNet uses 16 frames; 25 frames stalled early in training. — [LatentSync, arXiv 2412.09262](https://arxiv.org/abs/2412.09262)

### Inferences
- SPECULATIVE: coarticulation means lips anticipate upcoming phonemes by roughly 100 to 200 ms. A centred k=5 conv over 50 Hz features spans only ±40 ms (±2 feature steps) of direct per-frame audio. The bidirectional transformer does see neighbouring frames' audio through self-attention, but only indirectly. Cheap tests: (a) widen the conv to k=9 to 13 (±80 to 120 ms) or add a dilated stack; (b) concatenate a per-frame window of ±4 to 6 frames of raw features (Sonic-style 0.2 s or larger), then project. Cost: minutes of code, one retrain.
- SPECULATIVE: since the model is bidirectional over 64 frames, "future lookahead" already exists at the sequence level. Gains are more likely from a stronger local audio→mouth pathway, such as per-frame cross-attention to a ±k audio window, than from a longer sequence window. The 64-frame window already matches KDTalker's best.

### Gaps
- No 2024–2026 motion-space paper found with a clean ablation of past/future audio context length (±m) against LSE-C.
- EMO and Hallo audio-window ablations were not located with numbers.

---

## Q3. Lip-sync losses usable without rendering pixels

### Takeaway
There is evidence of two kinds. (1) A pixel SyncNet loss is the single largest measured lip-sync lever in any setting (LatentSync: Sync-conf 4.6 → 8.9). The generator here has a frozen, differentiable renderer, so rendering the mouth crop and backpropagating a SyncNet loss is feasible, and LivePortrait is fast. (2) Motion-space sync experts (3D SyncNet on meshes, speech-mesh contrastive representations) give measured but smaller gains. They also trade off against reconstruction, and a plain CNN SyncNet on 3D is inconsistent. A good motion-space critic needs a strong pretrained audio-visual prior. DEMO's InfoNCE is used only inside its motion autoencoder, and the paper has no ablation of it.

### Cited Findings
- **MEASURED, LatentSync Tab. 2 (HDTF)**:

  | Variant | Sync-conf | FVD |
  |---|---|---|
  | No SyncNet | 4.6 | 220.37 |
  | Latent-space SyncNet | 7.9 | 180.45 |
  | Pixel-space SyncNet (decoded) | 8.9 | 162.74 |

  The latent-space SyncNet converges worse, "since the input to the latent space SyncNet is the compressed latents … some lip information may already be lost". Caveat: the authors argue image-to-video portrait animation does not suffer their masked-frame "shortcut" problem, so the size of the gain may be smaller for audio-to-motion. — [LatentSync, arXiv 2412.09262](https://arxiv.org/abs/2412.09262)
- **StableSyncNet convergence recipe** (useful if training a sync expert on renders or on motion): batch size 1024, 16 frames, 2048-d embeddings, audio-visual offset correction after affine alignment. It reaches 94% accuracy on HDTF against 91% for the prior state of the art. Batch 128 "may fail to converge, with the loss remaining stuck at 0.69". — [LatentSync](https://arxiv.org/abs/2412.09262)
- **MEASURED, AV-HuBERT expert vs Wav2Lip SyncNet as the sync loss (Tab. 4, LRS2)**:

  | Sync loss | LSE-C | LSE-D |
  |---|---|---|
  | SyncNet lip-expert baseline | 7.116 | 7.396 |
  | AV-HuBERT, visual-visual | 7.481 | 6.556 |
  | AV-HuBERT, multimodal | 6.998 | 6.794 |
  | AV-HuBERT audio-visual ("unsupervised") | 7.958 | 6.301 |

  The authors also found the Wav2Lip SyncNet unstable even on GT pairs. — [arXiv 2405.04327](https://arxiv.org/abs/2405.04327)
- **MEASURED, Learn2Talk Tab. V (3D meshes; LSE measured by their own SyncNet3D, so not comparable to 2D LSE)**:
  - VOCASET: without the 3D sync loss, LSE-D 11.717 / LSE-C 6.286; full model 10.593 / 9.838.
  - BIWI: without sync loss, 9.434 / 8.933; full 8.897 / 9.449.
  - Cost: LVE on BIWI worsens (4.6971 without sync → 5.0003 full).
  - SyncNet3D is trained on audio vs 3D-motion windows (W=5 at 25 fps) in about 12–15 h on one RTX 4090.
  - The authors report a trade-off between the lipread loss and the sync loss.

  — [Learn2Talk, arXiv 2404.12888](https://arxiv.org/abs/2404.12888)
- **MEASURED, speech-mesh representation as a perceptual loss (Tab. 2/3, VOCASET)**: FaceFormer baseline MTM 53.6 ms / LVE 3.357 / PLRS 0.368 → with their loss 52.2 / 3.091 / 0.463. A CNN "3D SyncNet" critic gives 55.6 / 3.316 / 0.435 (MTM worse). The version without the 2D audio-visual prior gives 55.3 / 3.278 / 0.400. On CodeTalker, the 3D SyncNet made LVE worse (3.700 → 4.319). Loss weight: InfoNCE at 1e-7 over 5-frame sliding windows. — [arXiv 2503.20308](https://arxiv.org/abs/2503.20308)
- **DEMO**: the lip encoder is trained with audio-visual InfoNCE (Eqs. 4–5) inside the motion autoencoder. Its ablation (Tab. 2) only compares VAE+Flow, FCME+Diff and FCME+Flow; "LSE-D" values are about 238–246, a non-standard scale; no LSE-C is reported and there is no ablation of the contrastive term itself. — [DEMO, arXiv 2510.10650](https://arxiv.org/abs/2510.10650)
- Lipread (AV-ASR) perceptual losses on rendered avatars: VisualSpeaker reports a 56.1% improvement in Lip Vertex Error from a lip-reading loss through 3DGS renders. — [arXiv 2507.06060 (abstract only)](https://arxiv.org/abs/2507.06060). See also [arXiv 2407.01034 (abstract only)](https://arxiv.org/abs/2407.01034).
- Metric caveat: FantasyTalking2 reports that "Sync-C tends to assign higher confidence to exaggerated lip movements". Sync-C alone matched human lip-sync preference at 72.34% accuracy (Tab. 2). — [FantasyTalking2, arXiv 2508.11255](https://arxiv.org/abs/2508.11255). Dimitra observed that freezing head pose changed SyncNet scores with identical lips (HDTF: 9.42/5.69 vs 9.60/5.49) and concluded "the metrics are instable". — [Dimitra, arXiv 2502.17198](https://arxiv.org/abs/2502.17198)

### Inferences
- SPECULATIVE, highest-leverage option: a pixel SyncNet loss through the frozen LivePortrait warp and decoder, applied to the one-step clean estimate x̂₁ = x_t + (1−t)·v̂.
  - Use a 16-frame mouth crop at 256² or the official SyncNet at 96×96 lower-half, only on low-noise t (e.g. t>0.5), on a sub-batch.
  - Cost on one GPU: LivePortrait decode is roughly 10–13 ms per frame at inference and memory-heavy in backward. Expect 2–4x slower steps. Use gradient checkpointing, and consider fine-tuning only (a few thousand steps) from the current checkpoint.
  - Risk: Sync-C "hacking" through exaggerated mouth opening (FantasyTalking2). Keep the FM loss dominant, and monitor LMD / mouth-opening statistics against GT.
- SPECULATIVE, cheaper option: a motion-space SyncNet on the 39 lip/brow/eye keypoints (lip subset) × 5–16 frames against audio features, trained contrastively on the existing cache (big batch per StableSyncNet findings). Then use its cosine as an auxiliary loss on x̂₁. Cost: one GPU for hours. Evidence suggests gains are real but smaller and fragile (arXiv 2503.20308 Tab. 3). It works best if the audio branch is anchored to a pretrained audio-visual representation (e.g. frozen AV-HuBERT audio features) rather than trained from scratch.

### Gaps
- No paper found that trains a SyncNet directly on LivePortrait implicit keypoints and reports 2D LSE-C gains.
- DEMO gives no numeric evidence isolating its contrastive term.

---

## Q4. Guidance: audio CFG scale, separate scales, schedules

### Takeaway
Audio CFG is the cheapest knob and is large when missing (LeapTalk: no audio CFG gives Sync-C 4.34 vs 8.38). Beyond about 2, gains in motion-latent flow models are small and saturating (FLOAT: γa 1→2 moves LSE-D 7.049→6.994). Too-strong guidance can hurt audio-motion alignment of pose (LeapTalk BAS falls with scale). Schedules can trade sync against fidelity: applying CFG in only some steps lowers Sync-C by about 0.3 but improves FVD. A sweep of γa ∈ {1.5, 2, 3, 4, 5} plus a separate reference/prefix scale is a free test.

### Cited Findings
- **MEASURED, FLOAT Tab. 6 (RAVDESS; LSE-D only)**:

  | Audio scale γa | Emotion scale γe | LSE-D |
  |---|---|---|
  | 1 | 1 | 7.049 |
  | 1 | 2 | 7.212 |
  | 2 | 1 (default) | 6.994 |
  | 2 | 2 | 6.994 |

  FLOAT uses nested CFG with separate audio and emotion scales (Eq. 14). — [FLOAT, arXiv 2412.01064](https://arxiv.org/abs/2412.01064)
- **MEASURED, LeapTalk Tab. 2 (HDTF)**: full model Sync-C 8.38 / Sync-D 7.69; without audio-driven CFG 4.34 / 10.21. Note that CFG here is applied inside one-step DMD distillation, so the "no CFG" collapse is partly distillation-specific. Tab. 7: raising the CFG scale from 1 to 7 increases pose std (1.655→6.323), and BAS (beat alignment) drops overall (0.723 at 1.0, 0.650 at 7.0), i.e. "over-strong guidance can hurt audio-motion alignment". — [LeapTalk, arXiv 2608.00079](https://arxiv.org/abs/2608.00079)
- **MEASURED, Lip Forcing Tab. 2/3 (HDTF, 2-step distilled lip-sync model)**:

  | Configuration | Sync-C | Sync-D | FVD |
  |---|---|---|---|
  | All-CFG (4.5) | 7.13 | 7.85 | 138.32 |
  | No CFG | 6.14 | 8.39 | 120.85 |
  | Windowed (CFG only in a mid-trajectory band) | 6.81 | 7.85 | 119.88 |
  | Reverse window | 6.98 | 7.81 | 126.62 |

  The authors identify a "CFG fidelity–sync tradeoff" with a sync-favoring band in the middle of the trajectory. — [Lip Forcing, arXiv 2606.11180](https://arxiv.org/abs/2606.11180)
- Sonic uses separate CFG scales: reference image 2.0, audio 7.5. "Higher r_a improve lip-audio synchronization, but excessively high values do not always yield better outcomes" (Fig. 6f, no table). — [Sonic, arXiv 2411.16331](https://arxiv.org/abs/2411.16331)
- **MEASURED, number of solver steps (related knob)**:
  - FLOAT Tab. 5 (HDTF): NFE 2 → LSE-D 7.559; NFE 5 → 7.155; NFE 10 → 7.290; NFE 20 → 7.343. — [FLOAT](https://arxiv.org/abs/2412.01064)
  - KDTalker Tab. 6 (DDIM steps): 5 steps LSE-C 7.455 / LSE-D 7.424; 10 → 7.448 / 7.436; 50 → 7.326 / 7.548; 200 → 7.221 / 7.633. — [KDTalker](https://arxiv.org/abs/2503.12963)
  - In both motion-space models, fewer steps (5–10) gave slightly better sync than more.

### Inferences
- SPECULATIVE, free test: sweep γa ∈ {1.5, 2, 2.5, 3, 4}, Euler steps ∈ {5, 8, 10}, and a separate guidance scale on the prefix/reference condition (FLOAT/Sonic-style nested CFG: uncond → +ref → +audio). This requires training with independent dropout of each condition (Sonic drops audio 5%, image 5%, both 5%). Also test an interval schedule, audio CFG only for t in the middle band, which Lip Forcing found trades about 0.3 Sync-C for FVD. Cost: inference only, if condition dropout is already independent.
- SPECULATIVE: expected gain at γa beyond 2 is small (FLOAT: 0.055 LSE-D from 1→2). Large Sync-C jumps from CFG appear only when a model is otherwise under-using audio (LeapTalk's one-step case).

### Gaps
- FLOAT Tab. 6 reports only LSE-D, on RAVDESS rather than HDTF, and no LSE-C.
- No motion-space paper found with an LSE-C sweep over γa above 2.

---

## Q5. Post-training with a sync reward (RL / DPO / reward fine-tuning)

### Takeaway
Preference or reward post-training gives measured Sync-C gains, all so far on pixel video DiTs: +0.26 (Playmate2), +0.36 (Hallo4), and +2.55 (FantasyTalking2 TLPO, where plain DPO gave only +1.23). A SyncNet-score reward inside distillation gives +0.07 to +0.11 (Lip Forcing). A motion-space generator with a fast frozen renderer is a good fit for a cheap offline Flow-DPO pass: sample N motions per clip, render, score with SyncNet, pick best vs worst.

### Cited Findings
- **MEASURED, Playmate2 Tab. 1**: DPO pairs are auto-labelled by SyncNet (5 segments per sample; best Sync-C = winner, worst = loser), trained with Flow-DPO (β_t = β(1−t)², λ=0.1).
  - HDTF: without DPO Sync-C 7.89 / Sync-D 7.53; with DPO 8.15 / 7.32.
  - CelebV-HQ: 5.28 / 7.84 → 5.49 / 7.66.
  - FID and FVD also improved.

  — [Playmate2, arXiv 2510.12089](https://arxiv.org/abs/2510.12089)
- **MEASURED, Hallo4 Tab. 4/5 (human-preference DPO)**: baseline Sync-C 5.326 / Sync-D 8.391; with the motion-alignment preference 5.651 / 7.873; with motion + fidelity 5.689 / 7.853. Pair construction (Tab. 5): best-vs-worst 5.689, better-vs-worse 5.341, so the most distinct pairs work best. The paper gives a Flow-DPO loss for flow-matching DiTs (Eq. 5). — [Hallo4, arXiv 2505.23525](https://arxiv.org/abs/2505.23525)
- **MEASURED, FantasyTalking2 Tab. 3**:

  | Variant | Sync-C |
  |---|---|
  | Baseline | 3.154 |
  | Plain DPO | 4.381 |
  | IPO | 4.375 |
  | SimPO | 4.546 |
  | TLPO (per-dimension LoRA experts + timestep/layer gated fusion) | 5.704 |

  The authors say plain DPO shows "negligible enhancement in motion naturalness and lip-sync" relative to fidelity, because objectives compete. — [FantasyTalking2, arXiv 2508.11255](https://arxiv.org/abs/2508.11255)
- **MEASURED, FlowPortrait Tab. 5 (Flow-GRPO; MLLM-judged 1–5 scores, not LSE)**:
  - Lip-sync: SFT 3.74 → RL 4.40.
  - Lip-sync-only reward: 4.36, but motion falls to 3.01.
  - MLLM-only rewards caused jitter and colour-drift "reward hacking", fixed by adding LPIPS and RAFT-flow consistency rewards.
  - Best settings: stochastic window W=1 step, noise η=0.5.
  - Human study (Tab. 4): lip-sync SFT 3.87 → RL 4.16.

  — [FlowPortrait, arXiv 2603.00159](https://arxiv.org/abs/2603.00159)
- **MEASURED, Lip Forcing Tab. 2**: SyncNet reward as an exp(β·R) weight on the DMD gradient (β=2): static CFG 7.13 → 7.24 Sync-C; windowed CFG 6.81 → 6.88. — [Lip Forcing, arXiv 2606.11180](https://arxiv.org/abs/2606.11180)

### Inferences
- SPECULATIVE, recommended recipe for this system:
  1. For each training clip, sample K=4–8 motion sequences at the current CFG.
  2. Render with LivePortrait and score with official SyncNet (LSE-C, minus a penalty on mouth-opening amplitude vs GT to avoid exaggeration hacking).
  3. Keep best-vs-worst pairs with a margin.
  4. Run Flow-DPO with a frozen reference copy plus an FM loss regulariser (Playmate2's λ=0.1).

  Cost: rendering and scoring 64-frame clips is cheap relative to video DiTs. A few thousand pairs is feasible in a day on one GPU. Training a 53M model with a reference copy is trivial.
- SPECULATIVE: online GRPO is possible but riskier. FlowPortrait shows single-aspect rewards degrade other axes, so pair a sync reward with a motion-realism or jitter term.

### Gaps
- No RL or DPO paper found that applies to a motion-latent or keypoint generator with LSE numbers; all are pixel DiTs.
- FlowPortrait reports only MLLM and human scores, not LSE-C/D.

---

## Q6. Phoneme / viseme / text conditioning

### Takeaway
There is weak direct evidence for adding phoneme tokens to a generator. Dimitra uses wav2vec plus aligned phonemes plus CLIP text but has no ablation, and its SyncNet scores trail DreamTalk. The strongest measured gain is from phoneme-aware audio encoders (PASE, +13.7% LSE-C over HuBERT, reaching near-GT). That suggests phonetic supervision works best when injected into the audio representation, for example as a phoneme-CTC auxiliary head, rather than as extra generator tokens.

### Cited Findings
- **Dimitra**: conditions on wav2vec features plus a frame-aligned phoneme sequence (Montreal-style aligner) plus a CLIP-encoded transcript. It trains separate models for lips (13 3DMM dims), expression (51) and pose (6). There is no phoneme ablation.
  - HDTF (Tab. II): Dimitra (HP) sync_dist 9.42 / sync_conf 5.69, vs DreamTalk 8.66 / 6.14 and GT 8.63 / 6.82.
  - VoxCeleb2 (Tab. I): Dimitra 9.43 / 4.93, vs DreamTalk 8.48 / 5.64.

  — [Dimitra, arXiv 2502.17198](https://arxiv.org/abs/2502.17198)
- **MEASURED, PASE Table (NeRF)**: PASE LSE-C 8.7098 / LSE-D 6.1255, vs HuBERT 7.6599 / 7.1402 and GT 8.8302 / 6.0570. On 3DGS: 8.3487 vs 5.0660. PASE uses phoneme embeddings as alignment anchors plus contrastive audio-visual alignment. Model: 37.7M parameters. — [PASE, arXiv 2504.05803](https://arxiv.org/abs/2504.05803)
- Phonetic-context coarticulation loss for 3D animation: "replacing the conventional reconstruction loss with ours improves both quantitative metrics and visual quality". — [arXiv 2507.20568 (abstract only)](https://arxiv.org/abs/2507.20568)
- AV-HuBERT, trained to match audio to lip shapes, beats DeepSpeech as a condition (SyncDiff Tab. 2: LSE-C 7.05 vs 6.11). — [SyncDiff, arXiv 2503.13371](https://arxiv.org/abs/2503.13371)

### Inferences
- SPECULATIVE: cheapest phonetic intervention is an auxiliary frame-level phoneme or viseme classification head on the audio pathway (after Conv1d), using MFA or wav2vec2-CTC forced alignments. This enforces phoneme-discriminative audio embeddings (PASE-like) at negligible cost. A heavier alternative: replace WavLM-last with ASR-supervised features (Whisper), which carry this information implicitly (Teller Tab. 3).

### Gaps
- No 2024–2026 motion-space paper found with a clean with/without-phoneme ablation on LSE-C.

---

## Q7. Mouth-specific modelling (separate lip head, region loss weighting, lip-only models)

### Takeaway
Evidence favours decoupling or protecting the lip subspace from other objectives rather than simply up-weighting it.
- Xemo-Talker, a KDTalker follow-up on LivePortrait keypoints, shows that emotion losses on principal (articulation-heavy) directions hurt LSE; restricting them to the tail PCA subspace helps.
- Dimitra trains separate lip, expression and pose models.
- The motion renderer itself caps achievable LSE: LivePortrait keypoints beat face-vid2vid by +1.75 LSE-C in KDTalker.

### Cited Findings
- **MEASURED, KDTalker Tab. 5**: renderer choice with the same generator. Face-vid2vid (15 keypoints) LSE-C 5.579 / LSE-D 9.228; LivePortrait (21 keypoints) 7.326 / 7.548. — [KDTalker, arXiv 2503.12963](https://arxiv.org/abs/2503.12963)
- **MEASURED, Xemo-Talker Tab. 5 (MEAD; LivePortrait-keypoint motion, 70-D)**: supervising only the least-principal 10% of PCA directions gives LSE-C 6.51 / LSE-D 8.14, vs full-space 6.28 / 8.34 and head 10% 6.31 / 8.32. Tab. 4: removing the classification loss raises LSE-C to 6.91 (from 6.37), i.e. auxiliary non-lip objectives cost lip sync. Real video on MEAD: 8.04 / 7.52. — [Xemo-Talker, arXiv 2608.14700](https://arxiv.org/abs/2608.14700)
- **MEASURED, Teller Tab. 4**: single-head decoding gives Sync-C 7.790, multi-head 7.696. A small effect of splitting prediction heads. — [Teller](https://arxiv.org/abs/2503.18429)
- Dimitra: "Best results … are obtained … in case that we train separate models for lip motion, facial expression and head pose", attributed to unbalanced dimensionality. No numbers. — [Dimitra](https://arxiv.org/abs/2502.17198)
- LatentSync (a lip-only inpainting model) argues SyncNet supervision is needed mainly because masked-frame inpainting creates a visual shortcut; image-to-video animation "do[es] not suffer from the shortcut learning problem". — [LatentSync](https://arxiv.org/abs/2412.09262). The motion-space analogue: a strong prefix or reference condition could still create a "copy the previous mouth" shortcut. SPECULATIVE.
- FLOAT's motion autoencoder uses lip-region adversarial and feature-style-matching losses (λ_lip-FSM = 100) to improve lip fidelity of the latent. — [FLOAT](https://arxiv.org/abs/2412.01064)
- Other mouth designs: MoCoTalk adds a "lip consistency loss" [arXiv 2605.08050 (abstract only)](https://arxiv.org/abs/2605.08050); SkyReels-Audio uses a facial mask loss plus audio CFG [arXiv 2506.00830 (abstract only)](https://arxiv.org/abs/2506.00830). No numbers were verified for either.

### Inferences
- SPECULATIVE: test a lip-dedicated pathway. Either (a) a separate small output head or transformer branch for lip keypoints with its own per-frame audio cross-attention to a wider audio window, or (b) a mouth-openness / lip-velocity auxiliary loss matching GT statistics. Cost: small architecture change, one retrain.
- SPECULATIVE: ensure the prefix/reference conditioning cannot leak mouth state for frames under generation, e.g. drop or zero the lip dims in prefix conditioning at some probability, to avoid the LatentSync-style shortcut.
- SPECULATIVE: if the region-balanced loss already up-weights lips, further re-weighting is unlikely to move LSE much. The motion-space proxy plateau suggests the bottleneck is audio-representation quality or sync-specific supervision, not MSE weighting.

### Gaps
- No paper found that isolates lip-region loss weighting in a keypoint or 3DMM generator with LSE numbers.
- MuseTalk not reviewed.

---

## Ranked shortlist (synthesis; grades as above)

1. **Audio CFG / solver-step sweep plus separate reference-vs-audio scales.** Inference-only. MEASURED effects small to large; FLOAT Tab. 5/6, KDTalker Tab. 6, LeapTalk Tab. 2, Lip Forcing Tab. 3.
2. **Replace WavLM-last with Whisper encoder or AV-HuBERT features, or a learned multi-layer WavLM sum.** Re-extract features and retrain. MEASURED encoder effects up to 3.4 Sync-C (Teller Tab. 3; PASE table; SyncDiff Tab. 2/6). The WavLM layer choice itself is SPECULATIVE.
3. **Offline SyncNet-scored Flow-DPO fine-tune.** Render → score → best-vs-worst pairs. MEASURED +0.26 Sync-C (Playmate2 Tab. 1), +0.36 (Hallo4 Tab. 4). Days on one GPU.
4. **Pixel SyncNet loss through the frozen LivePortrait decoder on x̂₁.** MEASURED largest lever in its own setting (LatentSync Tab. 2: 4.6 → 8.9), but transfer to audio-to-motion is SPECULATIVE. Costly in memory and time.
5. **Motion-space sync expert or contrastive critic** anchored to a pretrained audio-visual representation. MEASURED gains in 3D (Learn2Talk Tab. V; arXiv 2503.20308 Tab. 2/3), with reconstruction trade-offs.
6. **Wider per-frame audio window (±4–6 frames) and a phoneme/viseme auxiliary head.** SPECULATIVE, supported indirectly (KDTalker Tab. 7, Sonic, PASE, Hallo4 Tab. 6).
7. **Protect the lip subspace from non-lip objectives; consider a lip branch.** MEASURED small (Xemo-Talker Tab. 4/5, Teller Tab. 4).

Metric warning: SyncNet confidence rewards exaggerated mouth motion (FantasyTalking2), and SyncNet is unstable even on GT pairs (arXiv 2405.04327) and to head pose (Dimitra). When optimising LSE-C directly (items 3–5), also monitor mouth-landmark distance or opening amplitude against GT and a human check.
