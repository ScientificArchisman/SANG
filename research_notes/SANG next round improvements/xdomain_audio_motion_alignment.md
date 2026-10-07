# Cross-domain methods for tight audio-to-motion temporal correspondence (for SANG-M lip-trajectory accuracy)

Status: COMPLETE 2026-10-07 (skeleton by earlier agent; extended and consolidated in this pass; raw evidence in RAW FINDINGS LOG below)
Tags: [M] measured in cited paper (number from a table), [C] claim without matching table, [I] my inference. Home domain stated per method.
Scope note: sibling note `lipsync_accuracy.md` already covers UniTalker audio-encoder ablation, Wav2Sem, SubtleTalk, ARTalk, DEMO, reward fine-tuning. Not repeated here except where needed for ranking.
Metric caveat (applies throughout): only THUNDER reports a lip **correlation** (L-PCC), which is SANG's metric. Most 3D papers report LVE (max lip-vertex L2), pixel papers report LSE-C/D (SyncNet), and V2A papers report DeSync/offset in ms. Gains are therefore translated to SANG's lip-corr only as [I] estimates.

## Q1. Lip reading / AV speech representations: better conditioning or a perceptual/consistency loss?

### Takeaway
The measured evidence favours using a frozen, **audio-aware** lip-reading / speech-from-motion expert as a **loss on the generated motion** (about −9% to −20% LVE in deterministic 3D models; +0.04 to +0.07 lip PCC in a diffusion model with a frozen SSL encoder, THUNDER). Using AV-HuBERT **as conditioning** gives only a small measured gain over HuBERT (+4.6% to +5.8% LSE-C, PASE). A contrastively sync-trained audio encoder gives more (+10% to +14% in NeRF, larger in 3DGS). The expert must be pretrained on large real data and see audio as well as lips: visual-only or from-scratch experts lose most of the gain.

### Cited Findings
- [M] THUNDER (home: diffusion 3D FLAME heads). A frozen mesh-to-speech (M2S) regressor is applied to generated motion during diffusion training. With a frozen Wav2Vec2 encoder, L-PCC went 0.568 -> 0.639 with mouth-only M2S and LVE went 0.879 -> 0.802 cm. With a trainable encoder, L-PCC went 0.623 -> 0.66. As a plug-in it helped Media2Face (0.531 -> 0.609 PCC) and DiffPoseTalk (0.612 -> 0.653). Upper-face diversity S-DIV-U fell 0.0419 -> 0.0322. — [THUNDER 2504.13386](https://arxiv.org/html/2504.13386)
- [M] SelfTalk (home: 3D mesh). In Table 4 (BIWI), removing the CTC lip-reading text loss raised LVE from 4.2485 to 4.9168. The text reports this as "an increase of 13.6% in LVE". Removing the whole lip-reading / ASR "commutative diagram" gave 5.3772. Lip-reading precision fell 88.31% -> 83.08%. — [SelfTalk 2306.10799](https://arxiv.org/html/2306.10799)
- [M] AV Guidance (home: 3D mesh). An audio-visual lip-reading expert loss cut BIWI LVE for FaceFormer 6.0449 -> 5.5061 and for CodeTalker 5.3711 -> 4.8403, and lowered CER and VER. Ablation (FaceFormer): the visual-only expert ("w/o speech") gave 6.0352, an expert without 2D pretraining gave 5.9344, and the relative-lip-vertex loss alone gave 6.0976, i.e. no gain. — [AV Guidance 2407.01034](https://arxiv.org/html/2407.01034)
- [M] AV-HuBERT as lip-sync expert (home: 2D pixel dubbing, LRS2). Switching the lip-sync loss from SyncNet to AV-HuBERT audio-vs-generated-lip features moved LMD 2.423 -> 1.188, LSE-C 7.116 -> 7.958 and LSE-D 7.396 -> 6.301. The audio-anchored ("unsupervised") variant beat the GT-lip-matching ("visual-visual") variant (LMD 1.798) and the "multimodal" variant (1.774). — [Yaman et al. 2405.04327](https://arxiv.org/html/2405.04327)
- [M] Learn2Talk (home: 3D mesh). Its motion-space SyncNet3D loss and its lip-read loss trade off against each other. On BIWI, w/o sync gave LVE 4.6971 / LSE-D 9.434, while the full model gave 5.0003 / 8.897. On VOCA, w/o sync gave 2.3581 / 11.717, while the full model gave 2.5148 / 10.593. So the sync loss improved LSE but worsened LVE by about 6–7%. The authors say the two are "not harmonious in one optimization framework". — [Learn2Talk 2404.12888](https://arxiv.org/html/2404.12888)
- [M] PASE (home: person-specific NeRF/3DGS). As drop-in audio conditioning on NeRF (LSE-C): HuBERT 7.6599, AV-HuBERT 8.0136 (+4.62%), Wav2Lip-SyncNet audio encoder 8.4461 (+10.3%), PASE 8.7098 (+13.7%), GT 8.8302. On 3DGS: HuBERT 5.0660, AV-HuBERT 5.3618 (+5.84%), Wav2Lip 6.9973 (+38.1%), PASE 8.3487 (+64.8%). Wav2Vec2 and Whisper (last layer) collapsed to about 3.3–3.4. LMD gains were smaller (−3.0% for PASE vs HuBERT on NeRF). — [PASE 2504.05803](https://arxiv.org/html/2504.05803)
- [C] SPECTRE (home: 3D face reconstruction) and EMOTE (home: emotional 3D animation) both use frozen lip-reader perceptual losses. Their ablations are qualitative or perceptual only. SPECTRE uses frame-local ResNet-18 features rather than conformer features, and reports artifacts unless the loss is paired with geometric constraints. EMOTE needs a motion prior (FLINT) to avoid "uncanny artifacts" and uses λ_lip = 2.5e-5 in a 2-epoch fine-tuning stage. — [SPECTRE 2207.11094](https://arxiv.org/html/2207.11094); [EMOTE 2306.08990](https://arxiv.org/html/2306.08990)

### Inferences
- [I] THUNDER is the closest analogue to SANG: a generative diffusion model, a frozen SSL audio encoder, and a lip-PCC metric. Its measured +0.071 PCC (frozen encoder) suggests SANG's 0.64 could reach roughly 0.68–0.71 with an M2S-style loss. That is about a third of the gap to the 0.82–0.85 re-render ceiling.
- [I] A render-free, **motion-space** expert is the one-GPU option. Train a small temporal model on SANG's 42-d lip/jaw keypoints (TalkVid GT, all 70 h) that either (a) regresses WavLM/HuBERT features or log-mel (M2S, THUNDER-style) or (b) predicts CTC phone/character targets from Whisper pseudo-transcripts (SelfTalk-style). Make it **audio-aware**, i.e. motion + audio -> transcript, following AV Guidance and Yaman et al. Freeze it, then apply the loss to the flow model's one-step x̂0 estimate, weighted toward low-noise timesteps.
- [I] Pixel-space AV-HuBERT/SyncNet losses through the LivePortrait renderer are possible: the warp + decoder are differentiable. They are expensive at 256–512 px, though, and SPECTRE reports a render-domain gap and artifacts. Keep them as a later step.
- [I] Do not use a motion-space SyncNet loss alone. Learn2Talk shows it trades lip amplitude (LVE) for timing. SANG's correlation metric is timing-sensitive, so it may still help correlation, but monitor both lip-corr and lip-amplitude error.
- [I] Expect a diversity cost (THUNDER: −23% upper-face diversity). Restrict the loss to lip/jaw channels so head and eye motion are unaffected.

### Gaps
- No measured result found for AV-HuBERT, BRAVEn, RAVEn, Auto-AVSR, AV-data2vec or SyncVSR features used as **conditioning in a multi-speaker motion generator**. The only head-to-head (PASE) uses person-specific renderers and a SyncNet-family metric that favours sync-trained features. BRAVEn, RAVEn and SyncVSR papers were not fetched: they report VSR WER, not generation.
- Untested: whether 42-d LivePortrait lip keypoints carry enough information for a lip-reader or M2S model. A quick probe is to train one on GT motion and check held-out CER or feature-regression R².
- SPECTRE and EMOTE give no numeric on/off ablation of the lip loss.

## Q2. Speech-driven 3D facial animation: which design choices reduced LVE in ablations?

### Takeaway
The largest measured LVE reductions came from four changes. (1) Expert supervision through lip reading or speech-from-motion: −10% to −20%. (2) The choice and fine-tuning of the audio encoder: −10% to −20% LVE in UniTalker; THUNDER-T gained +0.055 PCC from fine-tuning alone. (3) Explicit phonetic timing: KMTalk was −6.7% vs SelfTalk on BIWI and −30% on VOCA. (4) Multi-dataset unification, which helped only tiny datasets: −9.8% at 0.33 h, but +6.5% at 5.49 h. The FaceFormer alignment bias and periodic PE have only qualitative ablations, and SANG already has frame-aligned conditioning. Discrete motion priors helped on 4D scans but were the **worst** model on in-the-wild pseudo-GT (TFHP).

### Cited Findings
- [M] UniTalker (multi-dataset; Table 5). Unified training changed LVE by −9.8% (0.33 h set), −9.3% (0.56 h), −2.6% (0.67 h), **+6.5% (5.49 h)** and +9.7% (Chinese, under-represented). Per-dataset fine-tuning afterwards gave an average −6.3%. Training only worked with stabilisers: PCA output heads and a decoder warm-up with the audio encoder frozen, followed by joint fine-tuning. — [UniTalker 2408.00762](https://arxiv.org/html/2408.00762)
- [M] THUNDER: fine-tuning the audio encoder alone raised L-PCC from 0.568 to 0.623. — [THUNDER](https://arxiv.org/html/2504.13386)
- [M] KMTalk (ASR + Montreal Forced Aligner phoneme-boundary key frames, then audio-guided completion). BIWI LVE went 4.2485 (SelfTalk) -> 3.9654; uniform key-frame sampling gave 4.1648, and removing audio guidance gave 4.8859. On VOCA: 3.2238 -> 2.2639. It was insensitive to the ASR choice (Whisper-tiny 4.0643) and to a 1-frame offset (3.9991). As a plug-in it helped FaceFormer (5.3077 -> 5.2793), CodeTalker (4.7914 -> 4.5096) and SelfTalk (4.2485 -> 4.1122). — [KMTalk 2409.01113](https://arxiv.org/html/2409.01113)
- [M] CodeTalker: a motion codebook vs a shape-entangled codebook gave LVE 4.79 vs 6.41 (BIWI), against FaceFormer's 5.3742. Predicting codes with cross-entropy only gave 9.6356, vs 5.1138 when combined with a regression loss. — [CodeTalker 2301.02379](https://arxiv.org/html/2301.02379); [FaceFormer 2112.05329](https://arxiv.org/html/2112.05329)
- [M] DiffPoseTalk on TFHP (26.5 h, in-the-wild FLAME pseudo-GT), LVE in mm: DiffPoseTalk 8.81, FaceFormer 9.90, FaceDiffuser 12.12, **CodeTalker 12.71**. — [DiffPoseTalk 2310.00434](https://arxiv.org/html/2310.00434)
- [C] FaceFormer, qualitative only: removing the alignment bias gives "muted facial expressions"; removing wav2vec init gives no sync; sinusoidal PE jitters in silences on long sequences; ALiBi alone "freezes" on VOCASET. — [FaceFormer](https://arxiv.org/html/2112.05329)

### Inferences
- [I] For SANG, the highest-evidence Q2 levers are fine-tuning WavLM-large after a warm-up (THUNDER-T, UniTalker DW), or a cheaper LoRA / learned layer-weighting, plus phonetic timing features (KMTalk). Both are compatible with a ~53M generator.
- [I] At 70 h, multi-dataset scaling is not a lever. UniTalker's gains vanish at ≥5 h per set. Language balance inside TalkVid matters more (Chinese +9.7% in UniTalker until fine-tuned).
- [I] De-prioritise VQ / discrete priors. They counter over-smoothing in deterministic regressors on clean scans, but SANG is already generative and trained on noisy tracked data, where CodeTalker was the worst model.
- [I] Make sure audio reaches the DiT only through strictly local, frame-aligned paths. FaceFormer (no diagonal mask -> muted) and MMAudio (soft cross-attention "hamper[s] precision", Q5) agree on this.

### Gaps
- Not fetched due to budget: Media2Face, ScanTalk, EmoTalk and FaceDiffuser ablations. DiffPoseTalk's ablation rows were not retrieved.
- FaceFormer's alignment-bias and periodic-PE ablations have no numbers.
- LVE (max-vertex L2) does not map directly to trajectory correlation, so no conversion is attempted.

## Q3. Co-speech gesture and dance: rhythm/beat losses, contrastive pretraining, multi-scale audio

### Takeaway
Gesture and dance "alignment" is measured at onset level (BC/BAS). These metrics saturate above ground truth (EDGE), and beat-reward gains are small (Bailando +3.9% BAS). The transferable items are therefore objectives and conditioning, not beat losses: (1) **contrastive flow matching** with mismatched-condition negatives (SemConFlow; ΔFM reports up to 9× faster training and up to 8.9 lower FID on images); (2) adaptive per-frame fusion of explicit rhythm features (onset, amplitude) with content features (EMAGE CRA); (3) audio classifier-free guidance (EDGE; V-AURA in Q5).

### Cited Findings
- [M] SemConFlow (home: holistic gesture incl. face, BEAT2/SHOW). Its loss is L = ||v − v̂||² − λ||v − ṽ||², where ṽ is a mismatched sample's velocity with the same noise and λ ∈ [0,1). [C] Replacing it with standard FM "degrades the FGD metric" and lowers BC and diversity. Audio-only alignment best preserves BC. (Table 2 numbers not retrieved.) — [SemConFlow 2603.26553](https://arxiv.org/html/2603.26553)
- [C, abstract] ΔFM (home: image flow matching): up to 9× faster training, up to 5× fewer steps, and up to −8.9 FID vs FM on ImageNet-1k / CC3M. — [Contrastive Flow Matching 2506.05350](https://arxiv.org/abs/2506.05350)
- [M] EMAGE (home: holistic gesture + FLAME face, BEAT2 60 h). Rhythm (onset + amplitude via TCN) and content (word embeddings) are fused per frame with learned α. [C] Fusion "shows improvement in both FGD and Alignment"; one VQ-VAE for the whole body hurts the face, so per-part VQ-VAEs are used. BEAT2 GT BC is 6.896. — [EMAGE 2401.00374](https://arxiv.org/html/2401.00374)
- [M] Bailando (home: dance). Actor-critic beat-align reward: BAS 0.2245 -> 0.2332 and FID_g 11.82 -> 9.62. — [Bailando 2203.13055](https://arxiv.org/abs/2203.13055)
- [M] EDGE (home: dance diffusion, **49M params**, 4×A100 for 16 h). Guidance w=2 vs w=1: Elo 1751 vs 1601, beat-align 0.26 vs 0.27, GT 0.24. [C] Beat alignment "loses its meaning when candidate examples reach quality on par with ground truth". — [EDGE 2211.10658](https://arxiv.org/html/2211.10658)

### Inferences
- [I] Contrastive FM is the most SANG-portable idea from this field. It needs zero extra parameters and no extra forward pass: reuse in-batch targets from other clips as negatives. It directly penalises the audio-agnostic "average mouth" mode that limits correlation. Confidence is medium, because there are no lip-specific numbers.
- [I] SANG conditions only on WavLM's last layer. Adding explicit low-level envelope channels (RMS energy, onset strength, optionally pitch) at 50 Hz next to WavLM, with per-frame gating, is a near-free experiment (EMAGE CRA). Mouth opening correlates with acoustic energy.
- [I] Beat rewards and BC/BAS losses are too coarse (onsets within ±σ) for phoneme-level lip timing. Skip them.

### Gaps
- Not fetched: DiffSHEG, TalkSHOW, Lodge, Bailando++, BEAT2 contrastive-pretraining papers (CSMP 2309.05455 noted but not read). Ablation numbers for SemConFlow and EMAGE were not retrieved.
- No gesture paper reports a lip-trajectory correlation.

## Q4. TTS / VC alignment: what transfers (CTC / phoneme heads, MAS, alignment-aware conditioning)?

### Takeaway
TTS alignment machinery (MAS, CTC forward-sum, beta-binomial priors, duration predictors) solves an **unknown monotonic text→frame alignment**. SANG does not have that problem: audio and motion share a clock. What does transfer from flow-matching TTS: (1) **Sway Sampling**, a training-free sampler schedule (F5-TTS: WER −15% to −18% at 32 NFE, and 16 NFE with it beats 32 without); (2) refining the conditioning stream with conv blocks before the DiT (F5 vs E2: WER 4.17 vs 9.63 at equal size; a pure adaLN DiT "failed to learn alignment"); (3) a staged CFG-dropout schedule. Phonetic auxiliary supervision (CTC phone heads) is better evidenced in the 3D-face papers (SelfTalk, KMTalk) than in TTS.

### Cited Findings
- [M] F5-TTS architecture (155M small models, Seed-TTS test-zh): F5-TTS (ConvNeXt text refinement) WER 4.17 / SIM 0.54 vs E2 TTS 9.63 / 0.53, with E2 failing on 7% of samples. [C] A pure adaLN DiT without refinement "failed to learn alignment"; MMDiT "learned fast and collapsed fast". GT durations improve WER only 4.17 -> 3.87. — [F5-TTS 2410.06885](https://arxiv.org/html/2410.06885)
- [M] Sway Sampling (s=−1), 32 NFE without -> with: LibriSpeech-PC WER 2.84 -> 2.41 and SIM 0.62 -> 0.66; Seed-TTS zh WER 1.93 -> 1.58; E2 TTS zh 1.97 -> 1.77. [C] The "leak and override" experiment shows that early flow steps decide adherence to the condition. — [F5-TTS](https://arxiv.org/html/2410.06885)
- [M] F5 CFG training: the prompt is dropped with p=0.3, then prompt + text with p=0.2. — [F5-TTS](https://arxiv.org/html/2410.06885)
- [C] One TTS Alignment (forward-sum CTC + static prior): parallel TTS converges as fast as with MFA durations, alignments are closer to hand-annotated durations, and fewer words are repeated or skipped (plots only). — [One TTS Alignment 2108.10447](https://arxiv.org/html/2108.10447)

### Inferences
- [I] Sway Sampling is a zero-cost inference experiment for SANG's flow sampler: sweep s ∈ {0, −0.4, −0.8, −1} at fixed NFE and check lip-corr. WER is the TTS analogue of "faithfulness to the condition", and lip-corr is SANG's analogue.
- [I] SANG's ±80 ms conv over WavLM is a minimal "Conv2Text". A 2–4-block depthwise ConvNeXt refinement of the audio stream (still local, ±160–240 ms receptive field), with per-frame injection, costs about 1–2M params and follows the F5 finding.
- [I] A CTC phone head on intermediate DiT features, or phone-posterior input features, gives the generator explicit phonetic targets. Prefer a multilingual phoneme recogniser (CTC) over MFA, since MFA is not available for many TalkVid languages.
- [I] Duration predictors and MAS do not transfer. Learned A/V offset handling transfers as preprocessing (Q5: LatentSync).

### Gaps
- Not retrieved: a flow-matching TTS ablation that measures a CTC auxiliary head on the DiT (e.g. DiTTo-TTS, A-DMA). E2 TTS 2406.18009 was not fetched separately.
- Sway Sampling has never been tested on motion generation.

## Q5. Video-to-audio / audio-to-video sync: frame-aligned conditioning, aligned RoPE, sync features

### Takeaway
V2A gives the cleanest **measured** sync levers, in order of effect size. Sync-trained frame-rate features injected per token halve the error (MMAudio DeSync 0.973 -> 0.483 s). Per-timestep fusion instead of prepending also halves it (V-AURA 105 -> 49 ms). Audio CFG scale cuts offset about 3× (155 -> 50 ms). Filtering training data by AV-correspondence score gives 60 -> 49 ms with 2.5× less training. ConvMLP instead of MLP gives 0.533 -> 0.483, and aligned RoPE gives 0.509 -> 0.483. For the inverse direction (audio -> motion), SANG already has frame-aligned fusion. The transferable pieces are a sync-trained audio stream, per-frame adaLN/gating injection, conv FFNs, CFG tuning, and data offset correction / filtering.

### Cited Findings
- [M] MMAudio (home: V2A flow-matching DiT), Table 6 DeSync (s): sync module 0.483; sync features summed into the visual branch 0.490; **no sync features 0.973**; aligned RoPE 0.483 vs none 0.509 vs non-aligned 0.496. Table 7: ConvMLP 0.483 vs MLP 0.533. Sync features come from Synchformer's visual encoder (a self-supervised AV-offset detector, 24 fps), passed through ConvMLP, upsampled, and injected per token via adaLN. [C] Cross-modal attention "aggregate[s] features via a soft distribution ... hamper[ing] precision". — [MMAudio 2412.15322](https://arxiv.org/html/2412.15322)
- [M] V-AURA (home: autoregressive V2A; Sync = mean |offset| in ms via Synchformer). Prepend 105 vs fusion 49 (Table III). AV-similarity data filtering at thresholds 0.0 / 0.2 / 0.3 / 0.4 gave Sync 60 / 59 / 49 / 71 ms, using 708 / 662 / 278 / 168 GPU-h (Table II). CFG 1 / 3 / 5 / 6 / 7 / 9 gave Sync 155 / 80 / 52 / 50 / 55 / 53 ms (Table IV). — [V-AURA 2409.13689](https://arxiv.org/html/2409.13689)
- [M] Diff-Foley (home: V2A LDM). CAVP contrastive pretraining with within-clip temporal negatives reached Align Acc 94.05%. [C] CAVP features beat CLIP on sync (Table 2 numbers not retrieved). Inference uses double guidance (CFG + alignment classifier). — [Diff-Foley 2306.17203](https://arxiv.org/html/2306.17203)
- [C/M] LatentSync (home: latent-diffusion lip-sync). In-the-wild clips are offset-corrected with SyncNet, and clips with Sync_conf < 3 are dropped. Without offset adjustment, SyncNet "convergence is significantly impaired" (plot). Without SyncNet supervision the diffusion model shows a "shortcut learning problem": it copies visual context instead of using audio. TREPA lowers FVD 176.35 -> 162.74. — [LatentSync 2412.09262](https://arxiv.org/html/2412.09262)

### Inferences
- [I] SANG's audio is WavLM's last layer, trained for masked-unit prediction, not synchrony. MMAudio and PASE both show that a **sync-trained** encoder stream adds timing information beyond semantic SSL features. Proposal: train a SyncNet-style contrastive model on TalkVid with within-clip temporal negatives at ±2–15 frame offsets (Diff-Foley, Learn2Talk SyncNet3D recipe; ~12–15 h on one GPU per Learn2Talk). Pair audio windows with GT 42-d lip-motion windows. Freeze its audio tower and feed its 25–50 Hz embeddings as a second frame-aligned stream, injected per frame through adaLN scale/shift or gating. The same model doubles as an evaluation or guidance scorer, but use a separately trained instance for evaluation, as Learn2Talk does, to avoid metric coupling.
- [I] Data hygiene on TalkVid is cheap and measured to matter in two fields. Run SyncNet (or the TalkVid-trained motion SyncNet) to estimate the per-clip AV offset, shift the audio, and drop the lowest-confidence ~10–30% (V-AURA's optimum dropped ~50%, but over-filtering hurt).
- [I] LatentSync's "shortcut" warning applies to SANG's reference conditioning. If the reference (or any prior-window motion) carries mouth state, the DiT can under-use audio. Check by measuring lip-corr with shuffled vs true references.
- [I] Aligned RoPE alone gives about −5% in MMAudio and is likely redundant for SANG, which already uses shared-clock additive conditioning.

### Gaps
- Not found: an audio-to-motion or audio-to-talking-head paper that reuses Synchformer-style sync features as **conditioning** with an ablation. PASE (person-specific NeRF/3DGS) is the closest.
- Diff-Foley's Table 2 numbers were not retrieved.

## Top-8 ranked for SANG-M (same param count, one GPU)
Ranking = (measured evidence strength × expected lip-corr gain) / (one-GPU cost × risk). Baseline lip-corr is 0.64; the re-render ceiling is 0.82–0.85. All expected deltas are [I] extrapolations unless marked. Gains will not add up linearly.

1. **Inference-only: audio CFG-scale sweep + Sway Sampling** (sampler change, 0 params, 0 training).
   - Change: sweep audio guidance w ∈ {1…7} × Sway s ∈ {0, −0.4, −0.8, −1} at fixed NFE, scoring lip-corr and jitter.
   - Expected: +0.00 to +0.04 [I]. It also tells you how much of the gap is a sampler problem.
   - Cost: a few GPU-hours of evaluation.
   - Risk: low. High w can over-articulate or jitter.
   - Evidence: [M] V-AURA CFG 1 -> 6, Sync 155 -> 50 ms; [M] F5-TTS Sway WER 2.84 -> 2.41 and 1.93 -> 1.58; [M] EDGE w=2 Elo +150. — [V-AURA](https://arxiv.org/html/2409.13689); [F5-TTS](https://arxiv.org/html/2410.06885); [EDGE](https://arxiv.org/html/2211.10658)

2. **Motion-space, audio-aware lip expert as a consistency loss** (loss; expert ~2–5M params, outside the generator).
   - Change: train a small temporal net on TalkVid GT 42-d lip/jaw keypoints (+ audio) to regress WavLM or log-mel features (M2S) and/or a CTC phone/character target (Whisper pseudo-transcripts). Freeze it. Apply it to the flow model's one-step x̂0, weighted toward low-noise timesteps, on lip channels only.
   - Expected: +0.04 to +0.07 lip-corr (THUNDER's measured range on L-PCC).
   - Cost: expert about 0.5–1 GPU-day; fine-tuning SANG about 1–2 GPU-days.
   - Risk: medium. Diversity drops (THUNDER −23% upper-face diversity); the generator can exploit the expert; the expert must be audio-aware and pretrained on all 70 h.
   - Evidence: [M] THUNDER +0.071 / +0.037 PCC; [M] SelfTalk −13.6% LVE from CTC; [M] AV Guidance −9% to −10% LVE, with visual-only or from-scratch experts losing most of the gain. — [THUNDER](https://arxiv.org/html/2504.13386); [SelfTalk](https://arxiv.org/html/2306.10799); [AV Guidance](https://arxiv.org/html/2407.01034)

3. **TalkVid AV-offset correction + sync-confidence filtering** (preprocessing).
   - Change: estimate the per-clip offset with SyncNet (on rendered or real crops), shift the audio, and drop clips below a confidence threshold, sweeping the cut at 10/20/30%.
   - Expected: +0.01 to +0.04 [I]. A constant offset in part of the data directly caps trajectory correlation, and this also cleans the evaluation set.
   - Cost: one SyncNet pass over 70 h (≈ a few GPU-hours) plus a retrain.
   - Risk: low. Over-filtering hurt V-AURA at the 0.4 threshold.
   - Evidence: [M] V-AURA filtering 60 -> 49 ms with 2.5× less training; [C] LatentSync: no offset adjustment "significantly impaired" convergence. — [V-AURA](https://arxiv.org/html/2409.13689); [LatentSync](https://arxiv.org/html/2412.09262)

4. **Fine-tune the audio encoder after decoder warm-up** (encoder; LoRA on WavLM-large top layers ≈ +1–3M params, or a learned layer-weighted sum).
   - Expected: +0.03 to +0.05 [I].
   - Cost: 1.5–3× step time if WavLM is unfrozen (cache features for the frozen layers).
   - Risk: medium. It can overfit 70 h or erode multilingual robustness; keep a low LR and freeze lower layers.
   - Evidence: [M] THUNDER-T PCC 0.568 -> 0.623 from encoder fine-tuning alone, and 0.66 combined with M2S; [M] UniTalker's DW -> joint FT recipe. — [THUNDER](https://arxiv.org/html/2504.13386); [UniTalker](https://arxiv.org/html/2408.00762)

5. **Sync-trained second audio stream with per-frame adaLN/gating injection** (encoder + conditioning; +1–3M params).
   - Change: train a SyncNet-style contrastive model on TalkVid with within-clip temporal negatives, pairing audio windows with GT lip-motion windows. Freeze its audio tower and inject its frame features per frame (scale/shift), alongside WavLM.
   - Expected: +0.02 to +0.05 [I].
   - Cost: about 12–15 GPU-h to train the SyncNet (Learn2Talk-scale) plus a retrain.
   - Risk: medium. Features may be redundant with WavLM; use a different instance for evaluation to avoid metric coupling.
   - Evidence: [M] MMAudio DeSync 0.973 -> 0.483 with sync features; [M] PASE / Wav2Lip-SyncNet features +10% to +14% LSE-C over HuBERT vs AV-HuBERT +5%. — [MMAudio](https://arxiv.org/html/2412.15322); [PASE](https://arxiv.org/html/2504.05803); [Learn2Talk](https://arxiv.org/html/2404.12888)

6. **Contrastive flow matching with mismatched-audio negatives** (loss; 0 params, ~0 compute).
   - Change: L = MSE(v, v̂) − λ·MSE(v, ṽ_other-clip), sweeping λ ∈ {0.02, 0.05, 0.1}.
   - Expected: +0.01 to +0.03 [I]. It may also allow fewer sampling steps.
   - Cost: a retrain or fine-tune.
   - Risk: low–medium. A high λ destabilises training; there is no lip-specific evidence.
   - Evidence: [C] SemConFlow (FM vs contrastive FM degrades FGD, BC, diversity; numbers not retrieved); [C, abstract] ΔFM up to 9× faster training and up to −8.9 FID. — [SemConFlow](https://arxiv.org/html/2603.26553); [ΔFM](https://arxiv.org/abs/2506.05350)

7. **Explicit phonetic + envelope conditioning channels** (conditioning/preprocessing; <1M params).
   - Change: add per-frame phone posteriors from a multilingual CTC phoneme recogniser, or phone-boundary flags, plus RMS/onset envelope channels at 50 Hz, concatenated to WavLM before the local conv. Optionally add an auxiliary CTC phone head on mid-DiT features.
   - Expected: +0.01 to +0.03 [I].
   - Cost: offline feature extraction (≈ 1 GPU-day for 70 h) plus a retrain.
   - Risk: low–medium. Recogniser quality varies by language, and MFA is unavailable for many TalkVid languages.
   - Evidence: [M] KMTalk −6.7% (BIWI) / −30% (VOCA) LVE from phoneme-boundary key motions, robust to ASR choice; [M] SelfTalk CTC −13.6%; [C] EMAGE rhythm+content fusion improves alignment. — [KMTalk](https://arxiv.org/html/2409.01113); [SelfTalk](https://arxiv.org/html/2306.10799); [EMAGE](https://arxiv.org/html/2401.00374)

8. **Local temporal inductive bias in the DiT** (architecture at equal params).
   - Change: kernel-3 temporal ConvMLP FFNs, a 2–4-block depthwise ConvNeXt refinement of the audio stream before injection, and no soft cross-attention path for audio. Also check reference-frame "shortcut" leakage with a shuffled-reference test.
   - Expected: +0.01 to +0.02 [I].
   - Cost: a retrain.
   - Risk: low.
   - Evidence: [M] MMAudio ConvMLP 0.533 -> 0.483 DeSync and aligned RoPE only 0.509 -> 0.483; [M] F5 ConvNeXt text refinement WER 4.17 vs E2 9.63, with pure adaLN DiT "failed to learn alignment"; [C] LatentSync shortcut problem. — [MMAudio](https://arxiv.org/html/2412.15322); [F5-TTS](https://arxiv.org/html/2410.06885); [LatentSync](https://arxiv.org/html/2412.09262)

**Deprioritised (evidence against or weak for SANG's regime):**
- VQ / discrete motion priors: CodeTalker was the worst on in-the-wild TFHP (12.71 vs 9.90 mm).
- Multi-dataset scaling: no gain at ≥5 h per set in UniTalker.
- A motion SyncNet loss on its own: Learn2Talk LVE +6% to +7%.
- Beat / BC rewards: onset-level, and they saturate above GT.
- Aligned RoPE on its own: −5%.
- Pixel-space AV-HuBERT loss through LivePortrait: plausible, since Yaman et al. measured LMD −51%, but costly and prone to render-domain-gap artifacts (SPECTRE). Revisit only after item 2.

**Suggested order:** do 1 and 3 first (cheap; they also fix the evaluation and data). Then 2 (largest measured gain). Then 4 or 5 (encoder side; pick one first to avoid confounding). Treat 6, 7 and 8 as low-cost add-ons to the next full retrain.

---
## RAW FINDINGS LOG (appended during research)

### SelfTalk (arXiv 2306.10799, ACM MM 2023; home: speech-driven 3D mesh animation, VOCASET/BIWI) — lip-reading consistency loss
- [M] Architecture: facial animator + frozen-ish speech recognizer (ASR gives pseudo-text) + lip-reading interpreter (lip encoder on generated mesh -> text decoder, CTC). Losses: L = 1000·L_rec + 1000·L_vel + 0.001·L_lat (MSE between audio latent and lip-encoder latent, per frame) + 0.0001·L_ctc (CTC between lip-reader output on generated mesh and ASR pseudo-transcript, vocab 33). — [SelfTalk](https://arxiv.org/html/2306.10799)
- [M] Table 4 ablation, BIWI-Test-A (LVE ×1e-4 ↓ / FDD ↓ / LRP lip-reading precision ↑): Full 4.2485 / 3.5761 / 88.31%; w/o L_vel 4.6655 / 4.0681 / 84.15%; w/o L_lat 4.6846 / 3.6114 / 86.88%; w/o L_text(CTC) 4.9168 / 3.7993 / 85.87%; w/o commutative diagram (animator only) 5.3772 / 4.0378 / 83.08%. Text: removing CTC "led to an increase of 13.6% in LVE"; removing the whole diagram ">20% in LVE". — [SelfTalk](https://arxiv.org/html/2306.10799)
- [M] User study lip-sync preference vs CodeTalker 62.1% (BIWI-B) / 67.2% (VOCA). — [SelfTalk](https://arxiv.org/html/2306.10799)
- [I] Caveat: small single-speaker-ish lab datasets; 13.6% gain is relative to a weak deterministic baseline; the lip-reader is trained jointly on GT meshes (not frozen). For SANG the analogue is a lip-reader on 42-d LivePortrait lip keypoint trajectories trained on TalkVid GT motion + ASR (Whisper) pseudo-transcripts.

### THUNDER (arXiv 2504.13386 v4, 27 Jan 2026; Daněček, Schmitt, Polikovsky, Black, MPI-IS; home: stochastic diffusion 3D talking heads, FLAME) — analysis-by-audio-synthesis
- [M] Train a mesh-to-speech (M2S) regressor (feed-forward, based on Choi et al. video-to-speech) on pseudo-GT FLAME from video; freeze it; during diffusion training decode generated animation -> audio representation and compare to input speech. Variants: face2s (all vertices), exp2s (FLAME expr params), mouth2s (mouth vertices). Audio encoder: Wav2Vec2 at 25 Hz. — [THUNDER](https://arxiv.org/html/2504.13386)
- [M] Table 2 (THUNDERSET; LVE cm ↓ / L-CCC ↑ / L-PCC ↑ / DTW ↓; plus S-DIV-U ↑ upper-face diversity):
  - THUNDER (frozen W2V) w/o m2s 0.879 / 0.359 / 0.568 / 0.329 (S-DIV-U 0.0419); w/ face2s 0.804 / 0.411 / 0.633 / 0.285 (0.0297); w/ exp2s 0.83 / 0.362 / 0.63 / 0.296 (0.0404); w/ mouth2s 0.802 / 0.426 / 0.639 / 0.29 (0.0322).
  - THUNDER-T (trainable W2V) w/o m2s 0.723 / 0.428 / 0.623 / 0.266; w/ mouth2s 0.709 / 0.445 / 0.66 / 0.256.
  - Plug-in to other methods: Media2Face* 0.96/0.308/0.531 -> w/ m2s 0.815/0.428/0.609; DiffPoseTalk* 0.68/0.464/0.612 -> 0.651/0.469/0.653; FlameFormer* 0.809/0.368/0.57 -> 0.794/0.411/0.614; FlameSelfTalk* 0.695/0.504/0.698 -> 0.674/0.518/0.705. — [THUNDER](https://arxiv.org/html/2504.13386)
- [M] Table 3/extended (TFHP, with head pose): THUNDER-F (frozen) w/o m2s LVE 1.2 / L-PCC 0.388 -> with m2s 1.07 / 0.435; THUNDER-T (trainable) w/o 1.12 / 0.488 -> with 0.987 / 0.55; DiffPoseTalk (HuBERT trainable) 1.01 / 0.541. — [THUNDER](https://arxiv.org/html/2504.13386)
- [M] Lip PCC gains from M2S: +0.071 (0.568->0.639, frozen encoder, THUNDERSET), +0.037 (trainable), +0.047 (TFHP frozen). Fine-tuning the audio encoder ALONE gives +0.055 PCC (0.568->0.623). Sensitivity: "The stronger the loss, the more the lip-sync metrics improve. However, increasing weights ... come at the expense of generation diversity." — [THUNDER](https://arxiv.org/html/2504.13386)
- [M] Note also SelfTalk-style deterministic regression (FlameSelfTalk*) reaches L-PCC 0.698 vs stochastic THUNDER 0.639–0.66 on the same data: deterministic+lip-reading still beats stochastic+M2S on raw correlation. — [THUNDER](https://arxiv.org/html/2504.13386)
- [I] Most SANG-like evidence found so far: frozen SSL audio encoder + generative (diffusion) model + PCC metric on lips. Gain ≈ +0.04–0.07 PCC absolute (~8–12% relative). For SANG: train a motion-to-speech-feature regressor (42-d lips -> WavLM/HuBERT-unit or mel features) on TalkVid GT motion; apply on the one-step x0 estimate of the flow model with timestep-weighting.

### AV-HuBERT as lip-sync expert (Yaman et al., arXiv 2405.04327, CVPRW 2024; home: 2D pixel lip-sync / visual dubbing, LRS2/LRW/HDTF)
- [M] Replaces Wav2Lip's SyncNet lip-expert with frozen AV-HuBERT (lip-reading-finetuned) features in a cross-entropy lip-sync loss. Motivation [M, Fig. 1/5]: SyncNet cosine similarity fluctuates even on GT LRS2 pairs and is not shift-invariant; AV-HuBERT features are stable under shift/rotation. — [Yaman et al.](https://arxiv.org/html/2405.04327)
- [M] Table 4 ablation, LRS2 (LMD ↓ / LSE-C ↑ / LSE-D ↓ / AVSu ↑ / AVSm ↑ / AVSv ↑): Baseline (SyncNet lip-expert loss) 2.423 / 7.116 / 7.396 / 0.301 / 0.637 / 0.423; AV-HuBERT "visual-visual" (gen lips vs GT lips, no audio) 1.798 / 7.481 / 6.556 / 0.381 / 0.765 / 0.545; "multimodal" (gen-lip+audio vs GT-lip+audio) 1.774 / 6.998 / 6.794 / 0.395 / 0.789 / 0.575; "unsupervised" (gen-lip feature vs audio feature, cross-entropy) 1.188 / 7.958 / 6.301 / 0.508 / 0.939 / 0.879. I.e. audio-vs-generated-lip AV-HuBERT loss halves LMD (2.423 -> 1.188) vs SyncNet loss. — [Yaman et al.](https://arxiv.org/html/2405.04327)
- [M] Caveat: AVS metrics are computed with the same AV-HuBERT used as the loss (metric/loss coupling); LMD and LSE are independent and still improve. HDTF Table 2: LSE-C 8.106 vs Wav2Lip 9.054 (Wav2Lip higher on LSE-C but loses user study 2.91 vs 3.92 sync). — [Yaman et al.](https://arxiv.org/html/2405.04327)
- [I] Transfer to SANG requires differentiable rendering of 42-d motion -> mouth crops (LivePortrait warp+decoder is differentiable but ~expensive) OR a motion-space lip-reader. The "unsupervised" (audio <-> generated-lip) variant is the one that wins, which argues for an audio-anchored consistency loss, not just a GT-matching one.

### SPECTRE (arXiv 2207.11094, Filntisis et al.; home: monocular 3D face reconstruction from video, FLAME/DECA) — perceptual lip-reading loss
- [M] Lip-reading loss = feature distance between frozen lip-reader (Ma et al. Auto-AVSR-family, LRS3-trained; 3D conv + ResNet-18 + 12-layer conformer) on input mouth crops vs differentiably rendered mouth crops. Uses ResNet-18 (frame-local) features, not conformer features, because "conformer ... features are largely affected by the sequence context" and gave weaker visual correspondence (Appendix B.1, qualitative only). — [SPECTRE](https://arxiv.org/html/2207.11094)
- [C] Without geometric (landmark) constraints the lip-read loss "from a threshold and lower ... tends to create artifacts" -> must be paired with a geometric term. Evaluation via AV-HuBERT CER/WER on renders (Table 1), but no numeric on/off ablation of the lip loss retrieved; only qualitative Fig. 6. — [SPECTRE](https://arxiv.org/html/2207.11094)

### EMOTE (arXiv 2306.08990, Daněček et al., SIGGRAPH Asia 2023; home: emotional speech-driven 3D FLAME animation, trained on MEAD pseudo-GT)
- [M] Two-stage: stage 1 vertex MSE only (λ_rec=1); stage 2 freezes wav2vec, adds differentiable rendering + lip-reading loss (λ_lip = λ_lip^dis = 2.5e-5) + video emotion loss (2.5e-6), 2 more epochs. Output decoded through FLINT (temporal VAE motion prior) to avoid artifacts from perceptual losses. — [EMOTE](https://arxiv.org/html/2306.08990)
- [M] Ablation is perceptual only (Fig. 7, MTurk Likert): "EMOTE w/o the lip-reading loss ... suffers from inaccurate lip-sync"; full EMOTE preferred over all ablations on lip-sync. No LVE/PCC numbers for the lip-reading-loss ablation. — [EMOTE](https://arxiv.org/html/2306.08990)
- [C] Key design lesson: perceptual losses need a motion prior (FLINT) or they create "uncanny artifacts"; same group later reports the M2S approach (THUNDER) with numbers. — [EMOTE](https://arxiv.org/html/2306.08990)

### Learn2Talk (arXiv 2404.12888, Zhuang et al., 2024; home: speech-driven 3D FLAME/mesh, VOCASET/BIWI) — motion-space SyncNet3D + lip-reading teacher
- [M] SyncNet3D: contrastive (margin) audio-mel vs 3D mouth-mesh window encoder (W=5 frames @25fps BIWI, 6 @30fps VOCA), trained separately (12–15 h on one RTX 4090); a frozen instance is used as a sync-loss discriminator, a different instance for LSE-D/C evaluation. Lipread loss = differentiable render + lip-reading network + 2D teacher (talking-face model). Student training 3–5 h on one RTX 4090. — [Learn2Talk](https://arxiv.org/html/2404.12888)
- [M] Table V ablation (LSE-D ↓ / LSE-C ↑ / LVE ↓ / FDD ↓): BIWI w/o sync 9.434 / 8.933 / 4.6971 / 3.4083; w/o lipread 8.563 / 9.652 / 5.0375 / 5.1844; full 8.897 / 9.449 / 5.0003 / 3.7756. VOCASET w/o sync 11.717 / 6.286 / 2.3581; w/o lipread 10.956 / 9.926 / 2.8233; full 10.593 / 9.838 / 2.5148. — [Learn2Talk](https://arxiv.org/html/2404.12888)
- [M] Authors' reading: "there is a tradeoff between the lipread loss and the 3D sync loss ... not harmonious in one optimization framework": sync loss improves LSE (temporal "open as quickly as possible") but raises LVE; lipread loss lowers LVE ("open as much as possible"); weights tuned over "4 or 5 parameter combinations" (λ1=5e-6, λ2=1e-2 BIWI; 5e-4, 5e-7 VOCA). — [Learn2Talk](https://arxiv.org/html/2404.12888)
- [M] Downstream: lip-reading WER (video-only, LRS3 subset) on synthesized video 50.28 (Learn2Talk) vs 56.99 FaceFormer / 56.45 CodeTalker (Table VI). — [Learn2Talk](https://arxiv.org/html/2404.12888)
- [I] Important negative for SANG: a motion-space SyncNet loss alone moved LVE the wrong way (+6.5% BIWI, +6.7% VOCA vs w/o sync). Since SANG's target metric is trajectory correlation (closer to "timing" than "amplitude"), a sync/contrastive loss may still help correlation while hurting L2; must monitor both.

### FaceFormer (arXiv 2112.05329, CVPR 2022; home: speech-driven 3D mesh, VOCASET/BIWI) — alignment bias, periodic PE, wav2vec init
- [M] Table 1 BIWI-Test-A LVE (×1e-4 mm): VOCA 7.6427, MeshTalk 6.7436, FaceFormer 5.3742. — [FaceFormer](https://arxiv.org/html/2112.05329)
- [C] Ablations are qualitative/video-only (no LVE table for them): w/o alignment bias (diagonal cross-attention mask) "tends to generate muted facial expressions across all frames" -> "indispensable"; w/o wav2vec init "can not produce synchronized mouth motions"; TCN without encoder self-attention "often fails to close the mouth"; Original sinusoidal PE gives jitter in silences for sequences longer than training; ALiBi alone "quickly freezes to a static facial expression" on VOCASET. — [FaceFormer](https://arxiv.org/html/2112.05329)
- [I] SANG already uses frame-aligned additive conditioning (2 WavLM ticks per frame), which is the strongest form of "alignment bias"; nothing new to transfer except that a cross-attention path WITHOUT a diagonal mask degrades lip motion — keep audio injection strictly local/aligned.

### CodeTalker (arXiv 2301.02379, CVPR 2023; home: speech-driven 3D mesh) — discrete motion prior
- [M] Table 3 (BIWI-Test-A LVE ×1e-4 mm): shape-entangled codebook 6.41 vs motion (speaker-agnostic) codebook 4.79; vs FaceFormer 5.3742 (FaceFormer Table 1). Codebook N=256, P=1 frame per code, H=8 face components (BIWI). Larger temporal unit P degrades both reconstruction and LVE (Fig. 7, values only in plot). Appendix Table 5: Instance Normalization in the VQ autoencoder lowers recon error 3.27 -> 2.83 (BIWI) and "predicted lip amplitudes are closer to those of the ground truth". — [CodeTalker](https://arxiv.org/html/2301.02379)
- [M] Appendix Table 6: alternative of predicting code indices with pure cross-entropy gives LVE 9.6356 vs 5.1138 when CE is combined with a regression loss -> regression in continuous space is needed. — [CodeTalker](https://arxiv.org/html/2301.02379)
- [I] Discrete priors mainly fight over-smoothing of deterministic regressors; SANG is already a generative flow model so the expected gain is smaller. Not a top pick.

### UniTalker multi-dataset + encoder fine-tuning (arXiv 2408.00762, ECCV 2024; home: speech-driven 3D mesh/blendshape) — complements sibling note (encoder ablation there)
- [M] Table 5 (LVE; D0 ×1e-4, D1–D3 ×1e-6 m², D4–D7 ×1e-5 m²; hours per set 0.33/0.56/0.67/5.49/1.48/3.65/1.24/5.11): single-dataset L-[D*] 4.279 / 9.153 / 8.881 / 8.445 / 1.370 / 2.040 / 1.043 / 1.235; unified L-[D0-D7] 3.859 (−9.8%) / 8.303 (−9.3%) / 8.648 (−2.6%) / 8.991 (+6.5%) / 1.326 (−3.2%) / 2.056 (+0.8%) / 1.145 (+9.7%) / 1.211 (−1.9%); fine-tuned from unified L-FT 3.816 (−11%) / 8.060 (−12%) / 8.56 / 8.417 / 1.30 / 1.848 (−9.4%) / 0.998 / 1.178; "average LVE drop across datasets is 6.3%". — [UniTalker](https://arxiv.org/html/2408.00762)
- [M] Multi-dataset gains concentrate on the smallest sets (0.33 h, 0.56 h); the 5.49 h set got WORSE (+6.5%) and the under-represented Chinese set got worse (+9.7%) until per-dataset fine-tuning. Stabilisers needed: PCA output heads (3V -> 512) and two-stage Decoder Warm-up (frozen audio encoder first, then joint fine-tune); without them the "vanilla multi-head model fails to gain advantages from increased data size". — [UniTalker](https://arxiv.org/html/2408.00762)
- [I] SANG already has ~70 h — an order of magnitude more than any single A2F-Bench set — so "more data of the same kind" is unlikely to be the lever; the transferable parts are (i) decoder-warm-up -> unfreeze audio encoder (THUNDER-T and UniTalker both fine-tune the SSL encoder), (ii) per-language balance (TalkVid is multilingual; under-represented languages lose).

### KMTalk (arXiv 2409.01113, ECCV 2024; home: speech-driven 3D mesh) — phoneme-boundary key motions (ASR + Montreal Forced Aligner)
- [M] Pipeline: ASR (Auto-AVSR or Whisper) + MFA -> phoneme boundaries -> key frames ("approximately capture the inflection points of the lip movement curve", App. C.1); key-motion decoder predicts meshes at those frames; cross-modal motion completion (CMC) fills the rest with audio guidance. — [KMTalk](https://arxiv.org/html/2409.01113)
- [M] Table 3 ablation BIWI-Test-A (LVE ×1e-4 mm / FDD): none (=SelfTalk baseline) 4.2485 / 3.5761; uniform 33% sampling instead of phoneme localisation 4.1648 / 2.8713; phoneme localisation w/o key-motion decoder 4.1381 / 2.9546; w/o audio guidance in CMC 4.8859 / 3.2780; full 3.9654 / 2.5446 (−6.7% LVE vs SelfTalk). VOCA (Table 8): 3.2238 -> 2.2639 (−29.8%). — [KMTalk](https://arxiv.org/html/2409.01113)
- [M] Table 4 robustness: Auto-AVSR 3.9654; Whisper-large 4.0718; Whisper-tiny 4.0643; Auto-AVSR + 1-frame index offset 3.9991 -> insensitive to ASR choice and 1-frame shifts. — [KMTalk](https://arxiv.org/html/2409.01113)
- [M] Plug-in (Table 5 BIWI / Table 7 VOCA, LVE): FaceFormer 5.3077->5.2793 / 4.1090->3.9608; CodeTalker 4.7914->4.5096 / 3.9445->3.8473; SelfTalk 4.2485->4.1122 / 3.2238->2.6608. — [KMTalk](https://arxiv.org/html/2409.01113)
- [I] Evidence that explicit phonetic timing (forced-aligned phoneme boundaries) adds information that SSL features alone don't expose to a small decoder. For SANG a cheap analogue: per-frame phoneme-boundary / phone-class features from MFA or a CTC phone recognizer as an extra conditioning channel or auxiliary target (see Q4).

### DiffPoseTalk (arXiv 2310.00434, SIGGRAPH 2024; home: diffusion 3D FLAME + head pose, TFHP 26.5 h in-the-wild) — the closest data regime to SANG
- [M] Windowed transformer diffusion with HuBERT features, previous-window motion+audio as condition, contrastive style encoder, incremental CFG over audio (w_a) and style (w_s); CFG dropout: style null p=0.45, audio+style null p=0.1. — [DiffPoseTalk](https://arxiv.org/html/2310.00434)
- [M] Table 1 (TFHP, no head pose; LVE mm ↓): FaceFormer 9.90, CodeTalker 12.71, FaceDiffuser 12.12, DiffPoseTalk 8.81. (Ablation rows of Table 1 were not retrieved — gap.) — [DiffPoseTalk](https://arxiv.org/html/2310.00434)
- [I] On in-the-wild pseudo-GT, the discrete-prior CodeTalker is the WORST (12.71 vs FaceFormer 9.90): discrete priors that help on clean 4D scans do not transfer to noisy tracked data. De-prioritise VQ priors for SANG.

### SemConFlow (arXiv 2603.26553, 2026; home: holistic co-speech gesture incl. face, BEAT2 25 speakers + SHOW) — contrastive flow matching with mismatched-condition negatives
- [M] Objective: L_CFM = E[ ||v_θ(z_t,t|O) − v̂||² − λ ||v_θ(z_t,t|O) − ṽ||² ], λ ∈ [0,1), where ṽ = z̃_1 − z_0 uses a mismatched sample's motion with the SAME noise seed. Plus SACM: per-frame cosine alignment of motion latents to a fused audio/text target + clip-level CLIP-style InfoNCE (motion<->audio, motion<->text). — [SemConFlow](https://arxiv.org/html/2603.26553)
- [C] Ablation (Table 2, numbers not retrieved): "Replacing contrastive flow matching with standard FM degrades the FGD metric" and lowers diversity and BC; audio-only alignment "tends to preserve temporal consistency measured through BC"; removing SACM has "the most impact on Beat Consistency and FGD". — [SemConFlow](https://arxiv.org/html/2603.26553)
- [I] Directly portable to SANG's flow-matching DiT at zero parameter cost: within a batch, pair each x_t with the velocity target of a different clip (same noise) and subtract λ·MSE. This explicitly penalises the "audio-agnostic average mouth" mode. Expected effect is on conditional sharpness (like CFG baked into training). Evidence is gesture-domain and qualitative-only for the numbers I could retrieve -> medium confidence.

### EMAGE / BEAT2 (arXiv 2401.00374, CVPR 2024; home: holistic co-speech gesture + FLAME face, BEAT2 60 h / 25 speakers)
- [M] Audio conditioning = explicit rhythm (onset + amplitude via TCN) + content (transcript word embeddings), adaptively fused per frame: f = α·r + (1−α)·c, α = softmax(MLP(r,c)) ("Content Rhythm Attention", CRA); separate CRA encoders for face and body. Ablation text: CRA "shows improvement in both FGD and Alignment"; one VQ-VAE for whole body "decreases performance in facial movements" vs separate per-part VQ-VAEs. (Table 6 numbers not retrieved.) — [EMAGE](https://arxiv.org/html/2401.00374)
- [M] BC metric definition: audio beats = speech onsets; motion beats = local minima of upper-body joint velocity; BC = mean exp(−min dist²/2σ²). BEAT2 GT BC 6.896 vs TalkSHOW data 6.104 (Table 8). — [EMAGE](https://arxiv.org/html/2401.00374)
- [I] Gesture "alignment" is onset-level (±σ tolerance) — much coarser than phoneme-level lip timing. The transferable idea is giving the generator explicit low-level acoustic envelope channels (amplitude/onset) next to the SSL features; WavLM last layer is known to be relatively weak on low-level energy (see sibling-note discussion of layer choice). Cheap to add.

### Bailando (arXiv 2203.13055, CVPR 2022; home: music-to-dance, AIST++) — beat-align reward via actor-critic fine-tuning (Bailando++ is the TPAMI extension; not separately fetched)
- [M] Table 2: motion GPT w/o actor-critic FID_k 28.75 / FID_g 11.82 / BAS 0.2245 -> with actor-critic beat-align reward 28.16 / 9.62 / 0.2332 (+3.9% BAS, FID_g −19%). — [Bailando](https://arxiv.org/abs/2203.13055)
- [I] Reward fine-tuning on an alignment reward gives small alignment gains (+4% rel.) but helps quality; the same pattern as sibling note's reward-FT discussion (Q5 there).

### EDGE (arXiv 2211.10658, CVPR 2023; home: music-to-dance diffusion, AIST++) — 49M params, CFG, Jukebox features
- [M] 49M-parameter transformer diffusion (close to SANG's 53.4M), trained 4×A100 16 h, batch 512. Table 1: guidance w=2 Elo 1751 / PFC 1.5363 / Beat Align 0.26; w=1 1601 / 1.6545 / 0.27; Bailando 1397 / 1.754 / 0.23; GT 1653 / 1.332 / 0.24. — [EDGE](https://arxiv.org/html/2211.10658)
- [C] Authors argue the beat-alignment score "loses its meaning when candidate examples reach quality on par with ground truth" (EDGE exceeds GT). Compared librosa vs Jukebox audio features (details in their ablation; numbers not retrieved). — [EDGE](https://arxiv.org/html/2211.10658)
- [I] Dance/gesture beat metrics saturate above GT and are onset-level; they are not a good proxy for lip-trajectory correlation. Q3's main transferable items are therefore (a) contrastive flow matching (SemConFlow), (b) explicit envelope/onset channels (EMAGE CRA), (c) audio-motion contrastive embeddings as auxiliary loss (SemConFlow SACM; Learn2Talk SyncNet3D), not beat rewards.

### F5-TTS (arXiv 2410.06885, 2024; home: zero-shot flow-matching TTS, DiT) — conditioning refinement + Sway Sampling
- [M] Architecture ablation (small 155M models, Seed-TTS test-zh, 800K updates): F5-TTS (ConvNeXt-V2 refinement of the padded text stream before the DiT) WER 4.17 / SIM 0.54 vs E2 TTS (flat U-Net transformer, no refinement) 9.63 / 0.53; E2 TTS "consistently failed ... on 7% test samples (WER>>50%)". "Pure adaLN DiT (F5-TTS−Conv2Text) failed to learn alignment given simply padded character sequences"; MMDiT "learned fast and collapsed fast". Adding the same ConvNeXt branch on the audio side (F5-TTS++Conv2Audio) costs +1.61 WER. Table 4: GT duration instead of estimated improves WER only 4.17 -> 3.87. — [F5-TTS](https://arxiv.org/html/2410.06885)
- [M] Sway Sampling (inference-only non-uniform flow-step schedule, s=−1 concentrates steps at the noisy/early end), Table 5 (32 NFE, w/o SS -> w/ SS): F5-TTS LibriSpeech-PC WER 2.84 -> 2.41, SIM 0.62 -> 0.66; Seed-TTS test-zh WER 1.93 -> 1.58, SIM 0.69 -> 0.75; E2 TTS test-zh 1.97 -> 1.77. 16 NFE with SS beats 32 NFE without SS everywhere. "Leak and override" experiment: early flow steps decide the conditional content ("sketching the silhouette"), so allocating more steps there improves faithfulness to the condition. — [F5-TTS](https://arxiv.org/html/2410.06885)
- [M] CFG training: drop masked-speech prompt p=0.3, then prompt+text together p=0.2 ("two-stage control of CFG training may have the model learn more with text alignment"). — [F5-TTS](https://arxiv.org/html/2410.06885)
- [I] For SANG: (1) Sway Sampling is a zero-cost inference change to the flow-matching sampler, measured to improve faithfulness-to-condition (WER is the TTS analogue of lip correlation) by 10–18% relative; test with s ∈ {−0.4, −0.8, −1} at fixed NFE. (2) A small depthwise-conv (ConvNeXt) refinement of the audio-feature stream before injection helped the DiT align; SANG's ±80 ms conv is a minimal version — widening/deepening it (still local) is cheap.

### One TTS Alignment To Rule Them All (arXiv 2108.10447, NVIDIA 2021; home: TTS, LJSpeech)
- [M] Unsupervised alignment via CTC-style forward-sum loss over text<->mel soft alignment + static beta-binomial prior; parallel models (RAD-TTS, FastPitch, FastSpeech 2) with it "converge at the same rate as their baseline models using a forced aligner", with alignments closer to hand-annotated durations (Fig. 4, plot only) and fewer repeated/missing words. — [One TTS Alignment](https://arxiv.org/html/2108.10447)
- [I] TTS alignment machinery (MAS, forward-sum, duration predictors) solves an unknown monotonic text->frame mapping. SANG's audio and motion already share a clock (25 fps, 2 WavLM ticks/frame), so MAS/duration predictors do not transfer directly. What transfers is (a) the static diagonal prior idea = FaceFormer alignment bias (already implicit in SANG), (b) CTC heads as auxiliary phonetic supervision (see SelfTalk CTC, KMTalk phoneme boundaries), and (c) learned A/V offset — TalkVid in-the-wild clips may have small AV offsets that a per-clip shift estimate (SyncNet/LatentSync-style offset search) could fix in preprocessing.

### MMAudio (arXiv 2412.15322, CVPR 2025; home: video-to-audio flow matching, multimodal DiT) — frame-aligned sync conditioning + aligned RoPE
- [M] Conditional synchronization module: Synchformer visual features (self-supervised AV-offset detector, AudioSet; 24 fps) -> ConvMLP -> nearest-neighbour upsample to audio-latent rate -> added to global cond -> injected **per token via adaLN** in the audio stream (c_f = Upsample(ConvMLP(F_syn)) + 1·c_g). Motivation: cross-modal attention "aggregate[s] features via a soft distribution, which we found to hamper precision". — [MMAudio](https://arxiv.org/html/2412.15322)
- [M] Table 6 (FD_PaSST ↓ / IS ↑ / IB ↑ / DeSync s ↓): with sync module 70.19 / 14.44 / 29.13 / 0.483; sum sync into visual branch 73.59 / 16.70 / 28.65 / 0.490; **no sync features 69.33 / 15.05 / 29.31 / 0.973** (DeSync halves with frame-aligned sync features); aligned RoPE 0.483 vs no RoPE 0.509 vs non-aligned RoPE 0.496. Table 7: ConvMLP (kernel 3) 0.483 vs plain MLP 0.533 DeSync. Authors: aligned RoPE is "beneficial yet insufficient for good synchrony". — [MMAudio](https://arxiv.org/html/2412.15322)
- [I] Ranking of sync levers from MMAudio: (1) a sync-specialised, frame-rate condition encoder + per-token adaLN injection (−50% DeSync) >> (2) local conv in the MLP (−9%) > (3) aligned RoPE (−5%). For SANG (audio -> motion): the inverse analogue is adding features from an encoder trained for AV synchrony (Synchformer audio branch, AV-HuBERT audio branch, or a SyncNet audio tower trained on TalkVid) as a second frame-aligned stream, injected per-frame (adaLN-style scale/shift or gating) rather than only as additive tokens. WavLM-last-layer is trained for masked unit prediction, not sync. Also: ConvMLP FFNs (kernel 3 over time) in the DiT blocks are a near-zero-param change with a measured sync gain.

### Diff-Foley (arXiv 2306.17203, NeurIPS 2023; home: video-to-audio latent diffusion) — contrastive audio-visual pretraining (CAVP)
- [M] Stage 1 CAVP: PANNs audio encoder + SlowOnly video encoder trained with semantic contrast (across videos) + temporal contrast (N_T=3 clips from the same video ≥2 s apart as negatives); Stage 2 LDM conditioned on CAVP visual features; inference "double guidance" = CFG + alignment-classifier guidance. VGGSound: Align Acc 94.05%, IS 62.37 (vs ~30 baselines). — [Diff-Foley](https://arxiv.org/html/2306.17203)
- [C] Table 2 (numbers not retrieved): CAVP visual features "significantly improve synchronization and audio-visual relevance (Align Acc)" vs CLIP etc.; CLIP better on IS/KL but not sync. — [Diff-Foley](https://arxiv.org/html/2306.17203)
- [I] Two transferable ideas for SANG: (1) the **within-clip temporal-contrast negative** (same speaker, different time) is exactly what teaches timing rather than identity/semantics -> a TalkVid-trained audio<->lip-motion contrastive model should sample negatives from the same clip at offsets (e.g., ±2–15 frames) like SyncNet; (2) **alignment-classifier guidance** at sampling time (gradient of a frozen sync scorer on x̂0) is a no-retrain inference lever.

### V-AURA (arXiv 2409.13689, 2024; home: autoregressive video-to-audio, codec tokens) — 25 fps sync-contrastive features, fusion, AV-correspondence data filtering
- [M] Visual encoder: Segment AVCLIP (TimeSformer, contrastively pretrained with audio at sub-clip level) at 25 fps, temporally upsampled to audio-token rate and fused per timestep. S3D / ResNet-50 features "did not yield temporally-aligned" results. Sync metric = mean |offset| (ms) predicted by Synchformer. — [V-AURA](https://arxiv.org/html/2409.13689)
- [M] Table III conditioning (VGGSound-Sparse; KLD / FAD / IB / Sync ms ↓): prepend condition tokens 1.94 / 4.11 / 26.44 / 105; per-timestep fusion 1.93 / 3.55 / 28.92 / 49 (−53% offset). — [V-AURA](https://arxiv.org/html/2409.13689)
- [M] Table II **training-data filtering by audio-visual embedding similarity** (threshold / #samples / train GPU-h / Sync ms): 0.0 / 155,591 / 708 / 60; 0.2 / 119,469 / 662 / 59; 0.3 / 77,265 / 278 / 49; 0.4 / 33,225 / 168 / 71. Dropping the ~50% least-corresponding clips improved sync 18% AND cut training 2.5×; over-filtering (0.4) hurt. — [V-AURA](https://arxiv.org/html/2409.13689)
- [M] Table IV CFG scale (Sync ms): 1 -> 155; 3 -> 80; 5 -> 52; 6 -> 50; 7 -> 55; 9 -> 53. — [V-AURA](https://arxiv.org/html/2409.13689)
- [I] Three transfers to SANG: (1) **filter TalkVid by an AV-sync score** (SyncNet LSE-C/offset or AV-HuBERT AVSu on the real clip) — the ~0.82–0.85 re-render ceiling suggests the tracked lip motion is good, but in-the-wild clips with dubbing, off-screen speakers, or AV offset are poison for timing; (2) per-timestep fusion > prepend (SANG already does per-frame fusion — keep it); (3) audio CFG scale is a first-order sync knob (3× offset reduction from w=1 to w=6) — sweep SANG's audio guidance scale against lip-corr before any retraining.

### LatentSync (arXiv 2412.09262, ByteDance 2024; home: pixel/latent-diffusion lip-sync dubbing) — SyncNet supervision + AV-offset preprocessing
- [M] Preprocessing: "In-the-wild videos naturally contain audio-visual offsets"; they shift each clip to zero offset using pretrained SyncNet and drop clips with Sync_conf < 3; "without offset adjustment, the model's convergence is significantly impaired" (Fig. 10, SyncNet training curves; plot only). StableSyncNet best settings: batch 1024, 16 frames, emb 2048; HDTF sync accuracy 94% vs prior SOTA 91%. — [LatentSync](https://arxiv.org/html/2412.09262)
- [C] Without SyncNet supervision the diffusion model's lip-sync is "significantly poor" (they call this the "shortcut learning problem": the model copies from the reference/context instead of listening to audio); pixel-space SyncNet > latent-space SyncNet because the latent SyncNet converges poorly ("some lip information may already be lost"). TREPA (VideoMAE-v2 temporal feature alignment) FVD 176.35 -> 162.74 (Table 3). Tab. 2 sync numbers not retrieved. — [LatentSync](https://arxiv.org/html/2412.09262)
- [I] For SANG: (1) offset-correct + Sync_conf-filter TalkVid before training (cheap, one pass of SyncNet over 70 h); (2) the "shortcut" warning matches SANG's reference conditioning — if the reference frame or prior-window motion carries lip information, the DiT can under-use audio; ensure reference lips are neutral/randomized or masked.

### Contrastive Flow Matching, ΔFM (arXiv 2506.05350, Stoica et al., 2025; home: class/text-conditional image flow matching) — ID verified
- [C, abstract] Adds a term maximising dissimilarity between predicted flows of arbitrary sample pairs to enforce uniqueness of conditional flows; on ImageNet-1k / CC3M: up to 9× faster training, up to 5× fewer denoising steps, FID lower by up to 8.9 vs plain FM. Code: github.com/gstoica27/DeltaFM. — [ΔFM](https://arxiv.org/abs/2506.05350)
- [I] Same objective SemConFlow applied to gestures (Q3). For SANG: negatives = other clips in the batch (different audio); λ ≈ 0.05 is the ImageNet default reported in the repo [unverified — check code]. Zero extra parameters, zero extra forward passes.

### AV Guidance from a lip-reading expert (arXiv 2407.01034, Interspeech 2024; home: speech-driven 3D mesh, VOCASET/BIWI) — audio-visual lip-reading perceptual loss
- [M] Lip-reading expert takes **generated lip motion + speech** (audio-visual, not visual-only) and predicts the transcript; pretrained on large 2D talking-face/lip-reading data, then fine-tuned on 3D; plus a relative lip-vertex loss. — [AV Guidance](https://arxiv.org/html/2407.01034)
- [M] Table 1 BIWI-Test-A (LVE ×1e-4 mm / CER / VER): FaceFormer 6.0449 / 72.588% / 68.777% -> +AV Guidance 5.5061 / 68.423% / 62.422%; CodeTalker 5.3711 / 72.592% / 65.593% -> 4.8403 / 70.711% / 63.299% ("LVE ... 10% lower"). Table 2 VOCASET (LVE ×1e-5): FaceFormer 3.2496 -> 3.0987; CodeTalker 4.0557 -> 3.9884. — [AV Guidance](https://arxiv.org/html/2407.01034)
- [M] Table 3 ablation (BIWI LVE): FaceFormer+AVG 5.5061; w/o 2D prior knowledge (expert trained from scratch on 3D) 5.9344; w/o relative lip-vertex loss 5.9023; **w/o speech information (visual-only lip reader) 6.0352**. CodeTalker+AVG 4.8403; w/o prior 5.4271; w/o L_rlv 5.6524; w/o speech 5.3155. Relative-lip-vertex loss alone (no expert) gives 6.0976 / 5.2639 — no gain. — [AV Guidance](https://arxiv.org/html/2407.01034)
- [I] Two consistent lessons with Yaman et al. (Q1): an **audio-aware** expert beats a visual-only one, and the expert must be **pretrained on large real data** (from-scratch expert loses most of the gain). For SANG: train the motion-space expert on all 70 h TalkVid GT motion (+ audio), freeze, then use as loss.

### PASE (arXiv 2504.05803, 2025; home: person-specific NeRF/3DGS talking heads) — phoneme-aware, contrastively AV-aligned speech encoder as conditioning
- [M] PASE = STFT -> 8-layer GRU encoder trained on LRS2 with MFA phoneme embeddings as alignment anchors + contrastive audio<->lip alignment + prediction/reconstruction tasks; used as a drop-in audio feature. — [PASE](https://arxiv.org/html/2504.05803)
- [M] Table III (drop-in audio features; LSE-C ↑ / LSE-D ↓, % vs HuBERT): NeRF (SyncTalk renderer): HuBERT 7.6599 / 7.1402; DeepSpeech 7.4268 / 7.5418; Wav2Vec2 3.3782 / 10.200; Whisper 3.3416 / 10.202; **AV-HuBERT 8.0136 (+4.62%) / 6.8022**; Wav2Lip SyncNet audio encoder 8.4461 (+10.3%) / 6.4028; PASE 8.7098 (+13.7%) / 6.1255; GT 8.8302 / 6.0570. 3DGS (TalkingGaussian): HuBERT 5.0660 / 9.2126; AV-HuBERT 5.3618 (+5.84%); Wav2Lip 6.9973 (+38.1%); PASE 8.3487 (+64.8%) / 6.6705. — [PASE](https://arxiv.org/html/2504.05803)
- [M] Caveat: LSE metrics come from a SyncNet-family scorer, which favours SyncNet-like (Wav2Lip) and contrastively AV-aligned features (metric coupling); LMD gains are smaller (NeRF: HuBERT 2.9616 -> PASE 2.8725, −3.0%). Person-specific renderers trained on minutes of data, not a multi-speaker generator. — [PASE](https://arxiv.org/html/2504.05803)
- [I] Ordering is consistent across both renderers: contrastively AV-aligned features (PASE, Wav2Lip-SyncNet) > AV-HuBERT > HuBERT/DeepSpeech >> wav2vec2/Whisper (last-layer). This is the only measured head-to-head I found with **AV-HuBERT as conditioning**: modest (+5–6% LSE-C) over HuBERT. A sync-trained audio tower helps more, matching MMAudio's Synchformer result (Q5). For SANG, the analogous move is concatenating a sync-trained audio embedding (own SyncNet audio tower trained on TalkVid motion) to WavLM, not replacing WavLM.
