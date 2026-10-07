# Cross-domain sequence modelling, pretraining and data/preprocessing techniques for SANG-M (53.4M flow-matching DiT, 42-d face motion)

Evidence tags: [M] = measured ablation in the source paper; [C] = claim by authors without a clean ablation (or training curves only); [I] = my inference for SANG (not measured anywhere).
Home domain is stated for every source. All arXiv IDs below were resolved through arXiv HTML or the Firecrawl paper index in this pass (2026-10-07), except where a Gap says "not verified".
SANG context used for the mapping: 53.4M bidirectional flow-matching DiT; 64-frame windows at 25 fps of 42-d motion (3 head angles + 39 keypoint coordinates for lips, eyes and brows); WavLM audio and a reference as conditions; chained with a 10-frame prefix. It trains on about 12.6k clips (about 70 h) of TalkVid's 1,244 h, with noisy LivePortrait motion-extractor targets. Measured: val flow loss plateaus at 1.146; lip-trajectory correlation 0.64 (about 0.82–0.85 achievable); head velocity 0.92x real; 3x more data removed overfitting.

## Q1. Human-motion generation: representation, normalisation, loss choices; would a learned motion latent help SANG?

### Takeaway
In human-motion generation, at small model sizes the biggest measured gains come from three things. First, the representation: drop redundant or derived dimensions from a diffusion target (19x FID in MARDM) and remove identity from the target (CodeTalker). Second, data scale and curation. Third, the parameterisation: x0 or v targets in raw space, a KL-regularised latent in latent space. Latent diffusion (MLD) beat the *old* raw-space MDM, but a well-designed raw-space model (MARDM) beat MLD. For a 42-d signal, a learned latent is therefore not the obvious next step, and it is risky for lip precision.

### Cited Findings
Home domain for this section: text-to-motion / kinematic human-motion generation (HumanML3D, mocap), unless stated otherwise.

- [M] **MARDM** ("Rethinking Diffusion for Text-Driven Human Motion Generation: Redundant Representations, Evaluation, and Masked Autoregression", Meng, Xie, Peng, Han, Jiang; arXiv 2411.16575). Ablation (their Table 6, HumanML3D): the full method gets FID 0.116 / R-Precision Top-1 0.492. *Without representation reformation* (i.e. keeping the redundant 263-d HumanML3D features) it gets FID 2.196 / Top-1 0.387. *Without autoregression* it gets FID 0.551 / Top-1 0.435. So removing redundant (derived) dimensions from the diffusion target was the single biggest lever (about 19x FID). — [arXiv 2411.16575](https://arxiv.org/abs/2411.16575)
- [M] MARDM Table 2: for MDM, predicting x0 gave FID 0.518, while epsilon-prediction (cosine schedule) collapsed to FID 31.265. The authors attribute this to "dimensional distribution mismatch and error amplification". Redundant features *help VQ tokenisers* (T2M-GPT reconstruction FID 0.095 -> 0.081 with redundancy, Table 1) but *hurt diffusion*. (Numbers extracted from the arXiv HTML via a summariser; the table is worth re-checking before quoting in a paper.) — [arXiv 2411.16575](https://arxiv.org/abs/2411.16575)
- [M] **MoMask** (Guo, Mu, Javed, Wang, Cheng; arXiv 2312.00063), Table 2, HumanML3D:
  - Reconstruction FID: plain VQ 0.091 vs residual VQ (6 layers) 0.019.
  - Generation FID by number of residual layers V: 0.093 (V=0) -> 0.073 (V=1) -> 0.051 (V=5), then it degrades again to 0.076 at V>=6.
  - Quantization dropout: q=0 gives 0.091 generation FID, q=0.2 gives 0.051 (optimal), q>=0.4 gives 0.082–0.083.
  - Codebooks: 6 layers x 512 codes x 512-d. — [arXiv 2312.00063](https://arxiv.org/abs/2312.00063)
- [M] **ScaMo** ("ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model", arXiv 2412.14559). Finite scalar quantisation (FSQ) "consistently outperforms VQ" on reconstruction loss and MPJPE across codebook sizes on HumanML3D and their larger MotionUnion set. FSQ codebook utilisation stays "close to the theoretical maximum" while VQ utilisation falls as the codebook grows (Fig. 6; numbers are in their appendix, not extracted here). — [arXiv 2412.14559](https://arxiv.org/abs/2412.14559)
- [M] **MLD** ("Executing your Commands via Motion Diffusion in Latent Space", Chen et al.; arXiv 2212.04048), HumanML3D.
  - Latent vs raw (Table 10): MLD-1 (a single 1x256 latent) gets FID 0.473 at 50 DDIM steps. Raw-coordinate MDM gets 7.334 at 50 DDIM steps and 0.544 at 1000 DDPM steps. FLOPs are 29.86G vs 597.97G at 50 steps.
  - *A KL-regularised VAE is essential* (Table 8): MLD with a plain autoencoder gets FID 5.033 / R@3 0.581; with the VAE it gets 0.473 / 0.772.
  - *Reconstruction != generation*: the 7-token latent reconstructs far better (MPJPE 14.7 mm vs 54.4 mm for 1 token, Table 4) but generates worse (unconditional FID 7.614 vs 1.055, Table 6).
  - In latent space, epsilon-prediction beat z0-prediction (FID 0.473 vs 0.513, Table 9). This is the opposite of MARDM's raw-space result. — [arXiv 2212.04048](https://arxiv.org/abs/2212.04048)
- [M] **HY-Motion 1.0** ("HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation", Tencent Hunyuan; arXiv 2512.23464). A flow-matching DiT on the rectified-flow / OT path, with velocity target x1 - x0.
  - Pretraining on 3,000 h of noisier motion and then fine-tuning on 400 h of curated motion beats training on the 400 h curated set alone: DiT-0.46B average instruction-following score 3.20 vs 3.05 for DiT-0.46B-400h (Table 3, rated scores).
  - Scaling: DiT-0.05B (50M) 3.10 instruction / 2.91 quality; 0.46B 3.20 / 3.26; 1B 3.34 / 3.34. "Motion quality reaches a saturation point beyond the 0.46B parameter size". — [arXiv 2512.23464](https://arxiv.org/abs/2512.23464)
- [C] HY-Motion representation: 201-d per frame (root translation 3, root orientation 6D, 21x6D local rotations, 22x3 local positions). They *removed explicit velocities and foot-contact labels from the representation* "as we observed faster training convergence" (no numeric ablation). — [arXiv 2512.23464](https://arxiv.org/abs/2512.23464)
- [M] **Kimodo** ("Kimodo: Scaling Controllable Human Motion Generation", NVIDIA, Rempe et al.; arXiv 2603.15546). Trained on 700 h of optical mocap, with an x0-predicting diffusion model.
  - Model scaling (Table 2): Small 56M R@3 64.0 / FID 3.10; Medium 148M 69.2 / 2.36; Large 282M 71.9 / 1.85.
  - Data scaling: 10% of data R@3 71.0 / FID 2.07; 50% 70.8 / 1.81; 100% 71.5 / 1.84. Text metrics saturate early, but "foot skate and constraint accuracy monotonically improve with more available training data". — [arXiv 2603.15546](https://arxiv.org/abs/2603.15546)
- [M] Kimodo Table 1:
  - They use a *heavily smoothed root trajectory* as the global anchor and express joints relative to it. Removing it raised foot skate from 3.87 to 4.39 cm/s.
  - A one-stage denoiser instead of their two-stage one (root first, then body conditioned on root) raised full-body constraint error from 2.67 to 8.37 cm and foot skate from 3.87 to 7.59 cm/s.
  - The loss is a weighted sum of L1 terms on positions, velocities (weight 3.0), rotations, foot contacts and an FK-consistency term (no per-term ablation). — [arXiv 2603.15546](https://arxiv.org/abs/2603.15546)
- [M] **CodeTalker** (home: speech-driven 3D face meshes; arXiv 2301.02379). Representing *motion relative to the neutral or identity face* rather than absolute shape gives BIWI lip vertex error 4.79 vs 6.41 (x1e-4 mm) and reconstruction 2.83 vs 4.07 (Table 3). Details are in Q2. — [arXiv 2301.02379](https://arxiv.org/abs/2301.02379)

### Inferences
- [I] **Representation hygiene is the cheapest likely large win for SANG.**
  - Express the 39 keypoint coordinates as *deltas from the reference frame's keypoints*, so identity is removed from the target (as in CodeTalker). Then z-score each dimension with dataset statistics.
  - Check for near-redundant dimensions (for example z-coordinates that are almost constant, or points that move rigidly together). Options: drop them, or move to a PCA/whitened basis (UniTalker used PCA, Q2) and train the flow model in the whitened basis.
  - Why this matters: MARDM shows that derived or redundant dimensions in a diffusion target strongly hurt sample quality. Also, under an isotropic flow-matching loss, dimensions with tiny variance (lip corners and lip opening in normalised coordinates) get very little gradient unless they are standardised.
- [I] **Head as a separate "root" stream**: Kimodo's two-stage root-then-body denoiser and its smoothed-root anchor map naturally onto SANG's 3 head angles vs 39 local keypoints.
  - A full two-stage model costs parameters, so it is not a fit at fixed size.
  - A cheap analogue: express the face keypoints in the head-canonical frame, with the head rotation removed. Also give the head its own loss weight and velocity term. This may help the 0.92x head-velocity deficit. Unmeasured for faces.
- [I] **Learned motion latent (small VAE over 42-d x 64-frame windows): not recommended as a near-term change.**
  - MLD's big gain was measured against an old raw-space baseline, and MARDM's raw-space model beats MLD (0.116 vs 0.473 FID on the same benchmark).
  - A latent gains most when the raw space is high-dimensional or redundant. SANG's raw space is only 2,688 numbers per window.
  - The MLD reconstruction-vs-generation trade-off (54.4 mm MPJPE for the best-generating latent) is dangerous for lip accuracy, which is SANG's main deficit.
  - Possible upside: a KL-regularised autoencoder trained on noisy extractor output acts as a learned denoiser. Naturalness and jitter could improve even if lip correlation does not.
  - If tried at all, keep temporal compression at <=2x and the latent per-frame (not one global token), and gate the change on decoder reconstruction lip correlation >= 0.95 against the targets.
- [I] **Discrete tokens (RVQ / FSQ, MoMask / ScaMo)** would mean replacing the flow model with a masked-token model, which is a large rewrite. The MoMask and EMAGE evidence (Q2) suggests token priors improve distribution realism but can worsen per-frame accuracy. Low priority for SANG.

### Gaps
- Measured representation and normalisation ablations specific to *face keypoint* targets in flow-matching models were not found in this pass.
- Not fetched or verified in this pass, so no numbers are quoted: MDM geometric-loss ablations, MotionLCM, BAMM, MotionGPT, T2M-GPT, Light-T2M, MotionStreamer.
- Kimodo and HY-Motion give no per-loss ablation (velocity, foot-contact or FK terms).

## Q2. Self-supervised motion pretraining and motion priors (low-data audio-to-motion)

### Takeaway
Across gesture and 3D-face work, three things gave measured gains: (a) more and broader motion data, including *motion-only* data; (b) masked-motion reconstruction as an auxiliary objective; (c) multi-dataset pretraining followed by fine-tuning. The gains range from 6–12% error reductions to about 20% FGD improvements. Discrete priors (VQ) improve realism, but they need a regression term and can raise per-frame error. Noisy pseudo-labelled data does not always help.

### Cited Findings
- [M] **EMAGE** ("EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling", arXiv 2401.00374; home: co-speech gesture + FLAME face generation).
  - Table 6 ablation on BEATv1.3 (lower FGD is better):

    | Variant | FGD | MSE |
    |---|---|---|
    | Teacher-forced transformer baseline | 13.080 | 1.442 |
    | + one VQ-VAE prior | 9.787 | 1.619 (distribution better, per-frame error *worse*) |
    | + 4 part-wise VQ-VAEs | 7.397 | 1.243 |
    | + content-rhythm attention | 6.833 | 1.186 |
    | + masked gesture modelling ("masked hints") | 5.423 | 1.180 |

    In masked gesture modelling the model is also trained to reconstruct masked motion from visible motion plus audio. Beat-alignment BC was highest for the baseline (6.941 vs 6.794).
  - Table 7: adding Trinity (upper body + audio) gives FGD 5.319. Adding AMASS (body + hands, *mocap without speech*) gives 5.174, with diversity rising from 13.057 to 14.318. So motion-only data helped an audio-conditioned model.
  - A single VQ-VAE over the whole body "decreases performance in facial movements"; the face needs its own prior.
  - Cost: the 5 VQ-VAEs took 22.4 h on 5x4090. — [arXiv 2401.00374](https://arxiv.org/abs/2401.00374)
- [M] **CodeTalker** (arXiv 2301.02379; home: speech-driven 3D face mesh animation, VOCASET/BIWI).
  - Table 3: a codebook over *motion relative to the neutral face* (m_t) rather than full shape (m_t + identity template h) gives BIWI lip vertex error 4.79 vs 6.41 (x1e-4 mm) and reconstruction 2.83 vs 4.07.
  - Table 6: pure code classification (cross-entropy only) gives LVE 9.6356, vs 5.1138 for cross-entropy plus a regression loss. A continuous regression term is essential.
  - Supplement Table 5: per-sequence instance normalisation over time in the prior's encoder lowers reconstruction error from 0.12 to 0.08 (VOCA) and from 3.27 to 2.83 (BIWI), and restores lip amplitudes.
  - Settings: codebook N=256, temporal unit P=1 frame, H=8–16 face parts. A larger temporal unit P hurt. — [arXiv 2301.02379](https://arxiv.org/abs/2301.02379)
- [M] **UniTalker** ("UniTalker: Scaling up Audio-Driven 3D Facial Animation through A Unified Model", Fan, Li, Lin, Xiao, Yang; arXiv 2408.00762; home: 3D face animation).
  - Joint training on 8 datasets (18.53 h, 934 speakers) vs single-dataset training (Table 5): BIWI LVE 4.279 -> 3.859 (-9.8%); VOCASET 9.153 -> 8.303 (-9.3%); Multiface -2.6%.
  - Exception: 3D-ETF-HDTF got *worse* by 6.5% (8.445 -> 8.991). This pseudo-labelled (reconstructed-from-video) data did not benefit uniformly.
  - Fine-tuning the pretrained model: average LVE -6.3% (BIWI -11%, VOCASET -12%).
  - Transfer to an unseen annotation (BIWI) with 50% of the data (95 sequences): LVE 4.197e-4, vs 4.249e-4 for SelfTalk trained on the full data.
  - Training tricks:
    - PCA to L=512, to equalise heterogeneous targets.
    - Decoder warm-up with the audio encoder frozen.
    - A "pivot identity embedding": the identity label is replaced by a pivot identity with probability 10%. — [arXiv 2408.00762](https://arxiv.org/abs/2408.00762)
- [M] **HY-Motion 1.0** (arXiv 2512.23464; text-to-motion): broad pretraining (3,000 h, noisier) plus curated fine-tuning (400 h) beats curated-only (instruction-following average 3.20 vs 3.05 at 0.46B). — [arXiv 2512.23464](https://arxiv.org/abs/2512.23464)
- [M] **ACT** (robotics, see Q3): removing the latent (CVAE) objective made no difference on deterministic scripted demos. On noisy human demos success fell from 35.3% to 2%. When targets are noisy or multimodal, the generative or latent part of the model carries the performance. — [arXiv 2304.13705](https://arxiv.org/abs/2304.13705)

### Inferences
- [I] **"Pretrain on all of TalkVid, fine-tune on the clean subset" is the direct SANG analogue of HY-Motion.**
  - Concretely: run the LivePortrait extractor over far more of the 1,244 h, not just the current about 70 h. Pretrain the same 53.4M flow model with audio dropped on a high fraction of samples (CFG-style unconditional training). This turns low-sync or noisy clips into motion-prior data rather than bad audio-motion pairs. Then fine-tune with full audio conditioning on the sync-filtered subset.
  - SANG's own observation that 3x data removed overfitting strongly predicts gains from going further. EMAGE (+AMASS without speech) shows that motion-only data helps audio-conditioned generation.
  - UniTalker's HDTF result warns that adding *noisily labelled* data can hurt the clean test set. That is why the noisy bulk goes into pretraining and unconditional training, not straight into the paired fine-tune.
- [I] **Masked-motion objective inside the flow model.**
  - With some probability, replace the 10-frame-prefix setup with a random temporal mask: a random set of frames, a span, or the prefix itself is given clean, and the rest is generated. This is EMAGE's "masked hints" implemented as flow-matching inpainting.
  - It costs nothing in parameters, trains the prefix-continuation skill SANG needs for chaining, and regularises against overfitting.
- [I] **VQ / FLINT-style priors**: the evidence (EMAGE MSE up, CodeTalker needing a regression term) suggests a discrete prior would trade lip accuracy for realism. SANG's main gap is lip accuracy, so this is low priority.

### Gaps
- FLINT (the temporal VAE motion prior used in EMOTE, arXiv 2306.08990) was not fetched or verified in this pass; no FLINT ablation numbers are given here.
- No paper was found that pretrains an *audio-conditioned flow or diffusion face-motion model* on large unlabelled face motion and reports a lip-sync ablation. The analogy rests on gesture (EMAGE) and text-to-motion (HY-Motion) evidence.
- Self-supervised encoders for audio-to-motion (ETHead, arXiv 2608.01605, which pre-trains a speech encoder by self-distillation on 2D talking videos) were seen only at abstract level, with no numbers.

## Q3. Robotics policy learning: chunking, temporal ensembling, conditioning designs

### Takeaway
Action chunking is the large measured win in robotics (1% -> 44% success in ACT), and SANG already chunks (64 frames). For *generative* (diffusion or flow) chunk policies, naive temporal ensembling of independently sampled chunks is harmful. Inpainting-style continuation with a soft-decaying overlap weight (Real-Time Chunking) is the measured best way to join chunks. For global conditioning in DiT policies, adaLN-Zero dramatically beats cross-attention and in-context tokens.

### Cited Findings
Home domain for this section: imitation-learning robot policies (visuomotor manipulation).

- [M] **ACT / ALOHA** ("Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", Zhao, Kumar, Levine, Finn; arXiv 2304.13705), Fig. 7.
  - Chunk size: success rises "drastically from 1% at k=1 to 44% at k=100" (k = action chunk size), then tapers slightly at higher k.
  - Temporal ensembling (averaging overlapping chunk predictions with exponential weights) adds +3.3% for ACT and +4% for BC-ConvMLP. It *hurts* VINN, a retrieval method that already returns real trajectories.
  - Removing the CVAE objective: no change on deterministic scripted data, but 35.3% -> 2% on human demonstrations.
  - L1 was used instead of L2 because it "leads to more precise modeling of the action sequence" (no numbers). — [arXiv 2304.13705](https://arxiv.org/abs/2304.13705)
- [M]/[C] **Diffusion Policy** (Chi et al.; arXiv 2303.04137).
  - An action horizon of 8 was found optimal for most tasks (Fig. 5 left): too short is jittery and inconsistent, too long is unresponsive.
  - It keeps peak performance with latency up to 4 steps under position control (Fig. 5 right).
  - A position (absolute) action space beats a velocity action space for Diffusion Policy, while it hurts BCRNN and BET (Fig. 4).
  - Receding-horizon execution (predict Tp, execute Ta, re-plan) is used for smoothness.
  - The CNN+FiLM backbone is recommended as a first try. The transformer (cross-attention to observations) is better when "rate of action change [is] high" but "more sensitive to hyperparameters". — [arXiv 2303.04137](https://arxiv.org/abs/2303.04137)
- [M] **Real-Time Chunking (RTC)** ("Real-Time Execution of Action Chunking Flow Policies", Black, Galliker, Levine; arXiv 2506.07339).
  - Method: for flow-matching action-chunk policies (pi0 / pi0.5), RTC treats the next chunk as *inpainting*. The first d steps (already committed) are frozen. The overlap region is softly guided, with an exponentially decaying weight (Eq. 5), using pseudo-inverse guidance. The guidance weight is clipped at beta=5 with 5 denoising steps.
  - Fig. 5 (sim): temporal ensembling "performs poorly across the board, even with an inference delay of d=0", because averaging multimodal samples is invalid. RTC beats bidirectional decoding (BID) while using less compute. Hard masking underperforms soft masking, especially at small d.
  - Fig. 6 (real robot): the TE variants produced oscillations that triggered the robot's protective stop. RTC was "completely robust to injected delay". Its overhead is 97 ms vs 76 ms. — [arXiv 2506.07339](https://arxiv.org/abs/2506.07339)
- [M] **"The Ingredients for Robotic Diffusion Transformers"** (Dasari et al.; arXiv 2410.10088), Table III conditioning ablation:

  | Conditioning | Pick-Place | Pen-Uncap |
  |---|---|---|
  | adaLN-Zero | 50% +-12% | 100% +-0% |
  | Cross-attention, 10 DDIM steps | 0% | 0% |
  | Cross-attention, 100 DDIM steps | 38% | 70% |
  | adaLN without zero-init | 38% | 80% |
  | In-context (observation tokens in joint self-attention) | 0% | 0% |

  — [arXiv 2410.10088](https://arxiv.org/abs/2410.10088)
- [M] **ScaleDP** ("Scaling Diffusion Policy in Transformer to 1 Billion Parameters for Robotic Manipulation", arXiv 2409.14411).
  - The cross-attention DP-T "suffers from large gradient issues".
  - Two changes let Diffusion Policy scale from 10M to 1B: factorising the observation embedding into per-block affine (adaLN) layers, and *non-causal* attention over the action sequence.
  - Results (abstract-level numbers): +21.6% average over DP-T on 50 MetaWorld tasks, +36.25% on four real single-arm tasks and +75% on three bimanual tasks. — [arXiv 2409.14411](https://arxiv.org/abs/2409.14411)

### Inferences
- [I] **Chunk joins in SANG.** SANG's 10-frame prefix is "hard-masked inpainting".
  - RTC's measured result that soft, decaying overlap weights beat hard masks argues for a soft schedule. Overlap the windows by more than 10 frames, freeze the first few frames, and guide the rest with decaying weights. Either as an inference-time guidance (training-free), or by training with randomly noised prefixes so the model learns to trust the prefix only partially.
  - Do **not** average overlapping independently sampled windows (ACT-style temporal ensembling). RTC shows that this fails for multimodal generative policies. For SANG it would also shrink lip and head amplitude, making the 0.92x head-velocity deficit worse.
- [I] **Conditioning.** The robotics evidence concerns *global* observation conditioning, where adaLN-Zero >> cross-attention or in-context.
  - SANG's audio is *frame-aligned*, so the natural design is per-frame additive or concatenated audio injection, plus adaLN-Zero for the diffusion time, the reference and other global codes.
  - If SANG currently uses cross-attention for audio, or adaLN without zero-init, switching is parameter-neutral and has strong robotics evidence. It is unmeasured for audio-to-face motion.
- [I] Diffusion Policy's position-vs-velocity result supports keeping SANG's absolute (reference-relative) coordinates rather than switching to frame-to-frame velocity targets. Velocity should be an auxiliary loss, not the representation.

### Gaps
- Not fetched or verified in this pass, so no numbers are given: BESO (score-based policy), Consistency Policy, and pi0's own ablations.
- No robotics paper directly compares per-timestep aligned conditioning vs adaLN, which is the case relevant to audio.

## Q4. Time-series / signal processing: frequency losses, multi-resolution losses, patching, target denoising, label-noise-robust regression

### Takeaway
A frequency-domain auxiliary loss (FreDF: L1 on FFT coefficients, alpha about 0.8) gives consistent 2.7–9.6% MSE reductions in forecasting, plus large data-efficiency gains (30% of the data matched the full time-domain model). Patching mainly buys compute. Smoothing the noisy *extracted* face coefficients measurably improves motion stability with neutral lip-sync (StableFace). No measured label-noise-robust *regression* method for generative sequence models was found.

### Cited Findings
- [M] **FreDF** ("FreDF: Learning to Forecast in the Frequency Domain", Wang, Pan, Shen, Chen, Yang, Yang, Zhang, Liu, Li, Tao; arXiv 2402.02399; home: long-horizon time-series forecasting).
  - Argument (Thm 3.1): per-step MSE assumes independent steps, so it is biased when the label sequence is autocorrelated.
  - Loss: L_alpha = alpha * |FFT(Y_hat) - FFT(Y)|_1 + (1 - alpha) * sum_t ||Y_t - Y_hat_t||^2. L1 is used in the frequency domain because frequency magnitudes differ widely.
  - Table 1, with iTransformer:

    | Dataset | Baseline | FreDF |
    |---|---|---|
    | ETTm1 MSE | 0.415 | 0.392 (-5.5%) |
    | ETTm1 MAE | 0.416 | 0.399 |
    | ETTh1 MSE | 0.449 | 0.437 |
    | ECL MSE | 0.176 | 0.170 |
    | Weather MSE | 0.281 | 0.254 (-9.6%) |

  - Table 2 (Weather MSE): time-only 0.280, frequency-only 0.257, both 0.253. alpha about 0.8 is typically best (Fig. 6).
  - Table 3 (ETTm1 MSE reduction): 2-D FFT (time x variables) 5.60%, time-axis FFT 5.49%, variable-axis FFT 4.77%. Orthogonal bases (Fourier, Legendre) work best.
  - Fig. 7: "With only 30% of the training data, [frequency-domain learning] achieves performance comparable to learning in the time domain using the full training dataset." — [arXiv 2402.02399](https://arxiv.org/abs/2402.02399)
- [M] **PatchTST** ("A Time Series is Worth 64 Words", arXiv 2211.14730; home: forecasting).
  - Table 7, Weather horizon 96: patching + channel-independence MSE 0.152, CI only 0.164, patching only 0.168, neither 0.177.
  - On Traffic with L=336, patching lowered MSE from 0.397 to 0.367 and cut training time by up to 22x.
  - The channel-mixing model "quickly overfits" (Fig. 7). — [arXiv 2211.14730](https://arxiv.org/abs/2211.14730)
- [M] **StableFace** (arXiv 2208.13717; home: audio-driven talking face via 3D face coefficients).
  - It diagnoses three jitter sources: jitter in the *extracted* 3D face representations, a train/inference mismatch, and missing inter-frame dependency modelling.
  - Table V, Testset 1, full model vs without Gaussian-based adaptive smoothing of the extracted coefficients:

    | Metric | Full | Without smoothing |
    |---|---|---|
    | MSI-Lip | 0.504 | 0.470 |
    | MSI-Jaw | 0.997 | 0.954 |
    | NLMD | 0.0108 | 0.0117 |
    | Sync-C | 5.77 | 5.80 |

    Smoothing slightly *lowered* Sync-C on Testset 1. On Testset 2, Sync-C was 5.31 vs 5.29.
  - Removing transformer dependency modelling hurt most: Sync-C 5.51 / 4.57 on the two test sets, MSI-Lip 0.439.
  - An adaptive (per-sequence, motion-dependent) smoothing weight beat fixed and global-learnable weights in user preference (Fig. 9).
  - MSI is the reciprocal of keypoint acceleration variance. It correlates with human stability scores at Pearson 0.424 (lip) and 0.438 (jaw). — [arXiv 2208.13717](https://arxiv.org/abs/2208.13717)
- [M] **Kimodo** (text-to-motion): heavily smoothing the root trajectory used as the representation anchor reduced foot skate (4.39 -> 3.87 cm/s). Smoothing the *slow* component of the target helped. — [arXiv 2603.15546](https://arxiv.org/abs/2603.15546)
- [M] Noisy-keypoint filtering (home: infant / COCO pose estimation; "Toward Reliable Infant Pose Estimation: A Training-Dynamics Approach to Noisy Annotation Detection", arXiv 2610.06423, abstract only). Training-dynamics-based detection of noisy keypoint labels reached up to 91.9% F1. Filtering the detected labels improved COCO pose estimation by "up to 7.4 AP points" at moderate-to-high synthetic noise. This is a detector, not a generative sequence model. — [arXiv 2610.06423](https://arxiv.org/abs/2610.06423)

### Inferences
- [I] **Frequency and velocity auxiliary loss for SANG's flow model.**
  - Compute the implied clean sample x̂1 = x_t + (1 - t) * v̂, using the rectified-flow convention with t=1 at data.
  - Add L1 losses on (a) temporal differences (velocity) and (b) rFFT magnitudes or complex coefficients along time, per channel group (lips, eyes, brows, head). Weight by t, so they act mainly at low noise.
  - **Band-limit** the FFT loss to below about 8–10 Hz (64 frames at 25 fps gives bins of 0.39 Hz). Above that band, LivePortrait extractor jitter dominates, and matching it would teach jitter.
  - Expected effect: it directly targets the amplitude and phase of syllable-rate lip motion (about 3–8 Hz), and so lip correlation and the 0.92x head velocity. FreDF's 30%-data result matters for SANG's data-limited regime.
  - Caveat: this is a deterministic loss on a stochastic generator, so keep the weight small (e.g. 0.1–0.3 of the flow loss) to avoid regression-to-mean.
- [I] **Target denoising**: apply mild, *region-adaptive* smoothing to the training targets. Suggested settings:
  - Savitzky–Golay (window 5, order 2) or a one-euro filter on eyes, brows and head.
  - Lighter or no smoothing on the lips, so that plosive closures survive.
  - Outlier masking on frames with jumps above k-sigma in velocity, or with face-detection failures (zero loss weight on those frames).
  - StableFace shows stability gains with neutral sync.
- [I] Patching 2 frames per token would halve attention cost but risks blurring 25 fps lip detail. Not worth it at 64 tokens. PatchTST's channel-independence does not transfer, because the 42 dimensions are strongly coupled.
- [I] Label-noise-robust training for SANG:
  - Small-loss clip filtering: after warm-up, down-weight the top 5–10% highest-loss clips per epoch.
  - Huber or L1 instead of L2 on any auxiliary x̂1 losses (ACT also preferred L1).
  - Neither is measured for flow-matching motion models.

### Gaps
- No measured multi-resolution temporal loss (e.g. multi-scale STFT) for motion generation was retrieved in this pass.
- No measured label-noise-robust regression method for diffusion or flow *generative* targets was found.
- The FreDF results are for deterministic forecasters. Transfer to flow matching is unmeasured.

## Q5. Data augmentation and preprocessing for audio-conditioned sequence models

### Takeaway
The best-evidenced items are preprocessing rather than augmentation:
- Correcting audio-visual offsets with SyncNet and dropping low-sync-confidence clips: LatentSync shows convergence "significantly impaired" without it.
- Velocity-outlier, static-clip and artifact filtering (HY-Motion pipeline).
- Mild adaptive smoothing of extracted coefficients (StableFace).

Audio augmentation has solid ASR evidence: speed perturbation gives an average 4.3% relative WER improvement, and SpecAugment took LibriSpeech test-other from 7.5% to 6.8% WER. There is no measured evidence for audio-to-face motion. No measured evidence was found for left-right mirroring of face motion.

### Cited Findings
- [M]/[C] **LatentSync** (arXiv 2412.09262; home: audio-conditioned latent-diffusion lip sync, VoxCeleb2 + HDTF).
  - Data preprocessing for in-the-wild video:
    1. Affine face alignment.
    2. Shift the audio by the SyncNet-estimated AV offset to zero.
    3. Drop clips with SyncNet confidence < 3.
  - "Without offset adjustment, the model's convergence is significantly impaired" (Fig. 10, training curves only), and aligning *after* affine normalisation works better.
  - With these choices their StableSyncNet reaches 94% accuracy on out-of-distribution HDTF, vs a prior 91%. They also filtered low-visual-quality clips with HyperIQA.
  - Table 2: the generator without SyncNet supervision gets Sync_conf 4.6 / FVD 220.37, vs 8.9 / 162.74 with pixel-space SyncNet supervision. A sync-expert loss matters, and the "shortcut learning" problem means models ignore audio when visual context predicts the lips. — [arXiv 2412.09262](https://arxiv.org/abs/2412.09262)
- [C] **HY-Motion 1.0** cleaning pipeline (text-to-motion): duplicate removal, abnormal-pose removal, *joint-velocity outlier* removal, anomalous-displacement detection, static-clip pruning, foot-slide artifact detection. No per-step ablation. — [arXiv 2512.23464](https://arxiv.org/abs/2512.23464)
- [M] **StableFace**: smoothing the extracted per-frame 3D face coefficients improved stability (MSI-Lip 0.470 -> 0.504) with neutral Sync-C (see Q4). — [arXiv 2208.13717](https://arxiv.org/abs/2208.13717)
- [M] **UniTalker**: adding a *pseudo-labelled* (video-reconstructed) dataset was the one case where joint training hurt its own test set (3D-ETF-HDTF LVE +6.5%). Noisy extracted targets do not mix in for free. — [arXiv 2408.00762](https://arxiv.org/abs/2408.00762)
- [M] **Speed perturbation** ("Audio augmentation for speech recognition", Ko, Peddinti, Povey, Khudanpur, Interspeech 2015; home: ASR; predates the 2021 window and is cited as the canonical measurement). Three copies at speed factors 0.9, 1.0 and 1.1 gave "an average relative improvement of 4.3%" WER across 4 LVCSR tasks with 100–960 h of training data. — [ISCA archive](https://www.isca-archive.org/interspeech_2015/ko15_interspeech.html)
- [M] **SpecAugment** (Park et al., arXiv 1904.08779; home: ASR; 2019, canonical). Time warping plus frequency-band and time-block masking on filterbank features gave LibriSpeech 960h test-other 6.8% WER without an LM (5.8% with shallow fusion), vs the previous state-of-the-art hybrid at 7.5%. Switchboard / CallHome: 7.2% / 14.6% vs 8.3% / 17.3%. — [arXiv 1904.08779](https://arxiv.org/abs/1904.08779)
- [M] **EMAGE** (gesture): mixing datasets that cover only some body parts, including motion-only AMASS, improved FGD from 5.423 to 5.174 (Table 7). — [arXiv 2401.00374](https://arxiv.org/abs/2401.00374)

### Inferences
- [I] **AV-offset correction is probably the highest-value preprocessing for SANG's lip correlation.**
  - In-the-wild TalkVid clips often carry 1–3 frame AV offsets (40–120 ms at 25 fps).
  - Training on misaligned pairs teaches the model to output temporally blurred lips, which lowers lip-trajectory correlation directly.
  - Procedure:
    1. Run SyncNet per clip.
    2. Shift the WavLM feature sequence by the estimated offset.
    3. Drop or down-weight clips with confidence < 3 (LatentSync's threshold), or keep them only for audio-dropped (unconditional) training.
  - Cost: a few GPU-hours for 12.6k clips (estimated).
- [I] **Speed perturbation for audio-to-motion** must be *matched*: resample the audio at factor s (0.9 / 1.1) *and* the motion sequence by the same factor, then re-extract WavLM features.
  - It is attractive because WavLM is pretrained on unperturbed speech, and the model must generalise across speaking rates.
  - Risks: it distorts head-motion dynamics (head velocity statistics scale with s), and coarticulation timing does not scale linearly in real speech.
  - Suggestion: apply it to 20–30% of samples, and evaluate lip correlation separately on perturbed and clean validation sets.
- [I] **SpecAugment-style masking** of WavLM feature frames or channels acts like partial condition dropout and may improve robustness. Masking long time spans hides phonemes and will reduce lip accuracy, so keep spans short (<= 2 frames).
- [I] **Mirroring** (a left-right flip of face motion with the audio unchanged) is *not* recommended without verification.
  - LivePortrait's keypoints are *implicit*, learned without supervision. A left-right index correspondence is not guaranteed, and simply negating yaw and roll plus the x-coordinates may give an off-manifold target for the renderer.
  - The safe version: flip the source *video frames* and re-run the extractor. This doubles extraction cost and is exact.
  - No measured gain was found in any talking-head paper.
- [I] **Noise and room-impulse augmentation** of the audio before WavLM is cheap and standard in ASR. For SANG it mainly buys robustness to in-the-wild test audio, not lip accuracy on clean test sets.

### Gaps
- No talking-head or face-motion paper with a measured ablation of audio speed, pitch, noise, RIR or voice-conversion augmentation was found in this pass. "Learning Audio-Driven Viseme Dynamics for 3D Face Animation" (arXiv 2301.06059) claims robustness to volume, pitch, speed and noise distortions, but this was seen only at abstract level, without numbers.
- No measured ablation of mirroring augmentation for face motion was found.
- LatentSync's offset-correction benefit is shown only as training curves for SyncNet convergence, not as a generator-level lip metric.

## Q6. Top-8 ranked techniques for SANG (synthesis)

### Takeaway
At fixed parameter count, the ranking puts SANG's binding constraints first: data quantity, target quality and alignment, and the loss/representation of the lips. Architecture changes come later. Items 1–4 are cheap, low-risk and backed by the strongest cross-domain measurements. Items 5–8 are worthwhile but either less certain or more expensive. A learned motion latent, discrete priors, mirroring and patching are deliberately ranked below the top 8.

### Cited Findings
Ranked list, each entry with home domain, evidence, mapping onto SANG, one-GPU cost and risk:

1. **Scale motion extraction to a much larger part of TalkVid; pretrain broad, fine-tune curated.**
   - Home: text-to-motion (HY-Motion), mocap (Kimodo), gesture (EMAGE).
   - Evidence:
     - [M] HY-Motion: 3,000 h pretrain + 400 h fine-tune beats 400 h only (3.20 vs 3.05).
     - [M] Kimodo: constraint accuracy and foot skate improve monotonically with data at fixed size.
     - [M] EMAGE: motion-only AMASS improved FGD (5.319 -> 5.174).
     - SANG: 3x data already removed overfitting.
   - Mapping: extract 300–1,000 h; pretrain with a high audio-dropout rate (noisy or low-sync clips used only as unconditional samples); fine-tune on the sync-filtered subset.
   - Cost [I]: extractor throughput-bound, roughly 1–3 GPU-days of extraction plus 2–4x longer pretraining.
   - Risk: low. — [HY-Motion](https://arxiv.org/abs/2512.23464); [Kimodo](https://arxiv.org/abs/2603.15546); [EMAGE](https://arxiv.org/abs/2401.00374)
2. **AV-offset correction + sync-confidence filtering + target outlier masking + mild region-adaptive smoothing.**
   - Home: lip sync (LatentSync), talking face (StableFace), text-to-motion (HY-Motion).
   - Evidence:
     - [M]/[C] Offset correction is needed for convergence (LatentSync).
     - [M] Smoothing gives MSI-Lip 0.470 -> 0.504 with neutral Sync-C (StableFace).
     - [M] Noisy pseudo-labels hurt joint training (UniTalker HDTF +6.5% LVE).
   - Mapping:
     - Per-clip SyncNet offset: shift the WavLM features; drop or down-weight clips with confidence < 3.
     - Velocity-outlier frames get loss weight 0.
     - Savitzky–Golay smoothing on eyes, brows and head; light or none on the lips.
   - Cost: hours.
   - Risk: low. Over-smoothing the lips is the main failure mode. — [LatentSync](https://arxiv.org/abs/2412.09262); [StableFace](https://arxiv.org/abs/2208.13717); [UniTalker](https://arxiv.org/abs/2408.00762)
3. **Representation hygiene: reference-relative keypoint deltas, per-dimension standardisation, removing redundant dimensions (or a PCA-whitened basis), head handled as a separate "root" stream.**
   - Home: text-to-motion (MARDM, Kimodo), 3D face (CodeTalker, UniTalker).
   - Evidence:
     - [M] MARDM: redundancy removal took FID from 2.196 to 0.116.
     - [M] CodeTalker: identity-free motion gave LVE 6.41 -> 4.79.
     - [C] UniTalker: PCA used to balance targets.
   - Cost: hours plus a retrain.
   - Risk: low–medium. The renderer must accept the inverse-transformed output exactly. — [MARDM](https://arxiv.org/abs/2411.16575); [CodeTalker](https://arxiv.org/abs/2301.02379)
4. **Auxiliary x̂1-space losses: band-limited (< about 10 Hz) FFT L1 loss plus a velocity L1 loss, weighted by t and by region (higher on the lips).**
   - Home: time-series forecasting (FreDF), motion generation (Kimodo velocity term).
   - Evidence: [M] FreDF gives 2.7–9.6% MSE reduction, alpha about 0.8 best, and 30% of the data matches the full time-domain model.
   - Mapping: x̂1 = x_t + (1 - t) v̂; total aux weight about 0.1–0.3 of the flow loss.
   - Cost: negligible compute.
   - Risk: low–medium (regression-to-mean if over-weighted). The most direct lever on lip correlation (0.64) and head velocity (0.92x). — [FreDF](https://arxiv.org/abs/2402.02399); [Kimodo](https://arxiv.org/abs/2603.15546)
5. **Masked-motion (inpainting) multi-task training inside the flow model.**
   - Mapping: random frames, spans or prefixes are given clean; the rest is generated. Audio dropout is used for CFG.
   - Home: gesture (EMAGE masked hints), robot flow policies (RTC's inpainting view).
   - Evidence: [M] EMAGE FGD 6.833 -> 5.423 from masked hints.
   - Mapping: it generalises SANG's fixed 10-frame prefix; it regularises against overfitting; it enables better chaining.
   - Cost: small code change, same parameters.
   - Risk: low. — [EMAGE](https://arxiv.org/abs/2401.00374)
6. **Soft-masked chunk continuation (RTC-style) and no temporal ensembling of independent samples.**
   - Home: robot flow-matching policies.
   - Evidence: [M] RTC beats TE and BID; TE caused oscillations severe enough to trigger the robot's protective stop; soft beats hard masking.
   - Mapping:
     - Overlap windows by more than 10 frames.
     - Freeze the first few frames and apply exponentially decaying guidance over the rest of the overlap.
     - Train with noise-augmented prefixes.
   - Cost: inference-side code, plus about 20–30% sampling overhead (RTC: 97 vs 76 ms).
   - Risk: low. It improves seams and naturalness, not lip correlation. — [RTC](https://arxiv.org/abs/2506.07339); [ACT](https://arxiv.org/abs/2304.13705)
7. **Conditioning audit: frame-aligned additive audio injection + adaLN-Zero for global conditions (time, reference).**
   - Home: robot diffusion transformers.
   - Evidence:
     - [M] Dasari et al.: adaLN-Zero 50% / 100% success vs cross-attention 0% / 0% (10 DDIM steps) and in-context 0% / 0%.
     - [M] ScaleDP: cross-attention gradient problems; adaLN + non-causal attention scaled with +21.6% on MetaWorld.
   - Mapping: only if SANG currently uses cross-attention for global conditions or non-zero-initialised modulation.
   - Cost: a retrain.
   - Risk: medium, because there is no in-domain audio evidence. — [Dasari et al.](https://arxiv.org/abs/2410.10088); [ScaleDP](https://arxiv.org/abs/2409.14411)
8. **Matched audio–motion speed perturbation (0.9 / 1.1) plus light SpecAugment-style masking and noise/RIR on the audio.**
   - Home: ASR.
   - Evidence:
     - [M] Speed perturbation: 4.3% average relative WER gain.
     - [M] SpecAugment: test-other 7.5% -> 6.8% WER.
   - Mapping: resample audio and motion together, then re-extract WavLM features; use short masks only.
   - Cost: 2–3x WavLM feature extraction and storage.
   - Risk: medium. Time-warping distorts head dynamics and coarticulation, and there is no face-motion evidence. — [Ko et al. 2015](https://www.isca-archive.org/interspeech_2015/ko15_interspeech.html); [SpecAugment](https://arxiv.org/abs/1904.08779)

Below the cut:
- **Learned motion VAE latent.** MLD gains were measured against a weak raw baseline, and MARDM's raw space beats MLD. The best-generating latent reconstructs poorly (54.4 mm MPJPE), which is high risk for the lips.
- **VQ / RVQ / FSQ token priors.** EMAGE: per-frame MSE went up (1.442 -> 1.619) with a single VQ prior. CodeTalker needed a regression term.
- **Mirroring.** No evidence, and LivePortrait's implicit keypoints have no guaranteed left-right correspondence.
- **Patching.** A compute saving only, with a lip-detail risk.

### Inferences
- [I] Suggested order of experiments:
  1. Run items 2 and 3 together (a data/target-only change): retrain once and measure val flow loss, lip correlation and head-velocity ratio.
  2. Then add item 4 (loss).
  3. Then item 5 (masked training).
  4. Then item 1 (the large extraction run), using the improved pipeline.
  5. Item 6 is inference-only and can be tested at any time.
- [I] Expectation management:
  - Items 2–4 are the ones plausibly moving lip correlation from 0.64 toward 0.7–0.75. That is a guess; no cross-domain paper measures this metric.
  - Item 1 is the most likely to move the val-loss plateau (1.146), because overfitting disappeared with 3x data.
  - Item 6 mainly affects seam artifacts and perceived naturalness.
- [I] Note that the val flow loss is computed on noisy LivePortrait targets. Target cleaning (item 2) can *raise or lower* flow loss without reflecting true quality. Judge items 2–4 on lip correlation against cleaned and against raw targets, and on LSE-C/D after rendering, not on flow loss alone.

### Gaps
- No source measures any of these techniques on a 42-d LivePortrait-keypoint flow model. Every transfer to SANG is [I].
- Cost estimates for extraction throughput are my estimates. They should be checked against SANG's actual extractor speed (frames per second on the target GPU).
- Not covered with numbers in this pass: MotionLCM, BAMM, MotionGPT, T2M-GPT, Light-T2M, MotionStreamer, FLINT/EMOTE, BESO, Consistency Policy, pi0 ablations.
