# Training-recipe and data improvements for small (~50M) flow-matching motion generators (audio → 42-d facial motion)

Evidence labels used throughout:
- **MEASURED-MOTION**: ablation/table numbers on motion/keypoint/talking-head generators.
- **MEASURED-IMAGE**: ablation/table numbers on image (or pixel/latent video) generators only.
- **USED-NOT-ABLATED**: a paper uses the technique in a motion model but reports no isolating ablation (in passages retrieved).
- **SPECULATIVE**: my inference; no direct measurement found.

All arXiv IDs below were retrieved in this session via the Firecrawl research index (inspect/read), unless flagged "not verified".

Important context match: AVTR-1 (arXiv 2609.22913) uses a **42-dimensional LivePortrait-derived motion vector** (axis-angle rotation + 39 expression coordinates), **LR 3e-4 with cosine decay, EMA 0.995** — essentially the same target space and several of the same hyperparameters as SANG. It is therefore the closest published recipe, but it is autoregressive (5-frame chunks), 153M params, trained on 926 h, and its paper (as retrieved) does not ablate individual recipe components.

---

## 1. Timestep sampling (logit-normal / mode sampling) and loss weighting

### Takeaway
Logit-normal t-sampling (m=0, s=1) was the most robust rectified-flow variant in SD3's 61-formulation study and gains most at few sampling steps — but that evidence is on images. On motion, AVTR-1 (same 42-d space) uses logit-normal sampling without an ablation. It is a near-zero-cost change worth an A/B, but judge it on sample metrics, not validation flow loss.

### Cited Findings
- MEASURED-IMAGE: SD3 Table 1 (global rank over 24 sampler/EMA/dataset settings; lower is better): rf/lognorm(0.00, 1.00) avg rank **1.54** (5 steps: **1.25**, 50 steps: 1.50) vs uniform rectified flow "rf" **5.67** (5 steps: 6.50, 50 steps: 5.75); rf/mode(1.29) 2.75; rf/cosmap 4.13; eps/linear 2.88. — [SD3, arXiv 2403.03206](https://arxiv.org/abs/2403.03206)
- MEASURED-IMAGE: SD3 Table 2 (25 steps): ImageNet FID rf **49.70** → rf/lognorm(0,1) **45.78**; CC12M FID 94.90 → 89.91; CLIP 0.247→0.250 (ImageNet). Some variants (rf/lognorm(0.50,0.60)) were best at 50 steps but rank 8.5 at 5 steps, i.e. the choice interacts with the NFE budget. — [arXiv 2403.03206](https://arxiv.org/abs/2403.03206)
- Changing the t-density π(t) is equivalent to a loss weighting w_t = t/(1−t)·π(t) (SD3 Eq. 18); logit-normal density vanishes at t=0,1, which is why SD3 also tested heavy-tailed "mode" sampling (rf/mode(1.75) was among best CLIP/FID). — [arXiv 2403.03206](https://arxiv.org/abs/2403.03206)
- SD3 notes that "losses of different approaches are incomparable and also do not necessarily correlate with the quality of output samples", and evaluates validation loss **stratified at eight equally spaced t values** (App. B.3). — [arXiv 2403.03206](https://arxiv.org/abs/2403.03206)
- USED-NOT-ABLATED (motion): AVTR-1 samples one timestep per five-frame chunk from a logit-normal distribution, shared across the chunk's frames. — [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- USED-NOT-ABLATED (motion): AVTR-1 computes MSE separately for 4 regions (head rotation, brow, eyes, mouth) and sums them so "all four regions equal weight regardless of their coordinate count". — [arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- Counter-example (talking head): FLOAT trains with t ~ U[0,1] and still reaches reasonable motion at ~10 NFE. — [FLOAT, arXiv 2412.01064](https://arxiv.org/abs/2412.01064)

### Inferences
- SPECULATIVE: SANG's 1.146 validation flow loss plateau is not a sample-quality measure; switching to logit-normal will *change* the average val loss simply by reweighting t, so any comparison must use (a) stratified per-t val loss at fixed t-grid, and (b) sample metrics (lip-sync e.g. LSE-C/D on rendered video, motion FID/diversity on the 42-d space).
- SPECULATIVE: For 42-d data, the "hard middle" argument (velocity target most ambiguous at intermediate t) should still hold, but low-dimensional data has high SNR at small noise, so a slightly positive location (m>0, toward noise) might matter less than on images. Try lognorm(0,1) and mode(1.29) as a 2-arm sweep.
- SPECULATIVE: Region-balanced loss (AVTR-1 style) is cheap and likely helps mouth fidelity because mouth coordinates may otherwise be dominated by larger-variance rotation/brow dims; SANG should check whether its 42-d loss is per-dim z-scored (AVTR-1 z-scores per coordinate with dataset-global stats).

### Gaps
- No ablation of logit-normal vs uniform on a motion/keypoint model was found (AVTR-1 uses it without ablation).
- No evidence on its effect for ~50M models specifically; SD3's comparison used its own (larger) MM-DiT.

---

## 2. Minibatch optimal-transport coupling (OT-CFM)

### Takeaway
OT-CFM measurably straightens paths and lowers objective variance in low dimensions and gives better few-step Euler results, but in high-dimensional image generation its gains are marginal (CIFAR-10 FID 4.461 → 4.443 at 100 Euler steps). For *conditional* (audio-driven) generation, the benefit is unproven; AVTR-1 uses a related per-region "immiscible" noise-reassignment trick but does not report its effect.

### Cited Findings
- MEASURED (2-D toy, low-dim): Tong et al. Table 2: normalized path energy (NPE) e.g. N→8gaussians OT-CFM **0.018** vs I-CFM **0.222**; moons→8gaussians 0.053 vs 2.738 — near-OT paths. — [Tong et al., arXiv 2302.00482](https://arxiv.org/abs/2302.00482)
- MEASURED (low-dim): OT-CFM converges faster in validation error for the same number of steps (Fig. 2) and has significantly lower objective variance than CFM/FM (Fig. D.4, batch 512). — [arXiv 2302.00482](https://arxiv.org/abs/2302.00482)
- MEASURED (10-d funnel): Table D.2, Euler N=10: log Z bias OT-CFM **−0.039±0.030** vs CFM 0.281±0.202 (RWIS) — OT-CFM "performs significantly better" at a fixed small solver budget; with adaptive solver OT-CFM needs about half the integration time. — [arXiv 2302.00482](https://arxiv.org/abs/2302.00482)
- MEASURED-IMAGE: Table 5 CIFAR-10 FID: Euler 100 steps I-CFM 4.461 vs OT-CFM 4.443; 1000 steps 3.643 vs 3.741 (I-CFM better); adaptive dopri5 3.659 (146 NFE) vs 3.577 (134 NFE). Training overhead "<<1%". — [arXiv 2302.00482](https://arxiv.org/abs/2302.00482)
- OT-CFM "requires surprisingly small batches to approximate the OT map well" (Fig. D.2). — [arXiv 2302.00482](https://arxiv.org/abs/2302.00482)
- USED-NOT-ABLATED (motion, same 42-d space): AVTR-1 applies "immiscible noise assignment ... independently to head rotation, brow, eyes, and mouth. Within each batch, it reassigns the noise trajectories to reduce their distance from the corresponding regional targets", plus progressive (temporally correlated) noise. — [arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- Note: FLOAT and DEMO describe "OT-based flow matching", but in FLOAT's text this is the standard linear (OT-path) conditional flow with independent noise; FLOAT Table 3 compares flow matching vs diffusion, not I-CFM vs minibatch-OT. — [FLOAT, arXiv 2412.01064](https://arxiv.org/abs/2412.01064); [DEMO, arXiv 2510.10650](https://arxiv.org/abs/2510.10650)

### Inferences
- SPECULATIVE: At 64 frames × 42 dims = 2,688 dims per sample, SANG sits between Tong's low-dim regime (big wins) and CIFAR (3,072 pixel dims in their setup; negligible wins). Expect small gains in 10-step quality; possibly larger gains if SANG moves to 1–4 steps.
- SPECULATIVE: In conditional generation, naive minibatch OT pairs noise to targets ignoring the audio condition; it can bias the conditional path. Safer variants: OT within the batch but only over the *noise* assignment (what AVTR-1's immiscible assignment does), or OT computed per region. Cost: a Hungarian/Sinkhorn on a 256×256 cost matrix per step — negligible on one GPU.
- SPECULATIVE: OT-CFM's lower objective variance could reduce the irreducible part of SANG's validation flow loss, making the loss look better without samples necessarily improving — again evaluate on samples.

### Gaps
- No talking-head or human-motion paper found that ablates minibatch-OT coupling vs independent coupling.
- Immiscible diffusion's own paper (cited as ref [20] in AVTR-1) was not retrieved; its ID is not verified here.

---

## 3. Representation alignment (REPA and follow-ups) for motion/audio

### Takeaway
REPA gives large convergence speedups on image DiTs, but the relative gain shrinks with model size (smallest tested SiT-B/2 at 130M still gained FID 33.0 → 24.4). No direct REPA application to audio-to-facial-motion was found; the motion literature uses in-domain alignment targets (contrastive motion encoders, DCT anchors) with modest measured gains, and one 2026 paper reports that naively using a frozen self-supervised motion encoder (Motion-JEPA) as a latent space "fails dramatically".

### Cited Findings
- MEASURED-IMAGE: REPA Table 3 (ImageNet 256, no CFG, 400K iters): SiT-B/2 (130M) FID **33.0 → 24.4**; SiT-L/2 18.8 → 9.7; SiT-XL/2 reaches FID 7.9 at 400K vs 8.3 for vanilla at 7M (>17.5× speedup). — [REPA, arXiv 2410.06940](https://arxiv.org/abs/2410.06940)
- MEASURED-IMAGE: "the convergence speed-up from REPA becomes more significant as the diffusion transformer model increases in size" (Fig. 5b); best alignment depth is early layers (layer 6–8 of SiT-L; Table 2: depth 8 FID 10.0 vs depth 16 12.1); cosine similarity ≈ NT-Xent at convergence (9.9 vs 10.0). — [arXiv 2410.06940](https://arxiv.org/abs/2410.06940)
- MEASURED-IMAGE: Stronger target encoders give better FID (Table 2: MAE-L 12.5 vs DINOv2-B 9.7). — [arXiv 2410.06940](https://arxiv.org/abs/2410.06940)
- MEASURED-MOTION (abstract-level): LUMA aligns a text-to-motion diffusion model to a lightweight contrastive MoCLIP plus low-frequency DCT anchors; reports HumanML3D FID 0.035 and **1.4× faster convergence** vs baseline. — [LUMA, arXiv 2509.25304](https://arxiv.org/abs/2509.25304)
- MEASURED-MOTION (abstract-level): MoRAE reports that transferring the RAE paradigm to motion with a frozen Motion-JEPA encoder "fails dramatically" (ill-conditioned spectrum; flow residuals aligned with decoder-sensitive directions), and needs a compact bottleneck + motion-coupled training. — [MoRAE, arXiv 2607.29180](https://arxiv.org/abs/2607.29180)
- Talking-head video (pixel/latent, not motion): LatentSync adds TREPA (temporal representation alignment) for temporal consistency; its table isolates SyncNet supervision rather than TREPA in the retrieved passage. — [LatentSync, arXiv 2412.09262](https://arxiv.org/abs/2412.09262)
- Human image animation: SemanticREPA aligns to depth and face-ID features (abstract). — [arXiv 2605.10523](https://arxiv.org/abs/2605.10523)

### Inferences
- SPECULATIVE: For a 53M model on 42-d input, there is no obvious strong frozen "visual" encoder whose features describe the target, which is REPA's key ingredient. Candidate targets: (a) features of a pretrained audio-visual sync model (e.g., SyncNet/AV-HuBERT visual stream) applied to the *rendered* GT frames — expensive preprocessing but precomputable once; (b) a small contrastive audio↔motion encoder trained on SANG's own data (LUMA-style, MoCLIP analogue). (b) is cheaper and in-domain.
- SPECULATIVE: Given REPA gains shrink with model size and SANG is data-limited rather than compute-limited (overfitting vanished when data tripled), expected benefit is lower than on ImageNet; rank it below data and loss changes.

### Gaps
- No paper found applying REPA-style hidden-state alignment in an audio-to-face-motion or co-speech-motion flow model.

---

## 4. Few-step / one-step generation

### Takeaway
Measured motion evidence: (i) plain flow-matching talking-head motion models degrade sharply below ~5 steps (FLOAT: NFE 2 → shaky head, static expression; KDTalker: 1 step collapses, 5 steps ≈ best lip sync); (ii) consistency distillation (MotionLCM) and MeanFlow (ARMFlow) achieve 1–2-step motion generation at or better than multi-step baselines on HumanML3D/InterHuman. For SANG (already at 10 Euler steps, 53M, offline), few-step methods are a latency optimization, not a quality lever — except that shortcut/MeanFlow objectives sometimes act as a mild regularizer.

### Cited Findings
- MEASURED-MOTION (talking head, motion latents): FLOAT Table 5 (HDTF): NFE 2: FID 21.785, FVD 178.831, E-FID 1.542, LSE-D 7.559; NFE 5: 21.440 / 164.463 / 1.331 / **7.155**; NFE 10 (default): **21.100 / 162.052 / 1.229** / 7.290; NFE 20: 21.158 / 164.392 / 1.293 / 7.343. Low NFE gives "shaky head motion and a static expression"; Euler vs midpoint/Dopri5 gave no significant improvement. — [FLOAT, arXiv 2412.01064](https://arxiv.org/abs/2412.01064)
- MEASURED-MOTION (LivePortrait keypoints, DDIM): KDTalker Table 6: 1 step LSE-C **0.817**, LSE-D 12.630, FID 19.058; 5 steps LSE-C **7.455**, LSE-D 7.424, FID 10.226; 10 steps 7.448 / 7.436 / 9.939; 50 steps 7.326 / 7.548 / **9.756**; 200 steps 7.221 / 7.633 / 9.901. Lip sync is best at 5–10 steps and slightly declines with more steps. — [KDTalker, arXiv 2503.12963](https://arxiv.org/abs/2503.12963)
- MEASURED-MOTION (AR talking/listening, same 42-d space): AVTR-1 uses 4 Euler steps per chunk at inference (no NFE ablation retrieved). — [arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- MEASURED-MOTION (human motion, consistency distillation): MotionLCM Table 1 (HumanML3D): MLD (50 DDIM) FID 0.473, AITS 0.217 s; MotionLCM 1-step FID **0.467**, AITS **0.030 s**; 2-step FID 0.368, R@3 0.805; 4-step best FID (0.304 in its control table, "w/o control"). — [MotionLCM, arXiv 2404.19759](https://arxiv.org/abs/2404.19759)
- MEASURED-MOTION (MeanFlow, reaction generation): ARMFlow Table 4 (online, InterHuman, same architecture, CFG tuned per method): DDIM 10 FID 3.528; DDIM 50 3.449; Rectified Flow 10 steps **2.449**; ARMFlow (MeanFlow, 1 step) **2.178**. InterX: RF-10 0.059 vs ARMFlow 0.042. Uses biased (r,t) sampling with a proportion r=t to strengthen instantaneous velocity learning. — [ARMFlow, arXiv 2512.16234](https://arxiv.org/abs/2512.16234)
- MEASURED-MOTION (abstract-level): FlowerDance applies MeanFlow + physical-consistency constraints to music-to-dance with few steps. — [arXiv 2511.21029](https://arxiv.org/abs/2511.21029)
- MEASURED-IMAGE: Shortcut models Table 1 (DiT-B, equal compute), CelebAHQ FID at 128/4/1 steps: Flow Matching 7.3 / (63.3) / (280.5); Consistency Training 53.7 / 19.0 / 33.2; Reflow (two-stage) 16.1 / 18.4 / 23.2; **Shortcut 6.9 / 13.8 / 20.5**. ImageNet-256: FM 17.3 / (108.2) / (324.8); Shortcut 15.5 / 28.3 / 40.3. Shortcut slightly *beats* FM at 128 steps ("self-consistency loss acts as a form of implicit regularization" — authors' hypothesis). — [Shortcut Models, arXiv 2410.12557](https://arxiv.org/abs/2410.12557)
- MEASURED-IMAGE: MeanFlow training analysis: improved schedule reaches 1-NFE ImageNet FID 2.87 vs 3.43 baseline MeanFlow (DiT-XL) — [arXiv 2511.19065](https://arxiv.org/abs/2511.19065); Rectified MeanFlow (reflow-straightened couplings) improves baseline MeanFlow FID 30.9 → 8.6 at equal budget — [arXiv 2511.23342](https://arxiv.org/abs/2511.23342).
- MEASURED (talking-head *video*, not motion): LeapTalk is a 1-NFE Brownian-bridge + heterogeneous DMD student of a video diffusion teacher. Table 1 (HDTF): LeapTalk Pro FID 21, FVD 197, Sync-C 8.38 at 1 NFE vs SoulX-FlashHead (4 NFE) FID 30, Sync-C 8.07. Ablation Table 2: w/o audio-driven CFG in the DMD teacher score → Sync-C **8.38 → 4.34**, FID 21 → 162; w/o Brownian bridge FID 21 → 217. — [LeapTalk, arXiv 2608.00079](https://arxiv.org/abs/2608.00079)
- Reflow in 2-D OT benchmarks: 2-RF NPE 0.069 vs OT-CFM 0.018 (Tong Table 2). — [arXiv 2302.00482](https://arxiv.org/abs/2302.00482)

### Inferences
- SPECULATIVE: SANG at 10 Euler steps is likely near its sampling-quality plateau (FLOAT and KDTalker both saturate around 5–10). Test 5 steps: it may *improve* lip sync (FLOAT LSE-D 7.155 at NFE 5 vs 7.290 at 10; KDTalker LSE-C peaks at 5) at the cost of some expressiveness.
- SPECULATIVE: If one-step is desired, LeapTalk's key lesson that transfers to motion is: bake audio CFG into the distillation target (otherwise lip sync collapses). For a 53M 42-d model, MeanFlow or shortcut training from scratch (single run, JVP cost ~1.5–2× per step for MeanFlow) is lower-engineering than DMD (needs a fake-score network and teacher).
- LeapTalk is a pixel/latent video model, so its numbers do not directly speak to motion-space generators.

### Gaps
- "Motar's consistency head" could not be located in the research index (searches for Motar returned nothing matching); unverified.
- No talking-head *motion-space* paper found that ablates shortcut/MeanFlow vs flow matching on lip-sync metrics.
- Original MeanFlow paper ID not verified in this session.

---

## 5. Model scaling for motion generators

### Takeaway
Talking-head motion generators span ~43M (KDTalker) to 153M (AVTR-1); human-motion evidence (Kimodo, HY-Motion) shows quality gains from 56M → 282M and saturation of motion quality around 0.46B — but always with far more data than SANG uses. With SANG's data-limited behaviour, scaling data (TalkVid has ~20× more hours available) should come before scaling parameters.

### Cited Findings
- MEASURED-MOTION: KDTalker's keypoint diffusion model is **42.93M parameters**, batch 256, AdamW, warmup + cosine to peak LR 5.12e-4, 64 audio frames per pass, trained on only 4,282 aligned VoxCeleb pairs; outperforms SadTalker/AniTalker/AniPortrait on HDTF lip sync (LSE-C 7.326 at 50 steps). Frame-window ablation Table 7: 8 → 64 frames improves LSE-C 6.875 → 7.326 and diversity 0.673 → 0.760. — [KDTalker, arXiv 2503.12963](https://arxiv.org/abs/2503.12963)
- AVTR-1: 153M params, 18 layers, width 512, FFN 512, dropout 0, trained 200k steps on one GH200 in ~35 h on 926 h of data; Adan optimizer with 3 weight-decay groups; LR warmup 5k → 3e-4, cosine to 1e-5; EMA 0.995. — [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- FLOAT: vector-field predictor with hidden 1024, 8 heads; batch size 8, LR 1e-5, 2,000k steps, ~2 days on one A100; trained on HDTF (11.3 h, 230 IDs) + RAVDESS. — [FLOAT, arXiv 2412.01064](https://arxiv.org/abs/2412.01064)
- MEASURED-MOTION: Kimodo Table 2 (700 h mocap): S (56M) overview FID 3.10, R@3 64.0; M (148M) 2.36 / 69.2; L (282M) 1.85 / 71.9 — "increasing model size improves performance across all metrics", but authors "speculate that without more data there will be diminishing returns". Data: 10% subset worsens foot skate 4.23 → 5.28 cm/s and constraint error (full-body pos 2.77 → 4.60 cm). Batch: 512 → 2048 improves FID 2.01 → 1.61. — [Kimodo, arXiv 2603.15546](https://arxiv.org/abs/2603.15546)
- MEASURED-MOTION: HY-Motion Tables 3–4 (human eval): motion quality DiT-0.05B 2.91, DiT-0.46B 3.26, DiT-1B 3.34 — "motion quality reaches a saturation point beyond the 0.46B"; instruction following keeps improving with size and with 3,000 h vs 400 h pretraining. — [HY-Motion 1.0, arXiv 2512.23464](https://arxiv.org/abs/2512.23464)
- MEASURED-MOTION (AR tokens): ScaMo reports a logarithmic law of normalized test loss vs compute and power laws for parameters/data. — [ScaMo, arXiv 2412.14559](https://arxiv.org/abs/2412.14559)
- TalkVid contains 1,244 h from 7,729 speakers; mean clip length 17.93 s. — [TalkVid, arXiv 2508.13618](https://arxiv.org/abs/2508.13618)

### Inferences
- SPECULATIVE: SANG's 12.6k clips ≈ 12.6k × ~18 s ≈ ~63 h (if clips match TalkVid mean length) — roughly 5% of TalkVid and ~7% of AVTR-1's 926 h. Kimodo's 10%-data result and SANG's own 3.7k → 12.6k experiment both point to data as the binding constraint. A 53M model is within the range where published talking-head motion models work (KDTalker 43M).
- SPECULATIVE: SANG uses dropout 0.1 while AVTR-1 (more data) uses dropout 0; once data grows substantially, dropout can likely be reduced.
- SPECULATIVE: Batch 256 on one GPU is reasonable; Kimodo's batch-size gains were at 512–2048 with multi-GPU; gradient accumulation is an option but trades wall-clock.

### Gaps
- Ditto parameter count and results not retrieved.
- MDM/MoMask/MotionGPT scaling numbers were not retrieved in this session (HumanML3D evidence here comes from MotionLCM, Kimodo, HY-Motion, ScaMo).
- No talking-head paper found with a controlled model-size sweep on lip-sync metrics.

---

## 6. Loss design: velocity/acceleration, geometric keypoint losses, cosine direction, L1 vs L2

### Takeaway
Two recipes relevant to SANG's exact 42-d space exist: FLOAT (L1 flow loss + velocity loss on frame differences, λ=1) and AVTR-1 (region-balanced MSE + cosine velocity-direction loss + 1st–3rd order smoothness on expression, relative rotation, and reconstructed 3D keypoints, with λ_rot=λ_3D=100, λ_exp=0.1). Neither paper's retrieved passages ablate these terms, so benefit sizes are unmeasured for facial motion; the only measured acceleration-loss number found is in IMU/human motion.

### Cited Findings
- USED-NOT-ABLATED: FLOAT uses **L1** distance for the flow objective and a velocity loss L_vel = ‖Δv_t − Δu_t‖ (one-frame differences of predicted vs target vector field), λ_OT = λ_vel = 1. — [FLOAT, arXiv 2412.01064](https://arxiv.org/abs/2412.01064)
- USED-NOT-ABLATED: AVTR-1 total loss = L_CFM (region-summed MSE) + L_cos (cosine distance between predicted and target velocities, per region, "constrains the direction of the predicted flow independently of its magnitude") + L_reg + 0.01·L_VAD. L_reg = λ_exp(‖Δê‖²+‖Δ²ê‖²+‖Δ³ê‖²) + λ_rot(‖ω‖²+‖Δω‖²+‖Δ²ω‖²) + λ_3D(‖Δq̂‖²+‖Δ²q̂‖²+‖Δ³q̂‖²), where ω is the axis-angle of R̂_{t−1}ᵀR̂_t and q̂ are 3D keypoints from adding predicted expression to reference canonical keypoints and applying R̂; λ_exp=0.1, λ_rot=λ_3D=100; the last three history frames are prepended so regularization crosses chunk boundaries. Regularizers are applied to the one-step clean estimate m̂ = m_t + (1−t)v_θ. — [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- MEASURED-MOTION (IMU/human motion): Fine-tuning a text-to-motion diffusion model with a second-order acceleration loss reduced L_acc by 12.7% and improved downstream HAR by 8.7% vs the earlier model; larger gains in high-dynamic activities. — [arXiv 2512.08859](https://arxiv.org/abs/2512.08859)
- LeapTalk's distillation uses spatially weighted losses W = 1 + λ_face·M_face + λ_lip·M_lip (face/lip masks) — pixel-space analogue of region weighting. — [LeapTalk, arXiv 2608.00079](https://arxiv.org/abs/2608.00079)

### Inferences
- SPECULATIVE: For SANG (non-AR, 64-frame windows), the cheapest transferable pieces are: (1) region-balanced loss (mouth vs rotation vs brow/eyes), (2) cosine velocity-direction term, (3) velocity (Δ) loss on the predicted clean motion x̂₀ = x_t + (1−t)v̂. The 3D-keypoint smoothness term requires the canonical keypoints (reference) at train time — SANG would need to store per-clip reference canonical keypoints; cost is a few matmuls.
- SPECULATIVE: Smoothness regularizers on x̂₀ at high noise (t≈0 in AVTR-1's convention where t=1 is data) are noisy; consider weighting them by t or applying only for t above a threshold. Heavy smoothness weights (AVTR-1's 100) risk damping lip articulation; tune against a lip-sync metric, not just jitter.
- SPECULATIVE: L1 (FLOAT) vs L2: no motion ablation found; L1 is more robust to outlier frames (tracking glitches), which matters because SANG's data is only shot-cut filtered. If SANG adopts outlier filtering (Section 7), L2 is probably fine.

### Gaps
- No ablation found isolating velocity/acceleration/cosine/3D-keypoint losses on lip-sync or motion-realism metrics for talking heads.
- No L1-vs-L2 ablation for flow matching on motion found.

---

## 7. Data curation for talking heads

### Takeaway
The best-documented pipelines filter on video quality (DOVER), point-tracking stability (CoTracker ratio band), head pose/rotation/completeness/resolution, audio-visual offset/SyncNet confidence, and motion-outlier windows. Measured *effects on a generator* of individual filters are rarely reported; TalkVid validates its filters against human judgment (95.1% accuracy) and shows that training on TalkVid generalizes better, and LatentSync shows offset correction + Sync-conf ≥ 3 filtering is needed for SyncNet convergence. AVTR-1's finite-difference outlier mask is directly implementable on SANG's 42-d motion.

### Cited Findings
- TalkVid Table 2 filters (clip retained only if all pass): DOVER ≥ 7.0; CoTracker stability ratio ∈ [0.85, 0.999] (lower bound removes blur/erratic motion; upper bound removes "frozen" faces with <0.3 px mean displacement); Movement score avg ≥ 80/min ≥ 60; Rotation avg ≥ 70/min ≥ 60; Orientation avg ≥ 70/min ≥ 30; Resolution avg ≥ 50/min ≥ 40; Completeness = 100 (eyes/nose/mouth keypoints within frame; "Occlusions are tolerated as long as keypoints are within the visible area"). — [TalkVid, arXiv 2508.13618](https://arxiv.org/abs/2508.13618)
- TalkVid human validation (Table 3, Table 7): avg Cohen's κ 0.79; automated filters vs human golden standard avg accuracy 95.1%, F1 95.3%. Final dataset mean DOVER 8.55, mean CoTracker ratio 0.92. TalkVid-Core: 160 h stricter, demographically balanced subset. — [arXiv 2508.13618](https://arxiv.org/abs/2508.13618)
- Implication for SANG: TalkVid's Completeness filter explicitly tolerates occlusion (e.g., hands) if keypoints are in frame — so hand-over-face clips can remain in TalkVid. — [arXiv 2508.13618](https://arxiv.org/abs/2508.13618)
- Pipeline cost: CoTracker filter RTF 64.21, DOVER RTF 87.36, head filter RTF 72.47 on 96 CPU + 8×A800. — [arXiv 2508.13618](https://arxiv.org/abs/2508.13618)
- AVTR-1 training-window filter: per track, compute 1st/2nd/3rd-order finite differences of rotation and expression coordinates, scale per coordinate, average absolute values, smooth over 21 frames; frame is outlier if score > **0.15**; mask extended ±5 frames; keep only 25-frame windows not overlapping the mask. Also: human annotators required faces "free of occlusion"; faces ≥160 px at 1080p; whole-body pose landmarks with mean confidence > 0.3. — [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- USED-NOT-ABLATED: AVTR-1 applies noise truncation at inference (|noise| ≤ 1.2); qualitatively "suppresses extreme expression deformations and head rotations" (Fig. 4). — [arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- MEASURED (SyncNet training, not generator): LatentSync corrects audio-visual offset with pretrained SyncNet and removes videos with Sync_conf < 3; "without offset adjustment, the model's convergence is significantly impaired" (Fig. 10); affine alignment before offset estimation works better. — [LatentSync, arXiv 2412.09262](https://arxiv.org/abs/2412.09262)
- MEASURED (generator, lip-sync video): LatentSync Table 2: w/o SyncNet supervision Sync_conf 4.6, FVD 220.37; + pixel-space SyncNet 8.9, FVD 162.74 (supervision effect, not filtering). — [arXiv 2412.09262](https://arxiv.org/abs/2412.09262)
- KDTalker filtered VoxCeleb to 4,282 aligned audio-video pairs due to misalignment. — [KDTalker, arXiv 2503.12963](https://arxiv.org/abs/2503.12963)
- SoulX-FlashHead introduces VividHead: 782 h of "strictly aligned footage" (abstract; pipeline details not retrieved). — [SoulX-FlashHead, arXiv 2602.07449](https://arxiv.org/abs/2602.07449)
- MEASURED-MOTION (curriculum on data quality): HY-Motion pretrains on 3,000 h mixed-quality (incl. noisy in-the-wild video extractions) then fine-tunes on 400 h curated data at 0.1× LR; fine-tuning "drastically reduc[es] high-frequency jitter and foot sliding"; model trained only on 400 h HQ had better quality (3.31 vs 3.26) but worse instruction-following (3.05 vs 3.20). — [HY-Motion, arXiv 2512.23464](https://arxiv.org/abs/2512.23464)

### Inferences
- SPECULATIVE (high priority, low cost): Implement the AVTR-1 finite-difference outlier mask on SANG's existing 42-d tracks — no new models needed, runs on CPU in seconds. LivePortrait tracking glitches (hands over mouth, profile turns) show up as 2nd/3rd-order spikes; masking them removes targets the model cannot explain from audio and likely lowers irreducible flow loss.
- SPECULATIVE: Offset correction/Sync-conf filtering matters more for a motion model than it seems — a constant A/V offset in a clip teaches systematically delayed/advanced lip motion. Run SyncNet offset estimation once per clip, shift audio features by the detected offset, drop Sync_conf < 3 (LatentSync threshold).
- SPECULATIVE: Hand-occlusion filtering (e.g., MediaPipe Hands or DWPose hand keypoints overlapping the face box) is not measured anywhere found; the outlier mask partially covers it. Consider it second-tier.
- SPECULATIVE: HY-Motion's "scale-then-refine" suggests a SANG curriculum: train on the full (loosely filtered) TalkVid, then fine-tune at 0.1× LR on a strictly filtered subset (Sync-conf high, no outliers, frontal).

### Gaps
- No talking-head paper found with a before/after ablation of SyncNet-score filtering or hand-occlusion filtering on the *generator's* metrics.
- SoulX VividHead filtering criteria were not retrievable in this session.

---

## 8. Test-time tricks: CFG scale/schedules, guidance interval, number of ODE steps

### Takeaway
On motion: audio CFG ≈ 2 is the FLOAT default and optimal for lip sync in its sweep; higher CFG increases head-pose diversity but reduces beat alignment (LeapTalk). Guidance interval (guidance only at mid noise levels) gives sizable FID gains on images and slightly cheaper sampling; untested on motion. Steps: 5–10 Euler is the empirically supported range.

### Cited Findings
- MEASURED-MOTION: FLOAT Table 6 (RAVDESS): γa=1, γe=1: FID 33.066, FVD 171.047, LSE-D 7.049; γa=2, γe=1 (default): 31.681 / 166.359 / **6.994**; γa=2, γe=2: 32.253 / **162.658** / 6.994. "increasing γa leads to better temporal consistency (FVD) and lip synchronization (LSE-D)". — [FLOAT, arXiv 2412.01064](https://arxiv.org/abs/2412.01064)
- MEASURED (talking-head video, 1-step student): LeapTalk Table 7 audio-CFG sweep on HDTF: scale 1.0 Avg pose Std 1.655, BAS 0.723; 3.0 3.394 / 0.658; 5.0 5.000 / 0.696; 7.0 6.323 / 0.650 — "over-strong guidance can hurt audio-motion alignment". — [LeapTalk, arXiv 2608.00079](https://arxiv.org/abs/2608.00079)
- USED-NOT-ABLATED (motion, same 42-d space): AVTR-1 trains multi-condition dropout (all conditions kept 45%, all dropped 15%, past motion kept 75%) and applies independent guidance weights per condition and per region (rotation/brow/eyes/mouth) in regional latent spaces. — [AVTR-1, arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- MEASURED-IMAGE: Guidance interval Table 1 (ImageNet-512): EDM2-S FID 2.23 (CFG) → **1.68** (interval σ∈(0.28, 2.90], w=2.1); EDM2-XXL 1.81 → 1.40; DiT-XL/2 3.04 → 2.40 (guidance in 75 of 250 steps). Guidance at high noise truncates the distribution; at low noise it adds nothing, so it can be skipped to save compute; interval allows higher w with less sensitivity. — [Kynkäänniemi et al., arXiv 2404.07724](https://arxiv.org/abs/2404.07724)
- REPA's best system result (FID 1.42) also used the guidance interval. — [arXiv 2410.06940](https://arxiv.org/abs/2410.06940)
- Steps: FLOAT NFE 5 best LSE-D (7.155), NFE 10 best FID/FVD/E-FID (Table 5); KDTalker 5 steps best LSE-C (7.455), 50 best FID (Table 6); AVTR-1 uses 4 Euler steps. — [arXiv 2412.01064](https://arxiv.org/abs/2412.01064); [arXiv 2503.12963](https://arxiv.org/abs/2503.12963); [arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- SD3: rectified-flow formulations degrade less than diffusion when steps are reduced (Fig. 3). — [arXiv 2403.03206](https://arxiv.org/abs/2403.03206)

### Inferences
- SPECULATIVE (zero training cost): Sweep SANG audio CFG ∈ {1.5, 2, 2.5, 3} × steps ∈ {5, 8, 10} × guidance interval (apply CFG only for mid-t, e.g. t∈[0.2,0.8] in SANG's convention) and pick per metric. Separate CFG weights for mouth vs head-pose dims (AVTR-1 style) require only masking the guidance residual by coordinate — can be done without retraining if the 42-d layout groups regions.
- SPECULATIVE: Noise truncation (AVTR-1, τ=1.2) is a zero-cost inference knob that trades diversity for fewer extreme poses.

### Gaps
- No motion-model ablation of guidance intervals or CFG schedules (e.g., linearly increasing) found.

---

## 9. Cross-cutting: ranked recipe for SANG (53M, 64×42, one GPU)

### Takeaway
Ranked by (expected impact on a data-limited 42-d audio→motion model) ÷ (cost). Evidence strength noted per item. Data work ranks first because SANG's own experiments show data-limited behaviour and every scaling study found here shows data gains.

### Cited Findings
- Data scale matters for motion quality and constraint accuracy (Kimodo Table 2; HY-Motion Table 3) — [arXiv 2603.15546](https://arxiv.org/abs/2603.15546); [arXiv 2512.23464](https://arxiv.org/abs/2512.23464)
- Outlier-window mask recipe (AVTR-1 §3.1) — [arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- Offset correction + Sync-conf ≥ 3 (LatentSync §4) — [arXiv 2412.09262](https://arxiv.org/abs/2412.09262)
- Logit-normal (SD3 Tables 1–2) — [arXiv 2403.03206](https://arxiv.org/abs/2403.03206)
- Steps/CFG (FLOAT Tables 5–6; KDTalker Table 6) — [arXiv 2412.01064](https://arxiv.org/abs/2412.01064); [arXiv 2503.12963](https://arxiv.org/abs/2503.12963)

### Inferences
Ranked list (all SPECULATIVE as applied to SANG; evidence type in brackets):
1. **More TalkVid data** (currently ~5% of the 1,244 h): biggest expected gain. Cost: extraction compute (LivePortrait motion + audio features), disk. [MEASURED-MOTION: Kimodo, HY-Motion; SANG's own 3.7k→12.6k result]
2. **Finite-difference outlier mask** on 42-d tracks (AVTR-1 thresholds: score>0.15 after 21-frame smoothing, ±5-frame dilation). Cost: minutes of CPU. [USED-NOT-ABLATED]
3. **A/V offset correction + SyncNet conf filter (<3 drop)**. Cost: one SyncNet pass per clip (GPU-hours for tens of thousands of clips). [MEASURED on SyncNet training only]
4. **Inference sweep: steps 5/8/10 × CFG 1.5–3 × guidance interval; noise truncation**. Cost: evaluation only. [MEASURED-MOTION for steps/CFG; MEASURED-IMAGE for interval]
5. **Region-balanced loss + velocity (Δ) loss + cosine velocity-direction loss** on x̂₀. Cost: trivial compute; one retrain. [USED-NOT-ABLATED: FLOAT, AVTR-1]
6. **Logit-normal (0,1) or mode(1.29) t-sampling**; evaluate with stratified-t val loss + sample metrics. Cost: one line; one retrain. [MEASURED-IMAGE; USED in AVTR-1]
7. **Batch-level noise reassignment (immiscible / minibatch OT), per region**. Cost: Hungarian on 256×256 per step, <1% overhead. [MEASURED low-dim/image small gains; USED in AVTR-1]
8. **Curriculum: pretrain on broad data, fine-tune at 0.1× LR on strictly filtered subset**. Cost: second training stage. [MEASURED-MOTION qualitative + human eval, HY-Motion]
9. **In-domain representation alignment** (small contrastive audio↔motion encoder as REPA target). Cost: train an auxiliary encoder; moderate engineering. [MEASURED-IMAGE REPA; abstract-level motion LUMA 1.4× convergence]
10. **One-step (MeanFlow/shortcut) only if latency is needed**; if distilling, include audio CFG in the teacher target. Cost: JVP overhead / extra networks. [MEASURED-MOTION MotionLCM, ARMFlow; MEASURED-IMAGE shortcut; LeapTalk video]
11. **Model scaling beyond ~50–150M**: defer until data is ≥ several hundred hours. [MEASURED-MOTION Kimodo/HY-Motion with far more data]

### Gaps
- None of the ranked items has a controlled ablation on an audio-driven *facial-motion* flow model with lip-sync metrics; SANG will need its own A/B runs with a fixed evaluation protocol (held-out speakers, rendered LSE-C/LSE-D, pose diversity, jitter/acceleration statistics, motion FID in 42-d space).
