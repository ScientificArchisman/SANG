# Raising lip-trajectory accuracy (correlation, not amplitude) in SANG-M

Status: COMPLETE, 2026-10-07. Tags: [M] measured in the cited paper; [C] claim without a matching table; [I] my inference. Space: (motion) means keypoint, 3DMM, mesh or latent-motion evidence; (pixel) means video-space evidence. Every arXiv ID below was verified by fetching its arXiv page: 2608.06408, 2510.10650, 2408.00762, 2505.23290, 2502.20323, 2305.19556, 2412.01064, 2412.09296, 2608.15296, 2508.16401, 2609.22913, 2609.10317, 2506.23552, 2406.14272, 2608.05218, 2311.18168. Material already in `reports/SANG talking head improvements.md` and `reports/SANG boundary and lip sync fixes.md` (Teller, Pasad, SyncDiff, PASE, LatentSync, the DPO papers, UniSync, THUNDER, DreamTalk, CFG variants) is not re-collected. Per-paper raw notes are in the RAW FINDINGS LOG at the end.

**Bottom line [I].** The strongest new evidence says that per-sample lip accuracy in generative motion models is limited by mouth *stochasticity* as much as by the audio features:
- A probabilistic mesh model's single samples had 37% higher lip-vertex error than a deterministic model (10.8 vs 7.9), yet the *mean* of its samples matched the deterministic model exactly (7.9) ([Yang et al.](https://arxiv.org/html/2311.18168)).
- SubtleTalk measures flow matching alone at LVE 14.60, a deterministic prior alone at 11.35, and prior plus residual flow at 12.46 ([SubtleTalk](https://arxiv.org/html/2608.06408)).

SANG's flat ~0.63 correlation across every guidance setting fits that picture. The pending two-seed test decides it: compare the seed-to-seed correlation r_ss with corr(sample, GT), both measured the same way (see the decision rule under Q4). The second concrete finding is that SANG's audio encoder, WavLM-Large, is English-only by its own model card ("pre-trained in English and should therefore perform well only in English"), while TalkVid spans 15 languages ([WavLM-Large card](https://huggingface.co/microsoft/wavlm-large)). In mesh models, the encoder alone moves lip error by 10–20% ([UniTalker](https://arxiv.org/abs/2408.00762)).

## Q1. Audio front ends with measured lip-accuracy gains (motion / mesh)

### Takeaway
In 3D-mesh models, swapping the frozen self-supervised encoder moves lip-vertex error by about 10–20%, and the multilingual, higher-capacity encoder (wav2vec2-XLSR-53) is best overall (UniTalker). Add-on semantic or phoneme features give only 0.2–3% (Wav2Sem, PD-GS). For SANG's 15-language data, the clearest mismatch is that WavLM-Large is an English-only model. No paper compares WavLM-Large, XLS-R, Whisper and mHuBERT on lip *correlation* in a multilingual motion model.

### Cited Findings
- [M] (motion: mesh) UniTalker Table 6, LVE across 8 test sets (D0=BIWI, 1e-4; D1–D3 1e-6 m²; D4–D7 1e-5 m²):
  - Wav2Vec2-Base-960h: 4.491 9.916 9.887 9.812 1.585 2.217 1.351 1.409
  - WavLM-Base: 4.033 8.269 9.253 9.117 1.417 2.044 1.184 1.340
  - WavLM-Base-Plus: 4.080 8.136 9.776 9.053 1.392 1.975 1.158 1.264
  - Wav2Vec2-XLSR-53 (56k h, 53 languages): 3.859 8.303 8.648 8.991 1.326 2.056 1.145 1.211
  - The authors say precision is "largely affected by the pre-trained audio encoder from three aspects, including the pre-training method, the scale and diversity of pre-training dataset and the capacity of pre-training backbone." — [UniTalker](https://arxiv.org/abs/2408.00762)
- [M] (motion: mesh) UniTalker's Chinese-speech test set (D6) had higher LVE "because the proportion of Chinese speeches in A2F-Bench is small." Lip accuracy is language-dependent when a language is under-represented. — [UniTalker](https://arxiv.org/abs/2408.00762)
- [C] WavLM-Large was pretrained on 60k h Libri-Light + 10k h GigaSpeech + 24k h VoxPopuli (94k h). The card states: "The model was pre-trained in English and should therefore perform well only in English." — [WavLM-Large model card](https://huggingface.co/microsoft/wavlm-large). This conflicts with UniTalker's description of WavLM-Base-Plus as trained on "94k hours of audios in 23 languages" ([UniTalker](https://arxiv.org/abs/2408.00762)). The Microsoft card is the primary source.
- [M] (motion: mesh, multilingual) MultiTalk uses an encoder "pretrained on 53 languages" plus a language-style embedding. Its Table 5 reports the multilingual encoder beating an English-only one across languages; exact Table 5 numbers were not extracted. Main-table LVE (×1e-4 mm) for MultiTalk / SelfTalk: English 1.16 / 1.99, Italian 1.06 / 2.59, French 1.39 / 1.98, Greek 1.26 / 2.11. This comparison is confounded by training data. — [MultiTalk](https://arxiv.org/html/2406.14272)
- [M] (motion: mesh) Wav2Sem (CVPR 2025) adds semantic decoupling of near-homophone syllables to wav2vec2 and HuBERT features. LVE, baseline → +Wav2Sem_c:
  - FaceFormer, VOCASET: 4.1090 → 3.9891 (×1e-5)
  - UniTalker: 3.5416 → 3.1476 (VOCASET); 4.0213 → 3.9112 (BIWI, ×1e-4)
  - FaceDiffuser (HuBERT), VOCASET: 3.7924 → 3.7739
  - Most gains are 0.2–3%. — [Wav2Sem](https://arxiv.org/pdf/2505.23290)
- [M] (3DGS with explicit lip-deformation stage; HDTF) PD-GS gates Whisper+MFA phoneme embeddings into HuBERT audio. LMD / Sync: audio only 2.73 / 8.71; concatenation 2.70 / 8.78; gated 2.66 / 8.85. — [PD-GS](https://arxiv.org/html/2608.05218)
- [M] (motion: FLAME) ARTalk at a 100-frame window: HuBERT LVE 9.34 vs the Mimi codec 9.97. "Mimi outperforms HuBERT when the input length is short, whereas HuBERT performs better on longer audio segments." — [ARTalk](https://arxiv.org/html/2502.20323)
- [M] Front ends of the recent motion systems: AVTR-1 (SANG's 42-d space) uses 24-layer HuBERT-Large at 50 Hz averaged to 25 fps ([AVTR-1](https://arxiv.org/html/2609.22913)); Motar uses wav2vec2-base ([Motar](https://arxiv.org/html/2609.10317)); FLOAT uses wav2vec2 ([FLOAT](https://arxiv.org/html/2412.01064)). None ablates its encoder.
- [M/C] (motion latent) DEMO trains its lip-motion encoder with bidirectional InfoNCE against per-frame audio features, but **no ablation isolates the contrastive term**. Its HDTF "LSE-D" values (238.577 vs 243.227 for disentangled vs plain VAE) are on an unusual scale. — [DEMO](https://arxiv.org/html/2510.10650)

### Inferences
- [I] SANG feeds 15-language speech through an English-only encoder's *last* layer. The last layer is the most pretext-specific; the prior report's probing evidence points to mid or late-but-not-last layers. Two cheap fixes follow: (a) a learned softmax-weighted sum of WavLM-Large layers, and (b) a multilingual encoder (Whisper-large-v3 encoder, which the prior report favoured on LivePortrait keypoints, or XLS-R/XLSR-53, which UniTalker and MultiTalk favour).
- [I] A free diagnostic decides whether the encoder is the bottleneck: split SANG's lip correlation and CCC by clip language. If English clips score clearly higher than the other 14 languages, at matched clip counts, the encoder is the limit and the encoder arm moves up the plan.
- [I] Phoneme and semantic add-ons have small measured effects (PD-GS LMD −2.6%; Wav2Sem mostly <3%). Forced alignment in 15 languages is costly. They come last.

### Gaps
- No source measures lip correlation or CCC for WavLM-Large vs XLS-R vs Whisper vs mHuBERT in a multilingual motion model. No data2vec result in a motion model was found.
- MultiTalk's Table 5 exact numbers were not extracted.
- No measured result was found for using a *pretrained contrastive audio–lip encoder* (SyncNet-like) as the conditioning feature in a motion model. DEMO uses InfoNCE but does not ablate it.

## Q2. Temporal receptive field and coarticulation

### Takeaway
More audio context consistently helps, but the evidence is indirect:
- ARTalk's window sweep, 8 → 100 frames, cuts LVE from 11.73 to 9.34 (motion).
- A phonetic-context lip-sync study finds an optimum of about ±13 frames, roughly 1.2 s in total (pixel).
- NVIDIA's production system gives each frame 0.52–1 s of audio.

The 2024–2026 motion DiTs (FLOAT, Motar) keep a ±2-frame direct audio view. FLOAT measured frame-wise AdaLN *beating* cross-attention (LSE-D 7.290 vs 7.757). No motion paper ablates the per-frame audio window at a fixed clip length.

### Cited Findings
- [M] (motion: FLAME) ARTalk window-length ablation, LVE: 8 frames 11.73; 25 frames 10.20; 50 frames 9.78; 100 frames 9.34. — [ARTalk](https://arxiv.org/html/2502.20323)
- [M] (pixel, landmark LMD) CALS audio-window sweep: optimum ±15 frames on LRW (LMD 1.162, the largest window available) and about ±13 frames on LRS2 (LMD 1.059). "the audio context of around 1.2 seconds assists in resolving ambiguities in the lip shapes of phones, improving spatio-temporal alignment." — [CALS](https://arxiv.org/html/2305.19556)
- [C] (motion: mesh, production) Audio2Face-3D: the regression net uses a "0.52 seconds" audio context; the diffusion net uses a "1-second audio chunk" (HuBERT). No window ablation. — [Audio2Face-3D](https://arxiv.org/html/2508.16401)
- [M] (motion latent, HDTF) FLOAT Table 3: frame-wise AdaLN vs cross-attention, LSE-D 7.290 vs 7.757; FID 21.100 vs 21.873; E-FID 1.229 vs 1.452. FLOAT's self-attention is masked to "attention window length T = 2", i.e. 2·T neighbouring frames. — [FLOAT](https://arxiv.org/html/2412.01064)
- [M] (motion latent) Motar's local branch uses "windowed cross-attention" with 2 frames per side for lip articulation. — [Motar](https://arxiv.org/html/2609.10317)

### Inferences
- [I] SANG's 64-frame bidirectional self-attention already carries neighbouring audio to each frame, but only indirectly: audio enters per frame through AdaLN, at ±80 ms. Widening the per-frame audio view to about ±8–12 frames (±320–480 ms) is the cheapest test. It needs no feature re-caching, since only the conv or audio-adapter changes. Do it *inside* the AdaLN path, with a dilated Conv1d stack or a 2-layer local transformer over the audio sequence, rather than switching to cross-attention, which FLOAT measured as worse.
- [I] Expected effect: small to moderate, based on ARTalk's 20% LVE drop for a 12× window and CALS's optimum. Neither is the same manipulation as SANG's, so the size is uncertain.

### Gaps
- No motion-space ablation of the per-frame audio receptive field at a fixed sequence length was found.
- No primary phonetics source was fetched for coarticulation time spans (for example, anticipatory lip rounding).

## Q3. Lip-specific supervision in motion space

### Takeaway
Only two new motion-space lip-supervision results were measured, and both act on a *deterministic* path:
- SubtleTalk's vertex+velocity-trained motion prior.
- GoHD's distillation of a lip expert into a regressor (MLD 2.012 → 1.792).

The prior report already covers generic sync-expert losses, which raise sync scores while worsening landmark error (DreamTalk), and THUNDER's mesh-to-speech loss (PCC 0.568 → 0.639). No paper was found that trains on lip correlation or CCC directly.

### Cited Findings
- [M] (motion: FLAME) SubtleTalk's DMP is trained with ℒ_vert + ℒ_vel in FLAME vertex space, with separate face and lip weighting. As a stand-alone deterministic model it reaches LVE 11.35 vs 14.60 for flow matching alone. — [SubtleTalk](https://arxiv.org/html/2608.06408)
- [M] (motion: 3DMM, HDTF) GoHD Table 3, MLD: full 1.792; without the deterministic Stage 1, 1.931; without Wav2Lip distillation, 2.012. — [GoHD](https://arxiv.org/html/2412.09296)
- [C] (motion: mesh) Audio2Face-3D lists phoneme, motion and lip-distance losses, with no ablation. — [Audio2Face-3D](https://arxiv.org/html/2508.16401)
- [M] (from the prior report) DreamTalk's lip expert raised sync 2.63 → 4.51 but worsened mouth-landmark error 3.07 → 3.42. THUNDER's mesh-to-speech loss gave lip PCC 0.568 → 0.639 and CCC 0.359 → 0.426. — [prior report](/Users/archismanchakraborti/Desktop/SANG/reports/SANG%20boundary%20and%20lip%20sync%20fixes.md)
- [M] DEMO's InfoNCE audio–lip alignment has no isolated ablation. — [DEMO](https://arxiv.org/html/2510.10650)

### Inferences
- [I] In flow matching, a lip loss on the predicted clean sample x̂₁ = x_t + (1−t)·v̂ at high noise pulls toward the conditional mean, which is regression-like and shrinks amplitude. Apply it only for t near 1 (low noise), or put it on a separate deterministic mouth head (Q4).
- [I] The loss that targets SANG's metric directly is a per-window **CCC loss on lip aperture** (upper–lower inner-lip distance from the 42-d vector) between x̂₁ and GT. CCC penalises timing (correlation) and amplitude or offset mismatch together, so it contains its own amplitude guard. This is untested in the literature.

### Gaps
- No 2025–2026 motion-space ablation of a keypoint-space sync expert used as a loss, with lip-error numbers, beyond what the prior report holds.
- No measured effect of mouth-specific acceleration terms was found.

## Q4. Deterministic or less stochastic mouth

### Takeaway
This is the strongest and most SANG-relevant evidence of the round:
- (a) **SubtleTalk**: deterministic prior alone LVE 11.35; flow matching alone 14.60; flow matching + prior 12.46. Its full model reaches 11.96 while upper-face and head diversity metrics improve (FDD 12.47 → 4.46 vs FM only).
- (b) **Yang et al.**: one probabilistic sample 10.8 vs deterministic 7.9, and the mean of the samples 7.9.
- (c) **GoHD**: a deterministic, lip-distilled stage is worth 7.8% MLD.

Per-sample lip accuracy falls as mouth stochasticity rises. A deterministic mouth prior plus a residual flow recovers most of the accuracy and keeps diversity.

### Cited Findings
- [M] (motion: FLAME) SubtleTalk Table 5 (LVE ×1e-4 / FDD ×1e-3 / HDD):

  | Variant | LVE | FDD | HDD |
  |---|---|---|---|
  | DMP only | 11.35 | 15.25 | N/A |
  | FM only | 14.60 | 12.47 | 27.34 |
  | FM+DMP | 12.46 | 11.44 | 25.57 |
  | FM+MultiCond | 15.05 | 8.02 | 12.58 |
  | FM+DMP+MultiCond | 12.59 | 5.21 | 6.96 |
  | Full | 11.96 | 4.46 | 6.61 |

  The authors: "DMP mainly reduces LVE, while its effect on FDD/HDD is limited, suggesting that it stabilizes strongly audio-correlated mouth motion." Design details:
  - The DMP predicts only expression + jaw from frozen WavLM-base, shape and past-motion context, using 4 transformer-decoder blocks (d=256).
  - The flow models the normalised residual M − M_prior.
  — [SubtleTalk](https://arxiv.org/html/2608.06408)
- [M] (motion: DECA mesh) Yang et al. Table 6:
  - FaceFormer+Style (deterministic): ℓ_vertex 7.9 / ℓ_cover 7.9 / ℓ_mean 7.9
  - probabilistic model: 10.8 / 6.0 / 7.9
  - "+Avg100": 8.3 / 7.1 / 8.2
  - "Sync" column: 0.369 / 0.463 / 0.684
  - ℓ_mean is the lip error of the mean of the sample set; ℓ_cover is the minimum over samples. "Deterministic methods ... achieve lower ℓ_vertex ... However, this does not take into account the diversity of probabilistic methods."
  — [Yang et al.](https://arxiv.org/html/2311.18168)
- [M] (motion: 3DMM) GoHD: removing the deterministic Stage 1 worsens MLD 1.792 → 1.931. — [GoHD](https://arxiv.org/html/2412.09296)
- [M] (motion: FLAME) ARTalk: removing the speaker style embedding worsens LVE 9.34 → 11.80. Much of "lip error" is speaker-specific articulation style. — [ARTalk](https://arxiv.org/html/2502.20323)

### Inferences
- [I] **Decision rule for the pending two-seed diagnostic.** Model GT as y = μ(a) + η and a sample as s = μ̂(a) + ε, with ε independent of η.
  - If the sampler is calibrated (μ̂ ≈ μ, Var ε ≈ Var η), then corr(s, y) ≈ r_ss, the seed-to-seed correlation. The conditional mean would reach corr(μ, y) = √r_ss. With r_ss ≈ 0.64, the ceiling is about 0.80 in the same measurement space.
  - **Case 1, r_ss ≈ corr(sample, GT) ≈ 0.63:** SANG is a calibrated sampler and the 0.63 is aleatoric. Only reducing mouth randomness (averaging, lower mouth temperature, a deterministic prior) can raise per-sample correlation, and only at some cost in amplitude and diversity.
  - **Case 2, corr(sample, GT) > r_ss:** the model is over-dispersed (more mouth noise than reality). Lower temperature or averaging helps strongly.
  - **Case 3, r_ss ≫ 0.63 (for example ≥ 0.85):** samples agree with each other but not with GT, so the problem is bias. Prioritise audio front end, context and supervision (Q1–Q3).
- [I] If 0.63 is measured on rendered video and the renderer ceiling (0.82–0.85) attenuates roughly multiplicatively, the motion-space correlation is about 0.63/0.835 ≈ 0.75. Compute r_ss in the same space as corr(sample, GT).
- [I] The zero-training equivalent of Yang's ℓ_mean is to average the 18 mouth coordinates over K seeds (K = 2, 4, 8, 16) and plot correlation, CCC and amplitude ratio against K. The curve shows directly how much a deterministic mouth prior could gain. Expect amplitude to shrink as K grows; mouth guidance at 1.25–1.5 or a std rescale can restore it.
- [I] A cheaper intermediate is to scale the initial noise only on the mouth coordinates (τ = 0.5–0.8) under rectified flow. Nothing in the literature measures this for motion models.

### Gaps
- No paper reports lip *correlation* (as opposed to LVE) against sampling temperature or K-sample averaging in a motion flow model.
- The meaning of Yang's "Avg100" and the direction of its "Sync" score were not verified beyond the table text.

## Q5. Reward fine-tuning or DPO in motion space

### Takeaway
The one new motion-space reward fine-tuning result with numbers (FMReward/FMFL, 2026) more than doubles a human-preference reward but barely moves lip error (−0.5%). Holistic preference rewards are naturalness tools, not lip-timing tools. A lip-specific reward with an amplitude guard is still unmeasured in motion space.

### Cited Findings
- [M] (motion: FLAME) FMReward is trained on FMPair: 65,574 human-annotated pairs from 8,834 audio clips; wav2vec2 + motion transformer with cross-attention; Bradley–Terry loss. FMFL alternates the diffusion loss with a reward step; the gradient flows only at the final denoising step, with a KL regulariser. After 240 iterations:
  - LVE 9.7750 → 9.7264 mm
  - FDD 0.5352 → 0.5655
  - diversity 26.088 → 25.412
  - reward 1.0453 → 2.4115
  - No DPO comparison.
  — [FMReward](https://arxiv.org/html/2608.15296)
- [M] (from the prior report) Avatar Forcing's DPO with synthetic losers *raised* motion variance (1.408 → 1.734). FantasyTalking2 reports that SyncNet-confidence rewards favour exaggerated lips (agreement with humans 72.34%). — [prior report](/Users/archismanchakraborti/Desktop/SANG/reports/SANG%20boundary%20and%20lip%20sync%20fixes.md)

### Inferences
- [I] For SANG, reward fine-tuning belongs after the structural fixes. If tried, the reward should be lip-aperture CCC to GT on paired training clips, or a motion-space sync expert gated by an amplitude-ratio penalty. DPO pairs should be GT or best-CCC sample (winner) against the over-guided or least-correlated sample (loser).

### Gaps
- No 2025–2026 motion-space DPO or RL with a sync reward plus an amplitude guard, with lip numbers, was found.

## Q6. 2025–2026 motion-space systems: lip numbers and drivers

### Takeaway
None of the recent LivePortrait or latent-motion systems isolates what drives its lip sync. Their shared choices are a wav2vec2 or HuBERT front end, a short direct audio view (±2 frames), per-frame audio injection, and per-region CFG (AVTR-1). The best-documented drivers come from 3D-mesh work: a deterministic mouth prior (SubtleTalk), context or window length and a speaker-style embedding (ARTalk), and the encoder (UniTalker).

### Cited Findings
- [M] AVTR-1 (SANG's 42-d space): HuBERT-Large averaged to 25 fps; audio by cross-attention; four region latents for per-region CFG. Dyadic Table 5a: LSE-D 7.08 / LSE-C 3.28 vs AvatarForcing* 7.32 / 2.41 vs DyStream 6.57 / 3.21. No lip-driver ablation. — [AVTR-1](https://arxiv.org/html/2609.22913)
- [M] Motar (2609.10317): MEAD / Hallo3 gap to GT, ΔSync-C 0.08 / 0.06 and ΔSync-D 0.42 / 0.37. Re-rendering GT motion (oracle) gives ΔSync-C 0.11 / 0.45 and ΔSync-D 0.28 / 0.53. Its recipe ablation (MSE, adversarial, DMD) reports no sync columns. — [Motar](https://arxiv.org/html/2609.10317)
- [M] FLOAT: frame-wise AdaLN beats cross-attention by 0.467 LSE-D on HDTF. — [FLOAT](https://arxiv.org/html/2412.01064)
- [M] SubtleTalk (2026, FLAME) beats ARTalk, DiffPoseTalk, FaceFormer and FaceDiffuser on LVE (11.96 vs 11.98 / 13.72 / 13.87 / 14.30), with FDD 4.46 vs 9.74–15.39. — [SubtleTalk](https://arxiv.org/html/2608.06408)
- [M] (from the prior report) HDTF LSE-C / LSE-D against same-table real video: KDTalker 7.326 / 7.548 vs real 8.243 / 6.929; Teller 7.696 / 7.536 vs real 8.094 / 6.976. — [KDTalker](https://arxiv.org/abs/2503.12963), [Teller](https://arxiv.org/abs/2503.18429) (via the prior report)
- [C] JAM-Flow (2506.23552) couples a Motion-DiT and an Audio-DiT in an MM-DiT for joint audio–motion flow matching. Lip numbers were not fetched. — [JAM-Flow](https://arxiv.org/abs/2506.23552)

### Inferences
- [I] SANG's architecture already matches the field's choices (frame-wise AdaLN, a short audio view, per-region guidance). The levers no 2025–2026 LivePortrait-space system has tested, and where mesh evidence is strongest, are: a deterministic mouth prior with residual flow; a wider per-frame audio view; and a multilingual or layer-mixed encoder.

### Gaps
- Lip-driver ablations for IMTalker, KDTalker, Ditto and JAM-Flow were not re-fetched; the prior report holds their main numbers.
- No 2025–2026 motion-space paper reports lip-trajectory correlation or CCC, so SANG's 0.63 cannot be compared with any published system.

## Ranked plan for SANG-M (one GPU; a retrain is about 20 min of steps)

All arms are judged on the same panel: lip-aperture corr and CCC, amplitude ratio, /p b m/ closure, audio_gain, two-seed r_ss, plus LSE-C/D on HDTF. An arm is kept only if corr or CCC rises without amplitude above 1.3× or audio_gain below 1.5.

| Rank | Arm | Cost | Evidence (space) | Expected |
|---|---|---|---|---|
| 0a | Two-seed r_ss (pending), plus **K-seed averaging of the 18 mouth coordinates** (K = 2, 4, 8, 16) with and without std rescale; apply the Q4 decision rule | Inference only, hours | Yang et al. ℓ_mean 7.9 vs single sample 10.8 (mesh) [M] | Measures the headroom of rank 1. If corr rises toward √r_ss, stochasticity is the limit |
| 0b | **Lip corr/CCC by language** (English vs the other 14) | Minutes, existing eval | WavLM-Large card "English only"; UniTalker D6 under-represented language (mesh) | If English clearly leads, rank 3 moves up |
| 0c | Mouth-only noise temperature τ ∈ {0.5, 0.7} combined with mouth guidance 1.25–1.5 | Inference only | None measured [I] | Cheap partial version of 0a in a single sample |
| 1 | **Deterministic mouth prior + residual flow** (SubtleTalk recipe), described below | 2 × ~20 min; no re-cache | SubtleTalk LVE FM-only 14.60 → FM+DMP 12.46 (−15%), DMP-only 11.35; GoHD MLD 1.931 → 1.792 (mesh/3DMM) [M] | Largest expected gain if 0a shows a stochastic limit. Risk: amplitude shrink (check closure and amplitude) |
| 2 | **Wider per-frame audio view inside AdaLN**: replace the k=5 conv with a dilated conv stack or a 2-layer local transformer, ±8–12 frames; keep AdaLN, not cross-attention | 1 retrain; no re-cache | ARTalk window LVE 11.73 → 9.34 (mesh); CALS ±13-frame optimum (pixel); A2F-3D 0.52–1 s [C]; FLOAT AdaLN > cross-attention (LSE-D 7.290 vs 7.757) | Small to moderate; most useful if 0a shows bias (Case 3) |
| 3 | **Encoder**, described below | Re-cache 12.6k clips per arm (hours) + 1 retrain each | UniTalker encoder swing 10–20% LVE, XLSR-53 best (mesh); MultiTalk multilingual > English encoder; Teller Whisper result (prior report) | Moderate; larger for non-English clips |
| 4 | **Lip-aperture CCC auxiliary loss** on x̂₁, weighted to low noise (t ≥ 0.7); optionally a motion-space sync expert as an evaluator | 1 retrain | Untested [I]; THUNDER PCC 0.568 → 0.639 is the closest analogue (prior report) | Targets the metric directly; CCC limits amplitude inflation |
| 5 | Reward or DPO fine-tune with a lip-CCC reward plus an amplitude penalty; pairs of GT or best-CCC sample against over-guided samples | Hours | FMReward: holistic reward gives LVE −0.5% only (mesh) | Low for correlation; polish only |

**Rank 1 in detail.** Train a small regressor, audio → 18 mouth coordinates or all 39 expression coordinates, with MSE + velocity loss; SubtleTalk uses 4 transformer blocks at d=256. Then either:
- (a) feed its output as an extra AdaLN condition, or
- (b) reparameterise the DiT's mouth target as the residual from the prior.

**Rank 3 in detail.** Arms: (a) a learned softmax layer mix over cached WavLM-Large layers (for example every 3rd layer); (b) the Whisper-large-v3 encoder; (c) XLS-R.

**Not recommended now:**
- Replacing AdaLN with long-context cross-attention (FLOAT measured it worse).
- InfoNCE motion-encoder alignment (DEMO has no isolated evidence).
- Phoneme forced-alignment pipelines across 15 languages (PD-GS gain: LMD −2.6%).
- Holistic preference RL (FMReward: −0.5% LVE).

**Data scale** (SANG uses about 5% of TalkVid) was outside this scope. UniTalker attributes gains to "scale and diversity" of data, but no source here isolates hours against lip correlation in a motion model.


---
## RAW FINDINGS LOG (appended during research; consolidated into sections at the end)

### SubtleTalk (arXiv 2608.06408, v1 2026-08-03; Ding, Tan, Lin, Jiang, Zeng, Pan) — (motion: FLAME mesh)
- [M] Two-stage: Deterministic Motion Prior (DMP) = 4 transformer-decoder blocks, 8-head cross-attn, d=256, frozen WavLM-base content feature + shape β + previous-motion context; outputs only expression ψ + jaw θ_jaw; loss = vertex loss + velocity loss in FLAME vertex space with separate face/lip weighting. Stage 2 Residual Flow Matching models M_res = M − M_prior (normalized residual space), conditioning via cross-attn (audio, affect) + AdaLN; adds velocity, smoothness, std losses. — [SubtleTalk](https://arxiv.org/html/2608.06408)
- [M] Table 5 ablation (LVE ×1e-4 ↓ / FDD ×1e-3 ↓ / HDD ↓): DMP only 11.35 / 15.25 / N/A; FM only 14.60 / 12.47 / 27.34; FM+DMP 12.46 / 11.44 / 25.57; FM+MultiCond 15.05 / 8.02 / 12.58; FM+DMP+MultiCond 12.59 / 5.21 / 6.96; Full 11.96 / 4.46 / 6.61. — [SubtleTalk](https://arxiv.org/html/2608.06408)
- [M] Authors: "DMP mainly reduces LVE, while its effect on FDD/HDD is limited, suggesting that it stabilizes strongly audio-correlated mouth motion." Pure deterministic regression gives the BEST lip error (11.35) — pure FM is 29% worse (14.60); adding the prior recovers most of it (12.46) while keeping/improving diversity-type metrics. Richer conditioning (MultiCond) without the prior worsens LVE (15.05). — [SubtleTalk](https://arxiv.org/html/2608.06408)
- [M] Table 3: Full 11.96 LVE vs DiffPoseTalk 13.72, ARTalk 11.98, FaceFormer 13.87, FaceDiffuser 14.30. — [SubtleTalk](https://arxiv.org/html/2608.06408)

### DEMO (arXiv 2510.10650, 2025-10-12; Chen, Yang, Feng, Jiang, Yan) — (motion latent → pixel eval)
- [M/C] 512-d motion latent with separate lip / pose (3 Euler + 3 translation) / eye encoders; bidirectional InfoNCE between per-frame audio features f^a and lip-motion features f^v used while training the motion encoder (Eqs. 4–5). OT flow matching + transformer predictor. — [DEMO](https://arxiv.org/html/2510.10650)
- [M] No ablation isolates the contrastive loss. Disentangled encoder ("FCME+Flow") vs plain VAE+Flow on HDTF: FID 94.050 vs 121.311; reported "LSE-D" 238.577 vs 243.227 (values on an unusual scale; treat as unreliable / possibly a different metric). → The InfoNCE term has NO isolated measured lip gain. — [DEMO](https://arxiv.org/html/2510.10650)

### UniTalker (arXiv 2408.00762, ECCV 2024; Fan et al.) — (motion: 3D mesh / blendshape, 8 datasets "A2F-Bench", incl. Chinese speech + multilingual song)
- [M] Table 6, audio-encoder ablation (LVE ↓; 1e-4 for D0=BIWI, 1e-6 m² D1–D3, 1e-5 m² D4–D7), D0..D7:
  - Wav2Vec2-Base-960h (960 h English): 4.491 9.916 9.887 9.812 1.585 2.217 1.351 1.409
  - WavLM-Base (same 960 h, different objective): 4.033 8.269 9.253 9.117 1.417 2.044 1.184 1.340
  - WavLM-Base-Plus (94k h, 23 languages): 4.080 8.136 9.776 9.053 1.392 1.975 1.158 1.264
  - Wav2Vec2-XLSR-53 (56k h, 53 languages, larger): 3.859 8.303 8.648 8.991 1.326 2.056 1.145 1.211
  - Authors: precision "largely affected by the pre-trained audio encoder from three aspects, including the pre-training method, the scale and diversity of pre-training dataset and the capacity of pre-training backbone." XLSR-53 best on 5/8 test sets. Encoders frozen in decoder warm-up, then fine-tuned jointly. — [UniTalker](https://arxiv.org/abs/2408.00762) (text extracted from arXiv PDF v1)
  - [M] Encoder swap alone moves LVE by ~10–20% (e.g., D0 4.491→3.859 = −14%; D1 9.916→8.136 = −18%). Multilingual pretraining + capacity matter. Note: UniTalker reports Chinese-speech set D6 had higher LVE "because the proportion of Chinese speeches in A2F-Bench is small." — [UniTalker](https://arxiv.org/abs/2408.00762)
  - [I] No Large (WavLM-Large) or Whisper arm; SANG already uses WavLM-Large (94k h, 23 langs as base-plus but large). Gains of base→large/XLSR are within-family; not directly a reason to swap away from WavLM-Large.

### Wav2Sem (arXiv 2505.23290, CVPR 2025; Li, Dai, Zhao, Zhou, Pan, Li) — (motion: 3D mesh)
- [M] Claim: SSL audio features couple near-homophone syllables → "averaging effect" in lip shapes. Plug-in semantic feature module (Wav2Sem_m / Wav2Sem_c variants) fused into Wav2Vec2 / HuBERT features. — [Wav2Sem](https://arxiv.org/abs/2505.23290)
- [M] Table 1 LVE (VOCASET ×1e-5; BIWI ×1e-4) baseline → +Wav2Sem_c: FaceFormer (wav2vec2) 4.1090→3.9891 / 4.9847→4.9571; CodeTalker 3.9445→3.8838 / 4.7914→4.7751; UniTalker 3.5416→3.1476 / 4.0213→3.9112; FaceDiffuser (HuBERT) 3.7924→3.7739 / 4.2985→4.2521; LG-LDM (HuBERT) 3.7925→3.7863 / 4.9869→4.9258. Gains mostly 0.2–3%, one 11% (UniTalker VOCASET). Phoneme recognition PER (TIMIT test) wav2vec2 8.3→8.0, HuBERT 8.0→7.8. — [Wav2Sem PDF](https://arxiv.org/pdf/2505.23290)
- [I] Adding utterance-level semantic context to frame features gives small but consistent lip gains; it is a "longer context" signal. Small effect size — low priority for SANG vs. stochasticity fixes.

### ARTalk (arXiv 2502.20323; speech-driven 3D head, FLAME, autoregressive multi-scale VQ) — (motion: mesh, TFHP dataset)
- [M] Main: LVE 9.34 vs FaceFormer 12.72, CodeTalker 11.78, DiffPoseTalk 10.39. — [ARTalk](https://arxiv.org/html/2502.20323)
- [M] Window-length ablation (the AR clip/context window over which audio+motion are modelled): 8 frames LVE 11.73; 25 → 10.20; 50 → 9.78; 100 → 9.34. A 12× longer window cuts LVE by 20%. — [ARTalk](https://arxiv.org/html/2502.20323)
- [M] Encoder: HuBERT (100-frame window) LVE 9.34 / FFD 18.15 / MOD 1.81 vs Mimi codec 9.97 / 18.48 / 1.93; authors: "Mimi outperforms HuBERT when the input length is short, whereas HuBERT performs better on longer audio segments." Removing multi-scale AR → LVE 14.14; removing style embedding → 11.80. — [ARTalk](https://arxiv.org/html/2502.20323)
- [I] Style embedding (speaker identity of motion) is worth 2.46 LVE (11.80 vs 9.34) — i.e. a large share of "lip error" is speaker-specific articulation style. SANG's reference/prefix conditioning partly provides this; worth checking that the 10-frame prefix actually carries speaking style.

### Phonetic context-aware lip-sync, CALS (arXiv 2305.19556 v3, 2024-04-01; ICASSP 2024) — (pixel/video generator, 128×128; evaluated with landmark LMD)
- [M] Input-audio-window sweep: optimum at ±15 frames on LRW (LMD 1.162; the full window available) and ≈ ±13 frames on LRS2 (LMD 1.059). Authors: "the audio context of around 1.2 seconds assists in resolving ambiguities in the lip shapes of phones, improving spatio-temporal alignment." Coarticulation framed explicitly ("change in articulation of the current speech segment due to the neighboring speech"). — [CALS](https://arxiv.org/html/2305.19556)
- [I] Evidence for a ~±0.5 s direct audio view per frame is pixel-domain and old-ish, but consistent with ARTalk's window ablation (motion) and with SANG's ±80 ms audio conv being much narrower than phonetic coarticulation spans (anticipatory lip rounding spans several hundred ms; no source fetched for the phonetics claim → see Gaps).

### FLOAT (arXiv 2412.01064) — (motion latent, pixel eval on HDTF)
- [M] Table 3 (HDTF): frame-wise AdaLN vs cross-attention for audio: LSE-D 7.290 vs 7.757; FID 21.100 vs 21.873; FVD 162.052 vs 162.702; E-FID 1.229 vs 1.452. Authors: "frame-wise AdaLN provides better expression generation and lip synchronization." — [FLOAT](https://arxiv.org/html/2412.01064)
- [M] FLOAT's transformer self-attention is MASKED to a local window, "attention window length T = 2", attending to 2·T neighbouring frames; wav2vec2 audio; L=50 frames + L′=10 preceding frames. No ablation on window size or encoder. — [FLOAT](https://arxiv.org/html/2412.01064)
- [I] Cross-attention over a long audio context is NOT automatically better in motion DiTs: the one measured head-to-head (FLOAT) favours frame-wise AdaLN by 0.47 LSE-D. If SANG widens audio context, it should do so INSIDE the per-frame AdaLN feature (wider conv / local audio transformer before AdaLN), not by replacing AdaLN with cross-attention.

### GoHD (arXiv 2412.09296, AAAI 2025) — (motion: 64-d 3DMM expression; evaluated on rendered HDTF frames)
- [M] Stage 1: audio(+GT eye features+initial expression) → MLP regression of full expression, distilled from Wav2Lip's "resynchronized results" (a pixel lip expert used as a teacher for a deterministic regressor); Stage 2: LSTM generates residual eye motion; audio encoder frozen in Stage 2. — [GoHD](https://arxiv.org/html/2412.09296)
- [M] Table 3 (HDTF) mouth landmark distance MLD ↓: Full-LSTM 1.792; Full-transformer 1.806; w/o Stage 2 & eye features 1.785; w/o Stage 1 1.931; w/o Distillation 2.012. Removing the deterministic, lip-expert-distilled stage worsens MLD by 7.8%; removing distillation by 12.3%. — [GoHD](https://arxiv.org/html/2412.09296)

### FMReward / FMFL (arXiv 2608.15296, 2026-08-15; Wu, Li, Gao, Duan, Zhu, Zhai, Le Callet) — (motion: FLAME 3D)
- [M] Reward model: wav2vec2 audio + transformer motion encoder fused by cross-attention → MLP score, Bradley–Terry on FMPair (65,574 human-annotated motion pairs from 8,834 audio clips; candidates from DiffPoseTalk, MM2Face, UniTalker, ProbTalk3D, GT). FMFL: alternate standard diffusion loss and reward step; denoise from a random late timestep without grad, gradient only at the final step; KL regulariser against drift. — [FMReward](https://arxiv.org/html/2608.15296)
- [M] On their DiffPoseTalk-style "TDM", 240 FMFL iterations: LVE 9.7750 → 9.7264 mm (−0.5%); FDD 0.5352 → 0.5655; BA 0.2275 → 0.2322; diversity 26.088 → 25.412; reward 1.0453 → 2.4115. No DPO comparison. — [FMReward](https://arxiv.org/html/2608.15296)
- [I] A holistic human-preference reward in motion space more than doubles the reward but barely moves lip error. Reward FT is a perceptual-naturalness tool, not a lip-timing tool, unless the reward is lip-specific (e.g. a motion-space sync expert or correlation-to-GT on paired data).

### Audio2Face-3D (NVIDIA, arXiv 2508.16401) — (motion: mesh/PCA; production system)
- [C] Regression net (v2.3): hybrid autocorrelation + wav2vec2 encoder, "0.52 seconds" audio context → 1-frame pose (PCA, ~140-d). Diffusion net (v3.0): HuBERT, "1-second audio chunk" → 30-frame raw-vertex animation. Losses listed include phoneme loss, motion loss and a lip-distance loss — but NO ablation or regression-vs-diffusion lip numbers; only the claim that diffusion "generally produces higher quality and more expressive facial animations." — [Audio2Face-3D](https://arxiv.org/html/2508.16401)
- [I] A production system deliberately gives each output frame ≥0.5 s of audio (vs SANG's ±80 ms per-frame conv).

### AVTR-1 (arXiv 2609.22913, "AVTR-1: Open Stack for Real-Time Interactive Avatars") — (motion: SANG's 42-d LivePortrait space)
- [M] Audio: 24-layer HuBERT-Large (~300M), 1024-d at 50 Hz, averaged to 25 fps; self/other audio channels on separate paths; audio via cross-attention; four linear projections give separate latents for head rotation / brow / eyes / mouth so CFG is applied per region. Table 5a (dyadic set): AVTR-1 LSE-D 7.08 / LSE-C 3.28 vs AvatarForcing* 7.32 / 2.41 vs DyStream 6.57 / 3.21. No ablation isolates lip-sync drivers. — [AVTR-1](https://arxiv.org/html/2609.22913)

### Decoupled Self-Forcing Distillation ("Motar", arXiv 2609.10317) — (motion: 512-d X-NeMo latent; pixel eval)
- [M] wav2vec2-base audio; a local branch uses "windowed cross-attention" with a 2-frame window per side for frame-resolution lip articulation; adversarial motion loss + unconditional DMD through a frozen renderer. Table 1 (MEAD / Hallo3): ΔSync-C 0.08 / 0.06 and ΔSync-D 0.42 / 0.37 vs GT (GT Sync-C 1.68 / 4.93; Sync-D 12.22 / 8.90); oracle GT-motion re-render ΔSync-C 0.11 / 0.45, ΔSync-D 0.28 / 0.53. Table 2 recipe ablation reports MSE / CosSim / FMD only — no sync per loss. — [Motar](https://arxiv.org/html/2609.10317)
- [I] Motar, FLOAT and AVTR-1 all keep a SHORT direct audio view per frame (±2 frames) and rely on sequence attention for context; none ablates a wider per-frame audio window. The only motion-space window evidence is ARTalk's clip-length sweep.

### JAM-Flow (arXiv 2506.23552, 2025-06-30; Kwon, Shin, Jung, Park, Uh) — ID verified
- [C] MM-DiT coupling a Motion-DiT and an Audio-DiT; joint audio+motion flow matching; text/audio/motion conditioning. (Lip numbers not fetched.) — [JAM-Flow](https://arxiv.org/abs/2506.23552)

### MultiTalk (arXiv 2406.14272 v1, 2024-06-20; Sung-Bin, Lee, Son, Oh, Ju, Nam, Oh) — (motion: 3D mesh, multilingual)
- [M] Uses a multilingual speech encoder "pretrained on 53 languages" plus a language-style embedding; Table 5 ablation reports the multilingual encoder outperforming an English-only one across languages (exact Table 5 numbers not extracted). LVE (×1e-4 mm) MultiTalk vs SelfTalk vs CodeTalker: English 1.16 / 1.99 / 1.98; Italian 1.06 / 2.59 / 2.56; French 1.39 / 1.98 / 1.99; Greek 1.26 / 2.11 / 2.09 (confounded by training data). Lip-readability AVLR WER at SNR −7.5: MultiTalk 42.4 / 50.5 / 63.0 / 74.2 vs SelfTalk 42.8 / 56.5 / 68.3 / 80.3. — [MultiTalk](https://arxiv.org/html/2406.14272)

### PD-GS (arXiv 2608.05218 v1, 2026-08-05; Fu, Zhou) — (person-specific 3DGS; lip deformation predicted before rendering; HDTF)
- [M] Linguistic Fusion Module: Whisper ASR + Montreal Forced Aligner → 40-phoneme frame tokens; learned channel-wise gate blends HuBERT audio context with phoneme embeddings. Ablation (HDTF) LMD↓ / Sync↑: audio only 2.73 / 8.71; concat fusion 2.70 / 8.78; gated 2.66 / 8.85. Small gain (LMD −2.6%). — [PD-GS](https://arxiv.org/html/2608.05218)

### Yang et al., "Probabilistic Speech-Driven 3D Facial Motion Synthesis: New Benchmarks, Methods, and Applications" (arXiv 2311.18168, 2023-11-30; Apple) — (motion: DECA meshes)
- [M] Metrics: coverage error ℓ_cover = min over a sample set of lip vertex error; mean-estimate error ℓ_mean = lip vertex error of the MEAN of the sample set; plus standard ℓ_vertex. Table 6: FaceFormer+Style (deterministic) Sync 0.369, ℓ_vertex 7.9, ℓ_cover 7.9, ℓ_mean 7.9; Ours (probabilistic) 0.463, 10.8, 6.0, 7.9; Ours+Avg100 0.684, 8.3, 7.1, 8.2. Authors: "Deterministic methods (VOCA, Faceformer, Faceformer+Style) achieve lower ℓ_vertex ... However, this does not take into account the diversity of probabilistic methods." — [Yang et al.](https://arxiv.org/html/2311.18168)
- [I] Key quantitative fact for SANG: a single sample from a good probabilistic model had 37% higher lip error than a deterministic model (10.8 vs 7.9), while the MEAN of its samples matched the deterministic model exactly (7.9). This is the signature of aleatoric spread, not bias. (Exact meaning of "Avg100" and the direction of their "Sync" score not verified beyond the table.)
