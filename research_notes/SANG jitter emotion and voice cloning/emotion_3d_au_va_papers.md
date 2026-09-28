# Emotion control: 3D / AU / valence–arousal papers (helper-agent notes)

Saved verbatim by the coordinator from a helper agent's hand-back (the helper was started by the first emotion researcher, which stalled). Each paper was read from its arXiv HTML full text. Tags: **[M] MEASURED** = a number copied from a table (table number given); **[C] CLAIMED** = paper prose; **[I] INFERENCE** = the helper's own reasoning. Where something was not found, it says so.

---

## 1. Cafe-Talk — arXiv 2503.14517, ICLR 2025
Source: https://arxiv.org/html/2503.14517 (proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/file/2c71b14637802ed08eaa3cf50342b2b9-Paper-Conference.pdf)

- **Output / generator:** 51-D ARKit blendshapes at 25 fps; 8-block DiT (512-d, 8 heads) that predicts x0. [C]
- **Coarse condition (global):** talking style (CLIP embedding of a face image or appearance attributes) + emotion (CLIP *text* embedding of the label with synonym augmentation → open vocabulary) + intensity (learnable embedding × numeric intensity factor, "to learn the ordinal nature"). Concatenated → linear → injected by **AdaLN** in a self-attention block. [C]
- **Audio:** XLS-R wav2vec2 (315M, frozen), FiLM cross-attention with a near-diagonal alignment mask. [C]
- **Fine-grained (per frame):** (binary AU set, start frame, end frame) triplets → **adapter** (FiLM self-attention + **zero-initialised conv**), trained only in stage 2 with the base frozen. Base 25.09M params, adapter 5.39M (Tab. 5). [C/M]
- **Labels:** emotion/intensity from MEAD and RAVDESS. Training AUs are **not** from an AU detector: rule-binarised from blendshape coefficients (threshold → random-kernel max-pool → random merging of close segments; Alg. 1, App. C.2); coefficients from an in-house video mocap model. OpenFace rejected for a *"semantical mismatch between the AU detector prediction and the ground-truth facial movements"* (Fig. 11). [C]
- **Stage 2 uses 157 h of internet video with no emotion labels:** *"Considering the unreliability of the concurrent video-based emotion recognition methods, we randomly arrange the emotion label and intensity as input"* (App. C.3). [C] — closest analogue to training on TalkVid.
- **Text-to-AU detector:** CLIP ViT-B/32 (frozen) + CLIP-Adapter (512→128→512) + BCE head, InfoNCE; trained on 228 GPT-generated text–AU pairs + AffectNet emotion–AU pairs (AUs from OpenFace); F1 0.92 / 0.98 with / without InfoNCE on in-domain templates (App. E). [M]
- **Mask-based CFG (Eq. 6):** M̂₀ = G_Φ + α(G_{A,C} − G_Φ) + β·Z_cfg ⊙ (G_{A,C,F} − G_Φ); Z_cfg = temporal mask of frames with an AU condition; β = AU intensity knob. Training dropout: audio and coarse conditions each masked independently with p = 20%. [C]
  - [I] The F term is relative to the unconditional G_Φ, so inside masked frames audio+coarse is effectively weighted (α+β). The nested form (G_{A,C,F} − G_{A,C}) used by Playmate and FLOAT isolates the AU term.
- **Swap-label training:** replace the coarse label with C′, keep F and the ground truth; loss masked to the active-AU region, L_swap = ‖Z_ctrl ⊙ (M − M̂′)‖; randomly alternated with the plain loss; 80% of AU triplets dropped, individual AUs randomly dropped. [C]
- **Tab. 3 fine-grained ablation** (CR = AU control rate; in-the-wild audio, fixed seed) [M]:

  | Variant | CR | SyncD | SyncC |
  |---|---|---|---|
  | w/o two-stage | 0.35 | 11.94 | 2.78 |
  | w/o swap-label | 0.12 | 11.02 | 4.07 |
  | w/o masked CFG | 0.27 | 10.66 | 4.74 |
  | **Full** | **0.51** | **10.14** | **5.38** |

- **Lip-sync cost of the AU adapter:** base model on LibriSpeech SyncD 9.76 / SyncC 5.55 (Tab. 1); authors say SyncNet "degrades when fine-grained control is incorporated" (Tab. 3 and Tab. 1 use different test sets — indirect). Tab. 4 (single AU, fixed audio): Null SyncD 9.667; upper-face AUs 9.613–10.365; lower-face AUs 10.066–10.788; caption "+0.156 on average" (upper) vs "+0.693" (lower). [I] Recomputing from the scraped cells gives +0.199 / +0.694 — a scrape column shift or a caption error. Authors' limitation: lower-face AUs that conflict with speech still break lips. [C]
- **Tab. 1 vs SOTA** (MEAD LVE / Acc / Div; RAVDESS LVE; LibriSpeech SyncD / SyncC) [M]: Ours 7.21 / **59.48%** / 119.36; 3.93; 9.76 / 5.55. EmoTalk 7.90 / 9.97% / 62.02; 4.31; 11.09 / 3.30. UniTalker 10.14 / 12.44% / 12.64; 4.93; 9.70 / 4.72. Acc from their own transformer classifier on ARKit motion (88.28% on a MEAD split, App. D.2).
- **Tab. 2 coarse-condition injection:** AdaLN LVE 7.21 / Acc 59.48% / Div 119.36 vs FiLM 7.64 / 48.19% / 114.26. [M]
- **Tab. 8 label vs audio emotion:** swapped label → matches the label 57.22%, the audio's emotion 10.86%; label masked 23.68%; paired 59.68% (disagrees with Tab. 1's 59.48%). The label dominates the audio's emotion. [M]
- **Tab. 9 intensity:** Acc I=1 51.68%, I=2 63.86%, I=3 66.26%; intensity = 0 → 19.07% (Div 67.09); no intensity input → 53.27%. [M]
- User study: 38 participants, figure only. [C]
- **Cost:** 8× V100; stage 1 400k it, stage 2 300k it, batch 16, ~4 days per stage; inference 14.71 s per 5 s of audio on a 2080Ti (Tab. 7). [M] Project page https://harryxd2018.github.io/cafe-talk/ — code/licence not verified.

---

## 2. SubtleTalk — arXiv 2608.06408, ACM MM '26 (DOI 10.1145/3767308.3834962)
Source: https://arxiv.org/html/2608.06408. Output: FLAME expression + pose.

- **Valence–arousal, per frame:** e_t = [v, a] ∈ [−1,1]², Fourier-encoded [e, sin(2^kπe), cos(2^kπe)] → f_E, injected by **cross-attention** with audio (local causal masks, RoPE). [C]
- **Regional intensity, per window:** 5 scalars (eye, brow, head X/Y/Z) = temporal std of the relevant FLAME vertices/pose inside the 100-frame target window, normalised by dataset statistics → global encoder → **AdaLN**. [C]
- **Prosody:** F0 + log-energy features; multi-scale WavLM layers 3–11. [C]
- **Labels:** frame-wise VA from **EmotiEffLib** (Savchenko, ICML 2023); 3D pseudo-GT from **TEASER** FLAME fits, Savitzky–Golay smoothed; SyncNet filter; clips dropped if yaw > 40° in > 10% of frames. [C]
- **VA label noise is not quantified anywhere** in the main text (targeted full-text query found nothing).
- **Inference:** a VA Dynamics Predictor gives dense VA from **emotion2vec** (frozen) + temporal convs; optional sparse user VA anchors via feature modulation, hard-overwritten at anchor frames (Eq. 17). No accuracy reported for this predictor. [C]
- [I] The paper doesn't say whether Tab. 3/5 use ground-truth-derived intensity/VA or predicted; if ground truth, FDD/HDD are optimistic (intensity is computed from the target window itself).
- **Dataset SubtleTalk-Face (Tab. 1):** 36,733 clips, **73.83 h**, 3,905 IDs — MEAD 34.59 h (45 IDs), HDTF 14.80 h, CelebV-HQ 7.75 h, VFHQ 7.51 h, TFHP 9.18 h; train/val/test IDs 2,456 / 724 / 725. [M]
- **Lip preservation:** stage 1 Deterministic Motion Prior (DMP; audio content only; expression + jaw with lip-weighted vertex/velocity losses); stage 2 residual flow matching on (M − M_prior); Implicit Disentangled Control (IDC): perturb intensity dimension k by α, control loss on k + stop-gradient invariance loss on the others. [C]
- **Tab. 5 ablation** (LVE ×1e-4, FDD/HDD ×1e-3) [M]:

  | Variant | FDD | HDD | LVE |
  |---|---|---|---|
  | DMP only | 15.25 | N/A | **11.35** |
  | FM only | 12.47 | 27.34 | 14.60 |
  | FM + DMP | 11.44 | 25.57 | 12.46 |
  | FM + MultiCond | 8.02 | 12.58 | **15.05** |
  | FM + MultiCond + IDC | 5.78 | 7.93 | 14.51 |
  | FM + DMP + MultiCond | 5.21 | 6.96 | 12.59 |
  | Full | 4.46 | 6.61 | 11.96 |

  Adding conditions **worsened LVE** (14.60 → 15.05); IDC and the prior recover it.
- **Tab. 3 baselines** (FDD / HDD / LVE): DiffPoseTalk 9.74 / 21.71 / 13.72; ARTalk 11.37 / 26.70 / 11.98. [M]
- **Tab. 4 user study** (27 participants, % preferring SubtleTalk): vs DiffPoseTalk face naturalness 94.81, head naturalness 74.44, lip-sync 69.63; vs ARTalk 86.30, 78.89, 51.48. [M]
- No CFG in the main text. Cost: 4× V100; DMP ~2 h; RFM 6 h warm-up + 2 h IDC. [C] Project page only; code not verified.

---

## 3. EmoZone-Talker — arXiv 2606.15848 (v2, 25 Sep 2026), ACM MM '26
Source: https://arxiv.org/html/2606.15848. Renderer: per-identity 3D Gaussian Splatting. Code: https://github.com/HMQH/Emozone-Talker.

- **Control:** 5-D continuous per-frame AU vector (0–5 scale): AU1, AU2, AU4, AU6, AU9 — chosen *"avoiding direct duplication of the speech-driven jaw and lip controls"*; AU6 kept as a cheek–mouth stress case. [C]
- **CIT-AE:** separate Conv1D per AU channel over a centred window → shared AU encoder; AU tokens concatenated with audio and camera tokens as K/V in cross-attention (queries = per-Gaussian tri-plane features). [C]
- **Labels:** target AUs from **OpenFace**; rendered-frame AUs from **JAA-Net** for an L1 AU-consistency loss; evaluation with OpenFace *"to reduce evaluator–optimizer coupling"*; MEAD emotion labels only to organise evaluation. [C]
- **Mouth conflict:** landmark zones (audio-dominant mouth, expression-dominant upper face, synergy zone); a learnable region-dependent attention bias B in softmax(QKᵀ/√d + B) (soft, masks neither modality); **RAAR** hinge losses on average attention mass (upper face: AU > audio; mouth: audio > AU); SyncNet loss λ = 0.1; λ_AU = 0.05, λ_RAAR = 0.01. [C]
- **Tab. 2 matched-AU ablation** (Sync↑ / AUE-U / AUE-L / AU-Jerk) [M]:

  | Config | Sync | AUE-U | AUE-L | AU-Jerk |
  |---|---|---|---|---|
  | (A) no AU | 5.741 | 0.309 | 0.314 | 0.137 |
  | (B) raw AU | **5.378** | 0.324 | 0.358 | 0.153 |
  | (C) + SZ-PAB | 5.569 | 0.152 | 0.235 | 0.126 |
  | (D) + CIT-AE | 5.571 | 0.305 | 0.276 | 0.098 |
  | Full | 5.712 | 0.143 | 0.231 | 0.095 |

  Naïve AU conditioning hurt both lip-sync and AU accuracy.
- **Tab. 3 representation:** binary AUs AUE-U 0.281 / AU-Jerk 0.101 / Sync 5.654 vs continuous 0.143 / 0.095 / 5.712. [M]
- **Tab. 1 MEAD self-reconstruction** (Sync / LSE-D / AUE-U / AUE-L): Ours 5.228 / 6.93 / 0.145 / 0.258; TalkingGaussian 4.794 / 7.54 / 0.163 / 0.314. [M]
- **Tab. 4 E-Score** (AffectNet classifier): Ours 0.653 vs EAT 0.552, DreamTalk 0.526, DICE-Talk 0.311 (authors caveat: different interfaces, AUs vs labels). [M]
- **Tab. 5 user study** (20 participants, 1–5): Emotion 3.89, Lip 3.67; best lip baseline 3.65. [M]
- Under injected noise, CIT-AE cuts jitter 61% vs 19% without (Fig. 6). [C]

---

## 4. MEDTalk — arXiv 2507.06071 (v4), ACM MM '25
Source: https://arxiv.org/html/2507.06071. Output: 174-D MetaHuman rig. Code: https://github.com/SJTU-Lucy/MEDTalk.

- **Disentanglement:** emotion and content encoders over **motion** (not audio), trained with cross-reconstruction: self-reconstruction, *overlap exchange* (same content or same emotion → swap the matching code), *cycle exchange* (two rounds); L2 reconstruction only. **Paired data is synthesised** by running pretrained **EmoFace** on the same speech with 7 emotion labels. GRL and vCLUB MI minimisation were insufficient. Encoders/decoder then frozen. [C]
- **Frame-wise intensity:** pseudo-label = L1 norm of hand-selected rig controllers per frame (Eq. 8); predictor = **emotion2vec** + **Whisper** transcript → RoBERTa, cross-attention fusion; applied by **rescaling the label embedding's norm**, f̃_t = Î_t · f/‖f‖, concatenated with the content code into the frozen decoder. [C]
- **Reference image / text:** frozen CLIP image and text encoders + projection heads into the label-embedding space; data = highest-intensity RAVDESS frames (by MediaPipe blendshapes), captioned by **Gemini 2.0 Flash Thinking**. [C]
- **Tab. 3 ablation** (MLE lip error / MEE emotion-region error / EIE intensity error / FRD upper-face std) [M]:

  | Variant | MLE | MEE | EIE | FRD |
  |---|---|---|---|---|
  | Full | 0.00596 | 0.00906 | 0.79055 | 0.00289 |
  | w/o disentangle (end-to-end) | 0.00628 | **0.00861** | 0.83728 | 0.00825 |
  | w/o overlap | 0.01026 | 0.01112 | 0.90941 | 0.00134 |
  | w/o intensity | 0.00665 | 0.01142 | 0.86488 | 0.00753 |
  | w/o text | 0.00630 | 0.00965 | 0.83899 | 0.00757 |

  Disentangling helps lip error slightly; the end-to-end variant has a better MEE.
- **Tab. 1:** MLE Ours 0.00596, EmoFace 0.00651, FaceFormer 0.00662. [M]
- **Tab. 2 user study** (42 participants, 95% CI): lip-sync 4.022 ± 0.137 vs EmoFace 3.886 ± 0.112; GT 4.421. [M]
- Cost: all four stages < 1 h on one RTX 3090. [C] Authors: lip generalisation (e.g. multilingual) limited by the frozen latent space. [C]

---

## 5. EmoVOCA — arXiv 2403.12886 (v3), WACV 2025
Source: https://arxiv.org/html/2403.12886 (CVF: https://openaccess.thecvf.com/content/WACV2025/html/Nocentini_EmoVOCA_Speech-Driven_Emotional_3D_Talking_Heads_WACV_2025_paper.html). Code: https://github.com/miccunifi/EmoVOCA.

- **"Neutral speech + expression offset" analogue:** per-frame displacements from each subject's neutral face — speech from **VOCAset**, expression from **Florence4D** (70 expressions, no speech). Double-encoder / shared-decoder (SpiralNet); each encoder's code duplicated and concatenated during training; at inference decode D(μ_t·f_t ⊕ μ_e·f_e), μ_e = intensity. Dataset: v1 5 emotions × 3 intensities (7,200 sequences); v2 11 emotions (15,840). [C]
- Generators trained on it: one-hot emotion + intensity, each linear → 64-D, **concatenated to audio features**. [C]
- **Tab. 3a naïve offset vs learned** (LVE / UVE) [M]: FaceFormer + Sᵉ 5.971 / 1.923 vs E-FaceFormer (v1) 3.425 / 0.927; S2L+S2D + Sᵉ 4.872 / 1.467 vs E-S2L+S2D (v1) 2.165 / 0.552. [I] Ground truth is DE-SD's own synthetic output → partly circular.
- **Tab. 2 code combination** (expression-classifier acc at intensity 3 / LVE) [M]: Sum 0.72 / 6.181; Mult 0.66 / 6.897; Zero-cat 0.69 / 6.324; DE-SD 0.82 / 5.861.
- **Tab. 3b VOCA-Test** (LVE / UVE) [M]: EMOTE 4.561 / 0.897; E-S2L+S2D (v2) 3.181 / 0.791; CodeTalker 3.651.
- **Tab. 5 user study vs EMOTE:** lip-sync preference 82.4% / 17.6%; emotion 55.5% / 45.5% as printed (sums to 101). [M]
- Fig. 3b: linear interpolation of displacements "introduces severe artifacts and leads to the loss of speech-related movements" (qualitative). [C]

---

## 6. EMOTE — arXiv 2306.08990 (v2), SIGGRAPH Asia 2023
Sources: https://arxiv.org/abs/2306.08990, https://arxiv.org/html/2306.08990v2. Output: FLAME expression + jaw. Code: in https://github.com/radekd91/inferno (research use; licence not verified).

- **Architecture:** temporal VAE motion prior (**FLINT**; q = 8 frames per latent; 128-d latent), frozen decoder. Condition: one-hot **emotion**, **intensity** (mild/medium/high), **speaker ID** → linear "styling" layer → **concatenated per frame** to wav2vec2 features. [C]
- **Labels:** MEAD emotion labels; emotion features from **EMOCA's emotion network** (ResNet-50 on AffectNet); a 1-layer transformer **video emotion classifier** trained on those with MEAD labels; 3D pseudo-GT from EMOCA + MICA shape + SPECTRE lip-reading loss + MediaPipe landmarks, fine-tuned on MEAD. [C]
- **Losses:** per-frame **lip-reading loss** (negative cosine distance between lip-reading features of differentiably rendered mouth crops, SPECTRE-style); sequence-level **video emotion loss**; **swap-emotion disentanglement** (swap two samples' conditions — lip loss must match the original audio, emotion loss the new label). λ_lip = 2.5e-5, λ_emo = 2.5e-6. Stage 1 vertex MSE for 20 epochs; stage 2 adds rendering + perceptual losses for 2 epochs. [C]
- **No numeric lip-reading-loss ablation** in v1 or v2: the ablation is a Mechanical Turk Likert study (14 LRS3 audios, 15 participants per pair), plots only (Fig. 7). Prose: *"w/o the lip-reading loss suffers from inaccurate lip-sync"*; without disentanglement bilabial closures are lost, *"especially during higher intensity emotions"*. Participants **preferred w/o-disentanglement on emotion** but disliked its lip-sync. [C]
- **Only numbers:** static per-frame emotion recognition 57.9% vs video classifier **90.8%** on MEAD validation (App. A.5, Fig. 13). [M]

---

## 7. EmoTalk — arXiv 2303.11089, ICCV 2023
Sources: https://arxiv.org/abs/2303.11089 (via https://ar5iv.labs.arxiv.org/html/2303.11089). Code: https://github.com/psyai-net/EmoTalk_release (licence not verified). Output: 52 ARKit blendshapes.

- **Emotion from audio:** two wav2vec2-large extractors fine-tuned for content and emotion; **cross-reconstruction** on RAVDESS pairs (same sentence, different emotion: content of one + emotion of the other → swapped GT); plus self-reconstruction, velocity loss, emotion **classification** loss on the emotion extractor. [C]
- **User controls:** emotion level 2-way one-hot (high/low); personal style 24-way one-hot; each linear → 32-D, concatenated with emotion (256-D) and content (512-D) features; decoder adds **emotion-guided cross-attention**. [C]
- **Labels:** 3D-ETF dataset = RAVDESS + HDTF, 6.5 h; blendshapes from an in-house ResNet trained on Live Link Face capture, "manually fine-tuned by professional animators". [C]
- **Tab. 5 ablation** (RAVDESS LVE / EVE, mm) [M]:

  | Variant | LVE | EVE |
  |---|---|---|
  | Full | 2.762 | 2.493 |
  | w/o disentangling encoder | 3.126 | 3.076 |
  | w/o emotion-guided attention | 2.907 | 2.832 |
  | w/o classification loss | 3.096 | 2.815 |
  | **w/o HDTF (neutral in-the-wild data)** | **3.254** | 2.806 |

- **Tab. 1:** RAVDESS LVE FaceFormer 3.247 vs Ours 2.762. [M] **Tab. 4 user study vs FaceFormer:** lip-sync 59.1%, emotion 69.2%. [M] Cost: 1× V100, ~8 h (80 epochs). [C]

---

## 8a. Playmate — arXiv 2502.07203, ICML 2025 (closest match to SANG's architecture)
Source: https://arxiv.org/html/2502.07203 (PMLR: https://proceedings.mlr.press/v267/ma25a.html).

- **Motion space:** a **LivePortrait implicit-keypoint space**, fine-tuned with a VASA-style pose/expression transfer loss; "adaptive normalisation" (global statistics for expression, per-clip for pose). [C]
- **Stage 1:** audio-only diffusion transformer (Conformer + Pose-MLP + Exp-MLP); ~80k clips (AVSpeech, CelebV-Text, Acappella, own data); 4× A100, 3 days. [C]
- **Stage 2:** base **frozen**; a 2-block **DiT emotion-control module** takes the Conformer output + emotion condition, and its output **replaces the Conformer output feeding only the Exp-MLP** (not the pose head). ~30k emotion-labelled clips (MEAD, MAFW, own); 2× A100, 2 days. [C]
- **Nested CFG (Eq. 8):** ε̂ = ε(∅,∅) + w_a[ε(a,∅) − ε(∅,∅)] + w_e[ε(a,e) − ε(a,∅)]; defaults w_a = w_e = 1.5. [C]
- **Tab. 4** [M]: audio-only (w_a = 1.5) Sync-C 8.141 / Sync-D 7.064; + emotion w_e = 1.5 → Sync-C 7.395 / Sync-D 7.644 / Emo-A 54.405; w_e = 2.0 → Emo-A 57.579, Sync-C 7.205; w_e = 3.0 → Emo-A 50.055; w_e = 3.5 → Sync-C 6.89 / Sync-D 8.313 / Emo-A 55.753. **Emotion guidance has a clearly measurable lip cost, and Emo-A is not monotonic in w_e.**
- **Tab. 2 MEAD Emo-A** (Savchenko classifier): 0.550 vs EAT 0.450, EDTalk 0.460. [M]

## 8b. Xemo-Talker — arXiv 2608.14700 (Aug 2026)
Source: https://arxiv.org/html/2608.14700. Code promised: https://github.com/chaolongy/Xemo-Talker.

- **Motion space:** a 70-D frozen-LivePortrait motion vector (scale, translation, rotation, 63-D deformation).
- **Stage 1:** 75.4M "GMP" temporal U-Net diffusion on **emotion-unlabelled** VoxCeleb (9,594 clips) + HDTF (1,963). **Stage 2:** base frozen, 40.3M mirrored branch; label as a CLIP text prompt; injected through **zero-initialised 1×1 conv residuals** into the frozen decoder; trained on MEAD. [C]
- **Tri-Loss** on the motion predicted from the diffusion output: CE classification; prototype alignment; a **less-principal contrastive** term — cosine similarity between predictions under label y and a different label y⁻ (stop-gradient on y⁻), computed only on the **lowest-variance 10% of PCA directions (K = 7)** of the 70-D motion. [C]
- **Tab. 2** (LSE-C / LSE-D / emotion acc) [M]: Stage-I only 6.15 / 8.02 / 15.84%; Full **6.37 / 8.21 / 85.28%**; real video 8.04 / 7.52 / 85.38%; EAT 7.29 / 8.35 / 75.43%.
- **Tab. 5, where to apply the contrastive term** [M]: whole space 6.28 / 8.34 / 75.84%; first 10% (principal) 6.31 / 8.32 / 79.80%; **last 10% (tail) 6.51 / 8.14 / 80.51%**.
- **Tab. 4 leave-one-out** [M]: w/o CE 6.91 / 8.09 / 72.89%; w/o zero-init feature injection → 82.23%.
- Cost: single RTX 4090, 500k iterations per stage, batch 64. [C]

## 8c. FLOAT — arXiv 2412.01064, ICCV 2025
Source: https://arxiv.org/html/2412.01064. Code: https://github.com/deepbrainai-research/float.

- **Emotion signal:** 7-D softmax from a pretrained speech emotion recogniser (**Pepino et al. 2021**, wav2vec2-based), concatenated with audio and the source latent, injected by **frame-wise AdaLN** (same vector shared across frames); at inference redirectable to any one-hot. [C]
- **Incremental CFG (Eq. 14):** ṽ = v(∅) + γ_a[v(a) − v(∅)] + γ_e[v(a,e) − v(a)]; defaults γ_a = 2, γ_e = 1. [C]
- **Tab. 6 RAVDESS** (E-FID / LSE-D) [M]: γ_a = 1, γ_e 1 → 2: 1.555 → 1.334, 7.049 → 7.212. γ_a = 2, γ_e 1 → 2: 1.367 → 1.351, LSE-D 6.994 unchanged.
- **Tab. 2 removing speech emotion** (HDTF / RAVDESS) [M]: E-FID 1.229 → 1.254 and 1.367 → 1.502; LSE-D 7.290 → 7.264 and 6.994 → 7.222. Image-derived emotion (**HSEmotion**) instead: E-FID 1.158 / 1.305.
- **Tab. 3:** frame-wise AdaLN LSE-D 7.290 vs cross-attention 7.757. [M]
- Cost: 1× A100, ~2 days, 2,000k steps. [C]

---

## 9. Transfer to SANG (helper's inferences, [I] unless tagged)

1. **Freeze the audio base, add a zero-initialised emotion branch.** Evidence: Cafe-Talk joint training SyncC 2.78 vs 5.38 two-stage (Tab. 3); Xemo-Talker's stage-2 branch moves LSE-D only 8.02 → 8.21 (Tab. 2); Playmate stage 2 with the base frozen. SANG's current TalkVid model is exactly "stage 1"; an adapter + zero-init output projection added to its AdaLN path is lowest-risk. Playmate routes emotion only into the expression head, not the pose head → maps onto SANG's 3 + 39 split.
2. **Signals obtainable on unlabelled TalkVid:** (a) global speech-emotion vector from a speech emotion recogniser (FLOAT) — no visual labels, native to frame-wise AdaLN; (b) frame-level VA from EmotiEffLib/HSEmotion on video frames (SubtleTalk) — noise never reported; EMOTE found static per-frame recognition unstable (57.9% vs 90.8% video) → smooth/aggregate over time first; (c) AUs — Cafe-Talk found OpenFace AUs misaligned with motion, EmoZone needed CIT-AE; derive from or validate against SANG's own motion space; continuous > binary (AUE-U 0.143 vs 0.281); (d) random/swapped labels on unlabelled clips with a masked loss (Cafe-Talk stage 2 on 157 h).
3. **Guidance:** nested CFG with separate scales (Playmate Eq. 8 / FLOAT Eq. 14), not Cafe-Talk's Eq. 6 (double-counts audio); add Cafe-Talk's temporal Z_cfg mask only for time-windowed emotion (without it: CR 0.27 vs 0.51, SyncC 4.74 vs 5.38); budget a lip cost (Playmate −0.75 Sync-C at w_e = 1.5); report Sync with emotion accuracy as a curve over w_e.
4. **Protecting the mouth in LivePortrait space:** Xemo-Talker's PCA-tail result is on the same 70-D LivePortrait parameterisation (tail supervision LSE-C 6.51 vs whole-space 6.28, Tab. 5; principal components hold jaw and pose). Alternatives: restrict emotion guidance/losses to non-mouth keypoints (a hard version of EmoZone's soft bias); masked swap loss (Cafe-Talk); residual over a deterministic lip prior (SubtleTalk: conditions alone raised LVE 14.60 → 15.05, IDC + DMP recovered). EMOTE's lip-reading loss has no numeric ablation and needs differentiable rendering (costly through LivePortrait).
5. **Evaluation:** train a motion-space emotion classifier on held-out labelled data (Cafe-Talk 88.28% on ARKit; EMOTE video-level) rather than per-frame image classifiers.
6. **Neutral motion + keypoint-space emotion offset (EmoVOCA):** naïve addition much worse than learned combination (LVE 5.971 vs 3.425) but the test is circular; treat it as a baseline, not a solution.

Firecrawl reported the account is low on credits.
