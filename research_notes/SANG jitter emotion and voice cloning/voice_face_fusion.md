# Fusing voice conversion / voice cloning with an audio-driven talking-head generator (SANG-M)

Evidence grades used below: **[M]** = measured number quoted from a paper table/figure (table ref given); **[A]** = author statement or abstract-level claim, no number checked; **[I]** = my inference (in the Inferences subsections only). All arXiv IDs below were checked against the paper index (Firecrawl research `inspect/read`) during this session. Where only an abstract was read, the finding is marked [A].

SANG facts assumed throughout: 53M bidirectional DiT flow-matching over LivePortrait keypoint motion (42-d, 25 fps), frame-wise AdaLN on WavLM-Large **last-layer** features (50 Hz, 2 ticks/frame, ±80 ms conv); frozen LivePortrait renderer; TalkVid (15 languages); single GPU; no transcripts.

---

## Q1. Pipeline order and where the face is driven from: (i) VC first, face driven by converted audio; (ii) face driven by original audio, output audio swapped for duration-preserving VC; (iii) one shared WavLM pass feeding both the face model and a kNN-style VC. Lip-sync of talking heads driven by TTS/VC vs natural audio; measured domain gap and fixes

### Takeaway
Talking heads trained on natural audio lose lip-sync when driven by synthetic speech in some measured settings (for example, LSE-C 7.59→6.18 for Wav2Lip and 8.41→6.30 for PLGAN on LRS2). But the direction of the change depends on the TTS system and on SyncNet itself: another paper measured **higher** LSE-C with TTS audio. So the only way to avoid the domain gap for certain is to drive the face with the audio it was trained on. kNN-VC works in WavLM-Large layer 6 at the same 20 ms frame rate that SANG uses, so a shared-encoder design (iii) that drives the face from the *original* audio's last-layer features and swaps in kNN-VC output audio (ii) keeps lip timing by construction. The recommended fusion is (ii)+(iii), with (i) available as an option once the face model is fine-tuned on VC-converted features.

### Cited Findings
**kNN-VC mechanics (why it fits SANG)**
- [M/A] kNN-VC uses WavLM-Large layer 6, "a single vector for every 20 ms of 16 kHz audio". Each source frame is replaced by the mean of its k=4 nearest (cosine) target frames, then vocoded with HiFi-GAN. The output is therefore frame-synchronous with the source. Later layers (22, 24, and the mean of the last layers) were tried and gave "worse pitch and energy reconstruction". Layer 6 was "necessary for good speaker similarity and retention of the prosody information from the source utterance" (Sec. 4.1) — [kNN-VC, arXiv 2305.18975](https://arxiv.org/abs/2305.18975)
- [M] Table 1 (LibriSpeech test-clean, ~8 min of target data): kNN-VC WER 7.36 / CER 2.96 / EER 37.15 / MOS 4.03±0.08 / SIM 2.91±0.11. FreeVC scored 7.61 / 3.17 / 8.97 / 4.07±0.07 / 2.38±0.11 and YourTTS 11.93 / 5.51 / 25.32 / 3.53 / 2.57. The topline (real speech) scored WER 5.96, MOS 4.24, SIM 3.19. Higher EER (max 50) means harder to tell converted speech from real target speech — [2305.18975](https://arxiv.org/abs/2305.18975)
- [A] Inference with 8 min of reference audio is "faster than real-time" on a "consumer 8GB VRAM GPU". A "prematched" vocoder (HiFi-GAN trained on kNN-reconstructed features) improves both WER and EER at every data size (Fig. 2) — [2305.18975](https://arxiv.org/abs/2305.18975)

**Measured TTS/synthetic-audio domain gap in talking heads**
- [M] Table 1 of *Shared Latent Representation for Joint Text-to-Audio-Visual Synthesis* (LRS2, matched pairs), LSE-C / LSE-D with real audio → the same models with TTS audio (HierSpeech++): Wav2Lip 7.59/6.75 → 6.18/8.12; TalkLip 8.53/6.08 → 7.05/7.21; AVTFG 7.95/6.30 → 6.19/8.16; PLGAN 8.41/6.03 → 6.30/7.99; Diff2Lip 7.87/6.46 → 7.06/6.84; IPLAP 6.49/7.16 → 6.51/7.08 (the only model with no drop). FID also worsens, e.g. Wav2Lip 7.05 → 10.85 — [arXiv 2511.05432](https://arxiv.org/abs/2511.05432)
- [M] Fix in the same paper: pretrain on real Wav2Vec2 features, then fine-tune on TTS-predicted features. Table 3 ablation: first stage only, real features LSE-C 8.39; first stage only, TTS features 3.24; two-stage without sync loss 3.31; full two-stage 4.14. Caveat: Table 3's "full" 4.14 does not match Table 1's "Ours (TTS)" 6.30, and the paper does not explain the difference in setting — [2511.05432](https://arxiv.org/abs/2511.05432)
- [M] Contradicting direction: *Faces that Speak* Table 1 (LRS2), LSE-C with GT audio → TTS audio: MakeItTalk 2.674 → 3.487; Audio2Head 4.607 → 5.478; SadTalker 4.978 → 6.256 (GT video 5.360). In this paper TTS audio *raised* SyncNet confidence — [arXiv 2405.10272](https://arxiv.org/abs/2405.10272)
- [A] JAM-Flow reports that SyncNet-derived metrics "often fail to operate reliably, showing instability under codec variations and, most critically, with TTS outputs". In its dubbing Table 3, GT gets LSE-C 6.73 and a zero-shot TTS baseline 2.78, yet users preferred JAM-Flow (LSE-C 3.43) over VoiceCraft-Dub (LSE-C 6.05) in 62.6% vs 37.4% of cases — [JAM-Flow, arXiv 2506.23552](https://arxiv.org/abs/2506.23552)
- [M] TTS data augmentation for a few-shot talking face (blendshapes; soft-DTW to absorb TTS/recording misalignment), Table 3 on 29 min of data: recorded audio MSE 0.00349 vs 1-speaker TTS 0.00420 vs 13-speaker TTS 0.00378. User preference was 61% for recorded vs 39% for TTS13. More TTS speakers → better. HuBERT features beat MFCC and PPG at every data size from 10 to 50 sequences (Fig. 2) — [arXiv 2303.05322](https://arxiv.org/abs/2303.05322)
- [M] Out-of-domain audio (cross-lingual, cross-gender, singing), GeneFace Table 1, SyncNet in-domain / OOD: Wav2Lip 9.212/9.645, LSP 6.119/4.320, AD-NeRF 4.894/4.225, GeneFace (HuBERT features, audio-to-motion trained on LRS3) 6.987/6.212, GT 8.733. Person-specific models collapse on OOD voices; a large multi-speaker audio-to-motion stage degrades less — [GeneFace, arXiv 2301.13430](https://arxiv.org/abs/2301.13430)
- [M] Cross-identity pairing also costs sync. 2511.05432 Table 2 ("cross-test", random audio–video pairs, real audio): Wav2Lip 7.35, TalkLip 6.04, AVTFG 6.84, PLGAN 7.58, Diff2Lip 6.71. These are lower than the matched Table 1 values (7.59, 8.53, 7.95, 8.41, 7.87). Confound: the video also changes, not just the voice — [2511.05432](https://arxiv.org/abs/2511.05432)

**Speaker/voice normalisation precedents in talking heads**
- [A] MakeItTalk splits audio into content, which "robustly controls the motion of lips", and a speaker embedding, which drives the rest of the head dynamics. Its content encoder is AutoVC-based (per the 2303.05322 related-work summary) — [MakeItTalk, arXiv 2004.12992](https://arxiv.org/abs/2004.12992); [2303.05322](https://arxiv.org/abs/2303.05322)
- [A] Live Speech Portraits projects deep audio features onto "the target person's speech space" (manifold projection) before predicting motion, for robustness to wild audio. This is the closest talking-head analogue of a kNN projection onto X's own features — [LSP, arXiv 2109.10595](https://arxiv.org/abs/2109.10595)
- [A] OPT uses an "Audio Feature Disentanglement Module" to remove "speaker-specific information contained in arbitrary driving audios" — [OPT, arXiv 2302.08197](https://arxiv.org/abs/2302.08197)
- [A] AVCT feeds phonemes instead of acoustic features so that the model "can inherently generalize to audio spoken by other identities" — [arXiv 2112.02749](https://arxiv.org/abs/2112.02749)

**Duration behaviour of alternative VCs**
- [A] Seed-VC README: V1 has a `length-adjust` factor that defaults to 1.0 (duration preserved). V2's optional AR model is "for accent & emotion conversion"; with `convert-style false` only timbre is converted. Both accept 1–30 s of reference — [Seed-VC GitHub](https://github.com/Plachtaa/seed-vc)
- [A] Seed-VC uses the Whisper-small encoder as its content encoder and a flow-matching DiT with full-reference in-context timbre. It reports SECS 0.8676 vs OpenVoice and CosyVoice, and WER 11.99% — [Seed-VC, arXiv 2411.09943](https://arxiv.org/abs/2411.09943)
- [A] Vevo: an AR content→content-style stage (which changes duration; the paper measures DDUR, the duration difference) followed by a flow-matching acoustic stage for timbre — [Vevo, arXiv 2502.07243](https://arxiv.org/abs/2502.07243)

### Inferences
- [I] **(i) VC first, face from the converted audio.** This adds two failure sources that SANG does not have today. First, the VC domain shift into the face encoder: both directions of LSE change are documented, and a 1–2 point LSE-C loss is typical in 2511.05432. Second, VC content errors become lip errors: kNN-VC's WER rises sharply below about 30 s of reference. The one potential upside is that the face then sees a voice matching the face's identity. That matches TalkVid training conditions, where voice and face come from the same person. No paper isolates this effect.
- [I] **(ii) Face from the original audio, output audio replaced.** The face model sees exactly its training distribution. Lip timing is identical as long as the VC preserves duration frame for frame. kNN-VC does this by construction (20 ms in, 20 ms out). Seed-VC V1 and Vevo's timbre-only stage should, but they are not guaranteed to be sample-accurate; check lengths. The standard one-shot setting already pairs a face with another person's voice, so this is SANG's current operating point.
- [I] **(iii) Shared WavLM pass.** One forward pass of WavLM-Large with `output_hidden_states=True` gives layer 6 (for kNN-VC) and layer 24 (for SANG) on the same 50 Hz grid. This costs no extra encoder compute and aligns frames exactly.
- [I] The SyncNet unreliability reported by JAM-Flow and *Faces that Speak* means an LSE change between original and VC audio may partly reflect SyncNet's audio branch reacting to a vocoder or timbre change, not true lip error. Under (ii) the video is identical for both audio tracks, so any LSE delta is purely an audio-side metric artefact. Report it as a metric-sensitivity check, not a lip-sync regression.

#### Recommended fusion for SANG (inference, grounded in the findings above)
1. **Encoder:** run WavLM-Large once on the driving audio. Keep h6 (kNN-VC) and h24 (SANG conditioning).
2. **Face:** drive the DiT with the *original* h24. This is unchanged from today and needs no retraining.
3. **Voice:** kNN-VC(h6 → X's matching set) → prematched HiFi-GAN → output waveform. Mux it with the rendered video; there is no re-alignment step.
4. **Fallback when X's recording is < ~30 s:** a short-reference variant in the same WavLM space (MKL-VC, 5 s; see Q5). Alternatively Seed-VC V1 timbre-only, with a length check.
5. **Optional robustness (enables (i) and TTS driving):** fine-tune the DiT on re-encoded h24 from kNN-VC-converted TalkVid audio. Pair converted audio with the original motion; timing is unchanged, so no soft-DTW is needed. This follows the two-stage recipe of 2511.05432 and the speaker-diversity result of 2303.05322.
6. **No recording of X at all:** optional face-derived voice (Q3); flag the lower speaker plausibility.

### Gaps
- I found **no paper that measures lip-sync of a talking head driven by VC-converted audio** as opposed to TTS audio. All measured gaps are for TTS, whose timing differs. For kNN-VC the timing is identical, so the TTS numbers are an upper bound on the harm, and (ii) avoids it anyway.
- I found no talking-head paper that trains with **VC-based speaker-perturbation augmentation** of the driving audio. TTS augmentation (2303.05322) and pitch-shift Siamese content training in VC (ACE-VC, arXiv 2302.08137) are the nearest analogues.
- The direction of SyncNet's bias under TTS/VC audio is unresolved (2511.05432 vs 2405.10272).

---

## Q2. Which WavLM/HuBERT layers carry content vs speaker identity, and would speaker-normalised content features improve cross-speaker lip-sync? Evidence from talking-head audio-encoder comparisons

### Takeaway
In WavLM-Large, speaker information is most accessible in early and middle layers (probing peak at layer 3). Content dominates the top layers, but large models "recover" speaker information in deep layers. So SANG's last-layer features are content-heavy but not speaker-free. Cheap linear speaker-removal methods exist (LinearVC rank-100 projection, Eta-WavLM, USCF). Talking-head evidence favours SSL or phoneme-aware features over MFCC/PPG and shows gains from sync-specialised encoders (PASE, AV-HuBERT). But no paper directly shows that speaker-normalising WavLM features improves cross-speaker lip-sync.

### Cited Findings
- [A] WavLM paper, Fig. 2 (SUPERB learned layer weights): in Base models "the bottom layers contribute more to speaker-related tasks… for ASR, PR… the top layers are more important". In Large models "the top layers contribute most to content and semantic tasks, while the middle layers have a great impact on speaker tasks". WavLM-Large has 24 layers, 1024-d, 316.62M parameters — [WavLM, arXiv 2110.13900](https://arxiv.org/abs/2110.13900)
- [M] Large-scale probing, Table 1, WavLM-Large peak layer (val/test %): Speaker L3 (98.7/99.5); Gender L4; Pitch L3; Energy L3; Tempo L20; **Emotion L10 (95.7/94.0)**. Emotion accessibility is "relatively stable from the initial to the final layers". "Larger models unexpectedly recover speaker-discriminative information in their deep layers" — [arXiv 2501.05310](https://arxiv.org/abs/2501.05310)
- [A] kNN-VC chose layer 6 over layers 22/24 because the later layers "give poorer predictions of pitch, prosody, and speaker identity" (citing Lin et al., SLT 2023). Layers 22/24 "perform well on linear phone recognition" — [2305.18975](https://arxiv.org/abs/2305.18975)
- [A] Comparative CCA layer analysis: property trends differ by pre-training objective, and "single-layer performance often matches or improves upon using all layers" — [Pasad et al., arXiv 2211.03929](https://arxiv.org/abs/2211.03929)
- [A] Cross-model SUPERB probing: "the capacity to represent content information is somewhat unrelated to enhanced speaker representation" — [arXiv 2401.17632](https://arxiv.org/abs/2401.17632)
- [A] Linear structure: LinearVC shows "just rotating the features is sufficient for high-quality voice conversion". An SVD content/speaker factorisation "with a rank of just 100 gives competitive conversion results" — [LinearVC, arXiv 2506.01510](https://arxiv.org/abs/2506.01510)
- [A] Eta-WavLM linearly decomposes SSL features into speaker-specific and speaker-independent parts — [arXiv 2505.19273](https://arxiv.org/abs/2505.19273). USCF learns a "universal speech-to-content mapping via least-squares" with speaker transforms from "only a few seconds of target speech" — [arXiv 2603.08977](https://arxiv.org/abs/2603.08977). In predictive-coding models, speaker and phone subspaces are "nearly orthogonal", and collapsing the speaker subspace generalises to unseen speakers — [arXiv 2305.12464](https://arxiv.org/abs/2305.12464)
- [A] ContentVec (HuBERT-derived) disentangles speakers "without severe loss of content" and helps content tasks — [arXiv 2204.09224](https://arxiv.org/abs/2204.09224). Soft speech units for VC — van Niekerk et al., ICASSP 2022 (cited in 2305.18975). I did not open the Soft-VC arXiv page.
- [A] Talking-head encoder evidence: PASE (phoneme-aware speech encoder) improves lip-sync by 13.7% (NeRF) and 14.2% (3DGS) over "conventional methods based on acoustic features" — [arXiv 2504.05803](https://arxiv.org/abs/2504.05803). SyncDiff uses AV-HuBERT audio features and reports sync scores 27.7% (LRS2) and 62.3% (LRS3) relatively higher than prior diffusion methods — [arXiv 2503.13371](https://arxiv.org/abs/2503.13371). A real-time portrait study found that swapping the audio feature extractor for Whisper sped processing and improved some rendering-quality aspects — [arXiv 2411.13209](https://arxiv.org/abs/2411.13209). Wav2Sem notes that near-homophone syllables are "coupled" in SSL feature space, which causes lip "averaging" — [arXiv 2505.23290](https://arxiv.org/abs/2505.23290)
- [M] HuBERT > MFCC and > PPG for few-shot talking-face regression at all training sizes (Fig. 2) — [2303.05322](https://arxiv.org/abs/2303.05322)
- [A] JAM-Flow's stage-1 Motion-DiT concatenates wav2vec2 features, following "standard talking head practice" — [2506.23552](https://arxiv.org/abs/2506.23552)

### Inferences
- [I] SANG's h24 is a sensible content choice, but it will still leak speaker cues (per 2501.05310). The DiT may therefore learn voice-identity→motion-style correlations from TalkVid, for example gender- or age-linked mouth aperture. Under pipeline (ii) that leak comes from the *source* speaker. Under (i) it comes from X. Neither is harmful for lip timing, but (i) makes the face's "voice-implied style" consistent with X.
- [I] A cheap, testable speaker-normalisation for the face stream: estimate per-speaker mean/covariance of h24 (or an Eta-WavLM/LinearVC-style speaker projection) on TalkVid, remove it before AdaLN, and retrain. Then compare cross-speaker LSE-C/D and lip-vertex/bilabial-closure errors on held-out speakers against the current model. Given the orthogonal-subspace evidence, this should cost little content. Whether it helps lips is unmeasured.
- [I] Do **not** switch the face stream to h6. It is the layer with the most speaker/pitch information in WavLM-Large, which is the opposite of what the lips need.
- [I] Emotion is accessible at all depths (peak L10), so speaker-normalising h24 should not strip the emotion cues SANG may rely on. A linear speaker projection could still remove some pitch-range information that correlates with arousal; check with an SER probe before and after.

### Gaps
- No layer-24 (final) speaker-probe number for WavLM-Large was found in 2501.05310's table; only peak layers are listed.
- No talking-head paper compares WavLM layers or ContentVec vs raw SSL features for **cross-speaker** lip-sync. I could not verify encoder ablations in Hallo, EchoMimic, FLOAT, LatentSync or JoyGen (I did not read those papers' ablation tables), so their encoder choices (wav2vec2 / Whisper) are not graded here.

---

## Q3. Joint audio–visual generators and face-driven voice generation: what they report, and practicality on one GPU vs a modular pipeline

### Takeaway
Joint AV generators now beat simple cascades on their own benchmarks. They are either large (Ovi 10.9B, JoVA on a 12B base, LTX-2 14B+5B, UniAVGen 7.1B) or text-conditioned (JAM-Flow ~500M, OmniTalker 0.8B, AV-Flow), which conflicts with SANG's transcript-free, multilingual, single-GPU constraints. JAM-Flow is architecturally the closest to SANG: LivePortrait keypoints and a flow-matching DiT. Face-driven voice generation (Face-TTS, FaceVC, HYFace, Vclip) yields only coarse, gender- and age-level voice plausibility. It is useful only as a fallback when there is no recording of X.

### Cited Findings
- [M] **JAM-Flow** generates only the mouth-related LivePortrait expression deltas (e_mouth ∈ R^{T×4×3}) conditioned on the other 17 keypoints and on audio features. The audio branch is F5-TTS (text + reference audio), with joint attention in half the layers. Table 1 (HDTF): I2V LSE-C/D 7.324/7.777; V2V 8.086/7.181; SadTalker 7.885/7.545; Hallo 7.750/7.659; GT 8.70/6.597. About 500M parameters jointly; 45 s for a 20 s video on one RTX A6000. Training took 4× RTX 6000 Ada, about 1 day per stage. TTS on LibriSpeech-PC (Table 2): WER 4.91% / SIM-o 0.64 vs F5-TTS 2.42% / 0.66. Trained on CelebV-Dub with Whisper pseudo-transcripts — [2506.23552](https://arxiv.org/abs/2506.23552)
- [M] **UniAVGen** Table 1 (100 test samples; LS = SyncNet confidence; TC/EC = timbre/emotion consistency scored 0–1 by Gemini-2.5-Pro). Ovi (10.9B params, 30.7M joint samples): LS 6.48, TC 0.828, EC 0.558. UniAVGen (7.1B, 1.3M): LS 5.95, TC 0.832, EC 0.573. Two-stage F5-TTS→OmniAvatar (21.1B): LS 6.34, TC 0.454, EC 0.349. F5-TTS→Wan-S2V (16.6B): LS 6.35, TC 0.481, EC 0.375. The TC rubric only grades gender and age match — [arXiv 2511.03334](https://arxiv.org/abs/2511.03334)
- [A] **Ovi**: symmetric twin DiT, "5B-per-branch", 5 s 720p/24 fps clips, 16 kHz audio through a 1D-VAE. The authors note it "requires significant time per sampling step" (plus the CFG pass). Evaluated by pairwise human preference against JavisDiT and UniVerse-1 — [arXiv 2510.01284](https://arxiv.org/abs/2510.01284)
- [M] **JoVA** (built on Waver 12B), JoVABench-Gen LSE-C: JoVA 6.65; Ovi 6.61; Wan-S2V (driven by GT audio) 6.49; FantasyTalking 2.68. Mouth-area-loss ablation (Table 4): λ=0 → LSE-C 1.39; λ=5 → 6.70. Without explicit mouth supervision, joint models barely lip-sync — [arXiv 2512.13677](https://arxiv.org/abs/2512.13677)
- [M] **AV-Flow** is person-specific: trained on Audio2Photoreal (4 identities, 8 h) plus 50 h of one actor, with personalised renderers. Ablation Table 1: cascaded audio-DiT→visual-DiT F1_lips 0.848 / WER 0.179 vs AV-Flow 0.964 / 0.157. Runs in real time with 8 Euler steps and ~120 ms latency — [arXiv 2502.13133](https://arxiv.org/abs/2502.13133)
- [M] **OmniTalker**: 0.8B parameters, trained on 8 A100s, real-time at 25 FPS. Table 2 VoxCeleb2 Sync-C: 6.37 vs Hallo 6.53 and EchoMimic 5.88. In-context reference is truncated to 1–10 s. Speech SIM-A 0.766 vs F5-TTS 0.741 (Seed ZH). MOS "speak style" 4.57 vs Hallo 3.94. The model is text-driven — [arXiv 2504.02433](https://arxiv.org/abs/2504.02433)
- [M] **Faces that Speak (TTSF)**: text-to-audio-visual with a face-conditioned Matcha-TTS. Voice similarity C-SIM (Table 3): Face-TTS 0.272; TTSF with motion features 0.451; TTSF identity-only 0.593. Removing motion from the face embedding stabilises the voice. MOS lip sync 4.09 vs SadTalker 2.78, where SadTalker is cascaded on the same TTS (Table 4). Trained on 8× 48GB A6000 — [2405.10272](https://arxiv.org/abs/2405.10272)
- [A] Other joint systems with voice cloning: MM-Sonate (zero-shot voice cloning inside joint AV flow matching) — [arXiv 2601.01568](https://arxiv.org/abs/2601.01568); UniTalking — [arXiv 2603.01418](https://arxiv.org/abs/2603.01418); Hallo-Live runs at 20.38 FPS on **two H200s** (distilled from Ovi) — [arXiv 2604.23632](https://arxiv.org/abs/2604.23632); LTX-2 (14B video + 5B audio) — [arXiv 2601.03233](https://arxiv.org/abs/2601.03233); Talker-T2AV claims better lip-sync than dual-branch and cascaded pipelines — [arXiv 2604.23586](https://arxiv.org/abs/2604.23586)
- [A] **Face→voice**: Face-TTS (first TTS conditioned on face images, LRS3) — [arXiv 2302.13700](https://arxiv.org/abs/2302.13700); Face-StyleSpeech (face encoder for timbre plus a separate prosody encoder) — [arXiv 2311.05844](https://arxiv.org/abs/2311.05844); zero-shot FaceVC with memory-based face–voice alignment — [arXiv 2309.09470](https://arxiv.org/abs/2309.09470); ID-FaceVC — [arXiv 2409.00700](https://arxiv.org/abs/2409.00700); HYFace ("Hear Your Face", predicts average F0 from the face) — [arXiv 2408.09802](https://arxiv.org/abs/2408.09802); Vclip (CLIP-based face–voice association, **89.63% cross-modal verification AUC** on VoxCeleb, retrieval + GMM speaker generation) — [arXiv 2601.02753](https://arxiv.org/abs/2601.02753); a StyleTTS2 face adapter gives UTMOS 3.7–4.0 (GT 3.61) with face→voice retrieval "consistently above chance" — [arXiv 2607.26742](https://arxiv.org/abs/2607.26742); Face2VoiceSync trains on "a single 40GB GPU" — [arXiv 2507.19225](https://arxiv.org/abs/2507.19225)
- [A] Text-driven joint precedents: NEUTART (non-cascaded text→AV) — [arXiv 2312.06613](https://arxiv.org/abs/2312.06613); AnyoneNet (face-conditioned TTS + talking head) — [arXiv 2108.04325](https://arxiv.org/abs/2108.04325)

### Inferences
- [I] **Practicality for SANG.** Every open joint model needs text for its audio branch (JAM-Flow, OmniTalker, TTSF, AV-Flow, UniAVGen, Ovi). SANG's task is *audio-driven* with arbitrary driving speech, and TalkVid has no verified transcripts across 15 languages. A joint model would therefore have to become a "speech-in → speech-out + motion" model, i.e. VC inside the joint model. None of the surveyed systems does this. The modular pipeline (kNN-VC + existing SANG) is the only option that satisfies all constraints today.
- [I] JAM-Flow's LivePortrait-mouth-subspace design (4 of 21 keypoints) is directly transferable to SANG's 42-d motion. So is its inpainting-style training with a reference-motion context (see Q5). Its audio branch is not transferable (text + F5-TTS).
- [I] Face-derived voices score C-SIM around 0.27–0.59 and cross-modal association AUC below 0.9. Use them only as a "no-recording" default voice, and label it as imagined.

### Gaps
- I could not verify arXiv IDs for "AVI-Talking", "TAVS", "Visual-aware TTS", "FVTTS" or "SP-FaceVC" in this session, so they are not graded.
- No joint model reports memory or throughput on a single 24–48 GB GPU except JAM-Flow (A6000, 45 s per 20 s) and OmniTalker (25 FPS; GPU class not stated in the passage read).

---

## Q4. Face–voice consistency: evidence that a mismatched voice hurts realism, and cross-modal face–voice matching models usable as a metric

### Takeaway
Perception studies show that a face–voice mismatch produces eeriness or lowers realism. But automatic face–voice matchers are weak (best FAME-2026 EER about 24%) and lean on gender, age and language cues. As a metric they can only catch gross mismatches (gender or age), not "is this X's voice". When X's own recording exists, speaker similarity to X's real voice (SECS/EER) is the meaningful consistency metric.

### Cited Findings
- [A] "A mismatch in the human realism of face and voice produces an uncanny valley" (i-Perception 2011; 48 participants; matched vs mismatched 14 s videos). Incongruence elicited eeriness — [Mitchell et al. 2011](https://journals.sagepub.com/doi/10.1068/i0415)
- [A] Child avatars (N=70): "silencing clips improved perceived realism by removing mismatches between voice and animation, especially when tone or age felt incongruent" — [arXiv 2506.13477](https://arxiv.org/abs/2506.13477)
- [M] FAME 2024 (MAV-Celeb, cross-modal verification): first place overall EER 19.9% — [arXiv 2407.17902](https://arxiv.org/abs/2407.17902); challenge plan — [arXiv 2404.09342](https://arxiv.org/abs/2404.09342). FAME 2026: first place average EER 23.99% — [arXiv 2512.04814](https://arxiv.org/abs/2512.04814); second place (ImageBind-LoRA) 24.73% — [arXiv 2512.02759](https://arxiv.org/abs/2512.02759)
- [A] FLAG 2027 plan: face–voice models "may rely on language or gender cues", and baseline performance "degrades under language shifts and in gender-constrained settings" — [arXiv 2609.17913](https://arxiv.org/abs/2609.17913)
- [M] Vclip reaches 89.63% cross-modal verification AUC on the VoxCeleb test set — [2601.02753](https://arxiv.org/abs/2601.02753)
- [A] UniAVGen's "timbre consistency" is scored by Gemini-2.5-Pro using a gender/age rubric (0 = gender mismatch … 1 = gender and age match). There is "no open-source methodology" to quantify it — [2511.03334](https://arxiv.org/abs/2511.03334)
- [A] VR agent studies: vocal type affected perceived realism and familiarity — [TVCG 2026, PMID 41941770](https://doi.org/10.1109/TVCG.2026.3680715)

### Inferences
- [I] For SANG with X's recording: face–voice consistency ≈ SECS(output, X's real voice) plus a gross-mismatch guard (a FAME-style or Vclip-style matcher, or an LLM rubric, used only to flag gender or age conflicts).
- [I] Without a recording (face-derived or default voice), report the FAME-style verification score of (face image, output voice) against a same-gender impostor gallery. The FLAG protocol exists precisely because unconstrained galleries overstate performance.

### Gaps
- No study quantifies how much a *subtly* wrong voice (right gender and age, wrong person) lowers the realism of photoreal talking heads. Mitchell 2011 manipulated human vs synthetic realism, not identity.

---

## Q5. Personalisation that improves with more target data: voice and face motion (speaking style); numbers vs amount of data

### Takeaway
On the voice side, kNN-VC quality rises steeply up to about 5 min of target speech and then plateaus (8 min ≈ 5 min). Below 30 s it falls behind parametric VCs, and MKL-VC, a WavLM-space optimal-transport variant, targets the 5 s regime. On the face side, a short *video* of X personalises motion effectively: in-context style prompts (MimicTalk, OmniTalker with 1–10 s, JAM-Flow-style inpainting) or LoRA adapters (MimicTalk 15-min adaptation; AdaMesh with ~10 s). An audio-only recording gives almost no evidence about face motion.

### Cited Findings
- [A/M] kNN-VC Fig. 2 (WER vs EER by target data size): quality degrades as data shrinks. "Using the maximum speaker data in LibriSpeech (roughly 8 minutes per speaker) gives very similar performance to only using 5 minutes". With "less than 30s", the "more complex baselines perform better". With 5 s it still retains "moderate intelligibility and speaker similarity". With ≥10 min, larger k (~20) "may even" improve quality — [2305.18975](https://arxiv.org/abs/2305.18975)
- [A] MKL-VC replaces kNN regression with a factorised Monge–Kantorovich linear OT map in WavLM subspaces. It gives "high quality any-to-any cross-lingual voice conversion with only 5 second of reference", outperforms kNN-VC on LibriSpeech and FLEURS, and is comparable to FACodec — [arXiv 2506.09709](https://arxiv.org/abs/2506.09709). kDOT (discrete OT, barycentric projection) "often outperforms averaging-based approaches" in WER, MOS and FAD — [arXiv 2505.04382](https://arxiv.org/abs/2505.04382). kNN-FM-VC (13M, one-step flow matching trained on kNN pairs) "substantially reduces WER" vs kNN and kDOT — [arXiv 2609.27230](https://arxiv.org/abs/2609.27230)
- [A] kNN-VC "retains high performance in stuttered and cross-lingual voice conversion" — [arXiv 2310.08104](https://arxiv.org/abs/2310.08104)
- [A] Seed-VC clones from "1~30 seconds" of reference (README) — [Seed-VC GitHub](https://github.com/Plachtaa/seed-vc). OmniTalker truncates its audio-visual reference to 1–10 s — [2504.02433](https://arxiv.org/abs/2504.02433)
- [M] **MimicTalk** Table 1: adaptation takes 0.26 h and 8.239 GB on one A100. Sync 8.072 vs StyleTalk 7.173, GeneFace 6.480, RAD-NeRF 4.916. Fig. 4(b): CSIM rises with more adaptation data, and **60 s matches a person-specific RAD-NeRF trained on 180 s**. The ICS-A2M motion model mimics style in context from a reference audio–motion pair. In the ablation, the in-context prompt beat a hand-crafted style vector and a StyleTalk-style encoder on motion reconstruction — [arXiv 2410.06734](https://arxiv.org/abs/2410.06734)
- [A] AdaMesh learns personalised expression and pose style "from a reference video of about 10 seconds", using mixture-of-LoRA (MoLoRA) for expressions and retrieval for poses — [arXiv 2310.07236](https://arxiv.org/abs/2310.07236)
- [A] Imitator optimises identity-specific style from a short video on top of a style-agnostic prior, with a **bilabial-consonant lip-closure loss** — [arXiv 2301.00023](https://arxiv.org/abs/2301.00023). StyleLipSync does few-shot adaptation "using a few seconds of target video" with a sync regulariser that preserves generalisation — [arXiv 2305.00521](https://arxiv.org/abs/2305.00521). StyleTalk — [arXiv 2301.01081](https://arxiv.org/abs/2301.01081); StyleTalk++ — [arXiv 2409.09292](https://arxiv.org/abs/2409.09292); PersonaTalk (style-aware audio encoding via cross-attention) — [arXiv 2409.05379](https://arxiv.org/abs/2409.05379)
- [M] GeneFace domain-adapts multi-speaker landmark predictions to the target person with a post-net trained on 4–5 min of target video — [2301.13430](https://arxiv.org/abs/2301.13430)
- [A] Audio-only style cues: MemoryTalker stylises 3D motion "only with audio input" — [arXiv 2507.20562](https://arxiv.org/abs/2507.20562)

### Inferences
- [I] **Voice scaling plan for SANG.** For less than 30 s, use MKL-VC or kDOT in the same WavLM-L6 space; it drops in where kNN-VC sits. For 30 s to 5 min, use kNN-VC with k=4. For 10 min or more, use kNN-VC with k≈8–20 and optionally fine-tune the prematched vocoder on X, which is my extrapolation of the paper's prematching result. For hours, a per-speaker parametric model trained on X becomes worthwhile; that is untested here.
- [I] **Face personalisation.** SANG is bidirectional flow matching over motion, so the cheapest personalisation mirrors MimicTalk and JAM-Flow. Train with random *motion-infilling masks* so that a clip of X's real motion plus its audio can be prepended as an in-context prompt. For longer video, add a LoRA on the DiT. Expect gains to saturate after about 60 s of video, by MimicTalk's CSIM curve (which is for a renderer, not motion).
- [I] If only X's *audio* is available, it helps the voice but gives the face at most weak cues, such as speech-rate or energy statistics. Do not claim face personalisation from voice alone.

### Gaps
- No paper gives lip-sync or style metrics for motion personalisation *vs seconds of video* in a LivePortrait-keypoint DiT. MimicTalk's curve is for renderer CSIM.
- kNN-VC has no measured curve beyond 8 min per speaker.

---

## Q6. Emotion/prosody coupling: if the driving audio is converted, is emotion/prosody preserved so face and voice agree?

### Takeaway
Frame-synchronous SSL VCs (kNN-VC) keep most, not all, of the source emotion. In one measurement IEMOCAP UAR fell from 70.1% (original) to 56.7%, versus 30.4% for ASR→TTS. Timing, and so speech rhythm, is kept exactly. Under the recommended pipeline (ii) the face reads emotion from the original audio, while the voice carries somewhat attenuated emotion. The residual mismatch is modest and measurable.

### Cited Findings
- [M] JHU Voice Privacy Challenge 2024, Table 1 (IEMOCAP emotion UAR, avg; LibriSpeech WER): original 70.07% / 1.831%; **kNN-VC 56.70% / 3.162%**; kNN-VC + length variation 56.29%; whisper-VITS (ASR→TTS) 30.35% / 3.751%. "Voice conversion systems better preserve emotional content" — [arXiv 2409.08913](https://arxiv.org/abs/2409.08913)
- [M] The same table shows kNN-VC leaks source identity under the semi-white-box attack (EER 7.95%) — [2409.08913](https://arxiv.org/abs/2409.08913)
- [A] kNN-VC chose layer 6 for "retention of the prosody information from the source utterance" — [2305.18975](https://arxiv.org/abs/2305.18975)
- [M] Emotion is probe-accessible across WavLM-Large layers (peak L10: 95.7/94.0) — [2501.05310](https://arxiv.org/abs/2501.05310)
- [A] In-context-learning VC "faces challenges in preserving the prosody of the source speech", and adding SER-model prosody embeddings fixes it (evaluated on ESD) — [arXiv 2409.05004](https://arxiv.org/abs/2409.05004). Existing zero-shot VCs "struggle to fully reproduce paralinguistic information… breathing, crying, and emotional nuances" — [Takin-VC, arXiv 2410.01350](https://arxiv.org/abs/2410.01350). StarGANv2-VC "fails to preserve emotion" until emotion-aware losses are added — [Emo-StarGAN, arXiv 2309.07586](https://arxiv.org/abs/2309.07586)
- [A] Style-converting VCs deliberately change emotion or accent, and with it duration: Vevo's AR stage — [2502.07243](https://arxiv.org/abs/2502.07243); Seed-VC V2's `convert-style` AR model — [Seed-VC GitHub](https://github.com/Plachtaa/seed-vc)
- [M] Joint AV generators treat "emotion consistency" as a metric: UniAVGen EC 0.573 vs 0.349–0.375 for TTS→talking-head cascades (Gemini rubric) — [2511.03334](https://arxiv.org/abs/2511.03334)

### Inferences
- [I] Use timbre-only conversion. Never enable AR style conversion (Seed-VC V2 `convert-style`, Vevo-Style) in SANG's pipeline. It would re-time and re-emote the voice while the face keeps the source emotion and timing.
- [I] kNN-VC's pitch contour is assembled from X's matched frames. If X's recording is emotionally flat (e.g. read speech), high-arousal source frames have no good neighbours and the output flattens. Recommend recordings of X with varied affect, and measure SER agreement between the output audio and the original.
- [I] If SANG later conditions the face on emotion *derived from audio*, derive it from the original audio in pipeline (ii). Deriving it from the converted audio would make the face inherit VC's emotion loss.

### Gaps
- No measurement of kNN-VC emotion preservation as a function of the matching set's emotional coverage.
- No study checks face–voice emotional agreement after VC in a talking-head system.

---

## Q7. End-to-end evaluation protocol for the combined system

### Takeaway
Evaluate on the final muxed output with a mixed battery. Lip-sync: LSE-C/D, plus a SyncNet-independent check, because SyncNet is unreliable on synthetic audio. Voice: speaker similarity to X's real voice (SECS with WavLM-TDNN or ECAPA, plus kNN-VC-style EER), multilingual WER/CER, UTMOS. Emotion: SER UAR agreement between original and output. Face–voice: a gross-mismatch guard. Finish with a user study. Always report "original-audio" and "converted-audio" versions of the same video.

### Cited Findings
- [A] SyncNet (LSE-C/D) instability under codec changes and TTS audio — [2506.23552](https://arxiv.org/abs/2506.23552). TTSF authors: LSE-C "relies significantly on a pretrained model", so they add human judgement — [2405.10272](https://arxiv.org/abs/2405.10272)
- [A] AV-HuBERT-based lip-sync losses and "three novel lip synchronization evaluation metrics" — [arXiv 2405.04327](https://arxiv.org/abs/2405.04327). A mesh-based lip-closure F1 is used by AV-Flow — [2502.13133](https://arxiv.org/abs/2502.13133)
- [M] Voice metric conventions: kNN-VC reports Whisper-base WER/CER, x-vector EER (higher = closer to real target, max 50%), MOS, and a VCC2020 1–4 similarity MOS — [2305.18975](https://arxiv.org/abs/2305.18975). JAM-Flow and F5-TTS-style SIM-o — [2506.23552](https://arxiv.org/abs/2506.23552). Resemblyzer SECS, Whisper-large-v3 WER and UTMOS (HierSpeech++: WER 1.51%, SECS 72%, UTMOS 4.22 vs GT 4.47% / – / 3.05) — [2511.05432](https://arxiv.org/abs/2511.05432). SECS with Seed-VC 0.8676 — [2411.09943](https://arxiv.org/abs/2411.09943)
- [M] Emotion preservation via SER UAR (IEMOCAP) and content via ASR WER, VoicePrivacy protocol — [2409.08913](https://arxiv.org/abs/2409.08913)
- [A] Face–voice: FAME/FLAG cross-modal verification EER with a gender-constrained gallery — [2609.17913](https://arxiv.org/abs/2609.17913); LLM-rubric timbre/emotion consistency — [2511.03334](https://arxiv.org/abs/2511.03334)
- [M] User studies: MOS for lip sync, motion naturalness and realness (TTSF Table 4) — [2405.10272](https://arxiv.org/abs/2405.10272); pairwise preference (JAM-Flow dubbing 62.6/37.4) — [2506.23552](https://arxiv.org/abs/2506.23552)

### Inferences — proposed SANG protocol [I]
1. **Test set:** unseen TalkVid validation speakers (the existing hash split) × target identities X with 5 s, 30 s, 5 min and ≥10 min of held-out recordings. Driving audio comes from a different speaker, including cross-gender and cross-language pairs.
2. **Lip-sync:** LSE-C/D on (a) video + original audio and (b) video + converted audio. The video is the same in (ii), so Δ = the metric's audio sensitivity. Add bilabial-closure F1 (SANG already has bilabial rules) and an AV-HuBERT-based sync score.
3. **Voice identity:** SECS (WavLM-TDNN and ECAPA; report both) against held-out real X utterances, with a real-X vs real-X ceiling. Also kNN-VC-style EER.
4. **Content:** multilingual Whisper-large-v3 WER/CER relative to the ASR of the original audio (no transcripts needed). CER for zh/ja.
5. **Quality:** UTMOS on the output audio (note the 16 kHz vocoder ceiling), and FVD/FID on the video.
6. **Emotion:** SER label/UAR agreement between original and converted audio, plus an expression-emotion classifier on the video.
7. **Face–voice:** a gender- and age-constrained cross-modal verification score as a guard only.
8. **User study:** MOS for lip-sync, voice similarity to X (with a real reference clip), realism, and face–voice match. Include an A/B of pipeline (i) vs (ii).

### Gaps
- No standard, validated automatic face–voice identity metric exists. Speaker-verification models are not calibrated on vocoded kNN output from WavLM features (kDOT shows that OT post-processing can fool spoof detectors, arXiv 2505.04382).

---

## Q8. Safety requirements for cloning a real person's face and voice (design constraints)

### Takeaway
Treat consent-gated target enrollment, generation-time watermarking of audio and video, and disclosure as hard requirements. The EU AI Act Article 50 requires machine-readable marking by providers and deepfake disclosure by deployers from 2 August 2026. Post-hoc detection is fragile, so watermark at generation.

### Cited Findings
- [A] EU AI Act Art. 50: providers of systems generating synthetic audio or video must mark outputs "in a machine-readable format and detectable as AI-generated". Deployers creating deepfakes must disclose them. The obligations apply from 2 August 2026 — [AI Act Art. 50 guide](https://artificialintelligenceact.eu/transparency-rules-article-50/); [Article 50 text](https://artificialintelligenceact.eu/article/50/)
- [A] Microsoft withheld VASA-1: "no plans to release an online demo, API, product, additional implementation details… until we are confident that the technology will be used responsibly and in accordance with proper regulations" — [VASA-1 project page](https://vasavatar.github.io/VASA-1/)
- [A] AudioSeal: localised (sample-level) watermark detection for AI-generated speech, robust to audio edits, with a fast single-pass detector — [arXiv 2401.17264](https://arxiv.org/abs/2401.17264)
- [A] Detection is brittle: applying discrete OT in embedding space "can transform spoofed speech into samples that are misclassified as bona fide by a state-of-the-art spoofing detector" — [kDOT, arXiv 2505.04382](https://arxiv.org/abs/2505.04382)
- [M] kNN-VC outputs keep some *source*-speaker identity (VPC semi-white-box EER 7.95%). The driving speaker's identity is therefore partly recoverable from outputs, which is itself a privacy consideration — [2409.08913](https://arxiv.org/abs/2409.08913)

### Inferences
- [I] Design constraints for SANG:
  - Enroll X's voice and face only with recorded consent. Bind the consent to the matching set and store matching sets encrypted.
  - Watermark the output audio (AudioSeal) and the video frames; attach C2PA-style provenance metadata.
  - Burn in or overlay an "AI-generated" disclosure by default.
  - Keep generation logs.
  - Do not release per-identity matching sets or LoRAs.
  - Block public-figure targets without verified consent.

### Gaps
- I did not verify a specific video-watermarking paper ID or the C2PA specification in this session.
