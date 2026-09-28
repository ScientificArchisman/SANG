# Steady the start, then add emotion and voice

SANG's start-up jitter comes from a mismatch between training and inference, not from a limit of the model, and fixing it needs no retraining. Emotion control needs one cheap retrain. Voice cloning needs none, because in the right design the target voice never touches the face network. On six unseen-speaker demos, **generated/real pixel acceleration is 2.7× (median) over frames 0–10**, falling from 5.1× at frame 1 to 1.4× at frame 8, then to 0.5–0.8×; there is no spike at the window seam (0.49×). The decay covers exactly the ten sequence positions that received no loss in training, and the old first window put target frames on them. The new default `--start null` reproduces training's dropped-prefix layout, and `--start source` starts the clip from the photo's own motion. Both are implemented and unit-tested but not yet measured. For emotion, add a soft 8-class face-derived pseudo-label plus intensity as a fourth term in the frame-wise AdaLN condition, with FLOAT's nested guidance (audio outer at 2, emotion inner). Keep the emotion push orthogonal to lip aperture, and budget a lip-sync cost that LivePortrait-space systems measure at up to 0.75 Sync-C. For voice, run WavLM-Large once. The last layer of the *original* driving audio drives the face as it does today. Layer 6 feeds kNN-VC against a growing, consented set of the target's clean speech. kNN-VC works frame by frame on the same 20 ms grid, so its output muxes onto the video without re-alignment. It is also the only method with a documented quality gain from seconds to about 5 minutes of target audio without any training. A duration-preserving zero-shot converter covers the first 10–30 s, and a per-speaker fine-tune becomes worth an A/B test beyond about 30 minutes. A full SANG-M training run takes under 20 minutes of step time on one GPU, so the whole programme fits on one GPU. The binding constraints are data (TalkVid's emotion distribution is unmeasured), licences, and consent plus watermarking under EU AI Act Article 50, which has applied since 2 August 2026. Throughout, "measured" marks numbers from SANG's own logs and scripts or from a paper's table, and "inferred" marks reasoning not yet tested on SANG.

## Ten never-supervised positions cause the first-second jitter

SANG-M generated its first window in a layout it had never been trained on. In training, every window has a 10-frame prefix slot at sequence positions 0–9. The slot holds either the clean previous frames or, with probability 0.5, learned null tokens with flow time 0 and null audio. Target frames therefore always sit at positions 10–73. The loss is taken only there, because the forward pass returns outputs from position P onward (`sang/motion_model.py`). At inference the first window of a clip had no prefix slot, so its 64 target frames sat at positions 0–63. Frames 0–9 landed on learned positional embeddings that had never carried a noisy frame or received a gradient, inside a 64-token sequence the model had never seen (training sequences have 74). The training-time evaluation always passed a clean prefix, so this case was never measured. The recipe already covered the first window: FLOAT, whose chaining SANG copies, drops the prefix half the time explicitly "for smooth transition in the initial window" ([FLOAT](https://arxiv.org/abs/2412.01064)). SANG learned the right configuration, and inference did not use it.

SANG's own measurement matches the diagnosis. `scripts/video_jitter.py` computes pixel acceleration |x[t+1] − 2x[t] + x[t−1]| on 128×128 grayscale copies of the real and generated halves of six unseen-speaker side-by-side demos from checkpoint `runs/motion_12k_anneal`. It reports generated/real ratios by frame range (measured, SANG):

| Frames | Generated / real pixel acceleration | Reading |
|---|---|---|
| 1 → 8, per frame | 5.1, 3.6, 2.3, 2.5, 2.0, 2.0, 1.7, 1.4× | Decays across the unsupervised positions |
| 0–10, median | **2.7×** | Visible start-up shake |
| 10 onward | 0.5–0.8× | Steady state |
| 64–74 (seam to window 2) | **0.49×** | No seam spike |

The shake dies out across exactly the ten positions that never received a loss, then stops. A different cause, such as an unlucky initial noise draw, would have no reason to stop at position 10 (inferred). The seam result matters as much. The prefix hand-off between windows already works, so chaining is not the problem, and fixes aimed at seams are aimed at the wrong place. One caveat on the steady-state figure: the real half contains camera and background motion that the generated half, rendered over a static background, lacks. So 0.5–0.8× overstates how calm the generated face is. SANG's separately measured velocity ratio of about 0.9 against ground truth says the steady-state motion is, if anything, slightly under-dynamic.

| Option | What changes | Retraining | Evidence | Verdict |
|---|---|---|---|---|
| **A** `--start null` (default) | First window uses training's dropped-prefix layout: null tokens at 0–9, t = 0, null audio, targets at 10–73 | None | Implemented and unit-tested, not yet run on the cluster. Training used this layout in 50% of windows | Ship first. Predicts frames 0–10 fall into the steady-state band |
| **B** `--start source` | The photo's own 42-d motion (lips closed by LivePortrait's lip normalisation) repeated as a clean 10-frame prefix over 0.4 s of prepended silence | None | Implemented, not yet measured. Motion-space analogue of Ditto's initial-motion anchor and LeapTalk's reference-filled chunks | Run alongside A. Preferred when frame 0 must match the photo |
| **C** Retrain with variable-length prefixes including a 1-frame source anchor, or predict motion as a residual from the source | Training distribution | One run | No direct evidence. FLOAT conditions every frame on the source motion latent | Only if A and B fail, or B shows a freeze |
| **D** Post-hoc smoothing | Output filter | None | Hides the symptom. SANG is already under-dynamic, and over-smoothing damps faces (TT-SAC) | Stopgap only: frames 0–10, non-mouth dimensions |
| **E** Overlapping windows with cross-fade (Ditto) | Window fusion | None | Ditto's fusion is not ablated. No paper compares it with prefix chaining | Does not apply: the first window has no predecessor, and SANG has no seam spike |

A reproduces training exactly, but it leaves frame 0's pose and expression free, so the video need not open on the photo's expression. B opens on the photo, which is what a user uploading a portrait expects. Ditto uses an initial-motion condition plus a loss that ties the first generated frame to it. LeapTalk fills each new chunk with copies of the reference; removing that raised FID from 21 to 217, in a video-space model ([Ditto](https://arxiv.org/abs/2411.19509), [LeapTalk](https://arxiv.org/abs/2608.00079)). B is the training-free, motion-space version of the same idea. It carries two risks (inferred). First, a perfectly still 10-frame hold is rarer in TalkVid than a natural pause, so the model could produce a brief "freeze, then start". Second, a photo with a strong expression turns the first 0.4–1 s into a transition, which will interact with emotion control: a smiling photo plus a "sad" label ramps rather than cuts. C fixes both properly. Training with prefix lengths drawn from 0 to 10, including a single clean source frame, teaches the model to start from the photo without a fake still hold. Predicting the target as a residual from the source's 42-d motion makes "start at the photo" the natural zero. C costs one run but is untested, so it should wait for the A and B numbers. D is a last resort. Beyond frame 10, SANG's motion is already smoother than real video, and TT-SAC reports that larger smoothing windows "over-smooth dynamic facial variations" ([TT-SAC](https://arxiv.org/abs/2605.25488)).

The acceptance gate for A and B has three parts:

- Rerun `video_jitter.py` on the six demos plus about 50 validation clips.
- Compute the same acceleration ratio directly on the 42-d motion, which takes the renderer out of the measurement.
- Accept when frames 0–10 fall inside the frames-10+ band with no LSE-C drop over the first 64 frames.

The literature review of first-window handling was not delivered. The evidence here is therefore SANG's own measurement plus details that appeared in the emotion and naturalness notes, and a dedicated review remains a gap.

## Emotion enters as a fourth condition term, kept away from lip aperture

FLOAT is the closest published analogue to SANG-M. It uses flow matching, frame-wise AdaLN on a summed per-frame condition, and a 10-frame prefix. It adds a 7-d speech-emotion *softmax* (soft probabilities, not an argmax), shared across all frames, to the same condition path. It also adds a nested guidance term, ṽ = v(∅) + γa[v(a) − v(∅)] + γe[v(a,e) − v(a)], with **γa = 2, γe = 1** ([FLOAT](https://arxiv.org/html/2412.01064)).

On RAVDESS, removing the emotion input worsens E-FID from 1.367 to 1.502 and LSE-D from 6.994 to 7.222 (measured). On neutral HDTF the effect is within noise. The most relevant result for SANG, which already runs audio guidance at 2, concerns guidance strength. At γa = 1, raising γe from 1 to 2 worsened LSE-D from 7.049 to 7.212. **At γa = 2, LSE-D stayed at 6.994**: strong audio guidance absorbs the emotion push. FLOAT's user override, replacing the predicted label with a one-hot, is shown only qualitatively, and it works because acted RAVDESS is in the training mix.

Ditto is closest to SANG's data regime: LivePortrait keypoints on about 50 h of broadcast video. It conditions on clip-level HSEmotion pseudo-labels, and at inference converts a user label into the same soft format with a softmax over logit 8 on the chosen class(es), defaulting to Neutral ([Ditto](https://arxiv.org/html/2411.19509v3), [code](https://github.com/antgroup/ditto-talkinghead/blob/main/core/atomic_components/condition_handler.py)). Its "w/o Emo" ablation drops Sync-C from 8.069 to 6.310. But the label is extracted from the evaluated video itself. That shows pseudo-labels explain variance in in-the-wild motion; it does not show they give control.

A user label does override the audio's emotion when a model is trained for it. In Cafe-Talk's swap test, where the label conflicts with the audio, **the output matches the label 57.22% of the time and the audio's emotion 10.86%**, against 59.68% when label and audio agree. Accuracy also rises with intensity: 51.68%, 63.86% and 66.26% at intensities 1, 2 and 3 ([Cafe-Talk](https://arxiv.org/html/2503.14517)). That is 3D blendshape output on acted data. Label dominance is what a user control needs, and it is also the source of the face–voice conflict discussed below.

Most systems that measured it show a lip cost from emotion conditioning. FLOAT, at γa = 2, is the exception. The systems that contain the cost freeze the lip-capable base or restrict where emotion acts:

| System (output space) | How emotion enters | Emotion gain (measured) | Lip-sync change (measured) |
|---|---|---|---|
| Xemo-Talker (LivePortrait 70-d) | Frozen neutral base trained on unlabelled VoxCeleb+HDTF; zero-initialised branch trained on MEAD | Accuracy 15.84% → **85.28%** (real video 85.38%) | LSE-C 6.15 → 6.37, LSE-D 8.02 → 8.21 |
| Playmate (LivePortrait keypoints) | Frozen base; emotion module feeds only the expression head; nested guidance | Emo-A 54.4% at w_e 1.5, not monotonic in w_e | Sync-C 8.141 → **7.395** (w_e 1.5), 6.89 (w_e 3.5) |
| FLOAT (motion latent) | Soft speech-emotion vector in AdaLN; nested guidance | RAVDESS E-FID 1.502 → 1.367 | LSE-D 7.222 → 6.994 (better) at γa = 2 |
| GemTalk (pixel) | Blendshape priors | Accuracy 17.1% → 59.3% | Sync-C 7.125 → 7.027 |
| AUHead (pixel) | Action-unit (AU) adapter | Not reported in this ablation | Sync 6.99 → 6.63 |
| SubtleTalk (FLAME) | Valence–arousal + intensity | Upper-face dynamics error 12.47 → 8.02 | Lip error 14.60 → 15.05; 11.96 with lip prior and decoupling |

Sources: [Xemo-Talker](https://arxiv.org/html/2608.14700v1), [Playmate](https://arxiv.org/html/2502.07203), [GemTalk](https://arxiv.org/html/2608.00663v2), [AUHead](https://arxiv.org/html/2602.09534v2), [SubtleTalk](https://arxiv.org/html/2608.06408v1). Xemo-Talker's setup is structurally SANG's: an emotion-agnostic LivePortrait-motion model trained on unlabelled video, extended on a single RTX 4090. It adds one more measured lever. Applying its emotion-contrastive loss only to the lowest-variance 10% of PCA directions of LivePortrait motion gave **LSE-C 6.51 and 80.51% accuracy, against 6.28 and 75.84% for the whole space**. The principal directions carry jaw opening and head pose. None of these lip-sync differences comes with seeds or error bars, so gaps of 0.1–0.2 may be run-to-run noise.

### Label from the face, and measure TalkVid before choosing classes

Pseudo-labels should come from faces, not voices. HSEmotion/EmotiEffLib, which Ditto used, scores **56.7–59.6% on in-the-wild AFEW clips** and outputs valence and arousal from the same 16 MB model ([EmotiEffLib](https://github.com/av-savchenko/face-emotion-recognition)). Speech recognisers collapse on in-the-wild speech. MEMO's detector scores about 42% on Mandarin TV dialogue (M3ED) while reaching 100% on acted corpora ([MEMO](https://arxiv.org/html/2412.04448v1)), and an emotion2vec probe reaches only 26.98% unweighted accuracy on the same set ([emotion2vec](https://arxiv.org/html/2312.15185v1)). DICE-Talk measures emotion-clustering strength of about 2.4 from audio alone against 6–8 from face video ([DICE-Talk](https://arxiv.org/html/2504.18087v2)). An audio-derived condition scored no better than a one-hot label (48.2% vs 47.6%) ([2509.19749](https://arxiv.org/html/2509.19749v1)).

The inferred recipe:

- Run HSEmotion `enet_b0_8_va_mtl` on cached TalkVid frames.
- Store the window-mean soft 8-d probability vector, plus valence/arousal and a confidence score.
- Treat low-confidence windows as "no label". This also supplies the dropout needed for guidance.

TalkVid's emotion distribution under this labeller is unknown and must be measured before committing to class-based control. If strong negative emotions are rare, a labelled acted set is needed, as FLOAT, DICE-Talk (1:1 mix), Cafe-Talk and SubtleTalk all use. Licences narrow that choice sharply:

- **CREMA-D** (91 actors, ODbL/DbCL) is the only one clearly usable commercially ([CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)).
- **RAVDESS** is CC BY-NC-SA, with paid commercial licences ([RAVDESS](https://zenodo.org/record/1188976)).
- **MAFW** is non-commercial ([MAFW](https://mafw-database.github.io/MAFW/)).
- **MEAD**'s data terms could not be confirmed.

HSEmotion's code is Apache-2.0, but its models are trained on AffectNet, whose licence is research-only, so labelling training data with them needs legal review. Cafe-Talk offers a fallback precedent: its second stage trained on 157 h of unlabelled internet video with *randomly assigned* labels and a masked loss ([Cafe-Talk](https://arxiv.org/html/2503.14517)).

The reference condition will compete with the label. MEMO found that the emotional tone of its output is "largely inferred from the facial expression of the reference image". Its fix draws the reference frame from a different-emotion clip of the same person ([MEMO](https://arxiv.org/html/2412.04448v1)). SANG's reference condition includes a random frame's 42-d motion from the *same* clip, which carries that clip's expression (inferred). The options, in order of preference:

1. Draw the reference from another clip of the same speaker when one exists.
2. Otherwise raise reference dropout on labelled windows.
3. Or subtract the clip's mean expression offset from the reference motion.

### Concrete injection, guidance and constraint order

In code terms (inferred design), the condition becomes c = c_time + c_audio + c_ref + c_emo:

- c_emo = MLP([soft 8-d vector, intensity]), broadcast to every frame, with a learned null token.
- The emotion token has its own independent dropout of about 0.15.

Two training arms are worth running, because a run is cheap:

- **E1** retrains the whole model with the new term, as FLOAT does.
- **E2** freezes the current checkpoint and trains a zero-initialised adapter that adds a condition delta on the 39 expression dimensions only. This mirrors Xemo-Talker's frozen base and Playmate's expression-head routing.

The user sees three modes:

- **"Auto"**: the null emotion token. The model reads emotion implicitly from WavLM, where it is probe-accessible at every depth, peaking at layer 10 with 95.7% ([2501.05310](https://arxiv.org/abs/2501.05310)).
- **A label, or a mixture of labels**, encoded Ditto-style.
- **An intensity knob** that blends the neutral vector toward the label and scales γe.

Each Euler step then runs three forward passes instead of today's two: no audio and no emotion, audio only, and audio plus emotion. The emotion delta γe[v(a,e) − v(a)] is projected onto the complement of the lip-openness readout vector that the bilabial constraint already uses (`sang/naturalness.py`). Emotion can then widen or lift the lip corners, where a smile lives on the same six lip keypoints, but it cannot open or close the mouth to first order. A whole-mouth mask would kill smiles (inferred).

The sampler then computes the clean estimate x0 = x − t·v, applies the existing projections for bilabial closure and scheduled blinks, and re-noises, exactly as today. The constraints are applied last, so they win conflicts. A "surprise" jaw drop cannot open a /p b m/ closure, and "fear"'s wide eyes cannot cancel a scheduled blink on its frames. Xemo-Talker's PCA-tail result supports keeping emotion out of the high-variance articulation directions. No paper has tested per-dimension masking of emotion guidance, so this remains SANG's own experiment.

Two diagnostics bound the achievable result from above and below:

- **Upper bound, from the representation.** SANG predicts 13 of LivePortrait's 21 keypoints; the cheek and nose keypoints stay at the source's values. The cheek raise (AU6) of a Duchenne smile and the nose wrinkle (AU9) of disgust may therefore be capped (inferred). Re-rendering real CREMA-D motion with those 8 keypoints frozen, then classifying the frames, gives the emotion analogue of the ground-truth-motion renderer ceiling.
- **Lower bound, training-free.** Add a per-speaker mean emotion offset in keypoint space, orthogonal to lip aperture and clamped to the dataset range. The literature warns this baseline is weak: EmoVOCA measured naïve offsets at lip error 5.971 against 3.425 for a learned combination, on a partly circular test, and PC-Talk found the mouth "over closed" without its neutral-subtraction decomposition ([EmoVOCA](https://arxiv.org/html/2403.12886), [PC-Talk](https://arxiv.org/html/2503.14295v3)).

## Voice quality grows with every clean minute on a kNN matching set

kNN-VC fits SANG unusually well. It encodes speech with WavLM-Large layer 6, one vector per 20 ms, and replaces each source frame with the mean of its k = 4 nearest frames from the target's pooled "matching set". A HiFi-GAN trained on such matched features (the "prematched" vocoder) then synthesises audio. Layer 6 was chosen because it is "necessary for good speaker similarity and retention of the prosody information from the source utterance" ([kNN-VC](https://arxiv.org/abs/2305.18975)). Output frames map one-to-one onto input frames, on the same 50 Hz grid SANG already computes. With about 8 minutes of target speech, kNN-VC scores **WER 7.36% and EER 37.15%** (maximum 50%; higher means harder to tell from real target speech). FreeVC scores 7.61% and 8.97%, at nearly equal MOS (4.03 vs 4.07). The code is MIT-licensed ([repo](https://github.com/bshall/knn-vc)).

Quality rises with target data and then plateaus. Across papers with different setups, WER goes from 45.9% at 3 s ([Phoneme Hallucinator](https://arxiv.org/abs/2308.06382)) to 32.3% for one 5–10 s utterance ([MKL-VC](https://arxiv.org/abs/2506.09709)) to 7.73% at 1 minute ([2310.08104](https://arxiv.org/abs/2310.08104)). Quality plateaus near 5 minutes. A single-protocol table makes the crossover explicit ([Palindromic VC](https://arxiv.org/abs/2606.08843)):

| Target audio | kNN-VC SIM | kNN-VC WER | Seed-VC SIM | Seed-VC WER | Vevo SIM |
|---|---|---|---|---|---|
| 3 s | 0.380 | 43.0% | 0.455 | 4.1% | 0.510 |
| 10 s | 0.552 | 11.5% | 0.578 | 4.2% | 0.639 |
| 30 s | 0.617 | 4.2% | 0.622 | 2.6% | 0.651 |
| 60 s | 0.631 | 4.6% | 0.630 | 3.3% | 0.332 |

Zero-shot models saturate by about 30 s, and Vevo collapses at 60 s. kNN-VC catches up at 30–60 s and keeps improving toward its roughly 5-minute plateau (measured). So "better with more audio", without training, holds only for the matching-set route. That drives the tiering:

| Clean target audio | Method | Measured anchor | Training |
|---|---|---|---|
| None | Keep the driving voice, or use a consented stock voice. A voice predicted from the face is allowed only as a labelled fallback | Face-conditioned voices reach speaker similarity of only 0.27–0.59 ([Faces that Speak](https://arxiv.org/abs/2405.10272)) | None |
| 3–10 s | Duration-preserving zero-shot converter (Seed-VC v1 timbre-only, or a permissive alternative), or matching-set expansion | Phoneme Hallucinator from 3 s: WER 5.10%, EER 44.62%. MKL-VC from 5–10 s: WER 8.13% | None at run time. Phoneme Hallucinator needs one 6.5 h speaker-independent training; MKL-VC is training-free. Neither repo has a licence file |
| 10–30 s | Run both routes and pick automatically by speaker-verifier cosine, CER and UTMOS | Crossover region in the table above | None |
| 30 s – 5 min | kNN-VC, k = 4, full matching set | kNN-VC ≈ Seed-VC on similarity at 30–60 s | None: one WavLM pass per new clip |
| 5 – 30 min | kNN-VC with k ≈ 8–20. Optionally fine-tune the prematched vocoder on the target (inferred) | 8 min ≈ 5 min. Larger k "may even" help at ≥10 min | None, or a short vocoder fine-tune |
| 30 min – hours | A/B a per-speaker model (Seed-VC fine-tune, or RVC with its retrieval index) against kNN-VC on held-out clips | RVC recommends 10–50 min; no peer-reviewed curve shows when fine-tuning overtakes kNN-VC | Minutes to hours per speaker |

Sources for the tiers: [Seed-VC README](https://github.com/Plachtaa/seed-vc) (fine-tuning from one utterance, 100 steps, about 2 min on a T4) and the [RVC FAQ](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/faq_en.md). The matching set costs about 6 MB per minute in fp16, and search cost grows linearly with it. At hours of audio, a FAISS index keeps retrieval fast (inferred).

The target audio's quality matters more for kNN-VC than for zero-shot models, because kNN-VC re-emits the target's own frames. Noise, reverb or a second speaker in the matching set become output artefacts, and stuttered references carry their disfluencies into the output ([2310.08104](https://arxiv.org/abs/2310.08104)). An enrolment pipeline, with permissive tools throughout, should:

1. Separate out music (Demucs).
2. Detect speech (Silero VAD).
3. Diarise (pyannote) and verify each segment against the consent recording with a speaker verifier.
4. Denoise lightly (DeepFilterNet3). Generative enhancement should stay off the matching set.
5. Normalise loudness and gate on DNSMOS.

"More" should also mean *varied*. kNN-VC can only emit frames the target actually produced, so a matching set of flat read speech will flatten a shouting or laughing driver (inferred).

Language adds a constraint. kNN-VC works across Multilingual LibriSpeech's eight languages at WER 33.9% against a 21.5% topline ([2310.08104](https://arxiv.org/abs/2310.08104)). But with about 10 s of reference in a different language, its WER reaches 96.7%, and it leaks the reference language's phones ([MKL-VC](https://arxiv.org/abs/2506.09709)). SANG covers 15 languages, so cross-language pairs with short matching sets should route to the zero-shot converter.

That converter must preserve duration and prosody. Seed-VC v1 is the strongest open option measured: speaker similarity (SECS) 0.8676 and WER 11.99%, against OpenVoice's 0.7547 and 15.46% ([Seed-VC](https://arxiv.org/abs/2411.09943)). It keeps duration at `length_adjust = 1.0`. But it is GPL-3.0 and archived, and it pulls intonation toward the reference prompt ([REF-VC](https://arxiv.org/abs/2508.04996)). Chatterbox VC (MIT, with a built-in watermark, no published VC metrics) and X-VC (MIT, English and Chinese) are the permissive alternatives. Their exact alignment to SANG's 20 ms grid is unmeasured, so each needs a DTW check: at most one frame of mean deviation.

Anything that regenerates timing is excluded from the lip-synced path: Vevo-Voice, Seed-VC v2 with `convert-style`, GenVC, StableVC and R-VC. Separately, the weights licences of Emilia-trained checkpoints (Seed-VC, X-VC) need checking before commercial use, and Vevo's weights are non-commercial (CC-BY-NC-4.0).

## One WavLM pass drives the face from the original audio and the voice from layer 6

### Only the emotion input changes the network's weights

| Input | Where it enters | Network change | Retraining |
|---|---|---|---|
| (a) Reference image | LivePortrait's source encoder gives appearance features for the frozen renderer. Source keypoints give c_ref (63 canonical-keypoint coordinates plus the photo's 42-d motion), broadcast to all frames. With `--start source` they also form the first prefix | None | None |
| (b) Driving audio | One WavLM-Large pass with all hidden states. The last layer (1024-d, 50 Hz, two ticks per frame, ±80 ms convolution) gives c_audio. Layer 6 gives the kNN-VC queries. The waveform also feeds the IPA phoneme recogniser (bilabial bounds) and pause detection (blink bounds) | None | None |
| (c) Target-voice audio (optional) | Cleaning and speaker verification, then a separate WavLM pass at enrolment. Layer-6 frames are cached as that identity's matching set. It never enters the DiT | None | None up to about 30 min. Optional per-speaker fine-tune beyond |
| (d) Emotion control | Auto, or label(s) plus intensity, becomes a soft 8-d vector, then MLP → c_emo in the AdaLN sum (arm E1) or through a zero-initialised adapter (arm E2). Enters as the inner nested-guidance term, projected off lip aperture | New embedding and null token, or adapter | One run per arm plus a labelling pass |

The shared pass is nearly free. Hugging Face's `WavLMModel` with `output_hidden_states=True` returns layer 6 and the last layer from the same forward pass. Two details need checking (inferred).

First, the prematched vocoder was trained on kNN-VC's own WavLM extractor. Hugging Face's `hidden_states[6]` should be compared numerically with that extractor, aiming for cosine near 1, before the two are trusted to match.

Second, under `--start source` the face pass includes 0.4 s of prepended silence. The 20 lead ticks must be dropped before kNN matching. WavLM attends over the whole utterance, so the padding shifts every frame's features slightly. If that shift is not negligible, the fix is one extra, unpadded encoder pass for the voice branch.

The converted track has exactly the original length, so it muxes onto the rendered video directly.

### Pipeline order: face from the original, voice swapped at the end

```
reference image ─► LivePortrait source encoder ─► appearance (renderer) + c_ref ─────┐
emotion control ─► soft 8-d + intensity ─► c_emo ─────────────────────────────────────┤
driving audio ─► WavLM-Large, one pass ─► last layer ─► c_audio ──────────────────────┤
                        │                                                             ▼
                        │          DiT: 10 Euler steps, nested guidance, bilabial/blink projections
                        │               ─► 42-d motion ─► frozen LivePortrait ─► video frames ──────┐
                        └─► layer 6 ─► kNN-VC ◄── target matching set (layer 6, cleaned, consented) │
                                         └─► prematched HiFi-GAN ─► bandwidth ext. (optional)        │
                                                  ─► AudioSeal watermark ─► mux ◄────────────────────┘
                                                                             └─► + disclosure, C2PA manifest
```

Driving the face from the *original* audio keeps the DiT on its training distribution. Because kNN-VC preserves timing exactly, the lips match the output voice's timing by construction.

Driving the face from converted audio would add two risks SANG does not have today. The first is domain shift. On LRS2, TTS-driven talking heads lost LSE-C: 7.59 → 6.18 for Wav2Lip and 8.41 → 6.30 for PLGAN ([2511.05432](https://arxiv.org/abs/2511.05432)). Another paper measured the opposite direction, with SadTalker rising from 4.978 to 6.256 ([Faces that Speak](https://arxiv.org/abs/2405.10272)). SyncNet itself is unreliable on synthetic audio: JAM-Flow's users preferred a system with LSE-C 3.43 over one with 6.05 in 62.6% of comparisons ([JAM-Flow](https://arxiv.org/abs/2506.23552)). The second risk is that content errors in the voice conversion become lip errors, and kNN-VC's WER nears 46% at 3 s.

Face-from-converted audio should therefore be an opt-in mode. It needs a fine-tune on kNN-converted TalkVid audio paired with the *original* motion; timing is identical, so no alignment step is needed. This follows the real-then-synthetic two-stage recipe of [2511.05432](https://arxiv.org/abs/2511.05432) and the finding that more synthetic speakers help (MSE 0.00420 with one TTS speaker vs 0.00378 with 13) ([2303.05322](https://arxiv.org/abs/2303.05322)). Its possible upside is that WavLM's deep layers still carry speaker cues ([2501.05310](https://arxiv.org/abs/2501.05310)), so the face's voice-implied motion style would match the target. That upside is unmeasured.

An audio-only recording of the target personalises the voice, not the face. Motion personalisation needs video of the target: MimicTalk's 60 s of video matched a person-specific model trained on 180 s ([MimicTalk](https://arxiv.org/abs/2410.06734)).

### A user emotion label drives the face, and timbre-only conversion cannot follow it

**Auto mode.** The face must read emotion from the *original* audio. Deriving it from converted audio would inherit the conversion's losses: emotion recognition (IEMOCAP UAR) falls from 70.07% on original speech to 56.70% after kNN-VC, still far above an ASR→TTS cascade at 30.35% ([2409.08913](https://arxiv.org/abs/2409.08913)). The output voice keeps the driver's timing and prosody, but weakened and limited by what the target's matching set contains. The face will therefore be somewhat more expressive than the voice (inferred). Varied-affect enrolment and an emotion-recogniser agreement check between original and converted audio keep that gap visible.

**A label that agrees with the audio** simply intensifies the face.

**A label that conflicts with the audio** is the hard case. The face follows the label, per Cafe-Talk's 57% vs 11%, while a timbre-only converter keeps the driver's delivery, so face and voice disagree. Converters that change emotion do exist: Vevo-Voice reaches emotion similarity 0.872 against Vevo-Timbre's 0.816 ([Vevo](https://arxiv.org/abs/2502.07243)). But they regenerate timing, which forces the face onto converted audio, the riskier path above.

The defensible default is that the label shapes the face and the voice keeps the driver's delivery. The system should warn when auto-detected audio emotion contradicts the label, and cap γe in that case. An emotion-converting voice can become a later, explicit option once the converted-audio fine-tune exists.

Face–voice mismatch is perceptible: it produces eeriness in viewers ([Mitchell et al.](https://journals.sagepub.com/doi/10.1068/i0415)). Automatic face–voice matchers are too weak to confirm identity; the best FAME 2026 EER is 23.99% ([FAME 2026](https://arxiv.org/abs/2512.04814)). Voice identity should be scored against the target's real voice, with a matcher used only to catch gross gender or age mismatches.

## Gate every phase on its own metric, and ship only with consent and watermarks

### One evaluation battery, reported on both audio tracks

**Start-up jitter.** Report the per-bin pixel acceleration ratio and the renderer-free 42-d acceleration ratio on the demos plus validation clips. Under B, also report frame 0's distance to the photo's motion. All bins must stay flat, including the seam bin.

**Emotion.** Check each of these:

- Accuracy from two independent frame classifiers, such as an Emotion-FAN-style model fine-tuned on MEAD, and a motion-space classifier trained on CREMA-D's 42-d motion. HSEmotion must not be the only judge, since it produced the training labels.
- Real-video and 8-keypoints-frozen ceilings, reported alongside.
- Neutral and non-neutral accuracy reported separately.
- Cafe-Talk's swap test, scored against both the label and the audio.
- Accuracy against intensity and γe, which should rise monotonically.
- LSE-C/D, mouth-opening statistics and bilabial-closure rate for every emotion and every γe.
- A regression check that the null-emotion path reproduces the current model's TalkVid-val numbers.

Published emotion accuracies are not comparable across protocols: DICE-Talk scores 42.03% under one protocol and 49.44% under another ([Xemo-Talker](https://arxiv.org/html/2608.14700v1), [GemTalk](https://arxiv.org/html/2608.00663v2)). SANG must therefore report its own ceiling rows.

**Voice.** The test grid is target-audio amount {5 s, 30 s, 5 min, ≥10 min} × {same, cross} language × {same, cross} gender, on unseen validation speakers. Measure:

- Speaker similarity from two verifiers against held-out real target clips, with a real-vs-real ceiling.
- Whisper-large-v3 CER against the ASR transcript of the original audio (no transcripts needed).
- UTMOS.
- Emotion-recogniser agreement between original and converted audio.
- DTW frame deviation, for any converter outside the kNN family.

**Fused output.** Compute LSE on the same video with the original and with the converted track. Under the recommended pipeline the video is identical, so any difference measures SyncNet's audio sensitivity, not a lip regression. Also test watermark survival through AAC muxing. Finish with a pairwise user study on out-of-domain multilingual audio, covering lip sync, similarity to a real target clip, emotion match, face–voice match and naturalness.

### A phased plan that fits one GPU

The anchor is measured: the current model (53.4M parameters, 12,584 training clips) trained 15k steps at about 15 it/s on one 80 GB-class GPU. That is under 20 minutes of step time (slurm log 172696). Every other cost below is an estimate.

| Phase | Work | GPU cost (est.) | Engineering (est.) | Gate to proceed |
|---|---|---|---|---|
| 0 | Run A and B on the demos plus ~50 validation clips; add 42-d acceleration | < 1 h | ½ day | Frames 0–10 inside the steady-state band, first-window LSE-C unchanged |
| 1 | Voice path: shared WavLM pass, kNN-VC with prematched vocoder, enrolment cleaning, consent gate, zero-shot fallback under 30 s, AudioSeal, mux | Hours of evaluation | 3–4 days | Speaker similarity and CER per tier; LSE on both tracks; watermark survives AAC |
| 2 | Emotion data: HSEmotion pass over the TalkVid cache; distribution report; CREMA-D motion extraction; 8-keypoint ceiling; offset baseline | A few hours, mostly video decoding | 2 days | Enough non-neutral mass, or a CREMA-D mix decided |
| 3 | Emotion model: arms E1/E2 × {TalkVid, +CREMA-D} × reference decoupling; γe sweep with the aperture projection | ~½ GPU-day for the grid | 3 days | Null path unchanged; swap-test accuracy; lip cost within budget |
| 4 (optional) | Face-from-converted robustness: kNN-convert TalkVid to other speakers' sets, re-extract WavLM, fine-tune | Several hours | 2 days | Converted-driven LSE ≈ original-driven LSE |
| 5 (optional) | Option C retrain for the start, only if Phase 0 fails | < 1 h | 1 day | As in Phase 0 |
| 6 | User study; per-speaker fine-tune A/B for targets with ≥30 min | Minutes to hours per speaker | Ongoing | Beats kNN-VC on held-out clips |

Voice ships before emotion because it touches no weights. Phases 0–3 together come to about two weeks of engineering and one to two GPU-days (estimate).

### Consent, watermarking and Article 50 are hard requirements

SANG will synthesise a real person's face and voice. EU AI Act Article 50, applicable since 2 August 2026, requires providers to mark synthetic audio and video "in a machine-readable format and detectable as artificially generated", and requires deployers to disclose deepfakes ([Art. 50](https://artificialintelligenceact.eu/article/50/)).

Post-hoc detection is fragile. Optimal-transport post-processing can make spoofed speech pass as bona fide ([kDOT](https://arxiv.org/abs/2505.04382)). Watermarking at generation should therefore be the primary control. AudioSeal (MIT) localises its watermark at sample level, with **IoU 0.99 against WavMark's 0.35** on partially edited speech ([AudioSeal](https://arxiv.org/abs/2401.17264)). It belongs on the converted waveform after bandwidth extension and loudness normalisation, and before AAC muxing.

The inferred requirements:

- Recorded, logged consent from the target for both face and voice. Any emotion edit should be disclosed as such.
- A live read of a random phrase, verified against the uploaded audio before conversion is enabled. Refuse the job on a verification mismatch.
- Encrypted matching sets bound to the consent record, never released.
- Public-figure targets blocked without verified consent.
- A visible "AI-generated" disclosure plus a C2PA-style provenance manifest on the video.
- Generation logs.

The driver needs protection too. kNN-VC output still reveals the *source* speaker (EER 7.95%) ([2409.08913](https://arxiv.org/abs/2409.08913)).

Several gaps remain. The notes verified no specific video-watermarking method and did not cover non-EU law. They also left licence questions open for HSEmotion's AffectNet-trained models, MEAD, Emilia-trained converters and GPL Seed-VC; each needs legal review before commercial use.

## Conclusion

The jitter diagnosis generalises into a rule for everything added next: every conditioning slot must be exercised at inference in exactly the configuration it was trained in. That includes its *absent* state. The prefix slot failed because its "absent" case at inference (no slot at all) was not its "absent" case in training (a dropped slot). Emotion and voice add new absent states: no label, no target recording, a target recording too short for kNN. So the first test of the emotion model is that the null-emotion path reproduces today's model, and the first test of the voice router is how it behaves at 0 s and 3 s. Evaluation on clean prefixes hid the start-up bug. The same blind spot will hide these bugs unless it is closed on purpose.

The fusion is also asymmetric in a way that decides the order of work. Voice is a post-process that shares an encoder: it inherits lip sync for free, improves with data without training, and can ship first. Emotion is a model change with a measured lip-sync cost and an unmeasured data supply. The deciding unknowns are therefore empirical and cheap to resolve: TalkVid's emotion distribution, the 13-keypoint expressiveness ceiling, kNN-VC's cross-lingual behaviour with short matching sets, and whether SyncNet reacts to converted audio. Resolving them costs hours, not training runs.
