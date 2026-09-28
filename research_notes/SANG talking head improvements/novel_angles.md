# Novel, defensible contribution angles for audio-driven one-shot talking-head generation in motion space (2026-2027 paper)

Search date: 2026-09-28. Tools used: Firecrawl research index (arXiv, date-filtered from 2025-01-01 or 2025-06-01), `inspect_paper` to check IDs, `read_paper` for full-text passages, and one web search. Every arXiv ID below was returned by the index in this session. IDs marked "(inspected)" were also retrieved individually to confirm their metadata.

**Baseline check on the "decomposition is not novel" premise.** It holds. Every listed prior work does audio → compact motion → renderer:
- FLOAT, arXiv 2412.01064 (inspected): flow matching in a learned orthogonal motion latent space. [arXiv](https://arxiv.org/abs/2412.01064)
- KDTalker, arXiv 2503.12963 (inspected): implicit 3D keypoints with spatiotemporal diffusion. [arXiv](https://arxiv.org/abs/2503.12963)
- Ditto, arXiv 2411.19509 (inspected): motion-space diffusion using an off-the-shelf motion extractor. [arXiv](https://arxiv.org/abs/2411.19509)
- Teller, arXiv 2503.18429 (inspected): autoregressive transformer over RVQ tokens of implicit-keypoint motion. [arXiv](https://arxiv.org/abs/2503.18429)
- AVTR-1, arXiv 2609.22913: 153M autoregressive flow-matching motion generator plus a renderer. [arXiv](https://arxiv.org/abs/2609.22913)
- IMTalker, arXiv 2511.22167: flow-matching motion generator. [arXiv](https://arxiv.org/abs/2511.22167)
- DEMO, arXiv 2510.10650: OT flow matching on a disentangled motion latent. [arXiv](https://arxiv.org/abs/2510.10650)
- MoDA, arXiv 2507.03256: flow matching in a joint motion/render parameter space. [arXiv](https://arxiv.org/abs/2507.03256)

**Naming discrepancy.** arXiv 2609.10317 (inspected) is titled "Decoupled Self-Forcing Distillation for Streaming Talking Head Generation" (An et al., 2026-09-09). Its abstract does not use the name "Motar". Check the name before citing it as "Motar". [arXiv](https://arxiv.org/abs/2609.10317)

---

## Angle 1 — Subgroup fairness of motion-space talking-head models (audit + subgroup-robust training)

### Takeaway
TalkVid established that subgroup disparities exist. Its only remedy was more diverse data, tested on one pixel-space diffusion model (V-Express). I found no talking-head paper that does any of the following:
- audits motion-space generators (FLOAT, Ditto, KDTalker and similar) by subgroup;
- separates the generator's disparity from bias in the evaluator itself (SyncNet or FID);
- tests a training objective such as group DRO or reweighting.

This appears open. It is the strongest candidate, especially combined with Angle 3.

### Cited Findings
- **TalkVid scope:** 1,244 h, 7,729 speakers. TalkVid-Bench has 500 clips stratified by language, ethnicity, gender and age, and the paper states it "reveals performance disparities across subgroups that are obscured by traditional aggregate metrics". — [TalkVid, arXiv 2508.13618](https://arxiv.org/abs/2508.13618)
- **TalkVid's experiment is narrow:** it trains one model, V-Express (a pixel-space diffusion model), on HDTF, Hallo3 and TalkVid-Core (a 160 h subset). Each run takes 3 days on 4×A100. Only FID, FVD, Sync-C and Sync-D are reported. — [TalkVid §4.1](https://arxiv.org/html/2508.13618)
- **Language gap (TalkVid-trained model, Table 4):**

  | Language | Sync-C | FVD |
  |---|---|---|
  | English | 4.567 | 357.6 |
  | Chinese | 4.041 | 306.1 |
  | Polish | 3.695 | 288.2 |

  With HDTF training, English Sync-C is 4.000 and Polish is 2.654. — [TalkVid Table 4](https://arxiv.org/html/2508.13618)
- **Age gap (TalkVid-trained model):** the 60+ group is worst, with Sync-C 3.942 and FVD 321.6, against 4.329 and 253.7 for ages 19-30. — [TalkVid Table 4](https://arxiv.org/html/2508.13618)
- **Ethnicity:** FID is 40.7 for White speakers, 44.4 for African and 48.5 for Asian speakers (TalkVid-trained). The paper frames the Hallo3→TalkVid gain on African speakers as "mitigating ethnic bias". — [TalkVid Table 4 and §4.2.1](https://arxiv.org/html/2508.13618)
- **Mitigation tested:** only a change of training data. There is no loss-level fairness objective and no ground-truth (real-video) row per subgroup to calibrate the metrics. — [TalkVid §4](https://arxiv.org/html/2508.13618)
- **General generative-model fairness theory exists but has not been applied to talking heads:**
  - "Equalized Generative Treatment" defines fairness as comparable generation quality across groups (an f-divergence per group). It shows min-max fine-tuning achieves this for image and text generation. — [arXiv 2602.08660](https://arxiv.org/abs/2602.08660)
  - Worst-group loss gave the best performance/fairness balance in a depression-detection study. — [arXiv 2509.25795](https://arxiv.org/abs/2509.25795)
- **Adjacent face and speech tasks already have 2025-2026 fairness audits.** This makes the talking-head gap more visible, not less:
  - gaze estimation — [arXiv 2604.10707](https://arxiv.org/abs/2604.10707)
  - face verification with MLLMs — [arXiv 2603.25613](https://arxiv.org/abs/2603.25613)
  - ASR decoders — [arXiv 2604.21276](https://arxiv.org/abs/2604.21276)
- **The need is acknowledged but not addressed:** a talking-head survey lists "demographic fairness" among open challenges. — [arXiv 2308.16041 (survey, updated)](https://arxiv.org/abs/2308.16041)
- **THEval** evaluates 17 models on 85k videos using a curated real dataset intended "to mitigate bias of training data". I found no subgroup breakdown in its abstract. — [arXiv 2511.04520](https://arxiv.org/abs/2511.04520)

### Inferences
- **Precise gap:**
  - (a) Disparity audit of motion-space models. Decomposing the pipeline lets you attribute disparity to the motion generator or to the frozen renderer: render ground-truth LivePortrait motion and compare it with generated motion, per subgroup. No end-to-end pixel model can do this attribution, so it is a natural contribution *specific to* the motion-space setup.
  - (b) Evaluator-bias control. Compute Sync-C/FID on *real* clips per subgroup and report generated-minus-real gaps. Otherwise a "disparity" may be SyncNet's own language or ethnicity bias.
  - (c) Subgroup-robust flow-matching objective, such as group-DRO over TalkVid metadata or loss reweighting, reporting worst-group error against average error.
- **One-GPU feasibility is high:**
  - motion-space training is cheap;
  - the TalkVid metadata supplies groups;
  - TalkVid-Bench (500 clips) is the test set;
  - the renderer is frozen, so the ablations are motion-only.
- **Crowdedness is low** for talking heads specifically. The risk is that reviewers see it as "applying group DRO". The attribution analysis (motion vs renderer, generator vs evaluator) is what makes it defensible.

### Gaps
- I did not find a TalkVid-Bench leaderboard or follow-up papers reporting per-subgroup results for FLOAT, Ditto or KDTalker. A web search surfaced only TalkVid itself and aggregator pages. This absence is evidence the gap is open, but a CVPR/ICCV 2026 proceedings sweep was not possible with these tools.
- I could not verify whether SyncNet (Chung & Zisserman 2016) has been audited for language or ethnicity bias on real video.

---

## Angle 2 — Cross-lingual / tonal-language lip sync in motion space

### Takeaway
This is partially covered:
- MuEx (2510.06612) builds a 12-language dataset and a phoneme-guided MoE with zero-shot transfer to unseen languages;
- DisentTalk and JoyGen address Chinese data scarcity;
- KoUniTalk (2609.19840) is a Korean-English 3D benchmark.

What appears open:
- a controlled study of *how* motion-space models transfer across languages as a function of training-language mix (TalkVid has 15 languages);
- per-phoneme/viseme error analysis in keypoint space;
- whether tonal languages need prosody-driven non-lip motion (head and brow).

### Cited Findings
- **MuEx:** "Current TFS models perform well in English but struggle with non-English languages". Introduces PG-MoE and PV-Align, and MTFD (12 languages, 95.04 h), and claims zero-shot generalization to unseen languages. — [MuEx, arXiv 2510.06612](https://arxiv.org/abs/2510.06612)
- **DisentTalk:** introduces CHDTF (Chinese HD talking-face dataset) to address Chinese data scarcity; operates in 3DMM parameter space. — [arXiv 2503.19001](https://arxiv.org/abs/2503.19001)
- **JoyGen:** builds a 130 h Chinese talking-face dataset. — [arXiv 2501.01798](https://arxiv.org/abs/2501.01798)
- **KoUniTalk (Sep 2026):** a Korean-English 3D articulation benchmark (22 speakers, 4,978 sequences) that enables cross-language evaluation in a shared mesh space. — [arXiv 2609.19840](https://arxiv.org/abs/2609.19840)
- **BioLip** trains lip-sync deepfake detection on English only and tests zero-shot on 7 languages. This shows kinematic lip features can be language-general. — [arXiv 2604.16808](https://arxiv.org/abs/2604.16808)
- **TalkVid** reports English > Chinese > Polish Sync-C for every training set. — [TalkVid Table 4](https://arxiv.org/html/2508.13618)
- **Phoneme-aware encoders already improve sync** (English-centric evaluation):
  - PASE — [arXiv 2504.05803](https://arxiv.org/abs/2504.05803)
  - PD-GS — [arXiv 2608.05218](https://arxiv.org/abs/2608.05218)

### Inferences
- **Open gap:** a "cross-lingual transfer curve" for motion-space models. Train on English-only, then English plus k languages, then all 15 TalkVid languages. Measure per-language lip-keypoint error and sync on held-out languages, including leave-one-language-out, and relate the results to phoneme-inventory overlap (e.g., via PHOIBLE-style inventories).
- A LivePortrait keypoint model lets you measure lip-aperture and closure errors directly in motion space, avoiding SyncNet's language confound.
- **One GPU:** yes. The main cost is extracting motion for language subsets.
- **Crowdedness:** moderate for methods (MuEx) and low for controlled transfer analysis.

### Gaps
- I did not read MuEx's full text to check whether it already reports a training-language-mix ablation or phoneme-inventory analysis. This must be checked before claiming novelty.
- I found no talking-head paper studying lexical tone specifically. The one Mandarin-tone paper found is lip-to-speech, the inverse task. — [LTA-L2S, arXiv 2509.25670](https://arxiv.org/abs/2509.25670)

---

## Angle 3 — Evaluation: beyond Sync-C/Sync-D; a motion-space sync metric validated against humans

### Takeaway
This area is crowded, with 5+ 2025-2026 papers on talking-head evaluation or learned sync. It is still open on two narrow points:
- a sync/naturalness metric computed *directly on motion keypoints* (renderer-independent) that is validated against human judgement on in-the-wild multilingual data;
- metric *fairness*, meaning whether the metrics themselves behave differently across subgroups.

### Cited Findings
- **THEval:** 8 metrics across quality, naturalness and synchronization, chosen for efficiency and alignment with human preference. Evaluates 85,000 videos from 17 models and finds that "many algorithms excel in lip synchronization" but struggle with expressiveness. — [arXiv 2511.04520](https://arxiv.org/abs/2511.04520)
- **Temporally-Aligned Evaluation:** argues frame-wise metrics wrongly penalise timing shifts. Reformulates evaluation with Soft-DTW and benchmarks 20 methods on 7 datasets. — [arXiv 2606.01031](https://arxiv.org/abs/2606.01031)
- **UniSync (audio-visual sync):** a learned sync evaluator that accepts landmarks, 3DMM and face-parsing inputs as visual representations. **This is close prior art for a keypoint-space sync metric.** — [arXiv 2503.16357](https://arxiv.org/abs/2503.16357)
- **SyncLipMAE:** a contrastive, sync-aware representation with token-level A/V sync. — [arXiv 2510.10069](https://arxiv.org/abs/2510.10069)
- **Perceptually Accurate 3D Talking Head:** a learned speech-mesh representation used as both perceptual metric and loss, plus physically grounded lip metrics. This is the 3D-mesh analogue of a motion-space metric. — [arXiv 2503.20308](https://arxiv.org/abs/2503.20308)
- **FlowPortrait:** says existing metrics "correlate poorly with human perception" and uses an MLLM-based human-aligned evaluator as an RL reward. — [arXiv 2603.00159](https://arxiv.org/abs/2603.00159)
- **AVTR-1:** introduces R-DGG (Granger-gain) to test whether speaker speech actually drives listener motion. This is a motion-space statistical metric of conditioning dependence. — [arXiv 2609.22913](https://arxiv.org/abs/2609.22913)
- **KeySync:** introduces LipLeak for expression leakage. — [arXiv 2505.00497](https://arxiv.org/abs/2505.00497)
- **Syncphony:** proposes CycleSync, a video-to-audio reconstruction sync metric. — [arXiv 2509.21893](https://arxiv.org/abs/2509.21893)
- **VSR models versus human lipreaders:** VSR models rely on language priors more than visual cues. This caution applies to lipreading-based sync metrics. — [arXiv 2606.07435](https://arxiv.org/abs/2606.07435)

### Inferences
- A generic "learned keypoint-space sync metric" is **partially covered** (UniSync accepts landmarks; the 3D speech-mesh metric exists).
- **Defensible gap:**
  - (i) a renderer-agnostic motion metric suite on LivePortrait keypoints, validated by a human study on TalkVid-Bench across languages and ethnicities;
  - (ii) showing Sync-C's subgroup bias on real video, i.e. metric fairness (see Angle 1).

  This pairs well with Angle 1 as one paper: "Auditing fairness of talking-head generation requires fair metrics."
- **One GPU:** yes. The metric is a small contrastive model on keypoints. A human study of about 20-40 raters over roughly 200 clips is feasible.

### Gaps
- I did not verify whether THEval or 2606.01031 report per-language or per-ethnicity metric behaviour. Their abstracts do not mention it.

---

## Angle 4 — Calibrated one-to-many generation (diversity vs accuracy, uncertainty)

### Takeaway
Diversity is widely claimed. KDTalker's title is "Unlock Pose Diversity", and 3DiFACE, THUNDER and MoDA all make diversity claims. There is also an uncertainty-learning paper in pixel space (JULNet). What appears open for 2D motion-space talking heads:
- *calibration*: does the spread of sampled motions match the real variability of human motion conditioned on the same audio?
- a diversity-accuracy Pareto protocol, using TalkVid's multiple speakers and clips as the empirical conditional distribution.

### Cited Findings
- **THUNDER:** frames the core tension — "deterministic models produce high-quality lip-sync but without rich expressions, whereas stochastic models generate diverse expressions but with lower lip-sync quality". Uses analysis-by-audio-synthesis (3D mesh). — [arXiv 2504.13386](https://arxiv.org/abs/2504.13386)
- **3DiFACE:** diverse lip and head motions per audio, with "control between high fidelity and diversity" (3D). — [arXiv 2509.26233](https://arxiv.org/abs/2509.26233)
- **KDTalker** targets pose diversity in implicit-keypoint space. — [arXiv 2503.12963](https://arxiv.org/abs/2503.12963)
- **JULNet** learns pixel-level error/uncertainty maps for talking faces, not sample-diversity calibration. — [arXiv 2504.18810](https://arxiv.org/abs/2504.18810)
- **Head motion from gaze** is modelled with a CVAE for diverse outputs. — [arXiv 2605.25810](https://arxiv.org/abs/2605.25810)

### Inferences
- **Gap:** evaluate generators as *probabilistic forecasters* of motion, using proper scoring rules (energy score or CRPS on keypoint trajectories), coverage of real takes, and a lip (low-entropy) versus head-pose (high-entropy) split.
- **Method hook:** region-dependent guidance or temperature in flow matching, e.g. low noise or high CFG for lip keypoints and higher spread for pose.
- **One GPU:** yes. Sampling many takes is cheap in motion space.
- **Crowdedness:** low for calibration; moderate for "diverse generation".

### Gaps
- I did not search for motion-forecasting calibration literature (e.g., human motion prediction with energy scores). A related-work sweep is needed.

---

## Angle 5 — Few-second personalization / test-time adaptation in motion space

### Takeaway
This is crowded. There are many 2025-2026 personalization papers, both 3DGS/NeRF few-shot and motion-style-from-reference:
- MirrorTalk (2601.22501) distils style from a brief reference video;
- Fallingwater (2604.23692) does retrieval-based personalization from casual clips;
- TT-SAC (2605.25488) is test-time conditioning adaptation on FLOAT and others.

What might remain: a principled study of *how many seconds* of target video are needed in LivePortrait keypoint space, using LoRA or adapter adaptation of a flow-matching prior. This is a weak standalone contribution.

### Cited Findings
- **MirrorTalk:** semantically disentangled style encoder from a brief reference video, with a diffusion model. — [arXiv 2601.22501](https://arxiv.org/abs/2601.22501)
- **Fallingwater:** causal, zero-lookahead personalized facial motion via a multi-modal style retriever over "a handful of casually recorded clips". — [arXiv 2604.23692](https://arxiv.org/abs/2604.23692)
- **TT-SAC:** a training-free test-time conditioning loop evaluated on FLOAT, AniTalker and Sonic. — [arXiv 2605.25488](https://arxiv.org/abs/2605.25488)
- **Few-second 3D personalization:**
  - InsTaG — [arXiv 2502.20387](https://arxiv.org/abs/2502.20387)
  - EmoTaG — [arXiv 2603.21332](https://arxiv.org/abs/2603.21332)
  - FIAG — [arXiv 2506.22044](https://arxiv.org/abs/2506.22044)
- **ISExplore:** a well-chosen few-second segment can match minutes of reference. — [arXiv 2511.07940](https://arxiv.org/abs/2511.07940)
- **3D style personalization:**
  - PTalker — [arXiv 2512.22602](https://arxiv.org/abs/2512.22602)
  - StyleSpeaker — [arXiv 2503.09852](https://arxiv.org/abs/2503.09852)
  - MemoryTalker — [arXiv 2507.20562](https://arxiv.org/abs/2507.20562)

### Inferences
- **Crowdedness:** high. Pursue only as a secondary experiment, e.g. "does personalization close subgroup gaps?", tying it to Angle 1.

### Gaps
- I did not confirm whether any paper does LoRA adaptation of a LivePortrait-keypoint flow model specifically. None surfaced.

---

## Angle 6 — Data-scaling behaviour of audio-to-motion (hours × speakers)

### Takeaway
This appears open. I found no paper reporting a scaling law or controlled scaling curve for audio-to-motion (audio → keypoint) generators. Large-dataset papers report single training points, not curves:
- TalkVid (1,244 h);
- TalkVerse (6.3k h);
- VividHead (782 h, in SoulX-FlashHead).

### Cited Findings
- **TalkVerse:** 6.3k hours and 2.3M clips; a 5B DiT baseline. No scaling-curve claim in the abstract. — [arXiv 2512.14938](https://arxiv.org/abs/2512.14938)
- **SoulX-FlashHead:** introduces VividHead, 782 h. — [arXiv 2602.07449](https://arxiv.org/abs/2602.07449)
- **TalkVid** trains on a 160 h "Core" subset only. No hours or speakers sweep. — [TalkVid §4.1](https://arxiv.org/html/2508.13618)
- **ISExplore** (per-identity data efficiency in 3DGS): informativeness matters more than duration. — [arXiv 2511.07940](https://arxiv.org/abs/2511.07940)
- **Hybrid knowledge distillation** addresses small-data 3D facial animation. — [arXiv 2507.18352](https://arxiv.org/abs/2507.18352)

### Inferences
- **Gap:** a scaling study separating *hours per speaker* from *number of speakers* from *language/demographic diversity*, at fixed compute, in motion space. Report how lip error, pose FD and worst-subgroup error scale.
- The worst-subgroup axis links to Angle 1: "does diversity or quantity reduce disparity?" This is a question TalkVid raised but tested with only one data point.
- **One GPU:** feasible only because motion-space models are small (AVTR-1 is 153M; your model is presumably similar). Each run is hours, not days. Use hash-based speaker splits, which your repo already has (commit 95d75b1).
- **Crowdedness:** low.

### Gaps
- My search terms may have missed a scaling appendix in large-model papers such as OmniHuman or Wan-S2V. These were not checked.

---

## Angle 7 — Text/caption-controlled and disentangled control in keypoint space

### Takeaway
This is crowded. Several papers use text descriptions or captions for emotion and style:
- EmoCAST, CapTalk, SynchroRaMa and JAM-Flow;
- 2609.10317 routes "motion captions" into a motion-space generator;
- DEMO, EDTalk++ and MoCoTalk handle disentangled control.

Not recommended as the headline contribution.

### Cited Findings
- **2609.10317** fuses "audio and motion captions by their temporal granularity" in an identity-disentangled motion space. **This directly overlaps caption-controlled motion-space generation.** — [arXiv 2609.10317](https://arxiv.org/abs/2609.10317)
- **CapTalk:** text-described style and emotion control with dynamic emotion changes (3D). — [arXiv 2605.29316](https://arxiv.org/abs/2605.29316)
- **EmoCAST:** text-driven emotional talking portraits plus an in-the-wild emotional dataset with text. — [arXiv 2508.20615](https://arxiv.org/abs/2508.20615)
- **JAM-Flow:** joint audio-motion flow matching conditioned on text, audio or motion. — [arXiv 2506.23552](https://arxiv.org/abs/2506.23552)
- **Disentangled factors:**
  - DEMO: disentangled lip/pose/gaze flow matching — [arXiv 2510.10650](https://arxiv.org/abs/2510.10650)
  - EDTalk++: four orthogonal spaces — [arXiv 2508.13442](https://arxiv.org/abs/2508.13442)
  - MoCoTalk: multi-condition router — [arXiv 2605.08050](https://arxiv.org/abs/2605.08050)
- **Ditto** already maps motion representation to facial semantics for control. — [arXiv 2411.19509](https://arxiv.org/abs/2411.19509)
- **ConsistTalk** does frame-wise intensity control. — [arXiv 2511.06833](https://arxiv.org/abs/2511.06833)

### Inferences
- **Crowdedness:** high. The only residual niche is *demographic-conditioned control used as a fairness tool*, e.g. checking that controls work equally well across subgroups. That belongs under Angle 1.

### Gaps
- None critical.

---

## Angle 8 — Safety: watermarking / detectability of generated *motion*

### Takeaway
Proactive defenses against talking-head generation and audio-visual deepfake detection are active areas. Examples:
- Silencer, SyncBreaker, and a psychoacoustic audio defense (Aug 2026);
- BioLip detects lip-sync fakes from landmark kinematics.

I found **no work watermarking the motion trajectory itself**, i.e. embedding a signal in generated keypoints that survives the frozen renderer and re-extraction. This appears open, but it is a niche side contribution.

### Cited Findings
- **Silencer:** adversarial nullification of audio control in LDM talking heads. — [arXiv 2506.01591](https://arxiv.org/abs/2506.01591)
- **SyncBreaker:** joint image and audio perturbation. — [arXiv 2604.08405](https://arxiv.org/abs/2604.08405)
- **Audio-domain psychoacoustic defense** for 3D talking faces. — [arXiv 2608.30951](https://arxiv.org/abs/2608.30951)
- **Protective perturbations fail** under sequential real-world transforms. — [arXiv 2604.23688](https://arxiv.org/abs/2604.23688)
- **BioLip:** generators produce elevated velocity, acceleration and jerk variance in 64 perioral landmarks. A landmark-only detector trained on English generalizes zero-shot to 5 unseen generators and 7 languages. — [arXiv 2604.16808](https://arxiv.org/abs/2604.16808)
- **PIA:** phoneme-temporal and identity-dynamic deepfake detection. — [arXiv 2510.14241](https://arxiv.org/abs/2510.14241)
- **SAiW:** a source-attributable invisible *pixel* watermark. — [arXiv 2603.23178](https://arxiv.org/abs/2603.23178)

### Inferences
- **Gap:** a "motion watermark" embedded in keypoint trajectories, e.g. low-amplitude structured perturbation in head-pose spectra. Verify it after render → re-extract with the LivePortrait motion extractor, and under compression.
- Alternatively: measure whether your model's outputs show BioLip-style kinematic signatures and whether flow matching reduces them. This is a detectability audit.
- **One GPU:** yes.
- **Crowdedness:** low for motion watermarking, moderate for detection.

### Gaps
- Watermarking for 3D/skeletal *motion generation*, outside talking heads, was not searched.

---

## Overall ranking (for the report writer)

### Takeaway
The most defensible and least crowded package is Angles 1+3 (+6):

> **"Who does motion-space talking-head generation fail? A metric-controlled subgroup audit, renderer-vs-motion attribution, and subgroup-robust flow matching on TalkVid."**

Scaling-by-diversity curves (Angle 6) are the secondary result. Cross-lingual transfer (Angle 2) and calibrated diversity (Angle 4) are strong alternatives. Personalization (5) and text control (7) are crowded. Motion watermarking (8) is open but niche.

### Cited Findings
- TalkVid explicitly calls for "further research into auditing and mitigating bias in generative video models". Its own mitigation is data only. — [TalkVid §6](https://arxiv.org/html/2508.13618)
- EGT shows min-max (worst-group) fine-tuning balances per-group generation quality in image and text models. There is no video or talking-head application. — [arXiv 2602.08660](https://arxiv.org/abs/2602.08660)

### Inferences

| Angle | Crowdedness | Novelty status | 1-GPU feasibility |
|---|---|---|---|
| 1 Subgroup fairness audit + robust training | Low | Appears open (searched fairness/talking-head, TalkVid-Bench follow-ups, group DRO in generative models) | High |
| 2 Cross-lingual transfer | Moderate | Partially covered by MuEx 2510.06612, KoUniTalk 2609.19840 | High |
| 3 Motion-space metric + metric fairness | High (general) / Low (metric fairness) | Partially covered by UniSync 2503.16357, THEval, 2606.01031, 2503.20308 | High |
| 4 Calibrated one-to-many | Low–moderate | Appears open for calibration; diversity itself covered | High |
| 5 Few-second personalization | High | Largely covered (MirrorTalk, Fallingwater, TT-SAC) | High |
| 6 Scaling curves | Low | Appears open (searched scaling/data-size/talking-head) | Medium |
| 7 Text/caption control | High | Covered (2609.10317, CapTalk, EmoCAST, JAM-Flow, DEMO) | High |
| 8 Motion watermarking | Low | Appears open; detection side covered (BioLip) | High |

### Gaps
- There was no direct access to CVPR/ICCV/NeurIPS/ACM MM 2026 proceedings, so conference-only papers without arXiv versions may be missed.
- The index returned empty results for several long queries (retried with shorter ones). Coverage may be incomplete.
- The Firecrawl account reported low credits during the session.
