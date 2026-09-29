# SANG-M research log

**Living document: the single place for the plan, results, commands, ideas and references.**
Last updated 2026-09-29 (calibration done; voice path coded). Update it whenever a run finishes, a decision is taken or an idea is added.

Conventions:
- **measured** means a number from our own logs or scripts, with the job ID or file.
- **paper** means a number from a paper's table (arXiv ID given).
- **idea** means untested for SANG.
- Detailed evidence lives in the reports and notes listed in §11. This log summarises them and points to them.

---

## 0. Status at a glance (2026-09-29)

| Item | State |
|---|---|
| Best checkpoint | `runs/motion_12k_anneal/best.pt` (job 172696): val flow loss 1.1460 @ step 12k, audio_gain 1.50 |
| Renderer ceiling (M0) | Passed: mouth corr 0.871, mouth amp 0.963, CSIM 0.906, stitching off (job 171519) |
| Start-up jitter | **Fixed**, commit `a911f42`. Frames 0–10: 2.7× → 0.65× real (`--start null`) / 0.44× (`--start source`). §3.4 |
| Naturalness rules (lip closure, blinks) | Code in `f88d50d`; **`naturalness.py` next** (then the guided demo) |
| Openness calibration | **Done** (job 173053, 80 clips / 3,937 frames): lip R² 0.763, eye R² 0.706 (borderline: it ranged 0.61–0.72 while clips were added; treat blink numbers as rough) |
| Background warping | **Diagnosed** (jobs 173579–84): the renderer leaks ANY keypoint motion into the background (real 70-d: 1.25x; 42-d: 1.5x; +real s,t: 1.89x; generated demos 2.4x). The held-keypoint convention changes nothing (2.36x → 2.41x). Stitching held real motion still (0.95x) and costs ~0 on the 42-d target → **next: `--stitch` test**; then slot-0 background lock / compositing. See `reports/SANG background warping fix.md` (Update section) |
| HDTF benchmark | **Not started**. Blocked on: can the login node reach YouTube, or is HDTF already downloaded? |
| Voice cloning | **Coded** (kNN-VC + voice bank + `--voice`; 9 CPU tests); needs `install_voice.sh`, then `eval_voice.py` |
| Emotion control | Researched, design fixed (§7.4); **not coded**; first step is a data check |

**Next runs, in order** (full commands in §6):
1. `calibrate_openness.py`, then `naturalness.py`, then `demo_motion.py --guide both`.
2. Pick the default start mode (`null` or `source`) by watching `results/extras/demo_fix`.
3. Code and run the voice path (§7.3) and the emotion data check (§7.4).
4. HDTF protocol (§7.7).

---

## 1. Decisions taken (do not reopen without new evidence)

| Date | Decision | Why |
|---|---|---|
| 2026-09-23 | **Bidirectional, non-streaming** generator: whole 64-frame windows, no causal mask | User decision; streaming is not a goal |
| 2026-09-23 | **Motion space + frozen LivePortrait renderer**, inspired by LivePortrait / KDTalker / Ditto / FLOAT | Renderer trained on 69M frames; we can't match that |
| 2026-09-23 | Noisy trajectory is the transformer input (DiT). **No** per-token MLP diffusion head | Per-token head factorises frames: ∂v_i/∂x_j = 0 (tested) |
| 2026-09-23 | 42-d target: 3 head angles + 39 expression coords of EXPR_KP, head frame | AVTR-1 §2.1 + upstream code; M0 passed with it |
| 2026-09-23 (M0 v1, jobs 171503/4) | Crop = one fixed, unrotated box per clip (upstream *driving* convention) + shot-cut truncation | Per-frame source cropping caused "camera sway" (M0 v1, 14.7 dB) |
| M0 v2 → v3 | M0 gate = motion fidelity + CSIM, **not** PSNR | TalkVid freeze-frame baseline is already 15–16 dB, so the PSNR gate was invalid |
| M0 v3 (jobs 171518/9) | Stitching **off** at render | Stitching cost 5.4 dB in M0 (jobs 171518 vs 171519) |
| commit `95d75b1` | Speaker split by **hash**, stable as the cache grows | Val speakers never move to train |
| Before M0 | "Replicate SOTA first, paper later" | User |
| 2026-09-28 | **No audio watermarking** (AudioSeal removed from the plan) | User decision |
| Workflow | Push from the laptop only; pull on the cluster. Never push from the cluster | Cluster git auth goes through a stale VS Code askpass |

---

## 2. The system as built

### 2.1 Pipeline and shapes (one clip, n frames at 25 fps)

```
source photo ─► LivePortrait crop (DRIVE: scale 2.2, vy −0.1) ─► motion extractor
               ├─ x_c canonical keypoints [63]  ─┐
               └─ source motion m_src [70] ─► to_target ─► [42] ─┤ ref = norm([x_c, y_src]) [105]
                                                                  │
audio 16 kHz ─► WavLM-large last layer [2n, 1024] (50 Hz, 2 ticks/frame)
               └─ fold ticks ─► [n, 2048] ─► Conv1d k5 (±2 frames = ±80 ms) ─► [n, 512] = c_audio
                                                                  │
DiT (MotionFlowTransformer, 53.4M params):
   tokens = Linear(42→512)(x_t) + pos_emb (max 512) [+ prefix_tag on prefix frames]
   condition per frame c = time_emb(t) + c_audio + MLP(ref)           (frame-wise adaLN-Zero)
   8 × DiTBlock(dim 512, heads 8, SwiGLU, full bidirectional attention, dropout 0.1)
   out: adaLN → Linear(512→42) = velocity v [n, 42]
   window: 10 clean prefix frames (positions 0–9) + 64 target frames (positions 10–73)
sampling: rectified flow, t = 1 → 0, 10 Euler steps, audio-only CFG 2.0, EMA 0.995 weights
   optional bounds (lip closure, blinks) projected on x̂0 at every step (sang/naturalness.py)
long clips: windows of 64 new frames, each conditioned on the last 10 generated frames
   first window: --start null (dropped-prefix null tokens) | --start source (photo motion + 0.4 s silence)
y [n, 42] ─► from_target(y, m_src) [n, 70] (scale, translation, 8 shape keypoints from the source)
          ─► LivePortrait warp + SPADE (frozen, relative=False, stitch off) ─► 512×512 frames ─► mux original audio
```

- **Target layout:**
  - 42 = (pitch, yaw, roll) + head-frame expression of EXPR_KP = (1, 2, 6, 11–20).
  - Regions: rot [0, 1, 2]; brow (kp 1, 2) 6 cols; eyes (kp 11, 13, 15, 16, 18) 15 cols; mouth (kp 6, 12, 14, 17, 19, 20) 18 cols.
  - Keypoint transform: x = s·(x_c R + δ) + t, with R = (Rz·Ry·Rx)ᵀ in degrees (upstream `camera.py`).
- **Loss:** rectified flow, z_t = (1−t)x0 + tε, v = ε − x0, t ~ U(0,1). Region-balanced MSE (rot / brow / eyes / mouth, one share each) + velocity loss on x̂0 differences (λ_vel = 1).
- **Condition dropout:** audio 0.1, ref 0.1, prefix 0.5. A dropped prefix = null tokens + null audio + t = 0.
- **Optimiser:**
  - AdamW, lr 3e-4 cosine → 1e-5, warmup 5000 (2000 in runs 3–4), wd 0.05, betas (0.9, 0.99), grad clip 1.
  - Batch 256, bf16, EMA 0.995.
  - Eval every 1000 steps, patience 8.
- **Output:** audio is **not** changed. The output video carries the input audio (re-encoded to AAC, trimmed with `-shortest`).

### 2.2 Data

- **Source:** TalkVid (in the wild, 15 languages, ~7.7k speakers available); clip list `data/clips_filtered_all.txt`.
- **Cache:** `cache/motion_lp/<speaker>/<clip>.pt` holds:
  - `m` [T,70] fp16;
  - `kp` [T,63];
  - `audio` [2T,1024] fp16 (WavLM-large);
  - `n`.
  - Indexes are `index_{shard:03d}.jsonl` with rows `{clip, path, n, start}`.
- **Cached so far:** ~12.6k clips (run 4 used 12,584). `--max-frames 250` for later shards (default 500), `--min-frames 74`.
- **Normalisation:** `cache/motion_lp/norm42.pt` (fitted on train speakers).
- **Split:** hash of speaker name, val_frac 0.05. Run 3 had 11,477 train / 412 val clips from 70 val speakers.
- **Throughput:** ~90–125 s per clip (landmarks + decode dominate). Preloading into RAM is needed for training: BeeGFS per-sample reads capped run 1 at 1.4 it/s; preloaded runs reach ~15 it/s.

### 2.3 Code map

| File | What it does |
|---|---|
| `sang/motion.py` | Frozen LivePortrait codec (`MotionCodec`: landmarks, fixed-box crop, extract, source_motion with lip normalisation, render), pure-torch transforms (`to_target`, `from_target`, rotation), `motion_fidelity`, `first_cut`, `decode_clip` |
| `sang/motion_model.py` | `MotionFlowTransformer`, `flow_loss`, `sample` (Euler + CFG + bound projection + `prefix_keep`), `generate` (windows, `start`, bounds), `project`, `Norm`, `build` |
| `sang/naturalness.py` | Landmark lip/eye ratios, linear `Readout`s, phoneme recogniser (wav2vec2 XLSR-53 espeak), bilabial events, pauses, blink detector, beat alignment, closure/blink bounds, blink scheduler, `Guide`, `guided_generate` |
| `sang/bench.py` | PSNR, SSIM, CSIM (ArcFace buffalo_l), LSE via syncnet_python, FID, `write_mp4` |
| `scripts/motion_ceiling.py` | M0: extract → render → re-extract; gates |
| `scripts/cache_motion.py` | Sharded, resumable motion + WavLM cache |
| `scripts/train_motion.py` | Training with preload, hash split, eval ratios (std_r, vc_r, audio_gain), early stop |
| `scripts/infer_motion.py` | Photo + audio → mp4 (`--guide`, `--start`, `--cfg`, `--steps`, `--stitch`, `--raw`) |
| `scripts/demo_motion.py` | Real‖generated side-by-side on unseen val speakers (`--guide`, `--start`) |
| `scripts/calibrate_openness.py` | Fits lip/eye openness readouts → `openness.json` (resumable) |
| `scripts/naturalness.py` | Rule metrics, real vs none/lips/blinks/both → `naturalness_stats.json` + `results/naturalness/*.json` |
| `scripts/video_jitter.py` | Start-up jitter: per-frame-range pixel-acceleration ratio gen/real |
| `sang/voice.py` | Voice cloning: kNN-VC loading, `VoiceBank` (enrol → VAD → 3 s chunks → speaker filter → layer-6 frames), `convert`, `knn_features`, `SpeakerEncoder`, CER |
| `scripts/enroll_voice.py` | Build or grow `voices/<name>/` from a person's recordings |
| `scripts/eval_voice.py` | Speaker similarity to the target vs bank size (5/10/30/60 s/all), leakage, timing, optional Whisper CER, on unseen val speakers |
| `bash_scripts/install_voice.sh` | Login node: clone kNN-VC, cache its weights, fetch WavLM-base-plus-sv (`--asr`: Whisper-large-v3) |
| `tests/test_motion_model.py`, `tests/test_naturalness.py` | 23 CPU tests (all pass) |
| `configs/train_motion.yaml` | All hyperparameters, each with its source |

---

## 3. Measured results

### 3.1 M0: renderer ceiling on TalkVid (re-extract motion from the render)

Gates: mouth corr ≥ 0.85, mouth amp 0.75–1.25, CSIM ≥ 0.85, LSE-C within 0.5 of real.

| Job | Setting | Mouth corr | Mouth amp | CSIM | Verdict |
|---|---|---|---|---|---|
| 171518 | stitch on | 0.852 | 0.958 | 0.9025 | pass (−5.4 dB vs 171519) |
| **171519** | **stitch off** | **0.871** | **0.963** | **0.9058** | **pass** |
| 171520 | `--lse` | 0.869 | 0.934 | 0.9025 | LSE = nan (syncnet path / weights). **LSE never measured yet** |

History:
- v1: 14.7 dB PSNR, caused by the per-frame source crop.
- v2: ~16 dB, but the freeze-frame baseline is already 15–16 dB, so the PSNR gate was abandoned.

### 3.2 Training runs

| Job → dir | Clips | Change | Result |
|---|---|---|---|
| 172260 | 3,678 | no dropout, 60k steps, no preload | best val 1.418 @5k, then overfit to 6.67; 1.4–2.2 it/s |
| 172386 | 3,685 | dropout 0.1, eval every 1k, patience 8, preload | best 1.196 @5k (old split), early stop @13k; 15 it/s |
| 172630 → `runs/motion_12k` | 11,477 train / 412 val / 70 val speakers | hash split; `max_steps=40000 warmup_steps=2000` | best 1.154 @10k, plateau; audio_gain ≈ 1.49–1.51 |
| **172696 → `runs/motion_12k_anneal`** | 12,584 | `max_steps=15000 warmup_steps=2000 patience=0` | **best 1.1460 @12k** |

Best-checkpoint eval (step 12000, job 172696, unseen speakers, clean-prefix windows):

| Metric | Value | Meaning |
|---|---|---|
| val_loss | 1.1460 | Flow loss; not a quality measure by itself |
| mse_mouth / mse_rot | 0.5045 / 0.5115 | Sample error (normalised units) |
| std_r_mouth / vc_r_mouth | 1.058 / 0.975 | Amplitude / velocity vs GT (ideal 1) |
| std_r_rot / vc_r_rot | 0.963 / 0.918 | Head motion slightly under-dynamic |
| audio_gain | 1.500 | Mouth error with the wrong audio ÷ with the right audio (1 = audio ignored) |

Takeaways: 3× data fixed overfitting, and cosine annealing gave only −0.7%, so the model is plateaued at this scale.

### 3.3 Cache throughput

- ~90 s per clip.
- 24 h shards finished ~900–950 clips each.
- Array 172450 was re-submitted with 16 CPUs and a `%3` throttle.

### 3.4 Start-up jitter (`scripts/video_jitter.py`, 6 val clips, `motion_12k_anneal`)

Generated/real pixel acceleration (median over clips; lower = steadier):

| | Frames 0–10 | Frame 1 | Frames 10–25 | 25–64 | 64–74 (seam) | 74+ |
|---|---|---|---|---|---|---|
| Before (old first window) | **2.70** | 5.1 | 0.60 | 0.80 | 0.49 | 0.58 |
| `--start null` (`a911f42`) | **0.65** | 0.5 | 0.57 | 0.76 | 0.45 | 0.55 |
| `--start source` (`a911f42`) | **0.44** | 0.6 | 0.54 | 0.73 | 0.53 | 0.55 |

- **Cause:** in training, target frames always sat at positions 10–73. The prefix slot (positions 0–9) was either clean frames or dropped null tokens, and got no loss. The old inference ran the first window without the slot, so frames 0–9 sat on positions never trained as targets.
- **Why eval missed it:** training eval always passed a clean prefix.
- **Open:** choose the default between `null` and `source` by eye. Check whether `source` shows a "freeze then start".
- **General rule learned:** every conditioning slot must be exercised at inference exactly as in training, *including its absent state*. Apply this to the emotion and voice inputs too.

---

## 4. Bugs found and fixed

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | M_c2o shape errors | Upstream returns 3×3, not 2×3 | `as3x3`, `o2c` |
| 2 | CSIM could not load | HF LivePortrait insightface lacks the recognition model | Download full buffalo_l to `third_party/insightface_full` |
| 3 | "No CUDA GPUs" on the login node | LivePortrait picks its device from `flag_force_cpu`, not an argument | Set cfg/crop_cfg flags; API check only on GPU nodes |
| 4 | M0 v1 14.7 dB, "pan sway" | Per-frame `crop_source_image` re-centres and rotates the box | One fixed driving box per clip + shot-cut truncation |
| 5 | M0 v2 ~16 dB | PSNR gate invalid (freeze baseline 15–16 dB) | Gate on motion fidelity + CSIM |
| 6 | LSE nan | Relative path with cwd = syncnet; missing `sfd_face.pth` | Absolute path; install checks the weights. **LSE still unmeasured** |
| 7 | onnxruntime without CUDA | CPU wheel | `onnxruntime-gpu>=1.19` (user) |
| 8 | bench self-check tolerance | eps bias | 1e-4 tolerance |
| 9 | `lr=3e-3` override parsed as a string | YAML 1.1 | Float fallback in the override parser |
| 10 | Run 1 overfit and slow | No dropout; per-sample BeeGFS reads | dropout 0.1, eval 1k, patience, preload, fp16 audio |
| 11 | "no kernel image" on Blackwell | sm_120 needs torch cu128 | env `sang_bw` via `make_env_bw.sh`; `SANG_ENV` |
| 12 | Val set moving as data grew | Shuffle-based split | Hash-based split |
| 13 | Start-up jitter (first ~8 frames) | First window without the prefix slot | `generate(start=None)` = dropped-prefix layout (`a911f42`) |
| 14 | Calibration lost 4 h of work | Wrote only at the end; decoded every frame | Resumable, checkpoint every 10 clips, decode only landmarked frames (`eaadd54`) |
| 15 | Push 403 from the laptop | Wrong gh account active | `gh auth switch --user ScientificArchisman && gh auth setup-git` |
| 16 | Cluster git ECONNREFUSED | Stale VS Code askpass socket | New terminal, or `unset GIT_ASKPASS VSCODE_GIT_ASKPASS_MAIN VSCODE_GIT_IPC_HANDLE`; don't push from the cluster |
| 17 | Background (and torso, padding bars) wobbles; generated bg motion 2.35x real | The 42-d target holds scale, translation and 8 keypoints from the photo and rotated those keypoints' δ with the head; the 0.5 dB 42-vs-70 gate was only run with stitching ON, which hid a 7.5 dB gap (jobs 173138–41) | Convention change had no effect; widening the target improves placement (face PSNR 16.8→24.5) but moves the background MORE. Fix at render time: stitching (testing), then slot-0 lock / compositing |

---

## 5. Run book

### 5.1 Git workflow
```bash
# laptop
git add <files> && git commit -m "..." && git push
# cluster (/beegfs/work/achakraborti/SANG)
git pull && git log --oneline -1     # confirm the commit you expect is there
```
Always check that the commit is on the cluster before submitting a job that uses new flags.

### 5.2 Cluster facts

- **Partition:** `ifn`; QOS allows at most 3 running jobs.
- **A100 nodes:** `a100_80gb` (gpu08 etc.), env `avcodec` (the default in `env.sh`).
- **Blackwell nodes:** gpu10/gpu11 have `pro6000b_24gb/48gb/96gb` (sm_120) and need env `sang_bw` (torch cu128).
  - `train_motion.sh` defaults to `pro6000b_96gb` + `SANG_ENV=sang_bw`, 8 CPUs, 64 GB, 24 h.
  - Create the env once: `bash bash_scripts/make_env_bw.sh`.
- **`job.sh` defaults:** 2 CPUs, `a100_80gb`, 4 h. Override on the command line, e.g. `sbatch --time=08:00:00 --cpus-per-task=8 bash_scripts/job.sh ...`.
- **Logs:** `slurm_logs/sang_job_<id>.out` and `sang_motion_<id>.out`. Training also tees to `results/train/train_motion_<id>.txt`.
- **Useful commands:**
  - `squeue -u $USER`
  - `sinfo -p ifn`
  - `scontrol show node gpu10`
  - `scontrol update JobId=<array> ArrayTaskThrottle=2`
  - `scontrol release <id>`
  - Array tasks each get their own raw job ID (`%j`).

### 5.3 One-time install (login node, needs internet)
```bash
bash bash_scripts/install_motion.sh      # LivePortrait + weights, syncnet, buffalo_l, phoneme model, CPU tests
huggingface-cli download facebook/wav2vec2-xlsr-53-espeak-cv-ft   # (already in install_motion.sh)
# GPU API check (prints the exact srun line at the end of the install)
```

### 5.4 Commands by stage

| Stage | Command |
|---|---|
| M0 ceiling | `sbatch bash_scripts/job.sh scripts/motion_ceiling.py --n 50 --no-stitch` (`--target 42\|70`, `--lse`, `--dump results/extras/m0`) |
| Cache | `sbatch --cpus-per-task=16 --time=24:00:00 --array=0-7%3 bash_scripts/job.sh scripts/cache_motion.py --nshards 8 --max-frames 250` |
| Count cached clips | `cat cache/motion_lp/index_*.jsonl \| wc -l` |
| Train | `sbatch bash_scripts/train_motion.sh out_dir=runs/<name> max_steps=15000 warmup_steps=2000 patience=0` (any `key=value` overrides the yaml) |
| Resume | `sbatch bash_scripts/train_motion.sh resume=runs/<name>/last.pt` |
| Demo (side-by-side) | `sbatch bash_scripts/job.sh scripts/demo_motion.py --ckpt runs/motion_12k_anneal/best.pt --n 6 --out results/extras/<dir> [--start source] [--guide both]` |
| Photo + audio | `python scripts/infer_motion.py --ckpt runs/motion_12k_anneal/best.pt --image face.jpg --audio speech.wav --out out.mp4 [--start source] [--guide both]` |
| Openness calibration | `sbatch --time=08:00:00 --cpus-per-task=8 bash_scripts/job.sh scripts/calibrate_openness.py` (resumable; rerun the same line after a timeout) |
| Naturalness metrics | `sbatch bash_scripts/job.sh scripts/naturalness.py --ckpt runs/motion_12k_anneal/best.pt` |
| Jitter (laptop) | `python scripts/video_jitter.py results/extras/<dir>/*_sbs.mp4` |
| Voice install (login) | `bash bash_scripts/install_voice.sh [--asr]` |
| Voice eval | `sbatch bash_scripts/job.sh scripts/eval_voice.py --targets 20 [--asr openai/whisper-large-v3]` |
| Enrol a person | `sbatch bash_scripts/job.sh scripts/enroll_voice.py --name <name> --audio <files/folders> [--ref clean.wav]` (rerun with more audio to grow the bank) |
| Talk in their voice | `python scripts/infer_motion.py ... --voice voices/<name>` (or `--voice <audio folder>` to enrol on the fly); `demo_motion.py --voice voices/<name>` adds `*_voice-<name>_gen.mp4` |
| CPU tests | `python -m pytest tests/test_motion_model.py tests/test_naturalness.py -q` |

---

## 6. Next runs (queue with gates)

| # | Run | Command | Gate / what to look at |
|---|---|---|---|
| 1 | ~~Openness calibration~~ | done, job 173053 | lip 0.763, eye 0.706 |
| 2 | Naturalness metrics | `sbatch bash_scripts/job.sh scripts/naturalness.py --ckpt runs/motion_12k_anneal/best.pt` | Compare with the `real` column: `closure_viol` toward 0.10, `blinks_per_min` and `blink_at_pause` toward real, `lip_corr` not lower, `beat_align` vs `beat_chance` |
| 3 | Guided demo | `sbatch bash_scripts/job.sh scripts/demo_motion.py --ckpt runs/motion_12k_anneal/best.pt --n 6 --guide both --start source --out results/extras/demo_guided` | Watch p/b/m closures and blinks |
| 4 | Default start mode | Watch `results/extras/demo_fix/*_gen.mp4` vs `*_start-source_gen.mp4` | If `source` looks natural, make it the default |
| 5 | Voice install + eval | `bash bash_scripts/install_voice.sh --asr` (login), then `sbatch bash_scripts/job.sh scripts/eval_voice.py --targets 20 --asr openai/whisper-large-v3` | `sim_target` rises with bank seconds toward the 'real T vs T' ceiling; `sim_source` falls; `env_corr` ≈ 1; CER modest |
| 6 | Emotion data check (after coding, §7.4) | HSEmotion pass over the cache → distribution report | Enough non-neutral mass, or add CREMA-D |
| 7 | HDTF protocol (§7.7) | Needs HDTF on disk | GT row, GT-motion row, SANG row, baselines |

---

## 7. Idea backlog

Status tags: **DONE**, **CODED** (not measured), **NEXT**, **LATER**, **IDEA**, **DROPPED**.

### 7.1 Start-up and long-video stability

| Idea | Status | Notes |
|---|---|---|
| A. First window uses the dropped-prefix layout (`--start null`) | **DONE** `a911f42` | 2.7× → 0.65× over frames 0–10 |
| B. Start from the photo: its 42-d motion as a clean 10-frame prefix + 0.4 s silence (`--start source`) | **DONE** `a911f42` | 0.44×; opens on the photo. Risks: a brief "freeze"; a smiling photo + a "sad" label ramps rather than cuts |
| C. Retrain with variable-length prefixes (0–10, incl. one source frame), or predict the target as a residual from the source | IDEA | Only if A/B fail. FLOAT conditions every frame on the source latent |
| D. Post-hoc smoothing of the first frames | DROPPED | Hides the symptom; our motion is already slightly under-dynamic (vc_r_rot 0.92) |
| E. Overlap + cross-fade windows (Ditto) | DROPPED for the start | No seam spike (0.45–0.53×). Revisit for long clips |
| Renderer-free 42-d acceleration ratio in `naturalness.py` / training eval | NEXT | Removes background motion from the measurement |
| Add a no-prefix ("first window") case to the training eval | NEXT | The eval blind spot that hid the jitter |
| TT-SAC re-encoding / reference reset on 1–5 min clips | LATER | TT-SAC on FLOAT: CSIM 0.745 → 0.779, FVD 129 → 110 (arXiv 2605.25488) |

### 7.2 Naturalness: rule-based ("neuro-symbolic") priors

What they are: explicit knowledge of how faces move in speech, combined with the learned model.

| Idea | Status | Where in the code | Evidence |
|---|---|---|---|
| Rule metrics: closure depth / violation at /p b m/, lip corr, blink rate / IBI / duration / at-pause, head-beat vs loudness-onset alignment | **CODED** | `scripts/naturalness.py`, `sang/naturalness.py` | Bailando-style BAS; the others are ours |
| Linear lip/eye openness readouts from LivePortrait landmark ratios (lmk 90/102 over 48/66; eyes 6/18 over 0/12, 30/42 over 24/36) | **CODED**, calibration pending | `scripts/calibrate_openness.py` | Upstream `retargeting_utils.py` |
| Bilabial closure enforced at sampling: project x̂0 onto "lip openness ≤ τ" at /p b m/ frames, offset from real data | **CODED** | `project` in `sample`; `closure_bounds` | "Leaky mouth", PD-GS 2608.05218; ProjFlow 2602.22742 (exact linear constraints in flow matching) |
| Blink scheduler: keep the model's own blinks, add blinks in long gaps drawn from real inter-blink intervals, snap to pauses (±0.6 s), half-sine eye bound | **CODED** | `schedule_blinks`, `blink_bounds`, 2-pass `guided_generate` | TalkingEyes 2501.09921; DAWN 2410.13726; MoDiT 2507.05092; GoHD 2412.09296 |
| Language-agnostic phoneme recogniser (wav2vec2 XLSR-53 espeak, CTC 50 Hz) | **CODED** | `PhonemeRecognizer` | TalkVid has 15 languages and no transcripts, so MFA is out |
| Viseme / phoneme posteriors as an extra frame-wise condition (pooled to ~15–20 visemes) | IDEA (needs retrain) | New `phon` key in the cache; condition sum | PASE +13.7% / +14.2% lip-sync (2504.05803); PD-GS; FluentLip 2504.04427 |
| Prosody input (F0, energy, voicing) for head nods and brows | IDEA (needs retrain) | pyworld/torchaudio; condition sum | SubtleTalk 2608.06408; GoHD; Audio2Head 2107.09293; 2007.08547; ReFree 2606.13304 |
| Wider audio window: `audio_kernel` 5 → 9/11 (±160–200 ms) for coarticulation | IDEA (1 run) | config | CALS 2305.19556 (~1.2 s phonetic context) |
| AU detector (OpenFace/LibreFace) on renders to evaluate smile dynamics (AU12/AU6) | LATER | eval only | Cafe-Talk found OpenFace AUs misaligned with motion |
| Soft constraints / trust sampling instead of hard projection | IDEA | `sample` | Trust sampling 2411.10932; MIC 2607.01990; GMD 2305.12577 |
| Viseme-confusion metric | IDEA | eval | |

### 7.3 Voice cloning (optional target-voice input)

**Goal:**
- **Inputs:** (a) the photo of X, (b) driving audio of anyone, (c) optionally, recordings of X.
- **Output:** X's face speaking the driving content in X's voice, lip-synced.
- **Scaling:** it should get better as more of X's clean audio is added.

**Design** (report `reports/SANG jitter emotion and voice cloning.md`):
- **One shared WavLM-Large pass** with `output_hidden_states=True`:
  - the **last layer of the original audio drives the face**, unchanged, so there is no retraining;
  - **layer 6 feeds kNN-VC**.
- **kNN-VC** replaces each 20 ms source frame with the mean of its k = 4 nearest frames from X's matching set, then applies the prematched HiFi-GAN.
  - It is frame-synchronous on the same 50 Hz grid, so the converted audio muxes onto the video with lips aligned by construction.
  - Code: MIT, github.com/bshall/knn-vc.
- **Tiers by amount of clean target audio:**

| Target audio | Method | Measured anchor |
|---|---|---|
| none | Keep the driving voice | — |
| 3–10 s | Duration-preserving zero-shot converter: Seed-VC v1 (`length_adjust=1.0`, no convert-style; GPL-3.0, archived) or a permissive one (Chatterbox VC MIT, X-VC MIT). Or MKL-VC / Phoneme Hallucinator (no licence file) | MKL-VC WER 8.13% from 5–10 s vs kNN-VC 32.29%; Phoneme Hallucinator from 3 s WER 5.10%, EER 44.62% |
| 10–30 s | Run both, pick by speaker-verifier cosine + CER + UTMOS | Crossover region |
| 30 s – 5 min | kNN-VC, k = 4 | At 30/60 s kNN-VC SIM 0.617/0.631 ≈ Seed-VC 0.622/0.630 (Palindromic VC 2606.08843) |
| 5–30 min | kNN-VC, k ≈ 8–20; optional vocoder fine-tune | Quality plateaus near 5 min (kNN-VC 2305.18975) |
| 30 min + | A/B a per-speaker fine-tune (Seed-VC fine-tune, RVC) against kNN-VC | RVC FAQ recommends 10–50 min; no published crossover curve |

- **Enrolment (cleaning X's audio):** Demucs (remove music) → Silero VAD → pyannote diarisation + speaker verification against a reference clip → light DeepFilterNet3 (no generative enhancement) → loudness normalisation + DNSMOS gate. Cache the layer-6 frames (~6 MB/min fp16; FAISS index at hours of audio).
- **Varied speech matters:** kNN-VC can only emit frames X actually produced, so flat read speech flattens an expressive driver.
- **Languages:** cross-language pairs with short matching sets route to the zero-shot converter (kNN-VC with ~10 s of other-language reference: WER 96.7%).
- **Never use** timing-regenerating converters in the lip-synced path: Vevo-Voice/Style, Seed-VC v2 convert-style, GenVC, StableVC, R-VC.
- **Checks to write:**
  - HF `hidden_states[6]` vs kNN-VC's own extractor (cosine ≈ 1).
  - Under `--start source`, drop the 20 lead ticks before matching, or run one unpadded pass for the voice branch.
  - A DTW frame-drift check (≤ 1 frame) for any non-kNN converter.
- **Optional later:** fine-tune the face model on kNN-converted TalkVid audio paired with the original motion (timing identical), to allow "face from converted audio". Evidence on TTS audio is mixed: LSE-C drops 7.59 → 6.18 (Wav2Lip, 2511.05432) but rises for SadTalker (2405.10272).
- **Emotion interaction:** kNN-VC keeps prosody (IEMOCAP UAR 70.07% original → 56.70% after kNN-VC vs 30.35% for ASR→TTS, 2409.08913). A user emotion label shapes the face while the voice keeps the driver's delivery; warn on conflict.
- **Evaluation:**
  - Grid: {5 s, 30 s, 5 min, ≥ 10 min} × same/cross language × same/cross gender.
  - Speaker similarity (2 verifiers, real-vs-real ceiling); Whisper-large-v3 CER vs the original's transcript; UTMOS; emotion agreement original vs converted.
  - LSE on the same video with the original vs the converted track.
- **Decided:** no watermarking (2026-09-28).
- **Status:** CODED (2026-09-29). Not yet run on the cluster.
  - Deviation from the report: the voice branch uses **kNN-VC's own WavLM** (a second pass) instead of sharing the face model's HF WavLM pass. The prematched vocoder was trained on unilm WavLM-Large layer-6 features of raw audio, so this removes a silent feature mismatch and the `--start source` padding issue. It costs well under a second per clip.
  - Not yet built: the zero-shot fallback for banks under ~30 s (MKL-VC, training-free on the same features, or Seed-VC v1), Demucs / DeepFilterNet cleaning, and a second speaker verifier (ECAPA) for the metric.

### 7.4 Emotion control

**Design** (report §"Emotion enters as a fourth condition term"):
- **Labels:** from faces, not voices. Speech emotion recognisers collapse in the wild: MEMO ~42% on M3ED, emotion2vec 26.98%.
  - Run HSEmotion `enet_b0_8_va_mtl` on cached TalkVid frames.
  - Store the window-mean soft 8-d probabilities + valence/arousal + confidence; low confidence = "no label".
- **Data check first (NEXT):** measure TalkVid's emotion distribution. If negatives are rare, mix CREMA-D, the only clearly commercial labelled set (91 actors, ODbL/DbCL). RAVDESS is CC BY-NC-SA; MAFW is non-commercial; MEAD's terms are unconfirmed. HSEmotion models are AffectNet-trained (research licence), so they need legal review.
- **Injection:** c = c_time + c_audio + c_ref + **c_emo**, with c_emo = MLP([soft 8-d, intensity]) broadcast to all frames, a learned null token and ~0.15 dropout. Two arms:
  - **E1:** retrain the whole model (as FLOAT does).
  - **E2:** freeze the current checkpoint and train a zero-initialised adapter on the 39 expression dims only (Xemo-Talker / Playmate style).
- **Guidance:** nested, ṽ = v(∅) + γa[v(a) − v(∅)] + γe[v(a,e) − v(a)], with γa = 2 outside and emotion inside (FLOAT Eq. 14; Playmate Eq. 8). Use 3 forward passes per step. Do **not** use Cafe-Talk's Eq. 6 form, which double-counts audio.
- **Protect lip sync:** project the emotion delta off the lip-openness readout vector, so a smile can widen the lips but not open or close them.
  - The bilabial and blink projections run last and win conflicts.
  - Evidence: Xemo-Talker's PCA-tail loss (LivePortrait 70-d) gives LSE-C 6.51 vs 6.28 whole-space.
- **Reference leak:** the reference frame's expression will compete with the label (MEMO). In order of preference:
  1. draw the reference from another clip of the same speaker;
  2. raise ref dropout on labelled windows;
  3. subtract the clip's mean expression.
- **User modes:** auto (null token; WavLM carries emotion, probe-best at layer 10), label or mixture (Ditto-style soft vector), intensity knob.
- **Expected lip cost:**
  - Playmate: Sync-C 8.141 → 7.395 at w_e 1.5.
  - Xemo-Talker: LSE-D 8.02 → 8.21 while emotion accuracy went 15.84% → 85.28%.
  - FLOAT at γa = 2: LSE-D unchanged.
- **Ceilings:** SANG drives 13 of 21 keypoints, so cheek (AU6) and nose (AU9) may be capped. Measure by re-rendering real CREMA-D motion with those 8 keypoints frozen. Lower-bound baseline: a keypoint-space mean emotion offset (weak per EmoVOCA).
- **Evaluation:**
  - Two independent classifiers (not only HSEmotion): a MEAD-fine-tuned frame model and a motion-space classifier on CREMA-D 42-d motion.
  - Cafe-Talk swap test (label vs audio emotion).
  - Accuracy vs γe and intensity.
  - LSE-C/D + bilabial rate per emotion.
  - The null-emotion path must reproduce today's numbers.

### 7.5 Lip sync, audio and training recipe (report `SANG talking head improvements.md`, ranked)

| # | Idea | Cost | Evidence | Status |
|---|---|---|---|---|
| 1 | Inference sweep: steps {5, 8, 10} × CFG {1.5–4} × guidance interval × per-region CFG × truncation 1.2 | hours, no training | FLOAT T5/T6, KDTalker T6, EDM2 guidance interval 2404.07724 | NEXT (after HDTF protocol) |
| 2 | Long-video: overlap fusion, reference reset, TT-SAC re-encoding | hours | TT-SAC 2605.25488 | LATER |
| 3 | Data hygiene: finite-difference outlier mask (AVTR-1), SyncNet offset correction, drop Sync-conf < 3 | CPU min + GPU h + 1 run | LatentSync 2412.09262 | IDEA |
| 4 | Audio front end: Whisper-large-v3 / AV-HuBERT / learned WavLM layer mix; phoneme-CTC head; wider window | re-extract + 1 run per arm | Teller: Whisper Sync-C 7.696 vs 4.286 (2503.18429); SyncDiff 2503.13371; Pasad 2211.03929 | IDEA (largest measured motion-space lever) |
| 5 | Scale data 63 h → 250–500 h, then fine-tune on a strict subset | days | SANG: 3.7k → 12.6k fixed overfitting; Kimodo 2603.15546 | IDEA |
| 6 | Sync-reward Flow-DPO with a mouth-amplitude penalty | ~1 day + fine-tune | Playmate2 2510.12089; Hallo4 2505.23525; FantasyTalking2 2508.11255 (all pixel models) | IDEA |
| 7 | Expressiveness: emotion (§7.4), eye state | 1 run | Ditto T4; FLOAT T2; GoHD | see §7.4 |
| 8 | Motion discriminator; cosine velocity loss; logit-normal t | 1 run each | Motar T2; SD3 2403.03206 | IDEA |
| 9 | SyncNet loss through the differentiable LivePortrait decoder, or a keypoint SyncNet | 2–4× slower | LatentSync; Learn2Talk 2404.12888 | IDEA (risky) |
| 10 | Renderer: fine-tune warp+SPADE with a mouth/eye component loss, or swap to IMTalker | days–week | IMTalker 2511.22167 (FID 7.426 vs LivePortrait 9.049) | LATER, only if HDTF shows renderer-bound |

Also noted: track flow loss stratified on a fixed t-grid, since the average loss is not a quality measure (SD3). Our audio window is ±80 ms (FLOAT-equivalent).

### 7.6 Evaluation additions

- `audio_gain`, std_r / vc_r (have).
- Add: BAS, eye-region variance vs GT, idle-frame std, stratified-t loss, first-window ratios, 42-d acceleration.
- LSE: syncnet_python via `sang/bench.py`; **never successfully measured yet**. Fix before HDTF.

### 7.7 HDTF protocol (blocked on data access)

- **Rows:**
  - real video;
  - GT motion rendered through LivePortrait (renderer ceiling);
  - SANG;
  - baselines with weights (IMTalker Apache-2.0; check FLOAT, KDTalker, Ditto).
- **Freeze these:** split, crop, resolution, SyncNet checkpoint, FID/FVD implementation. Published numbers are context only: FLOAT's FID is 21.10 in its own paper vs 9.164 in IMTalker's.
- **Targets (paper):**
  - KDTalker HDTF: LSE-C 7.326 / LSE-D 7.548 / FID 9.756 / CSIM 0.949 (real 8.243 / 6.929).
  - FLOAT: FID 21.10, FVD 162, LSE-C 8.22.
- **To write:** `scripts/eval_hdtf.py` (loop `infer_motion` over the 349 clips, then `bench.py`) and FVD in `bench.py`.

### 7.8 Paper directions (for later; "replicate SOTA first")

1. **Subgroup audit with attribution:** a fairness audit that splits failures into generator vs renderer vs metric, using the GT-motion render and real clips per subgroup on TalkVid-Bench, plus group-DRO flow matching. Strongest option. TalkVid reports Sync-C 4.567 English vs 3.695 Polish, FID 40.7 vs 48.5 by ethnicity (2508.13618).
2. **Calibrated one-to-many generation:** energy score / CRPS on keypoint trajectories; tight lips, wide pose.
3. **Data-scaling curves:** hours vs speakers vs diversity at fixed compute, on one GPU.
4. **Sync-reward post-training in motion space** with a reward-hacking analysis.
5. **Cross-lingual transfer curves** in keypoint space (check MuEx 2510.06612 first).

---

## 8. Open questions

- HDTF access: YouTube reachable from the login node, or is HDTF already downloaded?
- Default start mode: `null` or `source`? Decide by eye (§6 #4).
- Are the openness readouts good enough (R² ≥ 0.7)? Does the model under-blink or leak at /p b m/? (`naturalness.py`)
- TalkVid's emotion distribution.
- Licences: HSEmotion (AffectNet-trained), MEAD, Emilia-trained VC checkpoints, Seed-VC GPL.
- Why LSE is nan in `bench.py`: syncnet weights and path were fixed, but it was never re-run successfully.

---

## 9. References

Grouped by use. The ID is arXiv unless noted. "abs" means only the abstract or search snippet was read; everything else was read in the research notes.

### Architecture and closest systems
| Paper | ID | Used for |
|---|---|---|
| LivePortrait | 2407.03168 + github.com/KwaiVGI/LivePortrait | Renderer, keypoint roles, crop conventions, lip normalisation, retargeting ratios |
| FLOAT (ICCV'25) | 2412.01064 | Frame-wise AdaLN (LSE-D 7.290 vs 7.757 cross-attn), prefix chaining + dropout, NFE / CFG ablations, emotion softmax + nested CFG |
| KDTalker (IJCV'25) | 2503.12963 | x_c prior, 64-frame window, HDTF targets, renderer ablation |
| Ditto (MM'25) | 2411.19509 | Canonical-keypoint identity, overlap fusion, HSEmotion pseudo-labels, emotion ablation |
| AVTR-1 | 2609.22913 | 42-d target, region-balanced loss, EMA 0.995, outlier mask |
| Decoupled Self-Forcing Distillation ("Motar") | 2609.10317 | Std-R / VC-R diagnostics, per-token-head collapse |
| JoyVASA | 2411.09209 | Diffusion over LivePortrait keypoints |
| Teller | 2503.18429 | Audio encoder effect on LivePortrait-keypoint model |
| IMTalker | 2511.22167 | Alternative renderer, FID comparisons |
| Playmate | 2502.07203 | LivePortrait-space emotion module + nested CFG |
| Xemo-Talker | 2608.14700 | Frozen base + zero-init emotion branch on LivePortrait 70-d |
| LeapTalk | 2608.00079 | Reference-filled chunks; CFG vs BAS |
| TT-SAC | 2605.25488 | Long-video stabilisation; over-smoothing warning |
| THEval | 2511.04520 | Metric sensitivity (abs) |

### Lip sync, audio, training
Pasad et al. 2211.03929 · SyncDiff 2503.13371 · PASE 2504.05803 · Hallo4 2505.23525 · LatentSync 2412.09262 · Sonic 2411.16331 · Playmate2 2510.12089 · FantasyTalking2 2508.11255 · FlowPortrait 2603.00159 · Learn2Talk 2404.12888 · 2503.20308 (mesh critic) · Dimitra 2502.17198 · UniSync 2503.16357 · THUNDER 2504.13386 · SD3 2403.03206 · guidance interval 2404.07724 · SynergyWarpNet 2512.17331 · Kimodo 2603.15546 · HY-Motion 2512.23464 · ScaMo 2412.14559

### Naturalness, phonemes, blinks, constraints
PD-GS 2608.05218 · coarticulation-weighted loss 2507.20568 (abs) · 3D dynamic visemes 2604.01756 (abs) · VedicTHG 2602.08775 (abs) · Text2Lip 2508.02362 (abs) · CALS 2305.19556 (abs) · MuEx 2510.06612 · Seeing Speech 2609.30517 (abs) · FluentLip 2504.04427 (abs) · TalkingEyes 2501.09921 (abs) · DAWN 2410.13726 · MoDiT 2507.05092 · GoHD 2412.09296 · CP-EB 2311.08673 (abs) · DFA-NeRF 2201.00791 (abs) · ProjFlow 2602.22742 (abs) · GMD 2305.12577 (abs) · Trust sampling 2411.10932 (abs) · MIC 2607.01990 (abs) · ConFlow 2607.14424 (abs) · Audio2Head 2107.09293 (abs) · Rhythmic head motion 2007.08547 (abs) · ReFree 2606.13304 (abs) · HM-Talker 2508.10566 (abs)

### Emotion
Cafe-Talk 2503.14517 · SubtleTalk 2608.06408 · EmoZone-Talker 2606.15848 · MEDTalk 2507.06071 · EmoVOCA 2403.12886 · EMOTE 2306.08990 · EmoTalk 2303.11089 · AUHead 2602.09534 · AU-guided landmarks 2509.19749 · GemTalk 2608.00663 · MEMO 2412.04448 · emotion2vec 2312.15185 · DICE-Talk 2504.18087 · PC-Talk 2503.14295 · WavLM layer probing 2501.05310 · EmotiEffLib/HSEmotion (github.com/av-savchenko/face-emotion-recognition) · datasets: CREMA-D, RAVDESS (zenodo 1188976), MAFW, MEAD

### Voice conversion and fusion
kNN-VC 2305.18975 (github.com/bshall/knn-vc, MIT) · Phoneme Hallucinator 2308.06382 · MKL-VC 2506.09709 · kNN-VC multilingual / stutter 2310.08104 · Palindromic VC 2606.08843 · kNN-FM-VC 2609.27230 · Seed-VC 2411.09943 (github.com/Plachtaa/seed-vc, GPL-3.0) · REF-VC 2508.04996 · Vevo 2502.07243 · RVC (github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) · Faces that Speak 2405.10272 · TTS-driven talking heads 2511.05432 · JAM-Flow 2506.23552 · synthetic-speaker training 2303.05322 · MimicTalk 2410.06734 · VoicePrivacy emotion/speaker leakage 2409.08913 · FAME 2026 2512.04814 · kDOT 2505.04382 · AudioSeal 2401.17264 (not used, see §1) · EU AI Act Art. 50 (artificialintelligenceact.eu/article/50)

### Data and benchmarks
TalkVid 2508.13618 · TalkVerse 2512.14938 · HDTF · EGT (group-robust generation) 2602.08660

---

## 10. How to add to this log

- A run finishes: add a row to §3.2 (or a new §3.x) with the job ID, command and numbers; update §0 and §6.
- A bug is found: add it to §4 with symptom, cause, fix and commit.
- An idea: add it to §7 with a status tag and its evidence, and add references to §9.
- A decision: add it to §1 with the date.

## 11. Where the detailed material lives

| File | Content |
|---|---|
| `docs/bidirectional_design_2026-09-23.md` | Why the bidirectional DiT; LivePortrait close reading; decision evidence |
| `reports/SANG talking head improvements.md` + `research_notes/SANG talking head improvements/` | Ranked improvements, paper directions, HDTF comparability |
| `reports/SANG background warping fix.md` + `research_notes/SANG background warping fix/` | Background-warping diagnosis (42-d round trip), fix ladder (held convention → wider target → slot-0 background lock → compositing → renderer fine-tune), metric upgrade, phased plan |
| `reports/SANG jitter emotion and voice cloning.md` + `research_notes/SANG jitter emotion and voice cloning/` | Jitter diagnosis, emotion design, voice-cloning tiers and fusion, evaluation, phased plan |
| `docs/sota_review_2026.md`, `docs/idea.md`, `docs/recovery_plan_2026-09-16.md`, `docs/v3_improvement_plan.md` | Earlier (pre-motion-space) history |
| Slides | https://claude.ai/artifact/WzsmgbTKsr1KyZEGEvEH7c (inspiration, architecture, SOTA + Ours) |
