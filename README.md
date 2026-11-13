# SANG

SANG turns one photo of a face and a speech recording into a talking-head video.
A flow-matching transformer reads the speech through a frozen WavLM model and generates LivePortrait motion, and the frozen LivePortrait renderer turns that motion into frames.
If you also give a short video of the person, the model reads it and moves the face more like that person.

## Installation

SANG needs Linux, an NVIDIA GPU and conda.
Create a conda environment with Python 3.10, and install PyTorch for your CUDA version into it.

```bash
conda create -n sang python=3.10 -y
conda activate sang
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

The setup script `bash_scripts/env.sh` installs everything else.
It clones LivePortrait and SyncNet into `third_party/`, installs the Python packages, downloads the LivePortrait and MediaPipe weights, and runs two self-checks.

```bash
bash bash_scripts/env.sh install
```

The script uses the conda environment named `sang`.
To use another environment, set `SANG_ENV`, as in `SANG_ENV=myenv bash bash_scripts/env.sh install`.
The WavLM audio encoder downloads itself from the Hugging Face Hub the first time it runs.

Run every command below from the repository root with the environment active.
`source bash_scripts/env.sh` activates it and sets the paths that the scripts need.

## Inference

The inference script needs a checkpoint, a reference of the face, and an audio file.
A pretrained checkpoint is not public yet, so train one with the steps in the next section.

```bash
python scripts/infer.py --ckpt runs/mbes/round4.pt --ref face.jpg --audio speech.wav --out out.mp4
```

The reference can be a photo or a video.
With a video, the first frame is the face that gets animated, and the first 30 seconds of the video show the model how the person moves.
Change that length with `--footage`, or set `--footage 0` to use the first frame only.

```bash
python scripts/infer.py --ckpt runs/mbes/round4.pt --ref person.mp4 --audio speech.wav --out out.mp4 --footage 30
```

The output is a 512 by 512 crop around the face at 25 frames per second, with the input audio.
Add `--paste` to paste the animated face back into the full frame.
The sampler defaults to `--sampler g=2,steps=6`, where `g` is the audio guidance scale and `steps` is the number of Euler steps.
Use `--seed` to get a different sample for the same input.

## Training

The model is trained on [TalkVid](https://github.com/FreedomIntelligence/TalkVid).
Download TalkVid and cut it into clips, so that every clip is an mp4 file with an m4a audio file of the same name next to it.
Store the clips as `data/talkvid/clips/<video>/<clip>.mp4`.
To keep the data somewhere else, set the environment variable `SANG_DATA` to that folder and change `cache_dir` and `sync_offsets` in `configs/train_motion.yaml` to match.

### Preprocessing

The clip filter drops clips where the face is missing, too small, moving out of frame, or cut.

```bash
find data/talkvid/clips -name "*.mp4" > data/clips_all.txt
python scripts/preprocess/filter_clips.py --clips data/clips_all.txt --out data/clips_filtered_all.txt
```

The motion cache stores the LivePortrait motion and the WavLM features of every clip.
A long clip is cut into up to ten segments of 20 seconds.
On several GPUs, give each process its own `--shard` out of `--nshards`.

```bash
python scripts/preprocess/cache_motion.py --clips data/clips_filtered_all.txt --chunks 10
```

Some segments run past the end of their audio file, and the audio check finds them.
SyncNet then measures how far the audio is shifted against the video in each source video, so that training can correct it.

```bash
python scripts/preprocess/check_audio.py
python scripts/preprocess/sync_offsets.py
python scripts/preprocess/sync_offsets.py --merge --exclude data/talkvid/preprocessed/audio_check.json
```

The last preprocessing step fits the motion latent that the model generates in.

```bash
python scripts/preprocess/fit_latent.py --out runs/latent
```

### Training the model

The first stage trains the flow model with the settings in `configs/train_motion.yaml` and writes `runs/demo/last.pt`.
It preloads every clip into memory, which takes about 128 GB of RAM for the full 97 hours of TalkVid.
Add `preload=false` to read the clips from disk instead.

```bash
python scripts/train/train_motion.py
```

Any setting can be changed on the command line, as in `python scripts/train/train_motion.py max_steps=20000 out_dir=runs/demo2`.

The second stage fine-tunes the sampler with energy scores in three runs.
Each run writes one checkpoint per round and stops by itself when the head or mouth motion moves away from the real data.
Its last log line names the round to continue from.
The round numbers below are the ones our best model used, and a new run may recommend different ones.

```bash
python scripts/train/train_set.py gen=runs/demo/last.pt score=es out_dir=runs/set_es
python scripts/train/train_set.py gen=runs/set_es/round5.pt score=es eval_batches=32 out_dir=runs/set_es2
python scripts/train/train_set.py gen=runs/set_es2/round2.pt score=mbes eval_batches=32 out_dir=runs/mbes
```

The final model is `runs/mbes/round4.pt`.

## Running on a Slurm cluster

The job script `bash_scripts/run.sh` runs any Python script as a Slurm job on one GPU.
It activates the environment through `bash_scripts/env.sh`, writes the log to `slurm_logs/`, and passes every argument on to Python.
It asks for 8 CPUs, 64 GB of memory and 24 hours, and any `sbatch` option overrides these.
Add the partition and account of your cluster when it needs them, as in `sbatch --partition=gpu --account=myproject bash_scripts/run.sh scripts/infer.py ...`.

```bash
sbatch bash_scripts/run.sh scripts/infer.py --ckpt runs/mbes/round4.pt --ref face.jpg --audio speech.wav --out out.mp4
```

These jobs run the whole training pipeline.
Submit each job after the one before it has finished.
The array jobs split the work into shards, and each task reads its shard number from `SLURM_ARRAY_TASK_ID`.

```bash
sbatch bash_scripts/run.sh scripts/preprocess/filter_clips.py --clips data/clips_all.txt --out data/clips_filtered_all.txt
sbatch --array=0-7 bash_scripts/run.sh scripts/preprocess/cache_motion.py --clips data/clips_filtered_all.txt --chunks 10 --nshards 8
sbatch bash_scripts/run.sh scripts/preprocess/check_audio.py
sbatch --array=0-2 bash_scripts/run.sh scripts/preprocess/sync_offsets.py --nshards 3
sbatch bash_scripts/run.sh scripts/preprocess/sync_offsets.py --merge --exclude data/talkvid/preprocessed/audio_check.json
sbatch bash_scripts/run.sh scripts/preprocess/fit_latent.py --out runs/latent
sbatch --mem=128gb bash_scripts/run.sh scripts/train/train_motion.py
sbatch --mem=128gb bash_scripts/run.sh scripts/train/train_set.py gen=runs/demo/last.pt score=es out_dir=runs/set_es
sbatch --mem=128gb bash_scripts/run.sh scripts/train/train_set.py gen=runs/set_es/round5.pt score=es eval_batches=32 out_dir=runs/set_es2
sbatch --mem=128gb bash_scripts/run.sh scripts/train/train_set.py gen=runs/set_es2/round2.pt score=mbes eval_batches=32 out_dir=runs/mbes
```

## Repository layout

The package `sang/data` reads audio, video and the motion cache.
The package `sang/model` holds the flow transformer, the motion latent and the energy scores.
The package `sang/render` wraps LivePortrait and runs the full generation pipeline.
The folder `scripts` holds the preprocessing, training and inference scripts, `bash_scripts` holds the setup and job scripts, and `configs` holds the training settings.
Evaluation code and current experiments live on the `dev` branch.

## License

SANG is released under the MIT License.
LivePortrait, WavLM, MediaPipe and SyncNet come with their own licenses.
The InsightFace face detector that LivePortrait uses is licensed for non-commercial research only.
