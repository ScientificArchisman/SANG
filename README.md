# SANG

SANG turns one photo of a face and a speech recording into a talking-head video.
A flow-matching transformer reads the speech through a frozen WavLM model and generates LivePortrait motion, and the frozen LivePortrait renderer turns that motion into frames.
If you also give a short video of the person, the model reads it and moves the face more like that person.

## Installation

SANG needs Linux, Python 3.10 and an NVIDIA GPU.
Install PyTorch for your CUDA version first, then run these commands from the repository root.

```bash
git clone https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait
pip install -r third_party/LivePortrait/requirements_base.txt
pip install -r requirements.txt
huggingface-cli download KlingTeam/LivePortrait --local-dir third_party/LivePortrait/pretrained_weights --exclude "*animal*" "*.git*"
```

The WavLM audio encoder downloads itself from the Hugging Face Hub the first time it runs.

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
It needs the MediaPipe face landmark model.

```bash
mkdir -p checkpoints
curl -L -o checkpoints/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
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
git clone https://github.com/joonson/syncnet_python third_party/syncnet_python
(cd third_party/syncnet_python && sh download_model.sh)
pip install scipy python_speech_features scenedetect
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

## Repository layout

The package `sang/data` reads audio, video and the motion cache.
The package `sang/model` holds the flow transformer, the motion latent and the energy scores.
The package `sang/render` wraps LivePortrait and runs the full generation pipeline.
The folder `scripts` holds the preprocessing, training and inference scripts, and `configs` holds the training settings.
Evaluation code and current experiments live on the `dev` branch.

## License

SANG is released under the MIT License.
LivePortrait, WavLM, MediaPipe and SyncNet come with their own licenses.
The InsightFace face detector that LivePortrait uses is licensed for non-commercial research only.
