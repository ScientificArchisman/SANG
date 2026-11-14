# Experiments

This branch holds the evaluation code and the experiments in progress.
New ideas start here, and code moves to `main` once it has passed its tests.
The commands below run on the cluster from the repository root, and every job goes through `bash_scripts/run.sh`.

## Setup

Run the installation once on the login node.
It installs LivePortrait, SyncNet, the ArcFace, I3D and Inception models for the metrics, and runs the tests.

```bash
bash bash_scripts/env.sh install
```

`bash_scripts/env.sh` points `SANG_DATA` at the shared TalkVid folder on BeeGFS and keeps the Hugging Face Hub offline inside jobs.
The motion cache, the sync offsets and the motion latent `runs/latent/pca.pt` already exist there, so training can start right away.
The preprocessing steps that build them are in the README of `main`.

The HDTF commands below use this folder, so set it in your shell first.

```bash
HDTF=/beegfs/work/$USER/HDTF
```

## 1. Ditto on HDTF

This experiment reproduces Ditto's metrics on our frozen HDTF test set of 75 clips in `configs/hdtf_test.csv`.
Download HDTF on the login node, because YouTube is not reachable from the GPU nodes.
Run the same command again to resume after a failure.

```bash
pip install yt-dlp
python scripts/data/download_hdtf.py --out $HDTF --workers 4
```

Install Ditto into its own conda environment, also on the login node.

```bash
bash bash_scripts/install_ditto.sh
```

Generate the Ditto videos in the Ditto environment, and then score them with the same metric code that SANG uses.

```bash
SANG_ENV=ditto sbatch --time=08:00:00 bash_scripts/run.sh scripts/eval/ditto_hdtf.py --data $HDTF
sbatch --time=08:00:00 bash_scripts/run.sh scripts/eval/eval_hdtf.py --data $HDTF --rows real ditto --real-stats --name ditto
```

The scores land in `results/hdtf/ditto/summary.json`.
Ditto animates the first 10 seconds of each clip without extra footage, so compare it with a SANG run that starts at second 0.

```bash
sbatch --time=16:00:00 bash_scripts/run.sh scripts/eval/eval_hdtf.py --data $HDTF --ckpt runs/mbes/round4.pt --rows sang --real-stats --sampler g=2,steps=6 --name r4_start0_s0
```

## 2. Training the latest model

The latest model is a flow model trained on TalkVid and then fine-tuned with energy scores in three runs.
Every job preloads 97 hours of TalkVid, so it needs 128 GB of memory.

```bash
sbatch --mem=128gb bash_scripts/run.sh scripts/train/train_motion.py out_dir=runs/demo
```

Each fine-tuning run writes one checkpoint per round and stops when the head or mouth motion moves away from the real data.
Its last log line names the round to continue from.
The round numbers below are the ones the current best model used, so check them against the logs of a new run.

```bash
sbatch --mem=128gb bash_scripts/run.sh scripts/train/train_set.py gen=runs/demo/last.pt score=es out_dir=runs/set_es
sbatch --mem=128gb bash_scripts/run.sh scripts/train/train_set.py gen=runs/set_es/round5.pt score=es eval_batches=32 out_dir=runs/set_es2
sbatch --mem=128gb bash_scripts/run.sh scripts/train/train_set.py gen=runs/set_es2/round2.pt score=mbes eval_batches=32 out_dir=runs/mbes
```

The final model is `runs/mbes/round4.pt`.
This recipe does not include the footage guidance, which is still in development.

## 3. Evaluating the latest model

### HDTF

The main HDTF result evaluates the 10 seconds that start at second 60 of each clip.
The model reads the 30 seconds before that segment as footage of the person.
Run both seeds and give every run a new name, because the script resumes any run whose name already exists.

```bash
E="bash_scripts/run.sh scripts/eval/eval_hdtf.py --data $HDTF --ckpt runs/mbes/round4.pt --rows sang --real-stats --sampler g=2,steps=6 --start 60 --demo 30"
sbatch --time=16:00:00 $E --seed 0 --name r4_s60_d30_s0
sbatch --time=16:00:00 $E --seed 1 --name r4_s60_d30_s1
```

Each run writes FID, FVD, CSIM, LSE-C, LSE-D, lip CCC, head motion and blink rate to `results/hdtf/<name>/summary.json`.
The paired script compares runs on the same clips, with the first run as the reference.

```bash
python scripts/eval/paired.py r4_s60_d30_s0 other_run_s0
```

### TalkVid

TalkVid is evaluated on validation speakers that training never sees.
A fine-tuning run with zero rounds only prints its evaluation line, which compares the head and mouth motion with the real motion while the model reads footage of each person.

```bash
sbatch --mem=128gb bash_scripts/run.sh scripts/train/train_set.py gen=runs/mbes/round4.pt rounds=0 eval_batches=32 out_dir=runs/eval_r4
```

The naturalness script measures lip sync, lip closures and blinks on 300 validation clips without footage.
It needs `openness.json` in the motion cache, which `scripts/eval/calibrate_openness.py` writes.

```bash
sbatch bash_scripts/run.sh scripts/eval/naturalness.py --ckpt runs/mbes/round4.pt --sync-offsets /beegfs/work_fast/shared/li_shared/archi_data/talkvid/preprocessed/sync_offsets_c10.json --variants g=2,steps=6 --name r4
```



## Moving work between branches

Run the tests before every commit.

```bash
python -m pytest tests -q
```

To bring a finished change to `main`, check out `main` and take the files or commits you need from `dev`.
Then merge `main` back into `dev`, so the two branches stay in step.

```bash
git checkout main
git checkout dev -- sang/model/flow.py
git commit -m "Describe the change"
git push origin main
git checkout dev
git merge main
```
