#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
# CPU-only. Runs scripts/filter_clips.py on a manifest in N parallel shards, then merges.
# Usage: sbatch bash_scripts/filter_clips.sh data/clips_all.txt data/clips_filtered_all.txt [extra filter args]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/bash_scripts/env.sh"
IN=${1:-data/clips_all.txt}; OUT=${2:-data/clips_filtered_all.txt}; shift 2 || true
N=${SLURM_CPUS_PER_TASK:-8}
TMP=$(mktemp -d "${OUT}.shards.XXXX")
split -n l/$N -d -a 2 "$IN" "$TMP/in."
for f in "$TMP"/in.*; do
  i=${f##*.}
  OMP_NUM_THREADS=1 python -u scripts/filter_clips.py --clips "$f" --out "$TMP/kept.$i" \
      --report "$TMP/report.$i.jsonl" "$@" > "$TMP/log.$i" 2>&1 &
done
wait
cat "$TMP"/kept.* | sort > "$OUT"
cat "$TMP"/report.*.jsonl > "${OUT%.txt}.report.jsonl"
echo "kept $(wc -l < "$OUT") / $(wc -l < "$IN") clips -> $OUT"
grep -h "dropped" "$TMP"/log.* | awk '{r=$3; for(i=4;i<=NF;i++) r=r" "$i; c[r]+=$2} END {for (k in c) printf "  dropped %6d  %s\n", c[k], k}' | sort -k2 -rn
rm -rf "$TMP"
