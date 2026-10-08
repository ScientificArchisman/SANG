#!/bin/bash
# Phase B: training-recipe batch on audio-video-aligned data (report 'SANG next round improvements' section 4,
# research log section 6 #12). Each run trains 15k steps and keeps two extra EMAs; its evaluation
# (naturalness.py on last.pt, aligned val set) starts automatically when training ends.
#
#   bash bash_scripts/phase_b.sh                          # all runs
#   bash bash_scripts/phase_b.sh b2_cfm05 b3_ccc          # only these
#   GRES=gpu:h100:1 bash bash_scripts/phase_b.sh          # training on another card type
#   EVAL_SPECS="g=2,mouth=1.25 g=2,mouth=1.25,tau=0.5,steps=6" bash bash_scripts/phase_b.sh
#
# Judge on last.pt, not best.pt: contrastive FM moves the optimum away from the flow MSE on purpose, so
# best-by-val-loss would favour early checkpoints (and last.pt was already >= best.pt in Phase A).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
S=cache/motion_lp/sync_offsets.json
COMMON=(max_steps=15000 warmup_steps=2000 patience=0 "sync_offsets=$S" "ema_extra=[0.999,0.9995]")
runs_for() {                                        # run name -> its overrides (bash 3 has no associative arrays)
  case "$1" in
    b0_control)   echo "" ;;                                  # same recipe as runs/motion_sync, plus the extra EMAs (B4)
    b1_lognorm)   echo "t_dist=logit_normal" ;;               # B1: logit-normal t, centred
    b1_lognorm03) echo "t_dist=logit_normal t_mean=0.3" ;;    # B1: shifted toward high noise
    b2_cfm05)     echo "lam_cfm=0.05" ;;                      # B2: contrastive FM, negative = other window of the same clip
    b2_cfm10)     echo "lam_cfm=0.1" ;;
    b3_ccc)       echo "lam_ccc=0.1" ;;                       # B3: lip-opening CCC on the clean estimate (t <= 0.3)
    b3_spec)      echo "lam_spec=0.2" ;;                      # B3: spectral L1 below 10 Hz on the mouth (t <= 0.3)
    *) return 1 ;;
  esac
}
ORDER=(b0_control b1_lognorm b1_lognorm03 b2_cfm05 b2_cfm10 b3_ccc b3_spec)
read -r -a SPECS <<< "${EVAL_SPECS:-g=2,mouth=1.25 g=2,mouth=1.25,tau=0.5,avg=4,steps=6}"
names=("$@")
[ ${#names[@]} -eq 0 ] && names=("${ORDER[@]}")
for name in "${names[@]}"; do                       # check every name before submitting anything
  runs_for "$name" > /dev/null || { echo "unknown run '$name' (known: ${ORDER[*]})"; exit 1; }
done

for name in "${names[@]}"; do
  ov=$(runs_for "$name")
  read -r -a extra <<< "$ov"
  jt=$(sbatch --parsable ${GRES:+--gres=$GRES} bash_scripts/train_motion.sh "out_dir=runs/$name" "${COMMON[@]}" ${extra[@]+"${extra[@]}"})
  jt=${jt%%;*}
  je=$(sbatch --parsable --dependency=afterok:$jt bash_scripts/job.sh scripts/naturalness.py --ckpt "runs/$name/last.pt" \
       --modes none --sync-offsets "$S" --name phaseB --variants "${SPECS[@]}")
  echo "$name: train ${jt}, eval ${je%%;*}  (${ov:-control})"
  if [ "$name" = b0_control ]; then                 # B4: the same run read through its slower EMAs
    for d in 0.999 0.9995; do
      je=$(sbatch --parsable --dependency=afterok:$jt bash_scripts/job.sh scripts/naturalness.py --ckpt "runs/$name/last.pt" \
           --ema $d --modes none --sync-offsets "$S" --name phaseB --variants "${SPECS[@]}")
      echo "  $name EMA $d: eval ${je%%;*}"
    done
  fi
done
echo "logs: slurm_logs/sang_motion_<train id>.out and slurm_logs/sang_job_<eval id>.out"
