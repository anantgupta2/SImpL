#!/bin/bash
# Re-evaluate EVERY trained LSAT run on the pooled valid+test joint set (461 Q),
# into evaluations/lsat-re-eval/<group>/<run>.csv. Groups mirror the source folders
# (hp_grid, hp_size, hp_simpl_base, hp_warmup, final). Each run -> a partition
# curve-eval (auto chunks ~15 ckpts, parts auto-deleted on merge).
# Skips run dirs whose training job is still RUNNING/PENDING (re-eval those later).
#   bash scripts/eval/reeval_lsat_all_joint.sh
set -euo pipefail
cd ~/scratch/SImpL
JOINT="data/lsat-ar/eval_joint_42_all.jsonl"
[[ -f "$JOINT" ]] || { echo "joint set missing: $JOINT" >&2; exit 1; }
OUTROOT="evaluations/lsat-re-eval"

# run dirs still training (checkpoints still growing) -> skip, re-eval after.
# (sbatch job names are generic, so match by run-name prefix instead.)
SKIP_RE='final-lsat50-simpl-pe2_|final-lsat50-simpl-pe4_|final-lsat50-simpl-pe2-us2_'

classify () {  # $1 = run dir basename (sans timestamp) -> echoes "group"
  case "$1" in
    lsat50-hp-*)            echo hp_grid ;;
    lsat-simpl-base-*)      echo hp_simpl_base ;;
    lsat-cot-warmup-*)      echo hp_warmup ;;
    final-lsat*)            echo final ;;
    lsat50-*|lsat100-*|lsat200-*) echo hp_size ;;
    *)                      echo other ;;
  esac
}

queued=0; skipped=0
for d in oat-output/lsat-ar/*/; do
  bn="$(basename "$d")"
  ls "$d/saved_models" 2>/dev/null | grep -q '^step_' || continue
  # strip trailing _<timestamp> (….T…) -> prefix used for glob + stem for csv
  stem="$(echo "$bn" | sed -E 's/_[0-9]{4}T[0-9:]+$//')"
  prefix="$stem"                                   # partition globs prefix_*
  short="${stem#Qwen3-4B-Base-}"                   # drop model prefix for paths
  grp="$(classify "$short")"
  # skip runs still training (re-eval after they finish)
  if grep -qE "$SKIP_RE" <<<"${short}_"; then
    echo "skip (training active): $short"; skipped=$((skipped+1)); continue
  fi
  out="$OUTROOT/$grp/${short}.csv"
  mkdir -p "$OUTROOT/$grp"
  TRAIN_DS=lsat-ar RUN_PREFIX="$prefix" DATASET_NAME=lsat-ar DATA_PATH="$JOINT" \
    OUTPUT_CSV="$out" COT_SAMPLES=8 bash scripts/eval/partition_curve_eval.sh >/dev/null
  echo "queued [$grp] $short"
  queued=$((queued+1))
done
echo "---- queued $queued runs, skipped $skipped active -> $OUTROOT/ ----"