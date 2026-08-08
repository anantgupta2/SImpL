#!/bin/bash
# Re-evaluate every final LSAT model on the pooled valid+test joint set (461 Q) for
# tighter CIs and one consistent eval across all runs. Queues a partition curve-eval
# per run dir (DATA_PATH = joint), writing evaluations/final/<name>_joint.csv.
# Parts auto-go to parts/ and are deleted on merge; chunk count auto-derives (~15/ckpt).
#   Run AFTER all training + test-only evals have finished:
#     bash scripts/eval/reeval_lsat_joint.sh
set -euo pipefail
cd ~/scratch/SImpL
JOINT="data/lsat-ar/eval_joint_42_all.jsonl"
[[ -f "$JOINT" ]] || { echo "joint set missing: $JOINT" >&2; exit 1; }

# run-dir prefix  ->  output csv stem   (extend as new final runs appear)
declare -A RUNS=(
  ["Qwen3-4B-Base-final-lsat50-cot_cot-only"]="lsat50_cot"
  ["Qwen3-4B-Base-final-lsat50-cotn16_cot-only"]="lsat50_cotn16"
  ["Qwen3-4B-Base-final-lsat50-simpl-us1_simpl-oat"]="lsat50_simpl_us1"
  ["Qwen3-4B-Base-final-lsat50-simpl-us2_simpl-oat"]="lsat50_simpl_us2"
  ["Qwen3-4B-Base-final-lsat50-simpl-us4_simpl-oat"]="lsat50_simpl_us4"
  ["Qwen3-4B-Base-final-lsat50-simpl-pe2_simpl-oat"]="lsat50_simpl_pe2"
  ["Qwen3-4B-Base-final-lsat50-simpl-pe4_simpl-oat"]="lsat50_simpl_pe4"
  ["Qwen3-4B-Base-final-lsat50-simpl-pe2-us2_simpl-oat"]="lsat50_simpl_pe2us2"
  ["Qwen3-4B-Base-final-lsat100-cot_cot-only"]="lsat100_cot"
)
mkdir -p evaluations/final/joint
for base in "${!RUNS[@]}"; do
  stem="${RUNS[$base]}"
  for s in 24 36 42; do
    prefix="${base}_${s}"
    # skip if no run dir yet
    ls -d oat-output/lsat-ar/${prefix}_* >/dev/null 2>&1 || { echo "skip (no run dir): $prefix"; continue; }
    out="evaluations/final/joint/${stem}_s${s}.csv"
    TRAIN_DS=lsat-ar RUN_PREFIX="$prefix" DATASET_NAME=lsat-ar DATA_PATH="$JOINT" \
      OUTPUT_CSV="$out" COT_SAMPLES=8 bash scripts/eval/partition_curve_eval.sh
  done
done
echo "queued joint re-evals -> evaluations/final/joint/"