#!/bin/bash
#SBATCH --job-name=fanout_curve_%j
#SBATCH --output=slurm/evals/outputs/fanout_curve_%j.out
#SBATCH --error=slurm/evals/errors/fanout_curve_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --qos=embers
#SBATCH --time=00:15:00
#
# Fan-out curve eval: resolve a run dir, then submit ONE small embers eval per
# saved checkpoint (each writes its own per-step CSV). Cheap + parallel + robust
# (a preempted single-step eval loses only that step). Run afterok the train job.
#   Env: TRAIN_DS, RUN_PREFIX, DATASET_NAME, [DATA_PATH], OUTPUT_DIR, [COT_SAMPLES=8]
set -euo pipefail
cd ~/scratch/SImpL

RUN_DIR="$(ls -dt oat-output/${TRAIN_DS}/${RUN_PREFIX}_* 2>/dev/null | head -1 || true)"
if [[ -z "$RUN_DIR" ]]; then echo "fanout: no run dir for ${RUN_PREFIX}" >&2; exit 1; fi
STEPS="$(ls "$RUN_DIR/saved_models" 2>/dev/null | grep '^step_' || true)"
echo "fanout: RUN_DIR=$RUN_DIR  steps: $STEPS"

for step in $STEPS; do
  sbatch --export=ALL,RUN_DIR="$RUN_DIR",STEP="$step",DATASET_NAME="$DATASET_NAME",DATA_PATH="${DATA_PATH:-}",OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATASET_NAME}/curves_steps}",COT_SAMPLES="${COT_SAMPLES:-8}" \
    scripts/eval/run_eval_final_saved_model.sh
done
echo "fanout: submitted $(echo "$STEPS" | wc -w) per-step evals"