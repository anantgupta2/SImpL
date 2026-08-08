#!/bin/bash
#SBATCH --job-name=auto_eval_%j
#SBATCH --output=slurm/evals/outputs/auto_eval_%j.out
#SBATCH --error=slurm/evals/errors/auto_eval_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=embers
#SBATCH --time=2:00:00
#
# Resolve a run dir by glob AT RUNTIME (after the training job finished, so the
# final timestamped dir exists) then run the standard avg@8 eval on it. Avoids
# hardcoding timestamps. Env in:
#   TRAIN_DS    dataset dir the run lives under (oat-output/$TRAIN_DS/...)
#   RUN_PREFIX  run-name prefix up to and incl. the seed (no timestamp)
#   DATASET_NAME / DATA_PATH / OUTPUT_DIR / COT_SAMPLES  -> passed to the eval
set -euo pipefail
cd ~/scratch/SImpL

RUN_DIR="$(ls -dt oat-output/${TRAIN_DS}/${RUN_PREFIX}_* 2>/dev/null | head -1 || true)"
if [[ -z "$RUN_DIR" ]]; then
    echo "auto_eval: no run dir matched oat-output/${TRAIN_DS}/${RUN_PREFIX}_*" >&2
    exit 1
fi
echo "auto_eval: resolved RUN_DIR=$RUN_DIR"
export RUN_DIR
exec bash scripts/eval/run_eval_final_saved_model.sh