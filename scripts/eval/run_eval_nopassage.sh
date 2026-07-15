#!/bin/bash
#SBATCH --job-name=nopassage_eval_%j
#SBATCH --output=slurm/evals/outputs/nopassage_eval_%j.out
#SBATCH --error=slurm/evals/errors/nopassage_eval_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=embers
#SBATCH --time=2:00:00
#
# Passage-less transfer eval (ARC-Challenge / GSM8K) -- see src/eval_nopassage.py.
# Deliberately NO resubmit-on-failure trap: auto_curve_eval.sh retries any exit=1 up to 31 times
# on the assumption it was preemption, which turns a deterministic data/config error into ~31
# wasted launches (this is exactly how the ARC failures stampeded). Fail loudly instead.
#   Env: TASK (mcq_nopassage|gsm8k), BASE_MODEL, [CHECKPOINT_DIR], [DATA_PATH], OUTPUT_CSV,
#        [COT_SAMPLES=8], [MAX_N], [RUN_NAME], [STEP]
set -euo pipefail
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
export HF_HUB_DISABLE_XET=1

CMD=(python -m src.eval_nopassage
     --task "${TASK}"
     --base_model "${BASE_MODEL}"
     --output_csv "${OUTPUT_CSV}"
     --cot_samples "${COT_SAMPLES:-8}")

[[ -n "${DATA_PATH:-}" ]]      && CMD+=(--data_path "${DATA_PATH}")
[[ -n "${CHECKPOINT_DIR:-}" ]] && CMD+=(--checkpoint_dir "${CHECKPOINT_DIR}")
[[ -n "${MAX_N:-}" ]]          && CMD+=(--max_n "${MAX_N}")
[[ -n "${RUN_NAME:-}" ]]       && CMD+=(--run_name "${RUN_NAME}")
[[ -n "${STEP:-}" ]]           && CMD+=(--step "${STEP}")

echo "[nopassage] ${CMD[*]}"
"${CMD[@]}"
