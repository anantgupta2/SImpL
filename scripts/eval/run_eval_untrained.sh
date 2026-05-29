#!/bin/bash
#SBATCH --job-name=reasoning_eval_%j
#SBATCH --output=slurm/evals/outputs/reasoning_eval_%j.out
#SBATCH --error=slurm/evals/errors/reasoning_eval_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=embers
#SBATCH --time=2:00:00

module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
set -euo pipefail

# Optional Input BASE_MODEL $1
if [[ $# -ge 1 ]]; then
    BASE_MODEL="$1"
else
    BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
fi

## remove anything before the last slash
MODEL_NAME="${BASE_MODEL##*/}"
BASELINE_TAG="${BASELINE_TAG:-$(echo "$MODEL_NAME" | tr '/:-' '_')_untrained}"
IS_INSTRUCT="${IS_INSTRUCT:-0}"

DATASET_NAME="${DATASET_NAME:-race-c}"
DATA_PATH="${DATA_PATH:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-evaluations/baselines/${BASELINE_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATASET_NAME}/baselines}"
OUTPUT_CSV="${OUTPUT_CSV:-}"

# Keep a concrete directory so eval_saved_models.py can iterate checkpoint_dir.
mkdir -p "$CHECKPOINT_DIR"

if [[ -z "$OUTPUT_CSV" ]]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_CSV="$OUTPUT_DIR/${BASELINE_TAG}.csv"
else
    mkdir -p "$(dirname "$OUTPUT_CSV")"
fi

echo "[eval-untrained] base_model=$BASE_MODEL checkpoint_dir=$CHECKPOINT_DIR"

echo "[eval-untrained] writing csv to $OUTPUT_CSV"

CMD=(
    python -m src.eval_saved_models
    --dataset_name "$DATASET_NAME"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --base_model "$BASE_MODEL"
    --no_merge_lora_before_eval
    --tensor_parallel_size 1
    --gpu_memory_utilization 0.95
    --reasoning_max_tokens 1024
    --answer_max_tokens 1024
    --output_csv "$OUTPUT_CSV"
)

if [[ -n "$DATA_PATH" ]]; then
    CMD+=(--data_path "$DATA_PATH")
fi
if [[ "$IS_INSTRUCT" -eq 1 ]]; then
    CMD+=(--is_instruct)
fi

"${CMD[@]}"
