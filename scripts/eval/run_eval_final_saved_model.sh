#!/bin/bash
#SBATCH --job-name=reasoning_eval_final_%j
#SBATCH --output=slurm/evals/outputs/reasoning_eval_final_%j.out
#SBATCH --error=slurm/evals/errors/reasoning_eval_final_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=embers
#SBATCH --time=2:00:00

# module load python/3.10.10 cuda/12.6.1
# source ~/r-nisha3-0/python-envs/oat-env/bin/activate
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
set -euo pipefail

if [[ $# -eq 1 ]]; then
    RUN_DIR="oat-output/$DATASET_NAME/${1}"
else
    RUN_DIR="${RUN_DIR:-}"
fi

DATASET_NAME="${DATASET_NAME:-race-c}"
DATA_PATH="${DATA_PATH:-}"
IS_INSTRUCT="${IS_INSTRUCT:-0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-oat-output/$DATASET_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATASET_NAME}/final_only}"
OUTPUT_CSV="${OUTPUT_CSV:-}"

if [[ -z "$RUN_DIR" ]]; then
    echo "RUN_DIR was not provided; expected a path like oat-output/$DATASET_NAME/${1}." >&2
    echo "You can also set CHECKPOINT_ROOT and pass RUN_DIR relative to it before calling this script." >&2
    exit 1
fi

if [[ ! -d "$RUN_DIR" ]] && [[ -d "$CHECKPOINT_ROOT/$RUN_DIR" ]]; then
    RUN_DIR="$CHECKPOINT_ROOT/$RUN_DIR"
fi

FINAL_CHECKPOINT="$(
    {
        find "$RUN_DIR/saved_models" -mindepth 1 -maxdepth 1 -type d -name 'step_*' 2>/dev/null || true
        find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null || true
    } | sort -V | tail -n 1
)"
if [[ -z "$FINAL_CHECKPOINT" ]]; then
    echo "No step_* or checkpoint-* checkpoint directories found under: $RUN_DIR" >&2
    exit 1
fi

FINAL_STEP_NAME="$(basename "$FINAL_CHECKPOINT")"
RUN_NAME="$(basename "$RUN_DIR")"

if [[ -z "$OUTPUT_CSV" ]]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_CSV="$OUTPUT_DIR/${RUN_NAME}_${FINAL_STEP_NAME}.csv"
else
    mkdir -p "$(dirname "$OUTPUT_CSV")"
fi

echo "[eval-final] run=$RUN_NAME final_checkpoint=$FINAL_CHECKPOINT"

CMD=(
    python -m src.eval_saved_models
    --dataset_name "$DATASET_NAME"
    --checkpoint_dir "$FINAL_CHECKPOINT"
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
