#!/bin/bash
#SBATCH --job-name=eval_full_dump_%j
#SBATCH --output=slurm/evals/outputs/eval_full_dump_%j.out
#SBATCH --error=slurm/evals/errors/eval_full_dump_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=embers
#SBATCH --time=2:00:00

# Like run_eval_final_saved_model.sh but dumps EVERY question's reasoning to JSON
# (--sample_output_count_per_checkpoint large) for mechanism analysis. Pass RUN_DIR (full
# path) via env, DATASET_NAME for the eval data/prompt, DATA_PATH for the test file.

module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
set -euo pipefail

RUN_DIR="${RUN_DIR:?set RUN_DIR to the full run dir path}"
DATASET_NAME="${DATASET_NAME:-race-c}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/analysis}"
SAMPLE_COUNT="${SAMPLE_COUNT:-2000}"

FINAL_CHECKPOINT="$(find "$RUN_DIR/saved_models" -mindepth 1 -maxdepth 1 -type d -name 'step_*' 2>/dev/null | sort -V | tail -n1)"
[[ -z "$FINAL_CHECKPOINT" ]] && { echo "No checkpoint under $RUN_DIR" >&2; exit 1; }
RUN_NAME="$(basename "$RUN_DIR")"; STEP="$(basename "$FINAL_CHECKPOINT")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_CSV="$OUTPUT_DIR/${RUN_NAME}_${STEP}.csv"

echo "[full-dump] run=$RUN_NAME ckpt=$STEP dataset=$DATASET_NAME -> $OUTPUT_CSV"
CMD=(
    python -m src.eval_saved_models
    --dataset_name "$DATASET_NAME"
    --checkpoint_dir "$FINAL_CHECKPOINT"
    --tensor_parallel_size 1 --gpu_memory_utilization 0.95
    --reasoning_max_tokens 1024 --answer_max_tokens 1024
    --sample_output_count_per_checkpoint "$SAMPLE_COUNT"
    --output_csv "$OUTPUT_CSV"
)
[[ -n "$DATA_PATH" ]] && CMD+=(--data_path "$DATA_PATH")
"${CMD[@]}"
