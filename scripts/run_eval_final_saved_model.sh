#!/bin/bash
#SBATCH --job-name=reasoning_eval_final_%j
#SBATCH --output=slurm/reasoning_eval_final_%j.out
#SBATCH --error=slurm/reasoning_eval_final_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=inferno
#SBATCH --time=2:00:00

module load python/3.10.10 cuda/12.6.1
source ~/scratch/python-envs/oat-env/bin/activate
cd ~/scratch/CS8803LLM_Self_Learning_Project/reasoning
set -euo pipefail

if [[ $# -eq 1 ]]; then
    RUN_DIR="oat-output/simpl/${1}"
else
    RUN_DIR="${RUN_DIR:-}"
fi

DATA_PATH="${DATA_PATH:-data/race_test_high_42_1000.jsonl}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-oat-output/simpl}"
OUTPUT_DIR="${OUTPUT_DIR:-oat-output/eval/final_only}"
OUTPUT_CSV="${OUTPUT_CSV:-}"

if [[ -z "$RUN_DIR" ]]; then
    echo "RUN_DIR was not provided; expected a path like oat-output/simpl/<run_name>." >&2
    echo "You can also set CHECKPOINT_ROOT and pass RUN_DIR relative to it before calling this script." >&2
    exit 1
fi

if [[ ! -d "$RUN_DIR" ]] && [[ -d "$CHECKPOINT_ROOT/$RUN_DIR" ]]; then
    RUN_DIR="$CHECKPOINT_ROOT/$RUN_DIR"
fi

if [[ ! -d "$RUN_DIR/saved_models" ]]; then
    echo "No saved_models directory in run: $RUN_DIR" >&2
    exit 1
fi

FINAL_CHECKPOINT="$(find "$RUN_DIR/saved_models" -mindepth 1 -maxdepth 1 -type d -name 'step_*' | sort -V | tail -n 1)"
if [[ -z "$FINAL_CHECKPOINT" ]]; then
    echo "No step_* checkpoint directories found under: $RUN_DIR/saved_models" >&2
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

python eval_saved_models.py \
    --data_path "$DATA_PATH" \
    --checkpoint_dir "$FINAL_CHECKPOINT" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.95 \
    --batch_size 128 \
    --output_csv "$OUTPUT_CSV"
