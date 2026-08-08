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

# BASE_MODEL from positional $1, else the BASE_MODEL env var, else a default.
BASE_MODEL="${1:-${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"

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
    --eval_seed "${EVAL_SEED:-42}"
    --output_csv "$OUTPUT_CSV"
)

if [[ -n "$DATA_PATH" ]]; then
    CMD+=(--data_path "$DATA_PATH")
fi
# Long-context (LongBench-v2 up to 128k): extend Qwen3 past native 32768 with YaRN.
# NOTE: build the rope JSON HERE, not via --export -- SLURM --export splits on commas, so a JSON
# value gets truncated at its first comma. Pass ROPE_YARN_FACTOR (a plain number) instead.
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    CMD+=(--max_model_len "$MAX_MODEL_LEN")
fi
if [[ -n "${ROPE_YARN_FACTOR:-}" ]]; then
    CMD+=(--rope_scaling "{\"rope_type\":\"yarn\",\"factor\":${ROPE_YARN_FACTOR},\"original_max_position_embeddings\":32768}")
    # vLLM validates max_model_len against the BASE max_position_embeddings (32768) before applying
    # the YaRN override, so it refuses 131072 without this. Safe here: YaRN is actually active (via
    # hf_overrides), so extended positions are handled and do not nan.
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi
# avg@N for reliable (low-variance) baselines; deterministic via --eval_seed.
COT_SAMPLES="${COT_SAMPLES:-1}"
if [[ "$COT_SAMPLES" -gt 1 ]]; then
    CMD+=(--cot_samples "$COT_SAMPLES")
fi
if [[ "$IS_INSTRUCT" -eq 1 ]]; then
    CMD+=(--is_instruct)
fi
# Base model has no understanding behaviour -> only score plain CoT (faster, meaningful).
if [[ "${COT_EVAL_ONLY:-0}" == "1" ]]; then
    CMD+=(--cot_eval_only)
fi

"${CMD[@]}"
