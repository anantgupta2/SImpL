#!/bin/bash
set -euo pipefail

cd ~/scratch/SImpL

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-oat-output/simpl}"
DATASET_NAME="${DATASET_NAME:-race-c}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/${DATASET_NAME}/final_only}"
SCRIPT_PATH="${SCRIPT_PATH:-scripts/run_eval_final_saved_model.sh}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
    echo "Checkpoint root not found: $CHECKPOINT_ROOT" >&2
    exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Per-run script not found: $SCRIPT_PATH" >&2
    exit 1
fi

mkdir -p slurm

mapfile -t RUN_DIRS < <(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo "No run directories found under: $CHECKPOINT_ROOT" >&2
    exit 1
fi

submitted=0
for run_dir in "${RUN_DIRS[@]}"; do
    if ! compgen -G "$run_dir/saved_models/step_*" > /dev/null \
        && ! compgen -G "$run_dir/checkpoint-*" > /dev/null; then
        echo "[skip] No checkpoints in $(basename "$run_dir")"
        continue
    fi

    run_name="$(basename "$run_dir")"
    run_slug="$(printf '%s' "$run_name" | tr -c '[:alnum:]_.-' '_')"

    sbatch_cmd=(
        sbatch
        --export=ALL
        --job-name "evalf_${run_slug}"
        --output "slurm/reasoning_eval_final_${run_slug}_%j.out"
        --error "slurm/reasoning_eval_final_${run_slug}_%j.err"
        "$SCRIPT_PATH"
    )

    echo "[queue] $run_name"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "        RUN_DIR=$run_dir DATASET_NAME=$DATASET_NAME DATA_PATH=$DATA_PATH OUTPUT_DIR=$OUTPUT_DIR ${sbatch_cmd[*]}"
    else
        RUN_DIR="$run_dir" \
        DATASET_NAME="$DATASET_NAME" \
        DATA_PATH="$DATA_PATH" \
        OUTPUT_DIR="$OUTPUT_DIR" \
        "${sbatch_cmd[@]}"
        submitted=$((submitted + 1))
    fi
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete."
else
    echo "Submitted $submitted jobs."
fi
