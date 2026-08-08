#!/bin/bash
#SBATCH --job-name=reasoning_eval
#SBATCH --output=slurm/evals/outputs/reasoning_eval.out
#SBATCH --error=slurm/evals/errors/reasoning_eval.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=embers
#SBATCH --time=3:00:00

module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
set -euo pipefail

if [[ $# -eq 1 ]]; then
    CHECKPOINT_ROOT="oat-output/simpl/${1}"
else
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-oat-output/simpl}"
fi

DATASET_NAME="${DATASET_NAME:-race-c}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_CSV="${OUTPUT_CSV:-}"

CMD=(
    python -m src.eval_saved_models
    --dataset_name "$DATASET_NAME"
    --checkpoint_root "$CHECKPOINT_ROOT"
    --tensor_parallel_size 1
    --gpu_memory_utilization 0.95
    --batch_size 128
)

if [[ -n "$DATA_PATH" ]]; then
    CMD+=(--data_path "$DATA_PATH")
fi

if [[ -n "$OUTPUT_CSV" ]]; then
    CMD+=(--output_csv "$OUTPUT_CSV")
fi

"${CMD[@]}"
