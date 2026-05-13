#!/bin/bash
#SBATCH --job-name=reasoning_eval
#SBATCH --output=slurm/reasoning_eval.out
#SBATCH --error=slurm/reasoning_eval.err
#SBATCH --account=gts-schava6-qcf
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=inferno
#SBATCH --time=12:00:00

module load python/3.10.10 cuda/12.6.1
source ~/scratch/python-envs/oat-env/bin/activate
cd ~/scratch/CS8803LLM_Self_Learning_Project/reasoning
set -euo pipefail

if [[ $# -eq 1 ]]; then
    CHECKPOINT_ROOT="oat-output/simpl/${1}"
else
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-oat-output/simpl}"
fi

DATA_PATH="${DATA_PATH:-data/race_train_high_42_1000.jsonl}"
OUTPUT_CSV="${OUTPUT_CSV:-}"

CMD=(
    python eval_saved_models.py
    --data_path "$DATA_PATH"
    --checkpoint_root "$CHECKPOINT_ROOT"
    --tensor_parallel_size 1
    --gpu_memory_utilization 0.95
    --batch_size 128
)

if [[ -n "$OUTPUT_CSV" ]]; then
    CMD+=(--output_csv "$OUTPUT_CSV")
fi

"${CMD[@]}"
