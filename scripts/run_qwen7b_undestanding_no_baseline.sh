#!/bin/bash
#SBATCH --job-name=understanding_only-no-baseline
#SBATCH --output=slurm/outputs/understanding_only-no-baseline.out
#SBATCH --error=slurm/errors/understanding_only-no-baseline.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a40:2
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short


source ~/.bashrc
conda activate oat-llm
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd ~/flash/SImpL
set -euo pipefail


CONFIG_PATH="${CONFIG_PATH:-configs/qwen7b-instruct-config.json}"
python -m src.run_with_config --config "$CONFIG_PATH" --understanding_only --wb_run_name skynet-understanding-no-baseline --use_baseline_reward false
