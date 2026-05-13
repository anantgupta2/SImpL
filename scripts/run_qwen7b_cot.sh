#!/bin/bash
#SBATCH --job-name=cot_only
#SBATCH --output=slurm/cot_only.out
#SBATCH --error=slurm/cot_only.err
#SBATCH --account=gts-schava6-qcf
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=inferno
#SBATCH --time=8:00:00

module load python/3.10.10 cuda/12.6.1
source ~/scratch/python-envs/oat-env/bin/activate
cd ~/scratch/CS8803LLM_Self_Learning_Project/reasoning
set -euo pipefail


CONFIG_PATH="${CONFIG_PATH:-configs/qwen7b-config.json}"
python run_with_config.py --config "$CONFIG_PATH" --cot_only --wb_run_name cot-only

