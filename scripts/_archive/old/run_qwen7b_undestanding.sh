#!/bin/bash
#SBATCH --job-name=understanding_only
#SBATCH --output=slurm/outputs/understanding_only.out
#SBATCH --error=slurm/errors/understanding_only.err
#SBATCH --account=gts-schava6-qcf
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=inferno
#SBATCH --time=10:00:00


module load python/3.10.10 cuda/12.6.1
source ~/r-nisha3-0/python-envs/oat-env/bin/activate
cd ~/scratch/SImpL
set -euo pipefail


CONFIG_PATH="${CONFIG_PATH:-configs/qwen7b-instruct-config.json}"
python -m src.run_with_config --config "$CONFIG_PATH" --understanding_only --wb_run_name understanding
