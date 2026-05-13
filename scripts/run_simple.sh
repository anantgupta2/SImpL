#!/bin/bash
#SBATCH --job-name=simpl_oat
#SBATCH --output=slurm/simpl_oat.out
#SBATCH --error=slurm/simpl_oat.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=embers
#SBATCH --time=8:00:00

module load python/3.10.10 cuda/12.6.1
source ~/scratch/python-envs/oat-env/bin/activate
cd ~/scratch/CS8803LLM_Self_Learning_Project/reasoning
set -euo pipefail


CONFIG_PATH="${CONFIG_PATH:-configs/test_config.json}"
python run_with_config.py --config "$CONFIG_PATH"

