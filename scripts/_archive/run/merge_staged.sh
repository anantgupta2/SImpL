#!/bin/bash
#SBATCH --job-name=merge_staged_%j
#SBATCH --output=slurm/evals/outputs/merge_staged_%j.out
#SBATCH --error=slurm/evals/errors/merge_staged_%j.err
#SBATCH --account=gts-schava6-qcf
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=inferno
#SBATCH --time=1:00:00
module load python/3.10.10 cuda/12.6.1
source ~/r-nisha3-0/oat-env/bin/activate
cd ~/scratch/SImpL
set -e
python scripts/claude/merge_cot_adapter.py Qwen/Qwen3-4B-Base "$1" "$2"
echo "MERGE_DONE $2"