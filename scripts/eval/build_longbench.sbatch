#!/bin/bash
#SBATCH --job-name=lb_build
#SBATCH --output=slurm/lb_build_%j.out
#SBATCH --error=slurm/lb_build_%j.err
#SBATCH --account=gts-schava6-qcf
#SBATCH --qos=inferno
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:00:00
module load python/3.12.5
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
export HF_HUB_DISABLE_XET=1
python scripts/claude/build_longbench.py
