#!/bin/bash
#SBATCH --job-name=cot_only_oat_%j
#SBATCH --output=slurm/cot_only/outputs/cot_only_oat_%j.out
#SBATCH --error=slurm/cot_only/errors/cot_only_oat_%j.err
#SBATCH --account=gts-schava6-qcf
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=inferno
#SBATCH --time=4:00:00


module load python/3.10.10 cuda/12.6.1
source ~/r-nisha3-0/oat-env/bin/activate
cd ~/scratch/SImpL
set -euo pipefail

MODEL_FAMILY="${2:-qwen}"
CONFIG_FILE="configs/$MODEL_FAMILY/${1:-Qwen4B-Base-race}.json"
echo "Using config file: $CONFIG_FILE"

python -m src.run_with_config --config "$CONFIG_FILE" --cot_only --wb_run_name cot-only

