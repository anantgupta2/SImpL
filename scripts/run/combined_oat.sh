#!/bin/bash
#SBATCH --job-name=combined_oat_%j
#SBATCH --output=slurm/combined/outputs/combined_oat_%j.out
#SBATCH --error=slurm/combined/errors/combined_oat_%j.err
#SBATCH --account=gts-schava6-qcf
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=inferno
#SBATCH --time=8:00:00


module load python/3.10.10 cuda/12.6.1
source ~/r-nisha3-0/oat-env/bin/activate
cd ~/scratch/SImpL
set -euo pipefail

MODEL_FAMILY="${3:-qwen}"
SEED="${2:-42}"
CONFIG_FILE="configs/$MODEL_FAMILY/${1:-Qwen4B-Base-race}.json"
echo "Using config file: $CONFIG_FILE"

python -m src.run_with_config --config "$CONFIG_FILE" --combined --wb_run_name "combined-oat_${SEED}" --seed "$SEED"

