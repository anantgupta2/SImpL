#!/bin/bash
#SBATCH --job-name=cot_only_trl_%j
#SBATCH --output=slurm/cot_only/outputs/cot_only_trl_%j.out
#SBATCH --error=slurm/cot_only/errors/cot_only_trl_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=inferno
#SBATCH --time=4:00:00


module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
set -euo pipefail


MODEL_FAMILY="${2:-qwen}"
CONFIG_FILE="configs/$MODEL_FAMILY/${1:-Qwen4B-Base-race}.json"
echo "Using config file: $CONFIG_FILE"

# Extract the number of GPUs from the JSON config using Python
GPUS=$(python -c "import json, sys; print(json.load(open(sys.argv[1])).get('oat_args', {}).get('gpus', 1))" "$CONFIG_FILE")

if [ "$GPUS" -gt 1 ]; then
    echo "Multiple GPUs detected ($GPUS). Launching with torchrun..."
    torchrun --nproc_per_node=$GPUS SImpL/src/algorithm/cot_only_trl.py --config "$CONFIG_FILE"
else
    echo "Single GPU detected. Launching standard python process..."
    python -m src.algorithm.cot_only_trl --config "$CONFIG_FILE"
fi
