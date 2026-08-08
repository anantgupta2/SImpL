#!/bin/bash
#SBATCH --job-name=hf_download_%j
#SBATCH --output=slurm/evals/outputs/hf_download_%j.out
#SBATCH --error=slurm/evals/errors/hf_download_%j.err
#SBATCH --account=gts-schava6-qcf
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=embers
#SBATCH --time=4:00:00
#
# Download the new baseline models (network-only, no GPU). Idempotent: snapshot_download
# skips files already present.
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
set -euo pipefail

python - <<'PY'
from huggingface_hub import snapshot_download
for m in [
    "Qwen/Qwen3-32B-Base",
    "OctoThinker/OctoThinker-3B-Hybrid-Base",
    "OctoThinker/OctoThinker-8B-Hybrid-Base",
]:
    print(f"[download] {m} ...", flush=True)
    p = snapshot_download(repo_id=m, ignore_patterns=["*.pth", "*.bin.index.json.tmp", "original/*"])
    print(f"[download] DONE {m} -> {p}", flush=True)
PY
echo "[download] all models present"