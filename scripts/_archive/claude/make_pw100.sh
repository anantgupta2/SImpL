#!/bin/bash
# Create a 100-datapoint ProofWriter-d5 train sample (seed 142, matching the LSAT/RACE
# 100-point sets) -> data/proofwriter-d5/train_142_100.jsonl. Pulls from HF (needs net).
set -euo pipefail
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
python - <<'PY'
from src.utils.preprocess_data import create_or_load_preprocessed_data
_, path = create_or_load_preprocessed_data(
    num_samples=100, split="train", subset=None, seed=142,
    output_dir="data", dataset_name="proofwriter-d5",
)
import json
n = sum(1 for _ in open(path))
print(f"[make_pw100] wrote {path} ({n} rows)")
PY