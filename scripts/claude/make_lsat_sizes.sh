#!/bin/bash
# Create LSAT-AR 50- and 200-passage train samples (seed 142, nested with train_142_100).
set -euo pipefail
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
python - <<'PY'
from src.utils.preprocess_data import create_or_load_preprocessed_data
for n in (50, 200):
    _, p = create_or_load_preprocessed_data(
        num_samples=n, split="train", subset=None, seed=142,
        output_dir="data", dataset_name="lsat-ar",
    )
    rows = sum(1 for _ in open(p))
    print(f"[make_lsat_sizes] wrote {p} ({rows} rows)")
PY