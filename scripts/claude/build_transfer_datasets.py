#!/usr/bin/env python3
"""Build the test jsonl for transfer/OOD datasets.

NOTE: `python -m src.utils.preprocess_data` is a hardcoded race-c smoke test that ignores argv --
it will silently rebuild race-c no matter what flags you pass. Use this instead.

  python scripts/claude/build_transfer_datasets.py clutrr folio cosmosqa bbh-logical-deduction
  python scripts/claude/build_transfer_datasets.py           # all of the below
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.preprocess_data import preprocess_race_data  # noqa: E402

DEFAULT = ["clutrr", "clutrr-mc4", "folio", "cosmosqa", "bbh-logical-deduction"]

for ds in (sys.argv[1:] or DEFAULT):
    try:
        data, path = preprocess_race_data(num_samples=None, split="test", subset=None,
                                          seed=42, dataset_name=ds)
        nq = sum(len(r["questions"]) for r in data)
        print(f"[ok]   {ds:<24} records={len(data):<6} questions={nq:<6} -> {path}")
    except Exception as e:
        print(f"[FAIL] {ds:<24} {type(e).__name__}: {str(e)[:120]}")
