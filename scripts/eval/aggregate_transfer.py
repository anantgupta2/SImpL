#!/usr/bin/env python3
"""Aggregate cross-dataset transfer evals (evaluations/transfer/*.csv).

Each CSV is a SINGLE row: the deployed (per-seed dev-argmax) checkpoint of one run, evaluated
(cot_accuracy, avg@8) on a transfer dataset it was NOT trained on. Files named
`{size}_{source}_{method}_to_{target}_s{seed}.csv` (e.g. 4B_lsat_u4c12_to_reclor_s123.csv).
Reports cot16 vs u4c12 (mean±std over seeds) and Δ = u4c12 − cot16 per (size, source→target).
"""
import csv, glob, os, re
from statistics import mean, pstdev

D = "evaluations/transfer"


def acc(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    # single deployed checkpoint -> take the (only) row's cot_accuracy
    try:
        return float(rows[-1]["cot_accuracy"]) * 100.0
    except (KeyError, ValueError):
        return None


def cell(size, source, method, target):
    vals = [acc(f"{D}/{size}_{source}_{method}_to_{target}_s{s}.csv") for s in ("123", "234", "345")]
    vals = [v for v in vals if v is not None]
    return vals


CELLS = [("4B", "lsat", "reclor"), ("8B", "lsat", "reclor"),
         ("4B", "race", "quail"), ("8B", "race", "quail"),
         ("4B", "lsat", "pwd2"), ("8B", "lsat", "pwd2"),
         ("4B", "race", "pwd2"), ("8B", "race", "pwd2"),
         ("4B", "lsat", "arc"), ("8B", "lsat", "arc"),
         ("4B", "race", "arc"), ("8B", "race", "arc")]
print(f"Transfer (deployed checkpoint, avg@8 cot_acc%)\n{'size src->tgt':<18}"
      f"{'cot16':>12}{'u4c12':>12}{'Δ(u4c12-cot16)':>16}   n(seeds)")
for size, src, tgt in CELLS:
    c = cell(size, src, "cotn16", tgt)
    u = cell(size, src, "u4c12", tgt)
    if not c or not u:
        print(f"{size} {src}->{tgt:<10}  (incomplete: cot16={len(c)}/3 u4c12={len(u)}/3)")
        continue
    cm, um = mean(c), mean(u)
    print(f"{size} {src}->{tgt:<8}{cm:>7.1f}±{pstdev(c):<4.1f}{um:>7.1f}±{pstdev(u):<4.1f}"
          f"{um-cm:>+13.1f}     {len(c)}/{len(u)}")