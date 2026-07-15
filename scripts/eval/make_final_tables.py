#!/usr/bin/env python3
"""Emit the paper-style final tables: one block per model size, rows = base / +CoT / +Understanding&CoT.

Base rows are the zero-shot untrained baselines (evaluations/baselines/base_{size}_{ds}.csv, a single
row). Trained rows use the CANONICAL convention: per-seed single-step dev-argmax (each seed deploys
its own best-dev checkpoint; read test there; mean +- pstdev over seeds).

Baseline split check: base csv total_questions must equal the trained runs' total_questions, else the
base row is on a different split and is reported as MISMATCH rather than silently compared.
"""
import csv, os, re, sys
from math import sqrt
from statistics import mean, stdev

SEEDS = ["s123", "s234", "s345"]


def sem(vals):
    """Standard error of the mean = sample std (ddof=1) / sqrt(n). Undefined for n<2."""
    return stdev(vals) / sqrt(len(vals)) if len(vals) > 1 else float("nan")

# (size, dev_dir, test_dir). 1.7B dropped: floored near random on LSAT, not part of the story.
SIZES = [
    ("4B", "evaluations/finals_dev", "evaluations/finals_test"),
    ("8B", "evaluations/final_8b/dev", "evaluations/final_8b/test"),
]

# dataset -> (pretty, baseline csv stem per size)
DATASETS = {"lsat": "LSAT-AR", "race": "RACE-C", "quail": "QuAIL"}
BASE_STEM = {"lsat": "lsat-ar", "race": "race-c", "quail": "quail"}

# (size, dataset) -> {row_label: method}. Only what actually exists / is the right control.
ROWS = {
    ("1.7B", "lsat"): [("+ CoT (cot16)", "cotn16"), ("+ Understanding & CoT (u4c12)", "flatsplit-u4c12")],
    ("1.7B", "race"): [("+ CoT (cot16)", "cotn16"), ("+ Understanding & CoT (u4c12)", "flatsplitv3-u4c12")],
    ("4B", "lsat"): [("+ CoT (cot16)", "cotn16"), ("+ Understanding & CoT (u4c12)", "flatsplit-u4c12")],
    ("4B", "race"): [("+ CoT (cot16)", "cotn16"), ("+ Understanding & CoT (u4c12)", "flatsplitv3-u4c12")],
    ("4B", "quail"): [("+ CoT (cot16)", "cotn16"), ("+ Understanding & CoT (flatsimpl)", "flatsimpl")],
    ("8B", "lsat"): [("+ CoT (cot16)", "cotn16"), ("+ Understanding & CoT (u4c12)", "flatsplit-u4c12")],
    # 8B-RACE uses the SHORT runs (matched to the 4B step budget), not -long: the long cot16
    # baseline is uneven (s234 died at step 248 vs 496 for s123/s345) and its dev csv is
    # contaminated with rows from the abandoned pre-OOM run dir. The short runs are all equal
    # length, and the transfer evals deploy the short checkpoints too -- so this keeps in-domain
    # and OOD reading from the same models.
    ("8B", "race"): [("+ CoT (cot16)", "cotn16-short"), ("+ Understanding & CoT (u4c12)", "flatsplitv3-u4c12-short")],
}


def load_curve(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path) as f:
        for r in csv.DictReader(f):
            m = re.search(r"(\d+)", r.get("step", ""))
            if not m:
                continue
            try:
                d[int(m.group(1))] = float(r["cot_accuracy"]) * 100.0
            except (KeyError, ValueError):
                pass
    return d


def nq(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return int(rows[-1]["total_questions"]) if rows else None


def devargmax_ps(dev_dir, test_dir, ds, method, cap=10**9):
    """Per-seed single-step dev-argmax -> test there. Returns (mean, sem, n_seeds, nq)."""
    vals, nqs = [], set()
    for s in SEEDS:
        dp = os.path.join(dev_dir, f"{ds}_{method}_{s}.csv")
        tp = os.path.join(test_dir, f"{ds}_{method}_{s}.csv")
        dev, test = load_curve(dp), load_curve(tp)
        common = [k for k in (set(dev) & set(test)) if k <= cap]
        if not common:
            continue
        best = max(common, key=lambda k: dev[k])
        vals.append(test[best])
        q = nq(tp)
        if q:
            nqs.add(q)
    if not vals:
        return None
    return mean(vals), sem(vals), len(vals), (nqs.pop() if len(nqs) == 1 else None)


def base_acc(size, ds):
    tag = {"1.7B": "1p7B", "4B": "4B", "8B": "8B"}[size]
    if ds not in BASE_STEM:
        return None
    p = f"evaluations/baselines/base_{tag}_{BASE_STEM[ds]}.csv"
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    return float(rows[-1]["cot_accuracy"]) * 100.0, int(rows[-1]["total_questions"])


def main():
    for size, dev_dir, test_dir in SIZES:
        dss = [d for d in DATASETS if (size, d) in ROWS]
        print(f"\n### Qwen3-{size}-Base\n")
        hdr = f"| {'Method':<34} |" + "".join(f" {DATASETS[d]:>14} |" for d in dss)
        print(hdr)
        print("|" + "-" * 36 + "|" + "".join("-" * 16 + "|" for _ in dss))

        # base row
        cells = []
        for d in dss:
            b = base_acc(size, d)
            if b is None:
                cells.append("     n/a".rjust(14))
                continue
            acc, bq = b
            # split check against the trained runs
            ref = devargmax_ps(dev_dir, test_dir, d, ROWS[(size, d)][0][1])
            tag = ""
            if ref and ref[3] and bq != ref[3]:
                tag = " !MISMATCH"
            cells.append(f"{acc:>10.1f}{tag}".rjust(14))
        print(f"| {'Base (zero-shot)':<34} |" + "".join(f" {c} |" for c in cells))

        # trained rows
        n_rows = max(len(ROWS[(size, d)]) for d in dss)
        for i in range(n_rows):
            label = None
            cells = []
            for d in dss:
                rows = ROWS[(size, d)]
                if i >= len(rows):
                    cells.append(" " * 14)
                    continue
                lbl, method = rows[i]
                label = label or lbl
                r = devargmax_ps(dev_dir, test_dir, d, method)
                if r is None:
                    cells.append("     --".rjust(14))
                    continue
                m, sd, n, _ = r
                star = "" if n == 3 else f" ({n}s)"
                cells.append(f"{m:>7.1f}±{sd:<4.1f}{star}".rjust(14))
            print(f"| {label:<34} |" + "".join(f" {c} |" for c in cells))


if __name__ == "__main__":
    main()
