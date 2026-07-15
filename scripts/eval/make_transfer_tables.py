#!/usr/bin/env python3
"""Paper-style TRANSFER/OOD tables: one block per model size.

Columns are grouped by the SOURCE dataset the model was RL-trained on (LSAT / RACE); the subcolumns
are the unseen targets it was evaluated on. Rows are Base (zero-shot on the target) / +CoT (cot16) /
+Understanding&CoT (u4c12).

Values = the DEPLOYED checkpoint (per-seed dev-argmax on the SOURCE dev set) evaluated on the target,
avg@8, mean +- pstdev over seeds. The Base row is the untrained model on that same target split, and
is identical across source groups by construction (it never saw either training set) -- it is the
common reference for "did RL on <source> transfer here at all".

Split safety: a target's baseline csv must have the same total_questions as the transfer csvs, else
the cell is flagged rather than silently compared across splits.
"""
import csv, glob, os
from math import sqrt
from statistics import mean, stdev


def sem(vals):
    """Standard error of the mean = sample std (ddof=1) / sqrt(n). Undefined for n<2."""
    return stdev(vals) / sqrt(len(vals)) if len(vals) > 1 else float("nan")

D = "evaluations/transfer"
SEEDS = ("123", "234", "345")

# source -> pretty
SOURCES = [("lsat", "LSAT-AR trained"), ("race", "RACE-C trained")]
# target key -> (pretty, baseline csv stem)
TARGETS = {
    "reclor": ("ReClor", "reclor"),
    "lsatlr": ("LSAT-LR", "lsat-lr"),
    "lsatrc": ("LSAT-RC", "lsat-rc"),
    "quail": ("QuAIL", "quail"),
    "clutrr": ("CLUTRR-18", "clutrr"),
    "clutrrmc4": ("CLUTRR-4", "clutrr-mc4"),
    "folio": ("FOLIO", "folio"),
    "cosmosqa": ("CosmosQA", "cosmosqa"),
    "bbhld": ("BBH-logic", "bbh-logical-deduction"),
    "zebra": ("ZebraLogic", "zebralogic"),
    "bbhtrack": ("BBH-track", "bbh-tracking"),
}
# DROPPED as uninformative (kept on disk, just not reported):
#   proofwriter-d2 - base already 76/82, no headroom, SEM 5.4
#   arc-challenge  - base 70.9/86.6; the +17 over base is FORMAT acquisition (identical for both
#                    methods, delta 0.0), and 8B is near ceiling
#   gsm8k          - base 89.9/92.1, no headroom; unstable at 4B (a cot16 seed scored 60.0 vs 90.8)
# reclor is LSAT-shaped and quail RACE-shaped, so each stays with its natural source; everything
# else is unseen by both and runs from both.
_BOTH = ["lsatlr", "lsatrc", "clutrr", "clutrrmc4", "folio", "cosmosqa", "bbhld",
         "zebra", "bbhtrack"]
SRC_TARGETS = {"lsat": ["reclor"] + _BOTH, "race": ["quail"] + _BOTH}
# NEAR = shares the source's TASK TYPE (what the claim is actually about), NOT merely the same exam.
#   lsat-ar = analytical/logical reasoning -> near: reclor, lsat-lr (both logical reasoning)
#   race-c  = passage reading comprehension -> near: quail, cosmosqa, lsat-rc (all passage RC)
# LSAT-RC is deliberately near for RACE but FAR for LSAT: it is the SAME EXAM as the lsat-ar
# training data but a DIFFERENT task type. That pair disentangles "exam familiarity" from
# "task-type similarity" -- if the near/far split is really about task type, RACE->LSAT-RC should
# be positive while LSAT->LSAT-RC should not, despite LSAT having seen that exam's AR section.
# LSAT-AR is ANALYTICAL reasoning (logic games: constraint satisfaction over a fixed entity
# set) -- NOT logical reasoning (argument analysis). reclor/lsat-lr are the latter, so they
# are FAR despite "feeling" LSAT-ish; lsat-lr came back null, consistent with that.
# The AR-style near targets are the constraint-puzzle ones: zebralogic, bbh-logical-deduction
# (ordering under constraints), bbh-tracking (state tracking).
NEAR = {("lsat", "zebra"), ("lsat", "bbhld"), ("lsat", "bbhtrack"),
        ("race", "quail"), ("race", "cosmosqa"), ("race", "lsatrc")}
METHODS = [("+ CoT (cot16)", "cotn16"), ("+ Understanding & CoT (u4c12)", "u4c12")]
SIZES = ["4B", "8B"]


def read(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    r = rows[-1]
    try:
        return float(r["cot_accuracy"]) * 100.0, int(r["total_questions"])
    except (KeyError, ValueError):
        return None


def cell(size, src, method, tgt):
    """(mean, std, n_seeds, nq) over seeds, or None."""
    vals, nqs = [], set()
    for s in SEEDS:
        r = read(f"{D}/{size}_{src}_{method}_to_{tgt}_s{s}.csv")
        if r:
            vals.append(r[0])
            nqs.add(r[1])
    if not vals:
        return None
    return mean(vals), sem(vals), len(vals), (nqs.pop() if len(nqs) == 1 else None)


def base(size, tgt):
    stem = TARGETS[tgt][1]
    r = read(f"evaluations/baselines/base_{size}_{stem}.csv")
    return r  # (acc, nq) or None


def fmt(c):
    if c is None:
        return "--"
    m, se, n, _ = c
    if n < 2:
        return f"{m:.1f} ({n}s)"
    return f"{m:.1f}±{se:.1f}" + ("" if n == 3 else f" ({n}s)")


def near_table():
    """NEAR transfer only, grouped by source: overarching source column, dataset subcolumns,
    rows = base / training method. These are the cells where understanding actually pays."""
    cols = [(src, t) for src in ("lsat", "race") for t in SRC_TARGETS[src] if (src, t) in NEAR]
    for size in SIZES:
        live = [(s, t) for s, t in cols if any(cell(size, s, m, t) for _, m in METHODS)]
        if not live:
            continue
        print(f"\n### Qwen3-{size}-Base — NEAR transfer (target shares the source's task family)")
        print("    deployed ckpt, avg@8 cot_acc%, ±SEM over 3 seeds\n")
        src_pretty = {"lsat": "LSAT-AR trained", "race": "RACE-C trained"}
        # two-row header: overarching source, then the dataset subcolumn
        h1 = f"| {'':<22} |"
        h2 = f"| {'Method':<22} |"
        sep = f"|{'-'*24}|"
        for s, t in live:
            h1 += f" {src_pretty[s].split()[0]:^13} |"
            h2 += f" {TARGETS[t][0]:^13} |"
            sep += f"{'-'*15}|"
        print(h1)
        print(h2)
        print(sep)

        line = f"| {'Base (zero-shot)':<22} |"
        for s, t in live:
            b = base(size, t)
            line += f" {(f'{b[0]:.1f}' if b else '--'):^13} |"
        print(line)
        for label, m in METHODS:
            line = f"| {label:<22} |"
            for s, t in live:
                line += f" {fmt(cell(size, s, m, t)):^13} |"
            print(line)
        line = f"| {'Δ (Und − CoT)':<22} |"
        for s, t in live:
            c, u = cell(size, s, "cotn16", t), cell(size, s, "u4c12", t)
            line += f" {(f'{u[0]-c[0]:+.1f}' if c and u else '--'):^13} |"
        print(line)


def main():
    """One table PER OOD DATASET: rows = (size, source). Makes each target readable on its own and
    puts the two sources side by side, which is where the near/far split shows up."""
    near_table()
    print("\n" + "=" * 78)
    order = [t for t in TARGETS if any(t in v for v in SRC_TARGETS.values())]
    for t in order:
        rows = [(size, src) for size in SIZES for src, _ in SOURCES
                if t in SRC_TARGETS[src] and any(cell(size, src, m, t) for _, m in METHODS)]
        if not rows:
            continue
        pretty, _ = TARGETS[t]
        print(f"\n### → {pretty}  (unseen target; deployed ckpt, avg@8 cot_acc%, ±SEM over 3 seeds)\n")
        print(f"| {'Model':<6} | {'Trained on':<10} | {'Base':>6} | {'+CoT':>13} | {'+Und&CoT':>13} "
              f"| {'Δ':>6} | {'kind':<4} |")
        print(f"|{'-'*8}|{'-'*12}|{'-'*8}|{'-'*15}|{'-'*15}|{'-'*8}|{'-'*6}|")
        for size, src in rows:
            b = base(size, t)
            c, u = cell(size, src, "cotn16", t), cell(size, src, "u4c12", t)
            bs = "--"
            if b is not None:
                ref = c or u
                bs = f"{b[0]:.1f}" + (" !NQ" if (ref and ref[3] and b[1] != ref[3]) else "")
            d = f"{u[0]-c[0]:+.1f}" if (c and u) else "--"
            kind = "near" if (src, t) in NEAR else "far"
            src_p = "LSAT-AR" if src == "lsat" else "RACE-C"
            print(f"| {size:<6} | {src_p:<10} | {bs:>6} | {fmt(c):>13} | {fmt(u):>13} | {d:>6} "
                  f"| {kind:<4} |")


if __name__ == "__main__":
    main()
