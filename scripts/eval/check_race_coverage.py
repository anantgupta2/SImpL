"""Coverage check for every cell the RACE tables read.

Verifies, per cell: 3 seeds present, file non-empty (the broad-except trap writes a header-only
csv with exit 0), and total_questions identical across seeds and across methods within a column.
Run before make_race_tables.py so no table silently averages 2 seeds.
"""
import csv, os, sys
from collections import defaultdict

ROOT = os.path.expanduser("~/scratch/SImpL")
os.chdir(ROOT)
SEEDS = ("123", "234", "345")
TARGETS = ["quail", "cosmosqa", "lsatrc", "lsatlr", "quality", "lbsmall"]

# (label, dev_dir, test_dir, method) -- in-domain curves
INDOM = [
    ("4B  race cot16", "evaluations/finals_dev", "evaluations/finals_test", "cotn16"),
    ("4B  race u4c12", "evaluations/finals_dev", "evaluations/finals_test", "flatsplitv3-u4c12"),
    ("4B  race flatsimpl", "evaluations/finals_dev", "evaluations/finals_test", "flatsimplv3"),
    ("4B  race u12c4", "evaluations/finals_dev", "evaluations/finals_test", "flatsplitv3-u12c4"),
    ("4B  race u16c0", "evaluations/finals_dev", "evaluations/finals_test", "flatsplitv3-u16c0"),
    ("8B  race cot16", "evaluations/final_8b/dev", "evaluations/final_8b/test", "cotn16-short"),
    ("8B  race u4c12", "evaluations/final_8b/dev", "evaluations/final_8b/test", "flatsplitv3-u4c12-short"),
    ("100 race cot16", "evaluations/scale100_dev", "evaluations/scale100_test", "cotn16"),
    ("100 race u4c12", "evaluations/scale100_dev", "evaluations/scale100_test", "flatsplitv3-u4c12"),
]

# (size, method, dir) -- transfer sources
XFER = [
    ("4B", "cotn16", "evaluations/transfer"),
    ("4B", "u4c12", "evaluations/transfer"),
    ("8B", "cotn16", "evaluations/transfer"),
    ("8B", "u4c12", "evaluations/transfer"),
    ("4Bs100c256", "cotn16", "evaluations/transfers_capped"),
    ("4Bs100c256", "u4c12", "evaluations/transfers_capped"),
    ("4Babl", "flatsimpl", "evaluations/transfer"),
    ("4Babl", "u12c4", "evaluations/transfer"),
    ("4Babl", "u16c0", "evaluations/transfer"),
]

BASELINES = [(m, s) for m in ("4B", "8B") for s in
             ("race-c", "quail", "cosmosqa", "lsat-rc", "lsat-lr", "quality", "longbench-small")]

bad = []


def rows_of(p):
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p)))


def nq_of(rows):
    v = {r.get("total_questions") for r in rows if r.get("total_questions")}
    return v.pop() if len(v) == 1 else (sorted(v)[0] + "?" if v else None)


print("=== in-domain curves (dev + test) ===")
for label, dd, td, m in INDOM:
    for which, d in (("dev", dd), ("test", td)):
        got, nsteps, nqs = [], [], set()
        for s in SEEDS:
            r = rows_of(f"{d}/race_{m}_s{s}.csv")
            if r:
                got.append(s)
                nsteps.append(len(r))
                nqs.add(nq_of(r))
        ok = len(got) == 3 and all(n > 0 for n in nsteps)
        flag = "" if ok else "  <== MISSING"
        if len(nqs) > 1:
            flag += f"  <== NQ MISMATCH {sorted(nqs)}"
        if flag:
            bad.append(f"{label} [{which}]")
        print(f"  {label:<20} {which:<4} seeds={len(got)}/3  steps={nsteps}  nq={sorted(nqs)}{flag}")

print("\n=== transfers ===")
nq_by_target = defaultdict(set)
for size, m, d in XFER:
    line = []
    for t in TARGETS:
        got, nqs = 0, set()
        for s in SEEDS:
            r = rows_of(f"{d}/{size}_race_{m}_to_{t}_s{s}.csv")
            if r:
                got += 1
                nqs.add(nq_of(r))
        nq_by_target[t] |= nqs
        line.append(f"{t}={got}" + ("" if got == 3 else "!"))
        if got != 3:
            bad.append(f"{size} {m} -> {t} ({got}/3 seeds)")
    print(f"  {size:<12} {m:<10} " + "  ".join(line))

print("\n  total_questions per target (must be one value each):")
for t in TARGETS:
    v = sorted(x for x in nq_by_target[t] if x)
    print(f"    {t:<10} {v}" + ("" if len(v) <= 1 else "   <== NQ MISMATCH"))
    if len(v) > 1:
        bad.append(f"nq mismatch on {t}: {v}")

print("\n=== baselines (n=1 by design) ===")
miss = [f"{m}/{s}" for m, s in BASELINES if not rows_of(f"evaluations/baselines/base_{m}_{s}.csv")]
print("  missing: " + (", ".join(miss) if miss else "none"))
bad += [f"baseline {x}" for x in miss]

print("\n" + ("ALL CELLS COMPLETE (3/3 seeds everywhere)" if not bad else
              f"{len(bad)} PROBLEM(S):\n  - " + "\n  - ".join(bad)))
sys.exit(1 if bad else 0)
