"""RACE-trained result tables: rows = methods, columns = datasets.

  1. scale table   : 4B + 8B, base / cot16 / u4c12
  2. data-scaling  : 50-passage vs 100-passage (cap 256), cot16 / u4c12
  3. rollout ratio : cot16 / u4c12 / flatsimpl / u12c4 / u16c0 (4B)

Convention: per-seed single-step dev-argmax, avg@8, mean +- SEM over 123/234/345.
"""
import csv, os, sys
from math import sqrt
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_final_tables import devargmax_ps, load_curve  # noqa: E402

ROOT = os.path.expanduser("~/scratch/SImpL")
os.chdir(ROOT)
SEEDS = ("123", "234", "345")

# in-domain (dev, test) dirs
D50 = ("evaluations/finals_dev", "evaluations/finals_test")
D100 = ("evaluations/scale100_dev", "evaluations/scale100_test")
D8B = ("evaluations/final_8b/dev", "evaluations/final_8b/test")

# column key -> (pretty name, transfer target key, baseline csv stem)
COLS = [
    ("RACE-C*", None, "race-c"),
    ("QuAIL", "quail", "quail"),
    ("CosmosQA", "cosmosqa", "cosmosqa"),
    ("LSAT-RC", "lsatrc", "lsat-rc"),
    ("LSAT-LR", "lsatlr", "lsat-lr"),
    ("QuALITY", "quality", "quality"),
    ("LB-32k", "lbsmall", "longbench-small"),
]


def sem(v):
    return stdev(v) / sqrt(len(v)) if len(v) > 1 else 0.0


def csv_val(p):
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    return float(rows[-1]["cot_accuracy"]) * 100 if rows else None


def xfer(size, method, target, d="evaluations/transfer"):
    """Mean +- SEM over seeds for one transfer cell."""
    v = [csv_val(f"{d}/{size}_race_{method}_to_{target}_s{s}.csv") for s in SEEDS]
    v = [x for x in v if x is not None]
    return (mean(v), sem(v)) if v else None


def baseline(model, stem):
    v = csv_val(f"evaluations/baselines/base_{model}_{stem}.csv")
    return (v, 0.0) if v is not None else None


def fmt(cell):
    return "  --  " if cell is None else f"{cell[0]:.1f}±{cell[1]:.1f}"


def render(title, note, rows):
    """rows: list of (label, [cell|None per column]); label None => separator."""
    w0 = max(len(r[0]) for r in rows if r[0]) + 1
    widths = [max(9, len(c[0]) + 1) for c in COLS]
    print(f"\n{title}\n{note}")
    head = "method".ljust(w0) + "".join(
        c[0].rjust(w) for c, w in zip(COLS, widths)) + "    mean"
    print(head)
    print("-" * len(head))
    for label, cells in rows:
        if label is None:
            print("-" * len(head))
            continue
        vals = [c[0] for c in cells if c is not None]
        m = f"{mean(vals):7.1f}" if len(vals) == len(COLS) else "      ."
        print(label.ljust(w0) + "".join(
            fmt(c).rjust(w) for c, w in zip(cells, widths)) + " " + m)


def delta_row(label, a, b):
    """b - a, elementwise; SEM of the difference is not defined here so show bare delta."""
    cells = [None if (x is None or y is None) else (y[0] - x[0], 0.0)
             for x, y in zip(a, b)]
    vals = [c[0] for c in cells if c is not None]
    txt = "".join(("  --  " if c is None else f"{c[0]:+.1f}").rjust(max(9, len(col[0]) + 1))
                  for c, col in zip(cells, COLS))
    npos = sum(1 for v in vals if v > 0)
    return label, txt, f"{mean(vals):+7.1f}  ({npos}/{len(vals)} pos)" if vals else ""


def print_delta(label, a, b, w0):
    lab, txt, m = delta_row(label, a, b)
    print(lab.ljust(w0) + txt + " " + m)


# --------------------------------------------------------------------------
def row_scale(model, size_key, dirs, cot_m, u_m):
    """Returns (base, cot16, u4c12) row-lists for one model size."""
    base = [baseline(model, stem) for _, _, stem in COLS]
    out = []
    for m_indom, m_xfer in ((cot_m, "cotn16"), (u_m, "u4c12")):
        cells = []
        for name, tgt, _ in COLS:
            if tgt is None:
                r = devargmax_ps(*dirs, "race", m_indom)
                cells.append((r[0], r[1]) if r and r[2] else None)
            else:
                cells.append(xfer(size_key, m_xfer, tgt))
        out.append(cells)
    return base, out[0], out[1]


def table1():
    rows = []
    for model, size_key, dirs, cm, um in (
        ("4B", "4B", D50, "cotn16", "flatsplitv3-u4c12"),
        ("8B", "8B", D8B, "cotn16-short", "flatsplitv3-u4c12-short"),
    ):
        b, c, u = row_scale(model, size_key, dirs, cm, um)
        if rows:
            rows.append((None, None))
        rows += [(f"{model} base", b), (f"{model} +cot16", c), (f"{model} +u4c12", u)]
    render("TABLE 1 - RACE-trained, RC + long context (4B / 8B)",
           "avg@8 cot-accuracy %, mean±SEM over 3 seeds. *in-domain.", rows)
    w0 = max(len(r[0]) for r in rows if r[0]) + 1
    print("-" * 40)
    for model, size_key, dirs, cm, um in (
        ("4B", "4B", D50, "cotn16", "flatsplitv3-u4c12"),
        ("8B", "8B", D8B, "cotn16-short", "flatsplitv3-u4c12-short"),
    ):
        b, c, u = row_scale(model, size_key, dirs, cm, um)
        print_delta(f"{model} Δ(u4c12-cot16)", c, u, w0)


def table2():
    """50-passage vs 100-passage, 4B, selection capped at 256 steps."""
    CAP = 256
    specs = [
        ("50p  +cot16", D50, "cotn16", "4B", "cotn16", "evaluations/transfer"),
        ("50p  +u4c12", D50, "flatsplitv3-u4c12", "4B", "u4c12", "evaluations/transfer"),
        ("100p +cot16", D100, "cotn16", "4Bs100c256", "cotn16", "evaluations/transfers_capped"),
        ("100p +u4c12", D100, "flatsplitv3-u4c12", "4Bs100c256", "u4c12", "evaluations/transfers_capped"),
    ]
    rows, keep = [], {}
    for label, dirs, m_ind, size_key, m_x, xd in specs:
        cells = []
        for name, tgt, _ in COLS:
            if tgt is None:
                r = devargmax_ps(*dirs, "race", m_ind, cap=CAP)
                cells.append((r[0], r[1]) if r and r[2] else None)
            else:
                cells.append(xfer(size_key, m_x, tgt, d=xd))
        rows.append((label, cells))
        keep[label] = cells
    render("TABLE 2 - data scaling: 50 vs 100 training passages (4B)",
           f"selection capped at step {CAP} so both are read at a matched budget.", rows)
    w0 = max(len(r[0]) for r in rows) + 1
    print("-" * 40)
    print_delta("Δ 50p  (u-cot)", keep["50p  +cot16"], keep["50p  +u4c12"], w0)
    print_delta("Δ 100p (u-cot)", keep["100p +cot16"], keep["100p +u4c12"], w0)
    print_delta("Δ cot16 100-50", keep["50p  +cot16"], keep["100p +cot16"], w0)
    print_delta("Δ u4c12 100-50", keep["50p  +u4c12"], keep["100p +u4c12"], w0)
    # the data-efficiency claim: half the data + understanding vs twice the data + cot only
    print_delta("Δ 50u4c12-100cot", keep["100p +cot16"], keep["50p  +u4c12"], w0)


def table3():
    """Understanding:CoT rollout ratio ablation, 4B."""
    specs = [
        ("cot16   (0:16)   0%", "cotn16", "4B", "cotn16"),
        ("u4c12   (4:12)  25%", "flatsplitv3-u4c12", "4B", "u4c12"),
        ("flatsimpl (8:8) 50%", "flatsimplv3", "4Babl", "flatsimpl"),
        ("u12c4   (12:4)  75%", "flatsplitv3-u12c4", "4Babl", "u12c4"),
        ("u16c0   (16:0) 100%", "flatsplitv3-u16c0", "4Babl", "u16c0"),
    ]
    rows, keep = [], {}
    for label, m_ind, size_key, m_x in specs:
        cells = []
        for name, tgt, _ in COLS:
            if tgt is None:
                r = devargmax_ps(*D50, "race", m_ind)
                cells.append((r[0], r[1]) if r and r[2] else None)
            else:
                cells.append(xfer(size_key, m_x, tgt))
        rows.append((label, cells))
        keep[label] = cells
    render("TABLE 3 - understanding:CoT rollout ratio (4B, 16 rollouts total)",
           "compute-matched; all arms use 16 rollouts per prompt.", rows)
    w0 = max(len(r[0]) for r in rows) + 1
    print("-" * 40)
    for label, *_ in specs[1:]:
        print_delta(f"Δ {label.split()[0]}-cot16", keep[specs[0][0]], keep[label], w0)


if __name__ == "__main__":
    table1()
    table2()
    table3()
    print()
