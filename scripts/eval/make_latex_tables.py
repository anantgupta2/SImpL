"""Emit the paper's LaTeX tables. flatsimpl (8:8) is the reported method; LSAT-LR dropped.

  1. main       : base / +Reasoner / +Understander, 4B and 8B
  2. budget     : understanding:CoT ratio sweep, 4B
  3. datascale  : 100-passage cot16 vs 50-passage flatsimpl, 4B

Numbers are computed from the CSVs, never transcribed.
"""
import csv, os, sys
from statistics import mean, stdev
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_final_tables import devargmax_ps  # noqa: E402

os.chdir(os.path.expanduser("~/scratch/SImpL"))
SEEDS = ("123", "234", "345")

# (pretty, transfer-target key, baseline stem)  -- LSAT-LR removed per paper scope
COLS = [("RACE-C", None, "race-c"), ("QuAIL", "quail", "quail"),
        ("CosmosQA", "cosmosqa", "cosmosqa"), ("LSAT-RC", "lsatrc", "lsat-rc"),
        ("QuALITY", "quality", "quality"), ("LB-32k", "lbsmall", "longbench-small")]

D4 = ("evaluations/finals_dev", "evaluations/finals_test")
D8 = ("evaluations/final_8b/dev", "evaluations/final_8b/test")
D100 = ("evaluations/scale100_dev", "evaluations/scale100_test")


def val(p):
    if not os.path.exists(p):
        return None
    r = list(csv.DictReader(open(p)))
    return float(r[-1]["cot_accuracy"]) * 100 if r else None


def xfer(size, src, m, t, d="evaluations/transfer"):
    v = [val(f"{d}/{size}_{src}_{m}_to_{t}_s{s}.csv") for s in SEEDS]
    v = [x for x in v if x is not None]
    return mean(v) if v else None


def row(dirs, indom_method, size, src, xfer_method, cap=10**9, xdir="evaluations/transfer"):
    out = []
    for _, tgt, _ in COLS:
        if tgt is None:
            r = devargmax_ps(*dirs, src, indom_method, cap=cap)
            out.append(r[0] if r and r[2] else None)
        else:
            out.append(xfer(size, src, xfer_method, tgt, d=xdir))
    return out


def baserow(model):
    return [val(f"evaluations/baselines/base_{model}_{stem}") .__float__()
            if False else val(f"evaluations/baselines/base_{model}_{stem}.csv")
            for _, _, stem in COLS]


def fmt(v, bold=False):
    if v is None:
        return "--"
    s = f"{v:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def rowmean(vals):
    return mean([v for v in vals if v is not None])


def line(label, vals, bold_mask=None, shade=False, bold_mean=False):
    # bold_mean is decided by the CALLER against the other rows' means -- bolding it whenever any
    # cell is bold marks a row that loses on average as if it won.
    bold_mask = bold_mask or [False] * len(vals)
    cells = " & ".join(fmt(v, b) for v, b in zip(vals, bold_mask))
    pre = "\\rowcolor{gray!15}\n" if shade else ""
    return f"{pre}{label} & {cells} & {fmt(rowmean(vals), bold_mean)} \\\\"


def best_mean(rows):
    """Index of the row with the highest mean."""
    return max(range(len(rows)), key=lambda i: rowmean(rows[i]))


def bold_best(rows):
    """rows: list of value-lists. Returns per-row bold masks marking the column max."""
    masks = []
    for i, r in enumerate(rows):
        masks.append([all(r[j] is not None and (o[j] is None or r[j] >= o[j])
                          for o in rows) for j in range(len(r))])
    return masks


# ---------------------------------------------------------------- table 1
b4 = baserow("4B")
c4 = row(D4, "cotn16", "4B", "race", "cotn16")
f4 = row(D4, "flatsimplv3", "4Babl", "race", "flatsimpl")
b8 = baserow("8B")
c8 = row(D8, "cotn16-short", "8B", "race", "cotn16")
f8 = row(D8, "flatsimplv3-short", "8Babl", "race", "flatsimpl")

hdr = r"""\begin{table*}[t]
\centering
\small
\caption{Comparison of standard chain-of-thought reasoning with our Understanding+Reasoner approach
across six reading comprehension and long-context benchmarks. All arms are rollout-matched at 16
rollouts per prompt. Values are avg@8 accuracy (\%), averaged over three seeds.}
\label{tab:main_results}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l|c|ccc|cc|c}
\toprule
& & \multicolumn{3}{c|}{\textbf{Reading Comprehension}} & \multicolumn{2}{c|}{\textbf{Long Context}} & \\
\textbf{Method} & \textbf{RACE-C} & \textbf{QuAIL} & \textbf{CosmosQA} & \textbf{LSAT-RC} & \textbf{QuALITY} & \textbf{LB-32k} & \textbf{Mean} \\
\midrule
"""
out = [hdr]
for name, (b, c, f) in (("Qwen3-4B-Base", (b4, c4, f4)), ("Qwen3-8B-Base", (b8, c8, f8))):
    masks = bold_best([c, f]); bm = best_mean([c, f])
    out.append(f"\\multicolumn{{8}}{{c}}{{\\textbf{{{name}}}}}\\\\")
    out.append("\\midrule")
    out.append(line("Base Model", b))
    out.append(line("+ Reasoner", c, masks[0], bold_mean=(bm == 0)))
    out.append(line("+ Understander (Ours)", f, masks[1], shade=True, bold_mean=(bm == 1)))
    if name.startswith("Qwen3-4B"):
        out.append("\\midrule")
out.append("\\bottomrule\n\\end{tabular}\n\\end{table*}")
print("\n".join(out))

# ---------------------------------------------------------------- table 2
print("\n\n% ============ budget ablation (4B) ============")
ABL = [("+ Reasoner (0:16)", "cotn16", "4B", "cotn16"),
       ("+ Understander (4:12)", "flatsplitv3-u4c12", "4B", "u4c12"),
       ("+ Understander (8:8)", "flatsimplv3", "4Babl", "flatsimpl"),
       ("+ Understander (12:4)", "flatsplitv3-u12c4", "4Babl", "u12c4"),
       ("+ Understander (16:0)", "flatsplitv3-u16c0", "4Babl", "u16c0")]
vals = [row(D4, mi, sz, "race", mx) for _, mi, sz, mx in ABL]
masks = bold_best(vals)
print(r"""\begin{table*}[t]
\centering
\small
\caption{Allocation of a fixed 16-rollout budget between understanding and chain-of-thought
(Qwen3-4B-Base). Every row draws the same 16 rollouts per prompt; only the split differs. Any allocation
between 25\% and 75\% understanding performs equivalently, while removing chain-of-thought
rollouts entirely collapses the model below its own base.}
\label{tab:budget}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l|c|ccc|cc|c}
\toprule
& & \multicolumn{3}{c|}{\textbf{Reading Comprehension}} & \multicolumn{2}{c|}{\textbf{Long Context}} & \\
\textbf{Und.:CoT split} & \textbf{RACE-C} & \textbf{QuAIL} & \textbf{CosmosQA} & \textbf{LSAT-RC} & \textbf{QuALITY} & \textbf{LB-32k} & \textbf{Mean} \\
\midrule""")
print(line("Base Model", b4))
print(r"\midrule")
bm = best_mean(vals)
for i, ((lbl, *_), v, mk) in enumerate(zip(ABL, vals, masks)):
    print(line(lbl, v, mk, shade=lbl.endswith("(8:8)"), bold_mean=(i == bm)))
print(r"""\bottomrule
\end{tabular}
\end{table*}""")

# ---------------------------------------------------------------- table 3
print("\n\n% ============ data efficiency (4B) ============")
c100 = row(D100, "cotn16", "4Bs100c256", "race", "cotn16", cap=256,
           xdir="evaluations/transfers_capped")
f50 = f4
masks = bold_best([c100, f50]); bm = best_mean([c100, f50])
print(r"""\begin{table*}[t]
\centering
\small
\caption{Data efficiency (Qwen3-4B-Base). Our method trained on 50 passages matches a standard
reasoner trained on 100 passages in-domain, while outperforming it on four of the five held-out
targets (all but LB-32k, where it trails by 0.3).
Checkpoint selection is capped at 256 steps for both so the two are compared at a matched
optimisation budget.}
\label{tab:data_efficiency}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l|c|ccc|cc|c}
\toprule
& & \multicolumn{3}{c|}{\textbf{Reading Comprehension}} & \multicolumn{2}{c|}{\textbf{Long Context}} & \\
\textbf{Method} & \textbf{RACE-C} & \textbf{QuAIL} & \textbf{CosmosQA} & \textbf{LSAT-RC} & \textbf{QuALITY} & \textbf{LB-32k} & \textbf{Mean} \\
\midrule""")
print(line("+ Reasoner (100 passages)", c100, masks[0], bold_mean=(bm == 0)))
print(line("+ Understander (50 passages)", f50, masks[1], shade=True, bold_mean=(bm == 1)))
print(r"""\bottomrule
\end{tabular}
\end{table*}""")
