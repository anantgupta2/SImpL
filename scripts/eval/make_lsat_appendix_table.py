"""Appendix table: LSAT-AR as a second training source. flatsimpl (8:8) vs cot16.

Groups: in-domain | LSAT-family transfer | very-OOD (ARC/GSM8K, no passage or non-MCQ).
"""
import csv, os, sys
from statistics import mean
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_final_tables import devargmax_ps  # noqa

os.chdir(os.path.expanduser("~/scratch/SImpL"))
S = ("123", "234", "345")
# (pretty, target key, baseline stem)  None target = in-domain
COLS = [("LSAT-AR", None, "lsat-ar"),
        ("LSAT-RC", "lsatrc", "lsat-rc"), ("LSAT-LR", "lsatlr", "lsat-lr"),
        ("ARC-C", "arc", "arc-challenge")]


def val(p):
    if not os.path.exists(p):
        return None
    r = list(csv.DictReader(open(p)))
    return float(r[-1]["cot_accuracy"]) * 100 if r else None


def xf(size, m, t):
    v = [val(f"evaluations/transfer/{size}_lsat_{m}_to_{t}_s{s}.csv") for s in S]
    v = [x for x in v if x is not None]
    return mean(v) if v else None


def rowvals(dirs, indom_m, size, xm):
    out = []
    for _, t, _ in COLS:
        if t is None:
            r = devargmax_ps(*dirs, "lsat", indom_m)
            out.append(r[0] if r and r[2] else None)
        else:
            out.append(xf(size, xm, t))
    return out


def f(v, b=False):
    if v is None:
        return "--"
    return f"\\textbf{{{v:.1f}}}" if b else f"{v:.1f}"


def line(lbl, v, other=None, shade=False):
    bold = [False]*len(v) if other is None else [
        (a is not None and (o is None or a > o)) for a, o in zip(v, other)]
    # a partial row must NOT report a mean -- averaging the cells that happen to exist
    # silently compares different column sets across rows.
    m = mean(v) if all(x is not None for x in v) else None
    bm = (other is not None and m is not None and all(x is not None for x in other)
          and mean(other) < m)
    pre = "\\rowcolor{gray!15}\n" if shade else ""
    return f"{pre}{lbl} & " + " & ".join(f(a, b) for a, b in zip(v, bold)) + f" & {f(m, bm)} \\\\"


D4 = ("evaluations/finals_dev", "evaluations/finals_test")
D8 = ("evaluations/final_8b/dev", "evaluations/final_8b/test")
print(r"""\begin{table*}[t]
\centering
\small
\caption{LSAT-AR as a second training source (Qwen3-4B-Base). The method reproduces its in-domain
gain on a second training set and improves both LSAT transfer targets. The ARC-C column is included
as a distant out-of-domain check; note that most of its base$\rightarrow$trained jump (70.9 to 88.2)
is answer-format acquisition and is identical for both arms, so only the between-arm difference is
informative. At 8B the in-domain gain persists (+0.5) but the transfer gains do not
(LSAT-RC $-$2.9, LSAT-LR $-$1.2); we attribute this to LSAT-AR's short passages (103 words on
average against RACE-C's 358), which leave the understanding little prose to interpret.}
\label{tab:lsat_appendix}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l|c|cc|c|c}
\toprule
& \textbf{In-domain} & \multicolumn{2}{c|}{\textbf{LSAT transfer}} & \textbf{Very OOD} & \\
\textbf{Method} & \textbf{LSAT-AR} & \textbf{LSAT-RC} & \textbf{LSAT-LR} & \textbf{ARC-C} & \textbf{Mean} \\
\midrule""")
for name, dirs, im, size, xm in (("Qwen3-4B-Base", D4, "cotn16", "4B", "cotn16"),):
    pass
for name, dirs, cm, fm, csize, fsize in (
        ("Qwen3-4B-Base", D4, "cotn16", "flatsimpl", "4B", "4Babl"),
) + ((("Qwen3-8B-Base", D8, "cotn16", "flatsimpl", "8B", "8Babl"),)
     if os.environ.get("LSAT_8B") else ()):
    b = [val(f"evaluations/baselines/base_{name.split('-')[1]}_{stem}.csv") for _, _, stem in COLS]
    c = rowvals(dirs, cm, csize, "cotn16")
    fl = rowvals(dirs, fm, fsize, "flatsimpl")
    print(f"\\multicolumn{{6}}{{c}}{{\\textbf{{{name}}}}}\\\\")
    print(r"\midrule")
    print(line("Base Model", b))
    print(line("+ Reasoner", c, fl))
    print(line("+ Understander (Ours)", fl, c, shade=True))
    if name.startswith("Qwen3-4B") and os.environ.get("LSAT_8B"):
        print(r"\midrule")
print(r"""\bottomrule
\end{tabular}
\end{table*}""")
