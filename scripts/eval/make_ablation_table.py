#!/usr/bin/env python3
"""Understanding:CoT rollout-ratio ablation (4B, total rollouts fixed at 16 -> compute-matched).

The split is the only thing that varies: cot16 is the 0%-understanding end of the same axis
(16 cot rollouts, 0 understanding), u16c0 is the 100% end (understanding only, no cot rollouts).
Values are the canonical convention: per-seed single-step dev-argmax, mean +- SEM over 3 seeds.

`uevery4` (8/8 but emitting understanding only every 4th step) was the off-axis point; it was
retired 2026-08-08 and its curves live in `evaluations/_archive/uevery4/`.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_final_tables import devargmax_ps  # noqa: E402

DEV, TEST = "evaluations/finals_dev", "evaluations/finals_test"

# (label, n_understanding, n_cot, lsat_method, race_method)
# flatsimpl sets reasoning_num_samples=8 with no explicit split; simpl_split_oat defaults both
# counts to reasoning_num_samples ("=> reproduces 8/8"), so flatsimpl IS the 8:8 / 50% point.
SPLITS = [
    ("cot16  (0% und)", 0, 16, "cotn16", "cotn16"),
    ("u4c12  (25% und)", 4, 12, "flatsplit-u4c12", "flatsplitv3-u4c12"),
    ("flatsimpl (50% und)", 8, 8, "flatsimpl", "flatsimplv3"),
    ("u12c4  (75% und)", 12, 4, "flatsplit-u12c4", "flatsplitv3-u12c4"),
    ("u16c0  (100% und)", 16, 0, "flatsplit-u16c0", "flatsplitv3-u16c0"),
]
OFF_AXIS = []


def cell(ds, method):
    return devargmax_ps(DEV, TEST, ds, method) if method else None


def fmt(c, base=None):
    if c is None:
        return "     --      "
    m, se, n, _ = c
    s = f"{m:5.1f}±{se:3.1f}" + ("" if n == 3 else f"({n}s)")
    if base is not None:
        s += f" ({m - base:+.1f})"
    return s


def main():
    b_lsat = cell("lsat", "cotn16")
    b_race = cell("race", "cotn16")
    print("\n### Understanding:CoT rollout ratio — Qwen3-4B, total rollouts = 16 (compute-matched)")
    print("    test cot_acc%, per-seed dev-argmax, ±SEM over 3 seeds, (Δ vs cot16)\n")
    print(f"| {'Split':<22} | {'und:cot':>8} | {'LSAT-AR':>22} | {'RACE-C':>22} |")
    print(f"|{'-'*24}|{'-'*10}|{'-'*24}|{'-'*24}|")
    for label, nu, nc, lm, rm in SPLITS:
        ratio = f"{nu}:{nc}"
        l = cell("lsat", lm)
        r = cell("race", rm)
        print(f"| {label:<22} | {ratio:>8} | {fmt(l, b_lsat[0] if b_lsat else None):>22} "
              f"| {fmt(r, b_race[0] if b_race else None):>22} |")
    print()
    for label, lm, rm in OFF_AXIS:
        l = cell("lsat", lm)
        r = cell("race", rm)
        if l or r:
            print(f"  off-axis: {label:<36} LSAT {fmt(l, b_lsat[0] if b_lsat else None)}"
                  f"   RACE {fmt(r, b_race[0] if b_race else None)}")


if __name__ == "__main__":
    main()
