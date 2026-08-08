"""Compare the LSAT-trained Reasoner vs Understander REASONING on LSAT-AR.

Both arms reason at full length here (~505 tok), so unlike the RACE panel there is real text on
both sides to compare. Finds questions where one arm is reliably right and the other reliably
wrong (pooled over seeds) and prints both chains, so the difference in *how* they reason is visible.

  python -m src.qualitative.lsat_reasoning_diff --k 3 --winner flatsimpl
"""
import argparse
import json
import os
import re
from collections import defaultdict
from statistics import mean

QD = "evaluations/qualitative"
SEEDS = ("123", "234", "345")


def load(tag):
    out = defaultdict(list)
    for s in SEEDS:
        p = f"{QD}/{tag}_lsat-ar_s{s}.jsonl"
        if not os.path.exists(p):
            continue
        for i, line in enumerate(open(p)):
            r = json.loads(line)
            out[(r["example_id"], r["question_index"])].append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--winner", default="flatsimpl", choices=["flatsimpl", "cot16"])
    ap.add_argument("--chars", type=int, default=1100)
    a = ap.parse_args()

    C, F = load("cot16"), load("flatsimpl")
    rows = []
    for key in set(C) & set(F):
        fc = mean(r["frac_correct"] for r in C[key])
        ff = mean(r["frac_correct"] for r in F[key])
        rows.append((ff - fc, key, fc, ff))
    sign = 1 if a.winner == "flatsimpl" else -1
    rows.sort(key=lambda x: -sign * x[0])

    print(f"LSAT-AR: {a.winner} wins these questions (pooled over {len(SEEDS)} seeds)\n")
    shown = 0
    for gap, key, fc, ff in rows:
        if sign * gap <= 0.5:
            break
        # pick a representative chain from each arm: one matching that arm's majority outcome
        def pick(store, want):
            for r in store[key]:
                for s in r["samples"]:
                    if s["correct"] == want:
                        return s
            return store[key][0]["samples"][0]
        c_s = pick(C, 1 if fc >= 0.5 else 0)
        f_s = pick(F, 1 if ff >= 0.5 else 0)
        gold = C[key][0]["gold"]
        print("=" * 100)
        print(f"q {key}  gold={gold}   cot16 {fc:.2f} correct | flatsimpl {ff:.2f} correct")
        for lbl, s in (("REASONER (cot16)", c_s), ("UNDERSTANDER (flatsimpl)", f_s)):
            mark = "CORRECT" if s["correct"] else "WRONG  "
            txt = re.sub(r"\n{2,}", "\n", s["text"]).strip()
            print(f"\n--- {lbl}  [{mark}] pred={s['pred']!r} {s['n_tokens']}tok ---")
            print(txt[:a.chars])
        print()
        shown += 1
        if shown >= a.k:
            break
    if not shown:
        print("no questions with a >0.5 reliability gap in that direction")


if __name__ == "__main__":
    main()
