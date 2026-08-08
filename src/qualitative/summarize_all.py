"""One table for the whole 'why does it work' investigation.

For every (model, dataset, prompt_mode) probe present in evaluations/qualitative/, report:
  acc        avg@8 accuracy (%)
  tok        mean generated tokens per sample
  %<=25      fraction of samples that are answer-only (<=25 tok) -> did the model reason?
  reasoned%  compliance for the reason interventions (100 - %<=25)
All are 3-seed means; SEM shown for accuracy and tokens.

Prefill note: reason_first prepends a forced 'Step-by-step reasoning:\\n1.' that is NOT part of the
generated text, so tok/%<=25 reflect the model's OWN continuation -- the compliance number is honest.
"""
import json
import os
from statistics import mean, stdev
from math import sqrt

QDIR = "evaluations/qualitative"
SEEDS = ("123", "234", "345")


def sem(v):
    return stdev(v) / sqrt(len(v)) if len(v) > 1 else 0.0


def load(tag):
    p = f"{QDIR}/{tag}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def run_stats(rows):
    toks = [s["n_tokens"] for r in rows for s in r["samples"]]
    n = len(toks)
    return {
        "acc": mean(r["frac_correct"] for r in rows) * 100,
        "tok": mean(toks),
        "short": 100 * sum(1 for t in toks if t <= 25) / max(n, 1),
    }


def cell(tag_prefix, ds):
    accs, toks, shorts = [], [], []
    for s in SEEDS:
        rows = load(f"{tag_prefix}_{ds}_s{s}")
        if not rows:
            continue
        st = run_stats(rows)
        accs.append(st["acc"]); toks.append(st["tok"]); shorts.append(st["short"])
    if not accs:
        return None
    return {"n": len(accs), "acc": mean(accs), "acc_sem": sem(accs),
            "tok": mean(toks), "tok_sem": sem(toks), "short": mean(shorts)}


# (row label, tag prefix)  grouped by base model
GROUPS = [
    ("Reasoner (cot16)", [
        ("  default  (reasons)", "cot16"),
        ("  forced direct", "cot16-direct"),
    ]),
    ("Understander 25% (u4c12)", [
        ("  default", "u4c12"),
        ("  forced reason-first", "u4c12-reasonfirst"),
    ]),
    ("Understander 50% (flatsimpl)", [
        ("  default", "flatsimpl"),
        ("  forced reason (weak)", "flatsimpl-reason"),
        ("  forced reason-first", "flatsimpl-reasonfirst"),
    ]),
]
DATASETS = ["race-c", "quail", "lsat-ar"]


def fmt(c):
    if c is None:
        return f"{'--':>22}"
    star = "" if c["n"] == 3 else f"*{c['n']}"
    return f"{c['acc']:>5.1f}±{c['acc_sem']:<3.1f} {c['tok']:>5.0f}±{c['tok_sem']:<3.0f} {c['short']:>3.0f}%{star:<2}"


hdr_ds = "".join(f"{d:>24}" for d in DATASETS)
print(f"\n8B qualitative probe — acc±SEM | tokens/sample±SEM | %answer-only(<=25tok)   (3-seed)\n")
print(f"{'condition':<30}{hdr_ds}")
print("-" * (30 + 24 * len(DATASETS)))
for group, rows in GROUPS:
    print(group)
    for label, tag in rows:
        line = "".join(fmt(cell(tag, ds)) for ds in DATASETS)
        print(f"{label:<30}{line}")
print("\n(%answer-only high = model answers directly with no reasoning; "
      "for a 'reason' row it is the NON-compliance rate)")
