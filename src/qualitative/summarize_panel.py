"""The decisive control table: is the Understander still ahead of a PROMPT-OPTIMIZED Reasoner?

Three conditions, all through the same token_probe pipeline (no cross-harness offset):
  cot16 default   -- Reasoner as reported in the paper
  cot16 direct    -- Reasoner told to answer with no reasoning (the control)
  flatsimpl       -- Understander (its default already answers directly)

Reports per target: accuracy of each, and the two gaps that matter:
  d_default = flatsimpl - cot16_default   (the paper's headline gap)
  d_direct  = flatsimpl - cot16_direct    (the gap that survives the control)
"""
import json
import os
from statistics import mean, stdev
from math import sqrt

QDIR = "evaluations/qualitative"
SEEDS = ("123", "234", "345")
TARGETS = [("RACE-C*", "race-c"), ("QuAIL", "quail"), ("CosmosQA", "cosmosqa"),
           ("LSAT-RC", "lsatrc"), ("QuALITY", "quality"), ("LB-32k", "lbsmall")]


def sem(v):
    return stdev(v) / sqrt(len(v)) if len(v) > 1 else 0.0


def acc_tok(tag, tgt):
    accs, toks = [], []
    for s in SEEDS:
        p = f"{QDIR}/{tag}_{tgt}_s{s}.jsonl"
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        if not rows:
            continue
        accs.append(mean(r["frac_correct"] for r in rows) * 100)
        toks.append(mean(x["n_tokens"] for r in rows for x in r["samples"]))
    if not accs:
        return None
    return {"acc": mean(accs), "sem": sem(accs), "tok": mean(toks), "n": len(accs)}


def run(scale, sfx):
    print(f"\n{'='*92}\n{scale}  —  accuracy (avg@8, %) and mean CoT tokens\n{'='*92}")
    print(f"{'target':<11}{'cot16 default':>20}{'cot16 DIRECT':>20}{'flatsimpl':>20}"
          f"{'d_def':>8}{'d_dir':>8}")
    dd, dr = [], []
    for label, tgt in TARGETS:
        c = acc_tok(f"cot16{sfx}", tgt)
        d = acc_tok(f"cot16-direct{sfx}", tgt)
        f = acc_tok(f"flatsimpl{sfx}", tgt)
        if not (c and d and f):
            print(f"{label:<11}  (incomplete)")
            continue
        star = lambda x: "" if x["n"] == 3 else f"!{x['n']}"
        gd, gr = f["acc"] - c["acc"], f["acc"] - d["acc"]
        dd.append(gd); dr.append(gr)
        print(f"{label:<11}"
              f"{c['acc']:>10.1f}±{c['sem']:<3.1f}{c['tok']:>5.0f}t{star(c):<2}"
              f"{d['acc']:>10.1f}±{d['sem']:<3.1f}{d['tok']:>5.0f}t{star(d):<2}"
              f"{f['acc']:>10.1f}±{f['sem']:<3.1f}{f['tok']:>5.0f}t{star(f):<2}"
              f"{gd:>+8.1f}{gr:>+8.1f}")
    if dd:
        print("-" * 92)
        print(f"{'MEAN':<11}{'':>60}{mean(dd):>+8.2f}{mean(dr):>+8.2f}")
        print(f"{'':<11}{'':>60}{f'({sum(1 for x in dd if x>0)}/{len(dd)})':>8}"
              f"{f'({sum(1 for x in dr if x>0)}/{len(dr)})':>8}")


run("Qwen3-8B", "")
run("Qwen3-4B", "-4b")
print("\nd_def = Understander - Reasoner(default)  [the paper's gap]")
print("d_dir = Understander - Reasoner(direct)   [gap vs prompt-optimized baseline]")
