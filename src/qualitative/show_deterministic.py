"""Deterministic (greedy, temp=0, n=1) panel: the numbers AND the actual model outputs.

Accuracy here is exact-match on a single greedy generation -- no sampling noise, so a cell either
gets a question right or not. Useful both as a cleaner number and for reading what the models
actually say.

  python -m src.qualitative.show_deterministic                 # accuracy table
  python -m src.qualitative.show_deterministic --show race-c   # + side-by-side outputs
  python -m src.qualitative.show_deterministic --show race-c --n 5 --scale 8b
"""
import argparse
import json
import os
import re
from statistics import mean, stdev
from math import sqrt

QD = "evaluations/qualitative_deterministic"
SEEDS = ("123", "234", "345")
TARGETS = [("RACE-C*", "race-c"), ("QuAIL", "quail"), ("CosmosQA", "cosmosqa"),
           ("LSAT-RC", "lsatrc"), ("QuALITY", "quality"), ("LB-32k", "lbsmall")]


def sem(v):
    return stdev(v) / sqrt(len(v)) if len(v) > 1 else 0.0


def load(tag, tgt, seed):
    p = f"{QD}/{tag}_{tgt}_s{seed}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def stat(tag, tgt):
    accs, toks, n = [], [], 0
    for s in SEEDS:
        rows = load(tag, tgt, s)
        if not rows:
            continue
        accs.append(mean(r["frac_correct"] for r in rows) * 100)
        toks.append(mean(x["n_tokens"] for r in rows for x in r["samples"]))
        n += 1
    if not accs:
        return None
    return {"acc": mean(accs), "sem": sem(accs), "tok": mean(toks), "n": n}


def table():
    for scale, sfx in (("Qwen3-8B", ""), ("Qwen3-4B", "-4b")):
        print(f"\n{'='*88}\n{scale} — GREEDY (temp=0, single sample)\n{'='*88}")
        print(f"{'target':<11}{'cot16 default':>19}{'cot16 DIRECT':>19}{'flatsimpl':>19}{'d_def':>8}{'d_dir':>8}")
        dd, dr = [], []
        for label, tgt in TARGETS:
            c, d, f = stat(f"cot16{sfx}", tgt), stat(f"cot16-direct{sfx}", tgt), stat(f"flatsimpl{sfx}", tgt)
            if not (c and d and f):
                print(f"{label:<11}  (incomplete)")
                continue
            gd, gr = f["acc"] - c["acc"], f["acc"] - d["acc"]
            dd.append(gd); dr.append(gr)
            mk = lambda x: "" if x["n"] == 3 else f"!{x['n']}"
            print(f"{label:<11}{c['acc']:>10.1f}±{c['sem']:<3.1f}{c['tok']:>4.0f}t{mk(c):<1}"
                  f"{d['acc']:>10.1f}±{d['sem']:<3.1f}{d['tok']:>4.0f}t{mk(d):<1}"
                  f"{f['acc']:>10.1f}±{f['sem']:<3.1f}{f['tok']:>4.0f}t{mk(f):<1}"
                  f"{gd:>+8.1f}{gr:>+8.1f}")
        if dd:
            print("-" * 88)
            print(f"{'MEAN':<11}{'':>57}{mean(dd):>+8.2f}{mean(dr):>+8.2f}")
            print(f"{'':<11}{'':>57}{f'({sum(1 for x in dd if x>0)}/{len(dd)})':>8}"
                  f"{f'({sum(1 for x in dr if x>0)}/{len(dr)})':>8}")


def show(tgt, scale, k, seed="123"):
    sfx = "-4b" if scale == "4b" else ""
    conds = [("cot16 default", f"cot16{sfx}"), ("cot16 DIRECT", f"cot16-direct{sfx}"),
             ("flatsimpl", f"flatsimpl{sfx}")]
    data = {lbl: load(tag, tgt, seed) for lbl, tag in conds}
    if any(v is None for v in data.values()):
        print(f"missing data for {tgt} {scale}")
        return
    base = data["cot16 default"]
    # prefer questions where the models disagree -- most informative to read
    idx = sorted(range(len(base)),
                 key=lambda i: -abs(data["flatsimpl"][i]["frac_correct"] - base[i]["frac_correct"]))
    for i in idx[:k]:
        print("\n" + "=" * 88)
        print(f"{tgt} s{seed}  q#{i}  gold={base[i]['gold']}")
        for lbl, _ in conds:
            r = data[lbl][i]
            s = r["samples"][0]
            mark = "OK " if s["correct"] else "XX "
            txt = re.sub(r"\s+", " ", s["text"]).strip()
            print(f"  {mark}{lbl:<14} pred={s['pred']!r:<5} {s['n_tokens']:>4}t | {txt[:220]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default=None, help="target key to print outputs for")
    ap.add_argument("--scale", default="8b", choices=["4b", "8b"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", default="123")
    a = ap.parse_args()
    table()
    if a.show:
        show(a.show, a.scale, a.n, a.seed)
