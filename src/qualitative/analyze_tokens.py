"""Aggregate the token_probe JSONL dumps into the two things the professor asked for:

  1. TOKEN USAGE: mean CoT tokens per sample, Reasoner (cot16) vs Understander (flatsimpl), per
     dataset, per seed and pooled. Also the accuracy so we can see it is the same eval.
  2. DISAGREEMENT SUBSET: questions where one model class reliably gets it right and the other
     reliably wrong (|frac_correct difference| large), dumped with representative CoT text from
     each so the difference can be read qualitatively.

Reads evaluations/qualitative/<model>_<dataset>_s<seed>.jsonl.
Usage: python -m src.qualitative.analyze_tokens [--disagree_out <path>] [--min_gap 0.75] [--k 40]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean, stdev
from math import sqrt

QDIR = "evaluations/qualitative"
SEEDS = ("123", "234", "345")
DATASETS = ("race-c", "quail")
MODELS = ("cot16", "flatsimpl")


def sem(v):
    return stdev(v) / sqrt(len(v)) if len(v) > 1 else 0.0


def load(tag):
    p = f"{QDIR}/{tag}.jsonl"
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p)]


def per_run_stats(rows):
    """mean tokens/sample, mean correct-sample tokens, avg@N accuracy, mean tokens/question."""
    tok_all, tok_correct, tok_wrong, accs, tok_perq = [], [], [], [], []
    for r in rows:
        qtok = [s["n_tokens"] for s in r["samples"]]
        tok_perq.append(mean(qtok))
        accs.append(r["frac_correct"])
        for s in r["samples"]:
            tok_all.append(s["n_tokens"])
            (tok_correct if s["correct"] else tok_wrong).append(s["n_tokens"])
    return {
        "n_q": len(rows),
        "acc": mean(accs) if accs else 0.0,
        "tok_mean": mean(tok_all) if tok_all else 0.0,
        "tok_correct": mean(tok_correct) if tok_correct else 0.0,
        "tok_wrong": mean(tok_wrong) if tok_wrong else 0.0,
        "tok_perq": tok_perq,  # for cross-seed SEM
    }


def token_table():
    print("=" * 78)
    print("TOKEN USAGE — mean CoT tokens per sample (avg@8), 8B deployed checkpoints")
    print("=" * 78)
    for ds in DATASETS:
        print(f"\n[{ds}]")
        print(f"  {'model':<10}{'acc':>8}{'tok/sample':>13}{'tok|correct':>13}{'tok|wrong':>12}")
        pooled = {}
        for m in MODELS:
            seed_tok, seed_acc, seed_c, seed_w = [], [], [], []
            for s in SEEDS:
                rows = load(f"{m}_{ds}_s{s}")
                if not rows:
                    continue
                st = per_run_stats(rows)
                seed_tok.append(st["tok_mean"]); seed_acc.append(st["acc"])
                seed_c.append(st["tok_correct"]); seed_w.append(st["tok_wrong"])
            if not seed_tok:
                print(f"  {m:<10}  (no data)")
                continue
            pooled[m] = seed_tok
            print(f"  {m:<10}{mean(seed_acc)*100:>7.1f}%"
                  f"{mean(seed_tok):>8.0f}±{sem(seed_tok):<3.0f}"
                  f"{mean(seed_c):>10.0f}   {mean(seed_w):>9.0f}")
        if "cot16" in pooled and "flatsimpl" in pooled:
            d = mean(pooled["flatsimpl"]) - mean(pooled["cot16"])
            rel = 100 * d / mean(pooled["cot16"])
            print(f"  {'Δ (flat-cot)':<10}{'':>8}{d:>+8.0f}     ({rel:+.1f}%)")


def _norm(t):
    return re.sub(r"\s+", " ", t).strip()


def disagreement(min_gap, k, out_path):
    """Per dataset, find questions where flatsimpl and cot16 disagree in reliability, pooled over
    seeds (mean frac_correct across the 3 seeds of each model). Dump the top-k by |gap|."""
    dumped = []
    for ds in DATASETS:
        # key: (example_id, q_index) -> per-model list of (frac_correct, sample_text_first)
        agg = defaultdict(lambda: {m: [] for m in MODELS})
        meta = {}
        for m in MODELS:
            for s in SEEDS:
                rows = load(f"{m}_{ds}_s{s}")
                if not rows:
                    continue
                for r in rows:
                    key = (r["example_id"], r["question_index"])
                    agg[key][m].append((r["frac_correct"], r["samples"]))
                    meta[key] = {"gold": r["gold"], "options_n": r["options_n"]}
        scored = []
        for key, bym in agg.items():
            if not bym["cot16"] or not bym["flatsimpl"]:
                continue
            fc_cot = mean(x[0] for x in bym["cot16"])
            fc_fs = mean(x[0] for x in bym["flatsimpl"])
            scored.append((fc_fs - fc_cot, key, fc_cot, fc_fs, bym))
        # both directions: flatsimpl-wins (gap>0) and cot16-wins (gap<0)
        scored.sort(key=lambda x: x[0])
        picks = [x for x in scored if x[0] <= -min_gap][:k] + \
                [x for x in scored if x[0] >= min_gap][-k:]
        print(f"\n[{ds}] disagreement pairs |gap|>={min_gap}: "
              f"flatsimpl-wins={sum(1 for x in scored if x[0]>=min_gap)}  "
              f"cot16-wins={sum(1 for x in scored if x[0]<=-min_gap)}  (dumping {len(picks)})")
        for gap, key, fc_cot, fc_fs, bym in picks:
            def rep(m, want_correct):
                # a representative sample: prefer one matching the model's majority outcome
                for _, samples in bym[m]:
                    for s in samples:
                        if s["correct"] == want_correct:
                            return s
                return bym[m][0][1][0]
            winner = "flatsimpl" if gap > 0 else "cot16"
            dumped.append({
                "dataset": ds, "example_id": key[0], "question_index": key[1],
                "gold": meta[key]["gold"], "winner": winner,
                "cot16_frac": round(fc_cot, 3), "flatsimpl_frac": round(fc_fs, 3),
                "cot16_sample": rep("cot16", 1 if fc_cot >= 0.5 else 0),
                "flatsimpl_sample": rep("flatsimpl", 1 if fc_fs >= 0.5 else 0),
            })
    if out_path:
        with open(out_path, "w") as f:
            json.dump(dumped, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {len(dumped)} disagreement cases -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disagree_out", default=f"{QDIR}/disagreements.json")
    ap.add_argument("--min_gap", type=float, default=0.75)
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args()
    token_table()
    disagreement(args.min_gap, args.k, args.disagree_out)


if __name__ == "__main__":
    main()
