#!/usr/bin/env python3
"""Build a JOINT LSAT-AR + RACE-C train set (both datasets in one file).

  25+25 -> 50 passages, compute-matched against the single-dataset 50-passage runs.
  50+50 -> 100 passages, compute-matched against the scale-100 runs (and a strict superset
           of 25+25), so "does mixing help?" can be asked at both data sizes.

Each record carries a `source_dataset` field ("lsat-ar" / "race-c") so training can pick the right
UNDERSTANDING prompt per example (LSAT-AR and RACE-C need different ones; see
oat_prompt_templates.UNDERSTANDING_PROMPT_REGISTRY). The CoT prompt is dataset-agnostic for these
two, so cot16 needs no dispatch.

Passages are taken from the FRONT of the existing train_*_50 pools, which are already verified
clean against both final_dev and final_test -- so the joint set inherits that guarantee and is a
subset of what the 50-runs trained on (50+50 is exactly their union).

Usage:
  python scripts/claude/build_joint_dataset.py          # 25+25 (default)
  python scripts/claude/build_joint_dataset.py 50 50    # 50 LSAT + 50 RACE
"""
import hashlib
import json
import os
import sys

POOL = {"lsat-ar": "data/lsat-ar/train_142_50.jsonl",
        "race-c": "data/race-c/train_92_50.jsonl"}

h = lambda s: hashlib.md5(s.strip().encode()).hexdigest()


def load(p):
    return [json.loads(l) for l in open(p)]


def main():
    n_lsat = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    n_race = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    SRC = [("lsat-ar", POOL["lsat-ar"], n_lsat), ("race-c", POOL["race-c"], n_race)]
    OUT = f"data/joint-lsat-race/train_{n_lsat}{n_race}.jsonl"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for src, path, n in SRC:
        recs = load(path)[:n]
        assert len(recs) == n, f"{path} has only {len(recs)} passages, need {n}"
        for r in recs:
            r["source_dataset"] = src
            rows.append(r)
        print(f"  {src:<8} took {len(recs)} passages from {path}")

    # Contamination guard. TEST overlap is fatal -- it would invalidate the reported number.
    # DEV overlap only affects checkpoint selection, and the train_*_50 pools already carry one
    # known LSAT near-duplicate (train `india1_1-G_3` ~ dev `200706_1-G_3`, 6/7 questions shared:
    # the same logic game reprinted across two exams under different ids, so id-based splitting
    # never caught it). 50+50 IS the union of both train-50 pools, so it inherits exactly that --
    # the same exposure as the LSAT-50 runs it is compared against. Warn, don't fail.
    for src, _, _ in SRC:
        mine = {h(r["article"]) for r in rows if r["source_dataset"] == src}
        for split in ("final_dev", "final_test"):
            other = {h(r["article"]) for r in load(f"data/{src}/{split}.jsonl")}
            n = len(mine & other)
            tag = ""
            if n and split == "final_test":
                tag = "  <<< FATAL: test contamination"
            elif n:
                tag = "  <<< dev only (selection), see comment above"
            print(f"  {src:<8} joint ∩ {split:<10} = {n}{tag}")
            if split == "final_test":
                assert n == 0, f"{src} joint set overlaps final_test"

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    nq = sum(len(r["questions"]) for r in rows)
    print(f"\nwrote {OUT}: {len(rows)} passages, {nq} questions")
    for src, _, _ in SRC:
        sub = [r for r in rows if r["source_dataset"] == src]
        print(f"   {src:<8} {len(sub):>3} passages, {sum(len(r['questions']) for r in sub):>4} questions")


if __name__ == "__main__":
    main()
