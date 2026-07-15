#!/usr/bin/env python3
"""Build the JOINT train set: 25 LSAT-AR + 25 RACE-C passages in one file.

Total = 50 passages, i.e. compute-matched against the existing single-dataset 50-passage runs --
the only thing that changes is that half the data comes from each dataset.

Each record carries a `source_dataset` field ("lsat-ar" / "race-c") so training can pick the right
UNDERSTANDING prompt per example (LSAT-AR and RACE-C need different ones; see
oat_prompt_templates.UNDERSTANDING_PROMPT_REGISTRY). The CoT prompt is dataset-agnostic for these
two, so cot16 needs no dispatch.

Passages are taken from the FRONT of the existing train_*_50 pools, which are already verified
clean against both final_dev and final_test -- so the joint set inherits that guarantee and is a
strict subset of what the 50-runs trained on.
"""
import hashlib
import json
import os

OUT = "data/joint-lsat-race/train_2525.jsonl"
SRC = [("lsat-ar", "data/lsat-ar/train_142_50.jsonl", 25),
       ("race-c", "data/race-c/train_92_50.jsonl", 25)]

h = lambda s: hashlib.md5(s.strip().encode()).hexdigest()


def load(p):
    return [json.loads(l) for l in open(p)]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for src, path, n in SRC:
        recs = load(path)[:n]
        for r in recs:
            r["source_dataset"] = src
            rows.append(r)
        print(f"  {src:<8} took {len(recs)} passages from {path}")

    # Contamination guard: the joint set must not touch either dataset's dev/test.
    for src, _, _ in SRC:
        mine = {h(r["article"]) for r in rows if r["source_dataset"] == src}
        for split in ("final_dev", "final_test"):
            other = {h(r["article"]) for r in load(f"data/{src}/{split}.jsonl")}
            n = len(mine & other)
            print(f"  {src:<8} joint ∩ {split:<10} = {n}" + ("  <<< CONTAMINATED" if n else ""))
            assert n == 0, f"{src} joint set overlaps {split}"

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
