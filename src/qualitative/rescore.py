"""Re-score saved probe JSONL with a fallback answer extractor.

The reason_first intervention makes the model reason and then conclude in PROSE ("the answer is D")
without a \\boxed{}, which the paper extractor (\\boxed{} only) scores as wrong -> a fake accuracy
crash. Here we re-extract from the saved text: try \\boxed{X}, else fall back to the LAST
'answer is X' / 'correct answer is X' / 'answer: X' style phrase. Reports paper-scored vs
fallback-scored accuracy so the size of the artifact is explicit.

Usage: python -m src.qualitative.rescore <tag_prefix> <dataset>   (e.g. flatsimpl-reasonfirst race-c)
       python -m src.qualitative.rescore --all
"""
import json
import os
import re
import sys
from statistics import mean, stdev
from math import sqrt

QDIR = "evaluations/qualitative"
SEEDS = ("123", "234", "345")
_LETTERS = "ABCDEFGH"

_BOXED = re.compile(r"\\boxed\{\s*([A-H])\s*\}")
# prose fallbacks, ordered; we take the LAST match across all patterns (final answer)
_PROSE = re.compile(
    r"(?:correct answer is|answer is|answer:|answer would be|option|choose)\s*\(?([A-H])\b",
    re.IGNORECASE)


def extract(text, n_opt):
    valid = set(_LETTERS[:n_opt])
    m = list(_BOXED.finditer(text))
    if m:
        c = m[-1].group(1).upper()
        return c if c in valid else ""
    m = list(_PROSE.finditer(text))
    for hit in reversed(m):
        c = hit.group(1).upper()
        if c in valid:
            return c
    return ""


def sem(v):
    return stdev(v) / sqrt(len(v)) if len(v) > 1 else 0.0


def score_file(tag):
    p = f"{QDIR}/{tag}.jsonl"
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    paper_q, fb_q = [], []
    boxed_missing = 0
    for r in rows:
        n_opt = r["options_n"]
        gold = r["gold"]
        paper_c = fb_c = 0
        for s in r["samples"]:
            # paper score = saved 'correct' (boxed-only)
            paper_c += s["correct"]
            fb_pred = extract(s["text"], n_opt)
            fb_c += int(fb_pred == gold)
            if s["pred"] == "" :
                boxed_missing += 1
        n = len(r["samples"])
        paper_q.append(paper_c / n)
        fb_q.append(fb_c / n)
    tot = sum(len(r["samples"]) for r in rows)
    return {"paper": mean(paper_q) * 100, "fb": mean(fb_q) * 100,
            "boxed_missing": 100 * boxed_missing / tot}


def cell(tag_prefix, ds):
    ps, fs = [], []
    miss = []
    for s in SEEDS:
        r = score_file(f"{tag_prefix}_{ds}_s{s}")
        if r:
            ps.append(r["paper"]); fs.append(r["fb"]); miss.append(r["boxed_missing"])
    if not ps:
        return None
    return {"paper": mean(ps), "paper_sem": sem(ps), "fb": mean(fs), "fb_sem": sem(fs),
            "miss": mean(miss), "n": len(ps)}


def main():
    if len(sys.argv) >= 3 and sys.argv[1] != "--all":
        c = cell(sys.argv[1], sys.argv[2])
        print(f"{sys.argv[1]} {sys.argv[2]}: paper={c['paper']:.1f}  fallback={c['fb']:.1f}  "
              f"(no-box {c['miss']:.0f}%)")
        return
    combos = [
        ("flatsimpl-reasonfirst", "race-c"), ("flatsimpl-reasonfirst", "quail"),
        ("u4c12-reasonfirst", "race-c"),
        ("flatsimpl-reason", "race-c"), ("flatsimpl-reason", "quail"),
        ("cot16", "lsat-ar"), ("flatsimpl", "lsat-ar"),
    ]
    print(f"{'condition':<34}{'paper':>10}{'fallback':>12}{'no-box%':>10}")
    print("-" * 66)
    for tag, ds in combos:
        c = cell(tag, ds)
        if not c:
            print(f"{tag+' '+ds:<34}  (missing)")
            continue
        print(f"{tag+' '+ds:<34}{c['paper']:>7.1f}±{c['paper_sem']:<2.1f}"
              f"{c['fb']:>8.1f}±{c['fb_sem']:<2.1f}{c['miss']:>9.0f}%")


if __name__ == "__main__":
    main()
