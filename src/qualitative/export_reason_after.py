"""Export readable reason_after traces: the u4c12 Understander answers first (boxed) then reasons.
Joins the deterministic token_probe output back to the source passages/questions so each example
shows passage -> question/options/gold -> the model's full answer+reasoning text (verbatim).

  python -m src.qualitative.export_reason_after --tag u4c12-reasonafter --seed 123 --k 3
"""
import argparse
import json
import os

from src.qualitative.export_examples import load_source

DET = "evaluations/qualitative_deterministic"
# target -> (display name, source jsonl); mirrors run_direct_panel.sh TGT
TARGETS = [
    ("race-c",   "RACE-C",        "data/race-c/final_test.jsonl"),
    ("quail",    "QuAIL",         "data/quail/test_42_all.jsonl"),
    ("cosmosqa", "CosmosQA",      "data/cosmosqa/test_42_all.jsonl"),
    ("lsatrc",   "LSAT-RC",       "data/lsat-rc/test_42_all.jsonl"),
    ("quality",  "QuALITY",       "data/quality/test_42_all.jsonl"),
    ("lbsmall",  "LongBench-v2",  "data/longbench-v2-small/test_42_all.jsonl"),
]
PASSAGE_CAP = 1200  # long-context passages (quality/lbsmall) get truncated for display


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="u4c12-reasonafter")
    ap.add_argument("--seed", default="123")
    ap.add_argument("--k", type=int, default=3, help="examples shown per target")
    ap.add_argument("--out", default="experiments/u4c12_reason_after.md")
    a = ap.parse_args()

    L = [f"# u4c12 (8B) — `reason_after`: answer first, then reason\n",
         "The 25%-understanding-share Understander, prompted to give its boxed answer first and THEN "
         "explain its reasoning. Deterministic (greedy), seed "
         f"{a.seed}. Its default behavior is to answer in ~5 tokens; this elicits the reasoning it "
         "otherwise skips. Passage/question/gold rejoined from source; model text verbatim "
         f"(long-context passages truncated to {PASSAGE_CAP} chars).\n"]
    total = 0
    for key, name, path in TARGETS:
        f = f"{DET}/{a.tag}_{key}_s{a.seed}.jsonl"
        if not os.path.exists(f):
            L.append(f"\n## {name}\n\n_(no file yet: {os.path.basename(f)})_\n")
            continue
        src = load_source(path)
        rows = [json.loads(l) for l in open(f)]
        # rank by amount of reasoning actually produced, so we SHOW it reasoning
        rows.sort(key=lambda r: -max((s["n_tokens"] for s in r["samples"]), default=0))
        L.append(f"\n## {name}\n")
        shown = 0
        for r in rows:
            meta = src.get((r["example_id"], r["question_index"]))
            if meta is None:
                continue
            s = r["samples"][0]
            art = meta["article"]
            if len(art) > PASSAGE_CAP:
                art = art[:PASSAGE_CAP].rstrip() + " …[truncated]"
            opts = " · ".join(f"{chr(65+i)}. {o}" for i, o in enumerate(meta["options"]))
            mark = "CORRECT" if s["correct"] else "WRONG"
            L.append(f"\n### {name} · ex {r['example_id']} q{r['question_index']} "
                     f"· gold {meta['answer']} · pred {s['pred'] or '(none)'} · {mark} · {s['n_tokens']} tok\n")
            L.append("> **Passage.** " + art.replace("\n", " ") + "\n")
            L.append(f"> **Q.** {meta['question']}  \n> {opts}\n")
            L.append("```\n" + (s["text"] or "").strip() + "\n```")
            shown += 1; total += 1
            if shown >= a.k:
                break
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(L))
    print(f"wrote {a.out}  ({total} examples)")


if __name__ == "__main__":
    main()
