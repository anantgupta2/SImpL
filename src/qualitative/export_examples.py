"""Export human-readable examples: the exact passage, question, options, gold answer, and each
model's verbatim output, as one markdown file per question.

The probe JSONL only stores example_id / question_index, so this rejoins against the source
dataset. It replicates eval_saved_models.load_eval_examples' FILTERING exactly (records with an
empty article are dropped; questions need text, >=2 options and a normalizable gold), because
question_index refers to the position in that filtered list -- indexing the raw file would
silently misalign question text with model outputs.

  python -m src.qualitative.export_examples --dataset lsat-ar --mode disagree --k 15
  python -m src.qualitative.export_examples --dataset race-c --tags cot16,flatsimpl --mode disagree
"""
import argparse
import json
import os
from collections import defaultdict
from statistics import mean

from src.utils.parsing_utils import normalize_gold_letter, parse_questions

QD = "evaluations/qualitative"
OUTROOT = "evaluations/qualitative_examples"
SEEDS = ("123", "234", "345")

DATASETS = {
    "lsat-ar":  "data/lsat-ar/final_test.jsonl",
    "race-c":   "data/race-c/final_test.jsonl",
    "quail":    "data/quail/test_42_all.jsonl",
    "cosmosqa": "data/cosmosqa/test_42_all.jsonl",
    "lsatrc":   "data/lsat-rc/test_42_all.jsonl",
    "quality":  "data/quality/test_42_all.jsonl",
}
# probe-tag suffix used in the filenames, per dataset
TAGSUF = {"lsat-ar": "lsat-ar", "race-c": "race-c", "quail": "quail",
          "cosmosqa": "cosmosqa", "lsatrc": "lsatrc", "quality": "quality"}


def load_source(data_path):
    """(example_id, q_index) -> {article, question, options, answer}; mirrors the eval loader."""
    out = {}
    with open(data_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            article = rec.get("article", "")
            if not isinstance(article, str) or not article.strip():
                continue                                   # loader drops these
            q_list = []
            for q in parse_questions(rec.get("questions", [])):
                qt = q.get("question", "")
                opts = q.get("options", [])
                if not isinstance(opts, list) or len(opts) < 2:
                    continue
                gold = normalize_gold_letter(q.get("answer", ""), len(opts))
                if not isinstance(qt, str) or not qt.strip() or not gold:
                    continue
                q_list.append({"question": qt, "options": [str(o) for o in opts], "answer": gold})
            if not q_list:
                continue
            ex_id = str(rec.get("example_id", f"example_{idx}"))
            for qi, q in enumerate(q_list):
                out[(ex_id, qi)] = {"article": article, **q}
    return out


def load_probe(tag, dsuf):
    store = defaultdict(list)
    for s in SEEDS:
        p = f"{QD}/{tag}_{dsuf}_s{s}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            store[(r["example_id"], r["question_index"])].append((s, r))
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lsat-ar", choices=sorted(DATASETS))
    ap.add_argument("--tags", default="cot16,flatsimpl",
                    help="comma-separated probe tags to include, in display order")
    ap.add_argument("--mode", default="disagree", choices=["disagree", "first"])
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--min_gap", type=float, default=0.5)
    ap.add_argument("--max_samples", type=int, default=2,
                    help="how many of the 8 sampled outputs to print per arm per seed")
    a = ap.parse_args()

    dsuf = TAGSUF[a.dataset]
    src = load_source(DATASETS[a.dataset])
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    probes = {t: load_probe(t, dsuf) for t in tags}
    missing = [t for t in tags if not probes[t]]
    if missing:
        raise SystemExit(f"no probe data for {missing} on {a.dataset}")

    keys = set.intersection(*(set(p) for p in probes.values()))
    frac = {t: {k: mean(r["frac_correct"] for _, r in probes[t][k]) for k in keys} for t in tags}
    if a.mode == "disagree" and len(tags) >= 2:
        t0, t1 = tags[0], tags[1]
        sel = sorted(keys, key=lambda k: -abs(frac[t1][k] - frac[t0][k]))
        sel = [k for k in sel if abs(frac[t1][k] - frac[t0][k]) >= a.min_gap][:a.k]
    else:
        sel = sorted(keys)[:a.k]

    outdir = os.path.join(OUTROOT, a.dataset)
    os.makedirs(outdir, exist_ok=True)
    index = []
    for n, key in enumerate(sel, 1):
        meta = src.get(key)
        if meta is None:
            continue
        ex_id, qi = key
        safe = f"{n:02d}_{str(ex_id).replace('/', '_')}_q{qi}"
        path = os.path.join(outdir, f"{safe}.md")
        L = []
        L.append(f"# {a.dataset}  |  example_id={ex_id}  question_index={qi}\n")
        L.append(f"**Gold answer: {meta['answer']}**\n")
        for t in tags:
            L.append(f"- {t}: {frac[t][key]*100:.0f}% of samples correct")
        L.append("\n---\n\n## Passage\n")
        L.append(meta["article"])
        L.append("\n\n## Question\n")
        L.append(meta["question"])
        L.append("\n")
        for i, o in enumerate(meta["options"]):
            L.append(f"{chr(ord('A')+i)}. {o}")
        L.append("\n---\n")
        for t in tags:
            L.append(f"\n## {t}\n")
            for seed, r in probes[t][key]:
                for j, s in enumerate(r["samples"][:a.max_samples]):
                    mark = "CORRECT" if s["correct"] else "WRONG"
                    L.append(f"\n### seed {seed} · sample {j+1} · pred={s['pred'] or '(none)'} "
                             f"· {mark} · {s['n_tokens']} tokens\n")
                    L.append("```\n" + s["text"].strip() + "\n```")
        open(path, "w").write("\n".join(L))
        index.append((safe, meta["answer"], {t: frac[t][key] for t in tags}))

    idx_path = os.path.join(outdir, "INDEX.md")
    with open(idx_path, "w") as f:
        f.write(f"# {a.dataset} — {len(index)} examples ({a.mode})\n\n")
        f.write("| file | gold | " + " | ".join(f"{t} %correct" for t in tags) + " |\n")
        f.write("|---|---|" + "---|" * len(tags) + "\n")
        for safe, gold, fr in index:
            f.write(f"| [{safe}.md]({safe}.md) | {gold} | "
                    + " | ".join(f"{fr[t]*100:.0f}%" for t in tags) + " |\n")
    print(f"wrote {len(index)} examples -> {outdir}/  (see INDEX.md)")


if __name__ == "__main__":
    main()
