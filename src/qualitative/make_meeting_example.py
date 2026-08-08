"""Build a meeting-ready markdown: one example contrasting CoT vs Understanding reasoning, plus the
two token tables. All model text is verbatim from the deterministic / understanding-probe JSONL.

  python -m src.qualitative.make_meeting_example
"""
import json
import os
import re
import statistics

from src.qualitative.export_examples import load_source

DET = "evaluations/qualitative_deterministic"
UD = "evaluations/qualitative_understandings"
OUT = "experiments/meeting_cot_vs_understanding.md"
EXAMPLE = ("696.txt", 0)   # England-weather, 141 words, gold A; both models correct


def det(tag, key, s="123"):
    for line in open(f"{DET}/{tag}_race-c_s{s}.jsonl"):
        r = json.loads(line)
        if (r["example_id"], r["question_index"]) == key:
            return r["samples"][0]


def und_body(eid):
    for line in open(f"{UD}/u4c12-8b_s123.jsonl"):
        r = json.loads(line)
        if r["example_id"] == eid:
            return r["samples"][0]


def uprobe(path):
    full = body = closed = fails = n = 0
    if not os.path.exists(path):
        return None
    for line in open(path):
        for s in json.loads(line)["samples"]:
            full += s["n_tokens_full"]; body += s["n_tokens_understanding"]
            closed += int(bool(s["closed_tag"])); fails += int(not (s["understanding"] or "").strip()); n += 1
    return dict(full=full/n, body=body/n, closed=100*closed/n, fails=100*fails/n)


def ans(tag, seeds=("123", "234", "345")):
    toks, acc = [], []
    for s in seeds:
        p = f"{DET}/{tag}_race-c_s{s}.jsonl"
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        acc.append(sum(r["frac_correct"] for r in rows)/len(rows)*100)
        toks += [x["n_tokens"] for r in rows for x in r["samples"]]
    return (statistics.mean(acc), sum(toks)/len(toks)) if toks else None


def main():
    src = load_source("data/race-c/final_test.jsonl")
    meta = src[EXAMPLE]; eid, qi = EXAMPLE
    cot = det("cot16", EXAMPLE)
    ra = det("u4c12-reasonafter", EXAMPLE)
    direct = det("u4c12", EXAMPLE)
    ub = und_body(eid)

    L = []
    L.append("# CoT vs Understanding — meeting example + token counts\n")
    L.append("Same 8B model family, same RACE-C question, deterministic (greedy). "
             "**Both answers are correct** — the contrast is the *style* of reasoning, not accuracy.\n")

    L.append("\n## The passage (141 words)\n")
    L.append("> " + meta["article"].replace("\n", "\n> ") + "\n")
    opts = "\n".join(f"> {chr(65+i)}. {o}" for i, o in enumerate(meta["options"]))
    L.append(f"\n**Q. {meta['question']}**\n\n{opts}\n\n> **Gold: {meta['answer']}**\n")

    L.append("\n---\n")
    L.append(f"\n## 1. The Reasoner (cot16) — chain-of-thought *about the question* ({cot['n_tokens']} tokens)\n")
    L.append("It runs a generic per-question procedure and re-derives everything from the passage each time.\n")
    L.append("```\n" + cot["text"].strip() + "\n```")

    L.append(f"\n## 2. The Understander — an understanding *of the passage* ({ub['n_tokens_understanding']} tokens), then a direct answer\n")
    L.append("The understanding is built **once per passage** (interpretation, attitude, pivotal words, "
             "referents); every question then becomes a lookup. Its actual answer is a bare letter:\n")
    L.append("```\n" + (ub["understanding"] or "").strip() + "\n```")
    L.append(f"**→ answer ({direct['n_tokens']} tokens):**")
    L.append("```\n" + direct["text"].strip() + "\n```")

    L.append(f"\n## 3. The Understander *forced to reason* (reason_after, {ra['n_tokens']} tokens)\n")
    L.append("Prompted to answer first, then justify. It can reason — grounded, quotes the passage, "
             "rules out each option — but it did not need to (same answer, same accuracy).\n")
    L.append("```\n" + ra["text"].strip() + "\n```")

    # ---- token tables ----
    fs, c16 = uprobe(f"{UD}/race-8b_s123.jsonl"), uprobe(f"{UD}/race-cot16-8b_s123.jsonl")
    u8 = uprobe(f"{UD}/u4c12-8b_s123.jsonl")
    L.append("\n---\n")
    L.append("\n## Token counts A — when *prompted to produce an understanding* (RACE-C, 8B)\n")
    L.append("| model | understanding tokens | well-formed (closed-tag) | fails to produce |")
    L.append("|---|---|---|---|")
    L.append(f"| **Understander** (flatsimpl 8:8) | {fs['body']:.0f} | {fs['closed']:.0f}% | {fs['fails']:.0f}% |")
    L.append(f"| Understander (u4c12, 25% share) | {u8['body']:.0f} | {u8['closed']:.0f}% | {u8['fails']:.0f}% |")
    L.append(f"| **Reasoner** (cot16) | {c16['body']:.0f} | {c16['closed']:.0f}% | {c16['fails']:.0f}% |")
    L.append("\n*The Reasoner, given the same prompt, produces a shorter understanding and fails to emit "
             "a well-formed one a third of the time — it was never trained to represent the passage.*\n")

    und_ra, cot_norm, und_norm = ans("u4c12-reasonafter"), ans("cot16"), ans("flatsimpl")
    L.append("\n## Token counts B — when *answering* (RACE-C, 8B, deterministic)\n")
    L.append("| condition | accuracy | tokens to answer |")
    L.append("|---|---|---|")
    L.append(f"| Reasoner (cot16), **normal** (CoT) | {cot_norm[0]:.1f}% | {cot_norm[1]:.0f} |")
    L.append(f"| Understander, **reason_after** (forced) | {und_ra[0]:.1f}% | {und_ra[1]:.0f} |")
    L.append(f"| **Understander, normal (direct)** | {und_norm[0]:.1f}% | **{und_norm[1]:.0f}** |")
    L.append(f"\n*Same accuracy (~86%) three ways. The Reasoner needs ~{cot_norm[1]:.0f} tokens of CoT; the "
             f"Understander, forced to reason, uses ~{und_ra[1]:.0f} — but its **normal** answer is "
             f"{und_norm[1]:.0f} tokens. The understanding did the work up front.*\n")

    open(OUT, "w").write("\n".join(L))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
