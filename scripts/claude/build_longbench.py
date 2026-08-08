#!/usr/bin/env python3
"""Build LongBench-v2 as an RC eval set, keeping every example whose prompt fits the 128k window.

LongBench-v2 spans short (<32k) to very long (>128k) contexts. We run Qwen3 with YaRN at
max_model_len=131072, so we keep each example whose full prompt (context + question + 4 choices)
fits with room for generation, and DROP the rest (per user: "fit whatever will fit, ignore the
rest"). This spans the whole ladder from the shortest examples up to the 128k ceiling.

Fast path: tokenizing a 16M-char context is slow and pointless (it is obviously too long), so
drop by char count first, then tokenize only the plausible candidates for the exact budget check.
"""
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer

WINDOW = 131072
GEN_BUDGET = 1024          # answer_max_tokens at eval
TEMPLATE_OVERHEAD = 200    # "Passage:", "Options:", letters, instructions
PROMPT_CAP = WINDOW - GEN_BUDGET - TEMPLATE_OVERHEAD   # ~129848
CHAR_PREFILTER = PROMPT_CAP * 5   # >5 char/token is impossible -> safe to drop without tokenizing
OUT = "data/longbench-v2/test_42_all.jsonl"


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
    d = load_dataset("THUDM/LongBench-v2", split="train")
    kept, dropped_char, dropped_tok = [], 0, 0
    for i, r in enumerate(d):
        ctx = str(r.get("context", "") or "")
        opts = [str(r.get(f"choice_{c}", "") or "") for c in "ABCD"]
        ans = str(r.get("answer", "") or "").strip().upper()
        if not ctx.strip() or not all(opts) or ans not in "ABCD":
            continue
        prompt_text = ctx + str(r.get("question", "")) + "".join(opts)
        if len(prompt_text) > CHAR_PREFILTER:      # obviously too long -> skip tokenizing
            dropped_char += 1
            continue
        ntok = len(tok(prompt_text).input_ids)
        if ntok > PROMPT_CAP:
            dropped_tok += 1
            continue
        kept.append({
            "example_id": str(r.get("_id", f"lb_{i}")),
            "article": ctx,
            "questions": [{"question": str(r.get("question", "")), "options": opts, "answer": ans}],
            "token_size": ntok,
            "length_band": r.get("length", ""),
        })
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(d)} scanned, {len(kept)} kept", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"\nwrote {OUT}: {len(kept)} kept  (dropped {dropped_char} by chars + {dropped_tok} by tokens)")
    ts = sorted(r["token_size"] for r in kept)
    if ts:
        print(f"  kept token range: {ts[0]} .. {ts[-1]}  median={ts[len(ts)//2]}")
        print(f"  length bands: {dict(Counter(r['length_band'] for r in kept))}")


if __name__ == "__main__":
    main()
