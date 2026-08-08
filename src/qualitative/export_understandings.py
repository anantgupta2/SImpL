"""Export understandings next to their passage + questions, so you can judge whether the
understanding pre-computes what the questions ask. One markdown file per passage."""
import json, os, sys
from src.utils.parsing_utils import normalize_gold_letter, parse_questions

SRC = {"race-8b": "data/race-c/final_test.jsonl", "lsat-8b": "data/lsat-ar/final_test.jsonl"}
UD = "evaluations/qualitative_understandings"; OUT = "evaluations/qualitative_understandings/readable"

def load_src(p):
    out = {}
    for idx, line in enumerate(open(p)):
        line = line.strip()
        if not line: continue
        rec = json.loads(line); art = rec.get("article", "")
        if not isinstance(art, str) or not art.strip(): continue
        qs = []
        for q in parse_questions(rec.get("questions", [])):
            opts = q.get("options", [])
            if not isinstance(opts, list) or len(opts) < 2: continue
            gold = normalize_gold_letter(q.get("answer", ""), len(opts))
            if not str(q.get("question","")).strip() or not gold: continue
            qs.append((q["question"], [str(o) for o in opts], gold))
        if qs: out[str(rec.get("example_id", f"example_{idx}"))] = (art, qs)
    return out

tag = sys.argv[1] if len(sys.argv) > 1 else "race-8b"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
src = load_src(SRC[tag]); rows = [json.loads(l) for l in open(f"{UD}/{tag}_s123.jsonl")]
od = os.path.join(OUT, tag); os.makedirs(od, exist_ok=True)
for n, r in enumerate(rows[:k], 1):
    m = src.get(r["example_id"])
    if not m: continue
    art, qs = m; L = [f"# {tag}  passage {r['example_id']}  ({len(qs)} questions)\n"]
    L.append(f"understanding = {r['samples'][0]['n_tokens_understanding']} tok\n\n## Passage\n\n{art}\n\n## Questions\n")
    for i,(qt,opts,g) in enumerate(qs,1):
        L.append(f"\n**Q{i} (gold {g}):** {qt}")
        for j,o in enumerate(opts): L.append(f"  {chr(ord('A')+j)}. {o}")
    L.append("\n\n## UNDERSTANDING (sample 1)\n\n```\n"+r["samples"][0]["understanding"]+"\n```")
    open(os.path.join(od, f"{n:02d}_{r['example_id'].replace('/','_')}.md"), "w").write("\n".join(L))
print(f"wrote {min(k,len(rows))} -> {od}/")
