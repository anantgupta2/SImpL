"""Mechanism analysis: why do LSAT-trained SImpL models transfer to RACE better than CoT?
Compares the DIRECT-CoT reasoning traces (cot_output) of LSAT-SImpL vs LSAT-CoT on the RACE test,
pooled across seeds. Run after the full-dump evals land in evaluations/analysis/lsat2race_full/.

    python scripts/claude/analyze_transfer.py [glob]
"""
import json, glob, os, sys, statistics as st, random

PAT = sys.argv[1] if len(sys.argv) > 1 else "evaluations/analysis/lsat2race_full/*.json"


def load(pat):
    models = {}
    for f in glob.glob(pat):
        name = os.path.basename(f)
        recs = json.load(open(f))
        models[name] = {(r["example_id"], r["question_index"]): r for r in recs}
    return models


models = load(PAT)
cot = {k: v for k, v in models.items() if "simpl" not in k and "-simpl" not in k}
sim = {k: v for k, v in models.items() if "simpl" in k}
print("CoT models:", len(cot), "| SImpL models:", len(sim))
if not cot or not sim:
    print("missing models; have:", list(models.keys())); sys.exit(0)

keys = set.intersection(*[set(m) for m in models.values()])
print("common questions:", len(keys))

def acc(ms):
    return st.mean(m[k]["cot_correct"] for m in ms.values() for k in keys)
print("pooled cot_accuracy: CoT=%.4f  SImpL=%.4f  (Δ=%+.4f)" % (acc(cot), acc(sim), acc(sim) - acc(cot)))

# per-question seed-averaged correctness
rows = []
for k in keys:
    c = st.mean(m[k]["cot_correct"] for m in cot.values())
    s = st.mean(m[k]["cot_correct"] for m in sim.values())
    rows.append((k, c, s))
simpl_wins = [k for k, c, s in rows if s - c >= 0.66]   # SImpL majority-right, CoT majority-wrong
cot_wins = [k for k, c, s in rows if c - s >= 0.66]
print("net-flip questions: SImpL>CoT=%d  CoT>SImpL=%d  (diff=%+d)"
      % (len(simpl_wins), len(cot_wins), len(simpl_wins) - len(cot_wins)))

def lens(ms):
    return [len(r["cot_output"]) for m in ms.values() for r in m.values()]
lc, ls = lens(cot), lens(sim)
print("cot_output length (chars): CoT median %d / SImpL median %d" % (st.median(lc), st.median(ls)))

# answer-letter distribution (does one over-pick a letter / refuse-style?)
def letterdist(ms):
    d = {}
    for m in ms.values():
        for r in m.values():
            d[r["cot_prediction"]] = d.get(r["cot_prediction"], 0) + 1
    return {k: d[k] for k in sorted(d)}
print("CoT pred dist:  ", letterdist(cot))
print("SImpL pred dist:", letterdist(sim))

# dump sample SImpL-win traces for qualitative read
cm, sm = list(cot.values())[0], list(sim.values())[0]
random.seed(0)
for k in random.sample(simpl_wins, min(6, len(simpl_wins))):
    rc, rs = cm[k], sm[k]
    print("\n===== Q", k, "| gold", rc["gold_answer"], "=====")
    print("Q:", rc["question"][:160])
    print("--- CoT pred=%s correct=%s ---" % (rc["cot_prediction"], rc["cot_correct"]))
    print(rc["cot_output"][:550])
    print("--- SImpL pred=%s correct=%s ---" % (rs["cot_prediction"], rs["cot_correct"]))
    print(rs["cot_output"][:550])
