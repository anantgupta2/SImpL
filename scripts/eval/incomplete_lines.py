#!/usr/bin/env python3
"""Print the 0-based manifest line indices whose OUTPUT_CSV does not yet cover every saved
checkpoint of its run (i.e. still need eval). Used by the orchestrator to decide what to submit."""
import sys, os, csv, re, glob

manifest = sys.argv[1]
def csv_steps(p):
    s = set()
    if os.path.exists(p):
        try:
            for r in csv.DictReader(open(p)):
                m = re.search(r'(\d+)', r.get('step', ''))
                if m: s.add(int(m.group(1)))
        except Exception:
            pass
    return s

incomplete = []
for idx, line in enumerate(open(manifest)):
    line = line.rstrip("\n")
    if not line:
        continue
    ds, prefix, dp, out = line.split("\t")
    rundirs = sorted(glob.glob(f"oat-output/{ds}/{prefix}_*"))
    nck = 0
    if rundirs:
        sm = os.path.join(rundirs[-1], "saved_models")
        if os.path.isdir(sm):
            nck = len([x for x in os.listdir(sm) if x.startswith("step")])
    if nck == 0:
        continue  # no checkpoints (deleted run) -> nothing to eval
    if len(csv_steps(out)) < nck:
        incomplete.append(idx)
print(" ".join(str(i) for i in incomplete))
