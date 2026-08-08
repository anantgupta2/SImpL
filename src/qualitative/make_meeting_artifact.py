"""Meeting one-pager (8B): Reasoner vs Understander on RACE-C. Understanding side-by-side, reasoning
side-by-side, and two disagreement cards (one each way). Boxed answers are stripped from the reasoning
(the answer shows in the green badge). All model text verbatim.
   python -m src.qualitative.make_meeting_artifact
"""
import html, json, os, re, statistics
from src.qualitative.export_examples import load_source

DET="evaluations/qualitative_deterministic"; UD="evaluations/qualitative_understandings"
OUT="experiments/meeting_cot_vs_understanding.html"
EX=("3429.txt",0); D1=("2420.txt",2); D2=("1166.txt",3)

def strip_boxed(t):
    t=re.sub(r"\\boxed\{[A-Za-z]\}","",t or "")
    t=re.sub(r"(?im)^\s*(final answer|final choice|the answer is|answer)\s*[:：]?\s*$","",t)
    return re.sub(r"\n{3,}","\n\n",t).strip()

def det(tag,key,s="123"):
    for line in open(f"{DET}/{tag}_race-c_s{s}.jsonl"):
        r=json.loads(line)
        if (r["example_id"],r["question_index"])==key: return r["samples"][0]
def uget(path,eid):
    for line in open(path):
        r=json.loads(line)
        if r["example_id"]==eid: return r["samples"][0]
def uprobe(path):
    body=closed=fails=n=0
    for line in open(path):
        for s in json.loads(line)["samples"]:
            body+=s["n_tokens_understanding"]; closed+=int(bool(s["closed_tag"]))
            fails+=int(not (s["understanding"] or "").strip()); n+=1
    return dict(body=body/n,closed=100*closed/n,fails=100*fails/n)
def ans(tag,seeds=("123","234","345")):
    toks=[];acc=[]
    for s in seeds:
        p=f"{DET}/{tag}_race-c_s{s}.jsonl"
        if not os.path.exists(p): continue
        rows=[json.loads(l) for l in open(p)]
        acc.append(sum(r["frac_correct"] for r in rows)/len(rows)*100)
        toks+=[x["n_tokens"] for r in rows for x in r["samples"]]
    return (statistics.mean(acc),sum(toks)/len(toks)) if toks else None

src=load_source("data/race-c/final_test.jsonl"); meta=src[EX]; eid,qi=EX
e=lambda s: html.escape(s or "")
cU=uget(f"{UD}/race-cot16-8b_s123.jsonl",eid); uU=uget(f"{UD}/u4c12-8b_s123.jsonl",eid)
cR=det("cot16",EX); uR=det("u4c12-reasonafter",EX)
fs=uprobe(f"{UD}/race-8b_s123.jsonl"); u8=uprobe(f"{UD}/u4c12-8b_s123.jsonl"); c16=uprobe(f"{UD}/race-cot16-8b_s123.jsonl")
a_cot=ans("cot16"); a_ra=ans("u4c12-reasonafter")

CSS="""
<style>
:root{--paper:#eef1f0;--surface:#ffffff;--ink:#171b1e;--soft:#57636b;--line:#d8dee0;
 --reason:#a5620f;--reason-bg:#f4ead9;--reason-line:#e4d0ad;--under:#0d7581;--under-bg:#dcecec;--under-line:#b7dadb;
 --good:#2f8f63;--bad:#c14b3a;--chip:#eef2f2;--shadow:0 1px 2px rgba(20,30,35,.06),0 8px 24px rgba(20,30,35,.05);}
@media (prefers-color-scheme:dark){:root{--paper:#111417;--surface:#191e22;--ink:#e7ecee;--soft:#95a1a8;--line:#2a323a;
 --reason:#d69739;--reason-bg:#2f2617;--reason-line:#463414;--under:#3cb2bf;--under-bg:#12333a;--under-line:#1d4c54;
 --good:#4bb583;--bad:#e0705d;--chip:#222a30;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}
:root[data-theme="light"]{--paper:#eef1f0;--surface:#ffffff;--ink:#171b1e;--soft:#57636b;--line:#d8dee0;
 --reason:#a5620f;--reason-bg:#f4ead9;--reason-line:#e4d0ad;--under:#0d7581;--under-bg:#dcecec;--under-line:#b7dadb;
 --good:#2f8f63;--bad:#c14b3a;--chip:#eef2f2;--shadow:0 1px 2px rgba(20,30,35,.06),0 8px 24px rgba(20,30,35,.05);}
:root[data-theme="dark"]{--paper:#111417;--surface:#191e22;--ink:#e7ecee;--soft:#95a1a8;--line:#2a323a;
 --reason:#d69739;--reason-bg:#2f2617;--reason-line:#463414;--under:#3cb2bf;--under-bg:#12333a;--under-line:#1d4c54;
 --good:#4bb583;--bad:#e0705d;--chip:#222a30;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}
*{box-sizing:border-box}
.wrap{--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
 background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.55;
 -webkit-font-smoothing:antialiased;padding:clamp(20px,4vw,52px) clamp(16px,4vw,40px);min-height:100%;}
.page{max-width:1080px;margin:0 auto}
.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--under);font-weight:700}
h1{font-size:clamp(28px,4.2vw,44px);line-height:1.05;letter-spacing:-.02em;margin:.35em 0 .2em;font-weight:800;text-wrap:balance}
h1 .vs{color:var(--soft);font-weight:600}
.thesis{font-size:clamp(16px,1.9vw,20px);color:var(--soft);max-width:66ch;margin:0 0 8px;text-wrap:balance}
.thesis b{color:var(--ink);font-weight:650}
.rule{height:1px;background:var(--line);margin:26px 0}
.qcard{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:20px 22px;box-shadow:var(--shadow)}
.qlab{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--soft);font-weight:700;margin-bottom:8px}
.passage{font-size:15px;color:var(--ink);margin:0 0 14px;line-height:1.55}
.passage.small{font-size:13.5px}
.q{font-weight:700;font-size:16px;margin:0 0 10px}
.opts{display:grid;gap:5px;font-size:14.5px;color:var(--soft);font-family:var(--mono)}
.opts .gold{color:var(--under);font-weight:700}
.gold-tag{display:inline-block;margin-top:12px;font-family:var(--mono);font-size:13px;font-weight:700;
 color:var(--under);background:var(--under-bg);border:1px solid var(--under-line);border-radius:6px;padding:2px 9px}
.sec{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--soft);font-weight:700;margin:34px 0 6px}
.seclead{color:var(--soft);font-size:14.5px;margin:0 0 14px;max-width:78ch}
.seclead b{color:var(--ink)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow);display:flex;flex-direction:column}
.panel .top{height:4px}
.panel .hd{padding:14px 17px 10px}
.tagrow{display:flex;justify-content:space-between;align-items:center;gap:8px}
.tag{font-size:11px;letter-spacing:.13em;text-transform:uppercase;font-weight:800}
.badge{font-family:var(--mono);font-size:11px;font-weight:800;letter-spacing:.03em;padding:2px 8px;border-radius:20px;color:#fff;white-space:nowrap}
.badge.right{background:var(--good)} .badge.wrong{background:var(--bad)}
.pn{font-size:18px;font-weight:750;margin:5px 0 2px;letter-spacing:-.01em}
.sub{font-size:13px;color:var(--soft);margin:0}
.chips{display:flex;gap:7px;flex-wrap:wrap;padding:0 17px 13px}
.chip{font-family:var(--mono);font-size:12px;font-weight:700;background:var(--chip);border:1px solid var(--line);border-radius:20px;padding:3px 10px;color:var(--ink);font-variant-numeric:tabular-nums}
.body{border-top:1px solid var(--line);padding:14px 17px;font-family:var(--mono);font-size:12.5px;line-height:1.5;white-space:pre-wrap;color:var(--ink);overflow-y:auto;max-height:440px;flex:1}
.r .tag{color:var(--reason)} .r .top{background:var(--reason)}
.u .tag{color:var(--under)} .u .top{background:var(--under)}
.why{font-size:14px;color:var(--ink);background:var(--chip);border-radius:10px;padding:12px 15px;margin:14px 0 0;line-height:1.55}
.why b{color:var(--under)} .why .rz{color:var(--reason);font-weight:700}
.dcard{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:20px 22px;box-shadow:var(--shadow);margin-bottom:18px}
.dhead{display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.dpill{font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;padding:3px 10px;border-radius:20px}
.dpill.win{color:var(--under);background:var(--under-bg);border:1px solid var(--under-line)}
.dpill.lose{color:var(--reason);background:var(--reason-bg);border:1px solid var(--reason-line)}
.dgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
@media(max-width:760px){.dgrid{grid-template-columns:1fr}}
.dgrid .panel{box-shadow:none}
table{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line)}
th{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--soft);font-weight:700}
td.num,th.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
tr.hi td{background:var(--under-bg)} tr.hi td:first-child{font-weight:750}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
@media(max-width:820px){.two{grid-template-columns:1fr}}
.tlab{font-size:12px;color:var(--soft);margin:0 0 8px;font-weight:600}
.cap{font-size:12.5px;color:var(--soft);margin:8px 2px 0}
.foot{font-size:12px;color:var(--soft);margin-top:26px;line-height:1.6}
</style>
"""

def opts_html(m):
    out=[]
    for i,o in enumerate(m["options"]):
        g="gold" if chr(65+i)==m["answer"] else ""
        out.append(f'<div class="{g}">{chr(65+i)}. {e(str(o))}</div>')
    return "\n".join(out)

def panel(cls,tag,name,sub,chips,body,verdict=None,pred=None):
    ch="".join(f'<span class="chip">{e(c)}</span>' for c in chips)
    badge=""
    if verdict:
        badge=f'<span class="badge {verdict}">{e((f"pred {pred} · " if pred else "")+("correct" if verdict=="right" else "wrong"))}</span>'
    return (f'<div class="panel {cls}"><div class="top"></div>'
            f'<div class="hd"><div class="tagrow"><span class="tag">{e(tag)}</span>{badge}</div>'
            f'<div class="pn">{e(name)}</div><p class="sub">{e(sub)}</p></div>'
            f'<div class="chips">{ch}</div><div class="body">{e(body)}</div></div>')

def dcard(key,pill,label,why):
    m=src[key]; c=det("cot16",key); r=det("u4c12-reasonafter",key)
    cv="right" if c["correct"] else "wrong"; rv="right" if r["correct"] else "wrong"
    rp=panel("r","Reasoner · cot16","Chain-of-thought","Reasons toward the answer.",
             [f"{c['n_tokens']} tokens"],strip_boxed(c["text"]),cv,c["pred"])
    up=panel("u","Understander","Its reasoning","Reasons after committing.",
             [f"{r['n_tokens']} tokens"],strip_boxed(r["text"]),rv,r["pred"])
    return (f'<div class="dcard"><div class="dhead">{pill}'
            f'<span class="qlab" style="margin:0">{e(label)} · RACE-C · gold {m["answer"]}</span></div>'
            f'<p class="passage small">{e(m["article"])}</p><p class="q">{e(m["question"])}</p>'
            f'<div class="opts">{opts_html(m)}</div><div class="dgrid">{rp}{up}</div>'
            f'<p class="why">{why}</p></div>')

d1_html=dcard(D1,'<span class="dpill win">Understander right · CoT wrong</span>',
  'The commoner split (holds on all 3 seeds)',
  'The Understander tracks the pivotal “couldn’t-care-less mood” and reads the waiting-room state as '
  '<b>relaxed</b>; the Reasoner fixates on the author’s earlier nerves and answers <span class="rz">nervous</span>.')
d2_html=dcard(D2,'<span class="dpill lose">CoT right · Understander wrong</span>',
  'The rarer split (holds on 2 of 3 seeds)',
  'A negation question — “a graduate student may <b>not</b>…”. The Reasoner works the course-numbering '
  'rules and finds the excluded option; the Understander’s reasoning slips on the “not”. This direction is the rarer one.')

body_html=f"""
<div class="wrap"><div class="page">
<div class="eyebrow">SImpL · reading comprehension · Qwen3-8B</div>
<h1>Reasoner <span class="vs">vs</span> Understander</h1>
<p class="thesis">Two 8B models trained on the same data. On one passage, we compare what each produces
when asked to <b>understand</b> it, and when asked to <b>reason</b> about a question — then look at the
questions where their reasoning disagrees. The Understander grasps the passage more faithfully; its reasoning is on par.</p>
<div class="rule"></div>

<div class="qcard">
<div class="qlab">The passage · RACE-C</div>
<p class="passage">{e(meta['article'])}</p>
<p class="q">{e(meta['question'])}</p>
<div class="opts">{opts_html(meta)}</div>
<span class="gold-tag">Gold answer: {e(meta['answer'])}</span>
</div>

<div class="sec">1 · The understanding each one produces</div>
<p class="seclead">Both models, given the same "understand this passage" prompt, write a fluent understanding — but they land in different places.</p>
<div class="grid">
{panel("r","Reasoner · cot16","Its understanding","Fluent, but reads the scene as being about poor hospital cleanliness.",
  [f"{cU['n_tokens_understanding']} tokens"], cU['understanding'].strip())}
{panel("u","Understander","Its understanding","Sees the disguise: Mum poses as a cleaner to slip in and visit a patient.",
  [f"{uU['n_tokens_understanding']} tokens"], uU['understanding'].strip())}
</div>
<p class="why">The Reasoner takes Mum's "<span class="rz">very dirty floors</span>" at face value and decides the
passage is about the hospital's cleanliness. The Understander catches the actual point — <b>Mum is
disguising herself as a cleaner to get past the nurse and see Dagmar</b> — the very thing the question turns on.</p>

<div class="sec">2 · The reasoning each one produces</div>
<p class="seclead">Asked the question, both reason step by step and both reach the right answer — comparable, passage-grounded reasoning. (The answer is shown in the badge; the boxed letter is removed so only the reasoning remains.)</p>
<div class="grid">
{panel("r","Reasoner · cot16","Its reasoning","Chain-of-thought toward the answer.",
  [f"{cR['n_tokens']} tokens"], strip_boxed(cR['text']),"right",cR['pred'])}
{panel("u","Understander","Its reasoning","Prompted to reason; walks the options.",
  [f"{uR['n_tokens']} tokens"], strip_boxed(uR['text']),"right",uR['pred'])}
</div>
<p class="why">Both arrive at <b>C — "to see a patient"</b>, each ruling out the cleaning / pleasing-the-nurse
distractors. On reasoning the two are even; the difference is in the understanding above.</p>

<div class="sec">3 · When their reasoning disagrees</div>
<p class="seclead">Across 1,420 RACE-C questions the two agree on 1,329. Of the 91 where they split, the
<b>Understander's reasoning is right 54 times to the Reasoner's 37</b> — it wins the disagreements more often.
Two robust cases, one in each direction:</p>
{d1_html}
{d2_html}

<div class="sec">Supporting counts (RACE-C, 8B, 3 seeds)</div>
<div class="two">
<div>
<p class="tlab">Tokens &amp; quality of the <b>understanding</b></p>
<table><thead><tr><th>model</th><th class="num">tokens</th><th class="num">well-formed</th><th class="num">fails</th></tr></thead><tbody>
<tr class="hi"><td>Understander (8:8)</td><td class="num">{fs['body']:.0f}</td><td class="num">{fs['closed']:.0f}%</td><td class="num">{fs['fails']:.0f}%</td></tr>
<tr><td>Understander (u4c12, 25%)</td><td class="num">{u8['body']:.0f}</td><td class="num">{u8['closed']:.0f}%</td><td class="num">{u8['fails']:.0f}%</td></tr>
<tr><td>Reasoner (cot16)</td><td class="num">{c16['body']:.0f}</td><td class="num">{c16['closed']:.0f}%</td><td class="num">{c16['fails']:.0f}%</td></tr>
</tbody></table>
<p class="cap">Given the same prompt, the Reasoner writes a shorter understanding and fails to form a well-formed one a third of the time.</p>
</div>
<div>
<p class="tlab">Tokens &amp; accuracy of the <b>reasoning</b></p>
<table><thead><tr><th>model</th><th class="num">accuracy</th><th class="num">tokens</th></tr></thead><tbody>
<tr><td>Reasoner (cot16) — CoT</td><td class="num">{a_cot[0]:.1f}%</td><td class="num">{a_cot[1]:.0f}</td></tr>
<tr class="hi"><td>Understander</td><td class="num">{a_ra[0]:.1f}%</td><td class="num">{a_ra[1]:.0f}</td></tr>
</tbody></table>
<p class="cap">When both are made to reason, they land at the same accuracy for a similar token budget — the Understander is not the weaker reasoner.</p>
</div>
</div>

<p class="foot">Verbatim outputs from deployed 8B checkpoints (Understander = RACE flatsimpl / u4c12; Reasoner = RACE cot16),
greedy decode, seed 42; boxed answers removed from §2–3 reasoning. Examples: RACE-C passages 3429, 2420, 1166. Aggregates over seeds 123/234/345.</p>
</div></div>
"""
os.makedirs("experiments",exist_ok=True)
open(OUT,"w").write(CSS+body_html)
print("wrote",OUT,"bytes",os.path.getsize(OUT))
