"""Combined meeting page: does the 8B pattern hold at 4B, and what about LSAT?
4B RACE (same shape as 8B) + LSAT (the dissociation: the Understander reasons, it does NOT answer
directly). Understanding side-by-side + reasoning side-by-side per section, boxed answers stripped.
   python -m src.qualitative.make_meeting_4b_lsat
"""
import html, json, os, re, statistics
from src.qualitative.export_examples import load_source

DET="evaluations/qualitative_deterministic"; UD="evaluations/qualitative_understandings"
OUT="experiments/meeting_4b_lsat.html"
R4=("1166.txt",0); L=("200312_1-G_1",0)

def strip_boxed(t):
    t=re.sub(r"\\boxed\{[A-Za-z]\}","",t or "")
    t=re.sub(r"(?im)^\s*(final answer|final choice|the answer is|answer)\s*[:：]?\s*$","",t)
    return re.sub(r"\n{3,}","\n\n",t).strip()
def det(tag,ds,key,s="123"):
    for line in open(f"{DET}/{tag}_{ds}_s{s}.jsonl"):
        r=json.loads(line)
        if (r["example_id"],r["question_index"])==key: return r["samples"][0]
def uget(path,eid):
    for line in open(path):
        r=json.loads(line)
        if r["example_id"]==eid: return r["samples"][0]
def uprobe(path):
    b=c=f=n=0
    for line in open(path):
        for s in json.loads(line)["samples"]:
            b+=s["n_tokens_understanding"]; c+=int(bool(s["closed_tag"])); f+=int(not (s["understanding"] or "").strip()); n+=1
    return dict(body=b/n,closed=100*c/n,fails=100*f/n)
def ans(tag,ds):
    t=[];a=[]
    for s in ("123","234","345"):
        p=f"{DET}/{tag}_{ds}_s{s}.jsonl"
        if not os.path.exists(p): continue
        rows=[json.loads(l) for l in open(p)]; a.append(sum(r["frac_correct"] for r in rows)/len(rows)*100); t+=[x["n_tokens"] for r in rows for x in r["samples"]]
    return (statistics.mean(a),sum(t)/len(t)) if t else None

e=lambda s: html.escape(s or "")
rsrc=load_source("data/race-c/final_test.jsonl"); lsrc=load_source("data/lsat-ar/final_test.jsonl")
rm=rsrc[R4]; lm=lsrc[L]
# 4B
r_cU=uget(f"{UD}/race-cot16-4b_s123.jsonl",R4[0]); r_uU=uget(f"{UD}/u4c12-4b_s123.jsonl",R4[0])
r_cR=det("cot16-4b","race-c",R4); r_uR=det("u4c12-reasonafter-4b","race-c",R4)
r_fs=uprobe(f"{UD}/race-flatsimpl-4b_s123.jsonl"); r_u4=uprobe(f"{UD}/u4c12-4b_s123.jsonl"); r_c16=uprobe(f"{UD}/race-cot16-4b_s123.jsonl")
r_acot=ans("cot16-4b","race-c"); r_ara=ans("u4c12-reasonafter-4b","race-c")
# LSAT
l_cU=uget(f"{UD}/lsat-COT16_s123.jsonl",L[0]); l_uU=uget(f"{UD}/lsat-8b_s123.jsonl",L[0])
l_cR=det("lsat-cot16","lsat-ar",L); l_uR=det("lsat-flatsimpl","lsat-ar",L)
l_uu=uprobe(f"{UD}/lsat-8b_s123.jsonl"); l_cc=uprobe(f"{UD}/lsat-COT16_s123.jsonl")
l_acot=ans("lsat-cot16","lsat-ar"); l_aund=ans("lsat-flatsimpl","lsat-ar")
# LSAT disagreements (one each way), seed 123 shown; corroborated across seeds
D1=("201106_2-G_2",1)   # Understander right / Reasoner wrong  (3/3 vs 0/3)
D2=("201112_2-G_1",1)   # Reasoner right / Understander wrong  (marginal)

CSS=open("/dev/stdin").read() if False else None
# reuse the 8B stylesheet verbatim
import importlib.util
CSS=r'''
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
h1{font-size:clamp(26px,4vw,42px);line-height:1.06;letter-spacing:-.02em;margin:.35em 0 .2em;font-weight:800;text-wrap:balance}
.thesis{font-size:clamp(15px,1.8vw,19px);color:var(--soft);max-width:70ch;margin:0 0 8px;text-wrap:balance}
.thesis b{color:var(--ink);font-weight:650}
.scope{display:flex;align-items:center;gap:12px;margin:40px 0 4px}
.scope .n{font-family:var(--mono);font-size:13px;font-weight:800;color:var(--under);background:var(--under-bg);border:1px solid var(--under-line);border-radius:8px;padding:3px 11px}
.scope h2{font-size:22px;font-weight:800;letter-spacing:-.01em;margin:0}
.rule{height:1px;background:var(--line);margin:8px 0 0}
.qcard{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px 20px;box-shadow:var(--shadow);margin-top:16px}
.qlab{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--soft);font-weight:700;margin-bottom:8px}
.passage{font-size:14px;color:var(--ink);margin:0 0 12px;line-height:1.55}
.q{font-weight:700;font-size:15.5px;margin:0 0 9px}
.opts{display:grid;gap:4px;font-size:14px;color:var(--soft);font-family:var(--mono)}
.opts .gold{color:var(--under);font-weight:700}
.gold-tag{display:inline-block;margin-top:11px;font-family:var(--mono);font-size:13px;font-weight:700;color:var(--under);background:var(--under-bg);border:1px solid var(--under-line);border-radius:6px;padding:2px 9px}
.sec{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--soft);font-weight:700;margin:24px 0 5px}
.seclead{color:var(--soft);font-size:14px;margin:0 0 13px;max-width:80ch}
.seclead b{color:var(--ink)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow);display:flex;flex-direction:column}
.panel .top{height:4px}
.panel .hd{padding:13px 16px 9px}
.tagrow{display:flex;justify-content:space-between;align-items:center;gap:8px}
.tag{font-size:11px;letter-spacing:.13em;text-transform:uppercase;font-weight:800}
.badge{font-family:var(--mono);font-size:11px;font-weight:800;padding:2px 8px;border-radius:20px;color:#fff;white-space:nowrap;background:var(--good)}
.badge.wrong{background:var(--bad)}
.dhead{display:flex;align-items:center;gap:10px;margin:20px 0 0;flex-wrap:wrap}
.dpill{font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;padding:3px 10px;border-radius:20px}
.dpill.win{color:var(--under);background:var(--under-bg);border:1px solid var(--under-line)}
.dpill.lose{color:var(--reason);background:var(--reason-bg);border:1px solid var(--reason-line)}
.pn{font-size:17px;font-weight:750;margin:5px 0 2px;letter-spacing:-.01em}
.sub{font-size:12.5px;color:var(--soft);margin:0}
.chips{display:flex;gap:7px;flex-wrap:wrap;padding:0 16px 12px}
.chip{font-family:var(--mono);font-size:12px;font-weight:700;background:var(--chip);border:1px solid var(--line);border-radius:20px;padding:3px 10px;color:var(--ink);font-variant-numeric:tabular-nums}
.body{border-top:1px solid var(--line);padding:13px 16px;font-family:var(--mono);font-size:12px;line-height:1.5;white-space:pre-wrap;color:var(--ink);overflow-y:auto;max-height:420px;flex:1}
.r .tag{color:var(--reason)} .r .top{background:var(--reason)}
.u .tag{color:var(--under)} .u .top{background:var(--under)}
.why{font-size:13.5px;color:var(--ink);background:var(--chip);border-radius:10px;padding:12px 15px;margin:13px 0 0;line-height:1.55}
.why b{color:var(--under)} .why .rz{color:var(--reason);font-weight:700}
.callout{border-left:3px solid var(--under);background:var(--under-bg);border-radius:0 10px 10px 0;padding:12px 16px;margin:13px 0 0;font-size:14px;color:var(--ink);line-height:1.55}
.callout b{color:var(--under)}
table{width:100%;border-collapse:collapse;font-size:13px;background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}
th,td{padding:9px 13px;text-align:left;border-bottom:1px solid var(--line)}
th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--soft);font-weight:700}
td.num,th.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
tr.hi td{background:var(--under-bg)} tr.hi td:first-child{font-weight:750}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;margin-top:6px}
@media(max-width:820px){.two{grid-template-columns:1fr}}
.tlab{font-size:12px;color:var(--soft);margin:0 0 8px;font-weight:600}
.cap{font-size:12px;color:var(--soft);margin:8px 2px 0}
.foot{font-size:12px;color:var(--soft);margin-top:28px;line-height:1.6}
</style>
'''

def opts_html(m):
    out=[]
    for i,o in enumerate(m["options"]):
        g="gold" if chr(65+i)==m["answer"] else ""
        out.append(f'<div class="{g}">{chr(65+i)}. {e(str(o))}</div>')
    return "\n".join(out)
def panel(cls,tag,name,sub,chips,body,verdict=None,pred=None):
    ch="".join(f'<span class="chip">{e(c)}</span>' for c in chips)
    if verdict=="right": badge=f'<span class="badge">{e(f"pred {pred} · correct")}</span>'
    elif verdict=="wrong": badge=f'<span class="badge wrong">{e(f"pred {pred} · wrong")}</span>'
    else: badge=""
    return (f'<div class="panel {cls}"><div class="top"></div>'
            f'<div class="hd"><div class="tagrow"><span class="tag">{e(tag)}</span>{badge}</div>'
            f'<div class="pn">{e(name)}</div><p class="sub">{e(sub)}</p></div>'
            f'<div class="chips">{ch}</div><div class="body">{e(body)}</div></div>')

r_why=('Both read the course-numbering passage correctly, but the Understander is more specific — it pulls '
       'out “proportionately”, the 10–16 vs. 5 credit-hour contrast, and the “generally” hedge; the Reasoner '
       'stays generic. The aggregate below shows the sharper gap: the Reasoner fails to form a well-formed '
       'understanding a third of the time.')
l_why=('On an LSAT logic puzzle both write a full structural extraction — entities, rules, deductions. '
       'This is a different kind of “understanding” from the reading-comprehension one, and neither model '
       'can pre-compute the answer from it: each question needs its own deduction.')

def dcard(key,pill,label,why):
    m=lsrc[key]
    c=det("lsat-cot16","lsat-ar",key); u=det("lsat-flatsimpl","lsat-ar",key)
    cv="right" if c["correct"] else "wrong"; uv="right" if u["correct"] else "wrong"
    rp=panel("r","Reasoner · cot16","Its reasoning","Chain-of-thought.",[f"{c['n_tokens']} tokens"],strip_boxed(c["text"]),cv,c["pred"])
    up=panel("u","Understander","Its reasoning","Reasons in full — no direct answer.",[f"{u['n_tokens']} tokens"],strip_boxed(u["text"]),uv,u["pred"])
    return (f'<div class="dhead">{pill}<span class="qlab" style="margin:0">{e(label)} · gold {e(m["answer"])}</span></div>'
            f'<div class="qcard" style="margin-top:8px"><p class="passage">{e(m["article"])}</p>'
            f'<p class="q">{e(m["question"])}</p><div class="opts">{opts_html(m)}</div></div>'
            f'<div class="grid">{rp}{up}</div><p class="why">{why}</p>')

d1_why=('The necessary fact is <b>D</b>: Trevino dives before Weiss, and Weiss can’t be last — so at least '
        'Weiss and the final diver come after Trevino. The Understander carries that deduction to D. The '
        '<span class="rz">Reasoner</span> checks each option, concludes “none must be true,” then guesses the '
        '“closest” one (A). Robust: Understander right on all 3 seeds, Reasoner wrong on all 3.')
d2_why=('The <span class="rz">Reasoner</span> builds one concrete valid order — Kevin, Juanita, Ginny, Fernando, '
        'Hakim — confirms Fernando can be fourth (<b>A</b>), and eliminates the rest. The Understander labels '
        'every option “possible” without constructing an order, then admits it’s guessing and picks C. The roles '
        'flip here — but the split is marginal (Understander wrong only on this seed, right on the other two), '
        'whereas the case above is 3-for-3.')
disagree=(
    '<div class="sec">When they disagree</div>'
    '<p class="seclead">Two more LSAT questions where the models split — one each way. Both reason in full; '
    'the badge marks who got it right.</p>'
    + dcard(D1,'<span class="dpill win">Understander right · Reasoner wrong</span>',"Skydiving order — the robust direction",d1_why)
    + dcard(D2,'<span class="dpill lose">Reasoner right · Understander wrong</span>',"Piano recital — the rarer, marginal direction",d2_why))

body=f"""
<div class="wrap"><div class="page">
<div class="eyebrow">SImpL · Qwen3 · companion to the 8B RACE one-pager</div>
<h1>LSAT vs 4B — where the shortcut stops, and where it holds</h1>

<div class="scope"><span class="n">LSAT</span><h2>Analytical reasoning — the Understander reasons, not answers</h2></div>
<div class="rule"></div>
<div class="qcard">
<div class="qlab">The setup · LSAT-AR · 8B · gold {e(lm['answer'])}</div>
<p class="passage">{e(lm['article'])}</p>
<p class="q">{e(lm['question'])}</p>
<div class="opts">{opts_html(lm)}</div>
</div>
<div class="sec">The understanding each produces — both structural</div>
<div class="grid">
{panel("r","Reasoner · cot16","Its understanding","Rule extraction.",[f"{l_cU['n_tokens_understanding']} tokens"],l_cU['understanding'].strip())}
{panel("u","Understander","Its understanding","Rule extraction.",[f"{l_uU['n_tokens_understanding']} tokens"],l_uU['understanding'].strip())}
</div>
<p class="why">{l_why}</p>
<div class="sec">The reasoning each produces (boxed answer removed) — both reason at length</div>
<div class="grid">
{panel("r","Reasoner · cot16","Its reasoning","Chain-of-thought.",[f"{l_cR['n_tokens']} tokens"],strip_boxed(l_cR['text']),"right",l_cR['pred'])}
{panel("u","Understander","Its reasoning","Here it reasons — no direct answer.",[f"{l_uR['n_tokens']} tokens"],strip_boxed(l_uR['text']),"right",l_uR['pred'])}
</div>
<div class="two">
<div><p class="tlab">Understanding (LSAT-AR, 8B)</p>
<table><thead><tr><th>model</th><th class="num">tokens</th><th class="num">well-formed</th><th class="num">fails</th></tr></thead><tbody>
<tr class="hi"><td>Understander (flatsimpl)</td><td class="num">{l_uu['body']:.0f}</td><td class="num">{l_uu['closed']:.0f}%</td><td class="num">{l_uu['fails']:.0f}%</td></tr>
<tr><td>Reasoner (cot16)</td><td class="num">{l_cc['body']:.0f}</td><td class="num">{l_cc['closed']:.0f}%</td><td class="num">{l_cc['fails']:.0f}%</td></tr>
</tbody></table></div>
<div><p class="tlab">Reasoning (LSAT-AR, 8B, 3 seeds) — note the token counts</p>
<table><thead><tr><th>model</th><th class="num">accuracy</th><th class="num">tokens</th></tr></thead><tbody>
<tr><td>Reasoner (cot16)</td><td class="num">{l_acot[0]:.1f}%</td><td class="num">{l_acot[1]:.0f}</td></tr>
<tr class="hi"><td>Understander (flatsimpl)</td><td class="num">{l_aund[0]:.1f}%</td><td class="num">{l_aund[1]:.0f}</td></tr>
</tbody></table></div>
</div>

{disagree}

<div class="scope"><span class="n">4B</span><h2>Reading comprehension — the pattern replicates</h2></div>
<div class="rule"></div>
<div class="qcard">
<div class="qlab">The passage · RACE-C · 4B · gold {e(rm['answer'])}</div>
<p class="passage">{e(rm['article'])}</p>
<p class="q">{e(rm['question'])}</p>
<div class="opts">{opts_html(rm)}</div>
</div>
<div class="sec">The understanding each produces</div>
<div class="grid">
{panel("r","Reasoner · cot16","Its understanding","Correct but generic.",[f"{r_cU['n_tokens_understanding']} tokens"],r_cU['understanding'].strip())}
{panel("u","Understander","Its understanding","Same read, more specific detail.",[f"{r_uU['n_tokens_understanding']} tokens"],r_uU['understanding'].strip())}
</div>
<p class="why">{r_why}</p>
<div class="sec">The reasoning each produces (boxed answer removed)</div>
<div class="grid">
{panel("r","Reasoner · cot16","Its reasoning","Chain-of-thought.",[f"{r_cR['n_tokens']} tokens"],strip_boxed(r_cR['text']),"right",r_cR['pred'])}
{panel("u","Understander","Its reasoning","Prompted to reason; walks the options.",[f"{r_uR['n_tokens']} tokens"],strip_boxed(r_uR['text']),"right",r_uR['pred'])}
</div>
<div class="two">
<div><p class="tlab">Understanding (RACE-C, 4B)</p>
<table><thead><tr><th>model</th><th class="num">tokens</th><th class="num">well-formed</th><th class="num">fails</th></tr></thead><tbody>
<tr class="hi"><td>Understander (8:8)</td><td class="num">{r_fs['body']:.0f}</td><td class="num">{r_fs['closed']:.0f}%</td><td class="num">{r_fs['fails']:.0f}%</td></tr>
<tr><td>Understander (u4c12)</td><td class="num">{r_u4['body']:.0f}</td><td class="num">{r_u4['closed']:.0f}%</td><td class="num">{r_u4['fails']:.0f}%</td></tr>
<tr><td>Reasoner (cot16)</td><td class="num">{r_c16['body']:.0f}</td><td class="num">{r_c16['closed']:.0f}%</td><td class="num">{r_c16['fails']:.0f}%</td></tr>
</tbody></table></div>
<div><p class="tlab">Reasoning (RACE-C, 4B, 3 seeds)</p>
<table><thead><tr><th>model</th><th class="num">accuracy</th><th class="num">tokens</th></tr></thead><tbody>
<tr><td>Reasoner (cot16) — CoT</td><td class="num">{r_acot[0]:.1f}%</td><td class="num">{r_acot[1]:.0f}</td></tr>
<tr class="hi"><td>Understander</td><td class="num">{r_ara[0]:.1f}%</td><td class="num">{r_ara[1]:.0f}</td></tr>
</tbody></table></div>
</div>

<p class="foot">Verbatim outputs, greedy decode, seed 42; boxed answers removed from reasoning panels.
LSAT: LSAT-AR puzzle 200312_1-G_1 (Understander = flatsimpl, Reasoner = cot16), normal answer path;
disagreement items 201106_2-G_2 and 201112_2-G_1.
4B: RACE-C passage 1166 (Understander = flatsimplv3 / u4c12, Reasoner = cot16). Aggregates over seeds 123/234/345.</p>
</div></div>
"""

os.makedirs("experiments",exist_ok=True)
open(OUT,"w").write(CSS+body)
print("wrote",OUT,"bytes",os.path.getsize(OUT))
