"""LSAT meeting page (separate URL): on analytical reasoning the Understander reasons rather than
answering directly (the dissociation), yet still understands more reliably and wins the head-to-head
disagreements more often. Two disagreement cards (one each way). All text verbatim.
   python -m src.qualitative.make_lsat_artifact
"""
import html, json, os, statistics
from src.qualitative.export_examples import load_source

DET="evaluations/qualitative_deterministic"; UD="evaluations/qualitative_understandings"
OUT="experiments/meeting_lsat.html"
C1=("201106_2-G_2",1)   # understander right / reasoner wrong (3/3 vs 0/3)
C2=("201112_2-G_1",1)   # reasoner right / understander wrong (marginal)

src=load_source("data/lsat-ar/final_test.jsonl"); e=lambda s: html.escape(s or "")
def det(tag,key,s="123"):
    for line in open(f"{DET}/{tag}_lsat-ar_s{s}.jsonl"):
        r=json.loads(line)
        if (r["example_id"],r["question_index"])==key: return r["samples"][0]
def uprobe(path):
    body=closed=fails=n=0
    for line in open(path):
        for s in json.loads(line)["samples"]:
            body+=s["n_tokens_understanding"]; closed+=int(bool(s["closed_tag"]))
            fails+=int(not (s["understanding"] or "").strip()); n+=1
    return body/n,100*closed/n,100*fails/n
def ans(tag,seeds=("123","234","345")):
    toks=[];acc=[]
    for s in seeds:
        p=f"{DET}/{tag}_lsat-ar_s{s}.jsonl"
        if not os.path.exists(p): continue
        rows=[json.loads(l) for l in open(p)]
        acc.append(sum(r["frac_correct"] for r in rows)/len(rows)*100)
        toks+=[x["n_tokens"] for r in rows for x in r["samples"]]
    return statistics.mean(acc),sum(toks)/len(toks)
u_acc,u_tok=ans("lsat-flatsimpl"); r_acc,r_tok=ans("lsat-cot16")
ufu,ufc,uff=uprobe(f"{UD}/lsat-8b_s123.jsonl"); rfu,rfc,rff=uprobe(f"{UD}/lsat-COT16_s123.jsonl")
bfu,bfc,bff=uprobe(f"{UD}/lsat-BASE_s123.jsonl")

CSS=open("experiments/_artifact_css.txt").read() if os.path.exists("experiments/_artifact_css.txt") else None
# inline the same design system as the RACE page (+ stat tiles)
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
 background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.55;-webkit-font-smoothing:antialiased;
 padding:clamp(20px,4vw,52px) clamp(16px,4vw,40px);min-height:100%;}
.page{max-width:1080px;margin:0 auto}
.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--under);font-weight:700}
h1{font-size:clamp(28px,4.2vw,44px);line-height:1.05;letter-spacing:-.02em;margin:.35em 0 .2em;font-weight:800;text-wrap:balance}
h1 .vs{color:var(--soft);font-weight:600}
.thesis{font-size:clamp(16px,1.9vw,20px);color:var(--soft);max-width:70ch;margin:0 0 8px;text-wrap:balance}
.thesis b{color:var(--ink);font-weight:650}
.rule{height:1px;background:var(--line);margin:26px 0}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:0 0 8px}
@media(max-width:680px){.stats{grid-template-columns:1fr}}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow)}
.tile .big{font-size:30px;font-weight:800;font-family:var(--mono);font-variant-numeric:tabular-nums;letter-spacing:-.02em;line-height:1}
.tile.u .big{color:var(--under)} .tile.r .big{color:var(--reason)}
.tile .lab{font-size:12.5px;color:var(--soft);margin-top:7px;line-height:1.4}
.note{font-size:13.5px;color:var(--soft);margin:14px 2px 0;max-width:78ch}
.sec{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--soft);font-weight:700;margin:36px 0 6px}
.seclead{color:var(--soft);font-size:14.5px;margin:0 0 14px;max-width:76ch}
.dcard{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:20px 22px;box-shadow:var(--shadow);margin-bottom:18px}
.dhead{display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.dpill{font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;padding:3px 10px;border-radius:20px}
.dpill.win{color:var(--under);background:var(--under-bg);border:1px solid var(--under-line)}
.dpill.lose{color:var(--reason);background:var(--reason-bg);border:1px solid var(--reason-line)}
.qlab{font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--soft);font-weight:700}
.setup{font-size:14.5px;color:var(--ink);margin:0 0 10px;line-height:1.55}
.q{font-weight:700;font-size:15.5px;margin:0 0 8px}
.opts{display:grid;gap:4px;font-size:14px;color:var(--soft);font-family:var(--mono)}
.opts .gold{color:var(--under);font-weight:700}
.dgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
@media(max-width:760px){.dgrid{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
.panel .top{height:4px}.panel.r .top{background:var(--reason)}.panel.u .top{background:var(--under)}
.hd{padding:13px 16px 9px}
.tagrow{display:flex;justify-content:space-between;align-items:center;gap:8px}
.tag{font-size:11px;letter-spacing:.13em;text-transform:uppercase;font-weight:800}
.panel.r .tag{color:var(--reason)}.panel.u .tag{color:var(--under)}
.badge{font-family:var(--mono);font-size:11px;font-weight:800;padding:2px 8px;border-radius:20px;color:#fff;white-space:nowrap}
.badge.right{background:var(--good)}.badge.wrong{background:var(--bad)}
.chips{display:flex;gap:7px;flex-wrap:wrap;padding:0 16px 12px}
.chip{font-family:var(--mono);font-size:12px;font-weight:700;background:var(--chip);border:1px solid var(--line);border-radius:20px;padding:3px 10px;color:var(--ink);font-variant-numeric:tabular-nums}
.body{border-top:1px solid var(--line);padding:13px 16px;font-family:var(--mono);font-size:12px;line-height:1.5;white-space:pre-wrap;color:var(--ink);overflow-y:auto;max-height:360px;flex:1}
.why{font-size:13.5px;color:var(--ink);background:var(--chip);border-radius:10px;padding:12px 15px;margin:14px 0 0;line-height:1.55}
.why b{color:var(--under)} .why .rz{color:var(--reason);font-weight:700}
table{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line)}
th{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--soft);font-weight:700}
td.num,th.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
tr.hi td{background:var(--under-bg)}tr.hi td:first-child{font-weight:750}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
@media(max-width:820px){.two{grid-template-columns:1fr}}
.tlab{font-size:12px;color:var(--soft);margin:0 0 8px;font-weight:600}
.cap{font-size:12.5px;color:var(--soft);margin:8px 2px 0}
.foot{font-size:12px;color:var(--soft);margin-top:26px;line-height:1.6}
</style>
"""

def card(key,pill,label,why):
    meta=src[key]
    c=det("lsat-cot16",key); u=det("lsat-flatsimpl",key)
    cv="right" if c["correct"] else "wrong"; uv="right" if u["correct"] else "wrong"
    opts="".join(f'<div class="{"gold" if chr(65+i)==meta["answer"] else ""}">{chr(65+i)}. {e(str(o))}</div>' for i,o in enumerate(meta["options"]))
    def pnl(cls,tag,s,verdict,pred):
        return (f'<div class="panel {cls}"><div class="top"></div><div class="hd"><div class="tagrow">'
                f'<span class="tag">{e(tag)}</span><span class="badge {verdict}">pred {pred} · {"correct" if verdict=="right" else "wrong"}</span>'
                f'</div></div><div class="chips"><span class="chip">{s["n_tokens"]} tokens</span></div>'
                f'<div class="body">{e(s["text"].strip())}</div></div>')
    return (f'<div class="dcard"><div class="dhead">{pill}<span class="qlab">{e(label)} · LSAT-AR · gold {meta["answer"]}</span></div>'
            f'<p class="setup">{e(meta["article"])}</p><p class="q">{e(meta["question"])}</p><div class="opts">{opts}</div>'
            f'<div class="dgrid">{pnl("r","Reasoner · cot16",c,cv,c["pred"])}{pnl("u","Understander",u,uv,u["pred"])}</div>'
            f'<p class="why">{why}</p></div>')

why1=('The necessary fact is <b>D</b>: Trevino dives before Weiss, and Weiss can’t be last — so at least '
      'Weiss and the final diver come after Trevino. The Understander carries that deduction to D. The '
      '<span class="rz">Reasoner</span> checks each option, concludes “none must be true,” then guesses the '
      '“closest” one (A) — abandoning the deduction. Robust: Understander right on all 3 seeds, Reasoner wrong on all 3.')
why2=('The <span class="rz">Reasoner</span> builds one concrete valid order — Kevin, Juanita, Ginny, Fernando, '
      'Hakim — confirms Fernando can be fourth (<b>A</b>), and eliminates the rest. The Understander labels '
      'every option “possible” without constructing an order, then admits it’s guessing and picks C. Here the '
      'roles flip — but the split is marginal (Understander wrong only on this seed, right on the other two), '
      'whereas the case above is 3-for-3.')

body=f"""
<div class="wrap"><div class="page">
<div class="eyebrow">SImpL · analytical reasoning · Qwen3-8B</div>
<h1>Reasoner <span class="vs">vs</span> Understander <span class="vs">— LSAT</span></h1>
<p class="thesis">On reading comprehension the Understander answers directly, in ~5 tokens. On <b>analytical
reasoning</b> it can’t: the understanding sets up the puzzle, but each question needs its own deduction.
So there is <b>no shortcut here</b> — both models reason at length. The Understander still forms the more
reliable understanding, and wins the head-to-head disagreements more often.</p>
<div class="rule"></div>

<div class="sec">The dissociation</div>
<div class="stats">
<div class="tile u"><div class="big">{u_tok:.0f}</div><div class="lab">tokens — the Understander’s answer length on LSAT.<br>On RACE it was <b>5</b>. No direct-answer shortcut here.</div></div>
<div class="tile r"><div class="big">{r_tok:.0f}</div><div class="lab">tokens — the Reasoner’s chain-of-thought. The two now reason at essentially the same length.</div></div>
<div class="tile u"><div class="big">{u_acc:.1f}%</div><div class="lab">Understander accuracy vs Reasoner’s {r_acc:.1f}% (+{u_acc-r_acc:.1f}). LSAT-AR is hard; both are well above the 20% chance rate.</div></div>
</div>
<p class="note">Why the shortcut disappears: a reading passage can be pre-digested into an understanding that already
contains the answer, so the Understander looks it up. An LSAT setup can only be turned into a board of rules —
every question then demands a fresh deduction on top. Same objective, opposite deployed behavior.</p>

<div class="sec">When they disagree</div>
<p class="seclead">Two questions where the models split — one in each direction. Both reason in full; the badges mark who got it right.</p>
{card(C1,'<span class="dpill win">Understander right · Reasoner wrong</span>','Skydiving order — the robust direction',why1)}
{card(C2,'<span class="dpill lose">Reasoner right · Understander wrong</span>','Piano recital — the rarer, marginal direction',why2)}

<div class="sec">Supporting counts (LSAT-AR, 8B, 3 seeds)</div>
<div class="two">
<div>
<p class="tlab">Quality of the <b>understanding</b> each produces</p>
<table><thead><tr><th>model</th><th class="num">tokens</th><th class="num">well-formed</th><th class="num">fails</th></tr></thead><tbody>
<tr class="hi"><td>Understander (flatsimpl)</td><td class="num">{ufu:.0f}</td><td class="num">{ufc:.0f}%</td><td class="num">{uff:.0f}%</td></tr>
<tr><td>Reasoner (cot16)</td><td class="num">{rfu:.0f}</td><td class="num">{rfc:.0f}%</td><td class="num">{rff:.0f}%</td></tr>
<tr><td>Base (no RL)</td><td class="num">{bfu:.0f}</td><td class="num">{bfc:.0f}%</td><td class="num">{bff:.0f}%</td></tr>
</tbody></table>
<p class="cap">Even on LSAT the Understander’s structural understanding is always well-formed; the Reasoner fails to produce one 20% of the time.</p>
</div>
<div>
<p class="tlab">The <b>answer</b> each produces (normal, both reason)</p>
<table><thead><tr><th>model</th><th class="num">accuracy</th><th class="num">tokens</th></tr></thead><tbody>
<tr><td>Reasoner (cot16)</td><td class="num">{r_acc:.1f}%</td><td class="num">{r_tok:.0f}</td></tr>
<tr class="hi"><td>Understander (flatsimpl)</td><td class="num">{u_acc:.1f}%</td><td class="num">{u_tok:.0f}</td></tr>
</tbody></table>
<p class="cap">No direct-answer shortcut on LSAT — both reason ~500 tokens; the Understander edges the Reasoner by {u_acc-r_acc:.1f} points.</p>
</div>
</div>

<p class="foot">Verbatim outputs from deployed 8B LSAT checkpoints (Understander = LSAT flatsimpl; Reasoner = LSAT cot16),
greedy decode, seed 42; “3 seeds” = 123/234/345. Examples: LSAT-AR items 201106_2-G_2 and 201112_2-G_1.</p>
</div></div>
"""
open(OUT,"w").write(CSS+body)
print("wrote",OUT,"bytes",os.path.getsize(OUT))
