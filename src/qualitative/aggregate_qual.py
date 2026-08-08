"""Aggregate the LSAT three-way understandings and the CosmosQA understanding+answer runs into
readable markdown, matching the format/metrics of experiments/qualitative_outputs.md.

Metrics (identical definitions to qualitative_outputs.md):
  closed-tag       = emitted </understanding>
  fails to produce = empty <understanding> body
  degenerates      = >30% repeated 5-grams in the raw generation

Everything is joined hands-free from the JSONL + the source datasets, so all quoted text is verbatim.

  python -m src.qualitative.aggregate_qual
"""
import json
import re
from statistics import mean

from src.utils.parsing_utils import parse_questions

QD = "evaluations/qualitative_understandings"


class _Ex:
    __slots__ = ("example_id", "article")

    def __init__(self, example_id, article):
        self.example_id = example_id
        self.article = article


def frac_rep_5grams(text: str) -> float:
    w = re.findall(r"\S+", text)
    if len(w) < 6:
        return 0.0
    grams = [" ".join(w[i:i + 5]) for i in range(len(w) - 4)]
    return 1.0 - len(set(grams)) / len(grams)


def load_u_probe(path):
    """understanding_probe schema: [{example_id, samples:[{raw, understanding, n_tokens_full, closed_tag}]}]"""
    rows = []
    for line in open(path):
        rows.append(json.loads(line))
    return rows


def u_stats(rows, keep=None):
    full, closed, fails, degen = [], [], [], []
    for r in rows:
        if keep is not None and r["example_id"] not in keep:
            continue
        for s in r["samples"]:
            full.append(s["n_tokens_full"])
            closed.append(int(bool(s["closed_tag"])))
            fails.append(int(not (s["understanding"] or "").strip()))
            degen.append(int(frac_rep_5grams(s["raw"]) > 0.30))
    n = max(len(full), 1)
    return dict(full=mean(full), closed=100 * sum(closed) / n,
               fails=100 * sum(fails) / n, degen=100 * sum(degen) / n, n=n)


def load_ua(path):
    return [json.loads(l) for l in open(path)]


def ua_stats(rows):
    ub, closed, fails, corr, nq = [], [], [], 0, 0
    for r in rows:
        ub.append(r["n_tokens_understanding"])
        closed.append(int(bool(r["closed_tag"])))
        fails.append(int(not (r["understanding"] or "").strip()))
        for q in r["questions"]:
            corr += q["correct"]; nq += 1
    n = max(len(ub), 1)
    return dict(ubody=mean(ub), closed=100 * sum(closed) / n, fails=100 * sum(fails) / n,
                acc=100 * corr / max(nq, 1), nq=nq, npass=len(rows))


def article_map(data_path, dataset_name):
    """Light re-implementation of load_eval_examples' example_id->article map (no vllm import).
    Mirrors its filtering: drop empty-article records and records with no valid question; example_id
    comes from the record's own field (the f'example_{idx}' fallback never triggers on these sets)."""
    out = {}
    for idx, line in enumerate(open(data_path)):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        art = rec.get("article", "")
        if not isinstance(art, str) or not art.strip():
            continue
        has_q = any(
            isinstance(q.get("question"), str) and q.get("question", "").strip()
            and isinstance(q.get("options"), list) and len(q.get("options", [])) >= 2
            for q in parse_questions(rec.get("questions", []))
        )
        if not has_q:
            continue
        out[str(rec.get("example_id", f"example_{idx}"))] = _Ex(str(rec.get("example_id", f"example_{idx}")), art)
    return out


# ---------------- LSAT three-way ----------------
def build_lsat():
    und = load_u_probe(f"{QD}/lsat-8b_s123.jsonl")
    base = load_u_probe(f"{QD}/lsat-BASE_s123.jsonl")
    cot = load_u_probe(f"{QD}/lsat-COT16_s123.jsonl")
    amap = article_map("data/lsat-ar/final_test.jsonl", "lsat-ar")

    # index by example_id -> first sample
    def idx(rows):
        return {r["example_id"]: r["samples"][0] for r in rows}
    iu, ib, ic = idx(und), idx(base), idx(cot)
    common = [r["example_id"] for r in und if r["example_id"] in ib and r["example_id"] in ic]
    cset = set(common)
    # stats restricted to the common passage set so all three arms cover the same passages
    su, sb, sc = u_stats(und, cset), u_stats(base, cset), u_stats(cot, cset)
    # shortest passages first, for display
    common.sort(key=lambda e: len(amap[e].article) if e in amap else 1e9)

    L = []
    L.append("# Qualitative outputs (LSAT-AR) — the understanding artifact across Base / Reasoner / Understander\n")
    L.append("All three are the **same 8B model family** given the **same LSAT understanding prompt** on the\n"
             "**same LSAT-AR test passages** (in-distribution here — the LSAT analogue of the CosmosQA panel in\n"
             "`qualitative_outputs.md`). Base = Qwen3-8B-Base (no RL). Reasoner = LSAT cot16 (RL on answer-\n"
             "correctness only, step 152). Understander = LSAT flatsimpl (RL with the understanding objective,\n"
             "step 316). Outputs verbatim; `[no tags]` = no `<understanding>` emitted.\n")
    L.append("Generated hands-free by `src/qualitative/aggregate_qual.py` from `*_s123.jsonl`.\n")
    L.append(f"\n## Aggregate over {len(common)} LSAT-AR passages (2 samples each)\n")
    L.append("| model | mean full toks | closed-tag | fails to produce | degenerates (repetition) |")
    L.append("|---|---|---|---|---|")
    L.append(f"| Understander | {su['full']:.0f} | {su['closed']:.0f}% | {su['fails']:.0f}% | {su['degen']:.0f}% |")
    L.append(f"| Base | {sb['full']:.0f} | {sb['closed']:.0f}% | {sb['fails']:.0f}% | {sb['degen']:.0f}% |")
    L.append(f"| Reasoner (cot16) | {sc['full']:.0f} | {sc['closed']:.0f}% | {sc['fails']:.0f}% | {sc['degen']:.0f}% |")
    L.append("\n*'fails to produce' = empty `<understanding>` body. 'degenerates' = >30% repeated 5-grams.*\n")

    for n, eid in enumerate(common[:3], 1):
        art = amap[eid].article if eid in amap else "(passage unavailable)"
        nwords = len(re.findall(r"\S+", art))
        L.append("\n---\n")
        L.append(f"## Example {n}  (LSAT-AR, {nwords}-word setup)\n")
        L.append("> **Setup.** " + art.replace("\n", "\n> ") + "\n")
        for label, s in (("Understander", iu[eid]), ("Base", ib[eid]), ("Reasoner (cot16)", ic[eid])):
            body = (s["understanding"] or "").strip()
            wf = bool(body)
            tks = s["n_tokens_understanding"]
            L.append(f"\n**{label}** ({tks} tok, well-formed={wf}):")
            shown = body if wf else "[no <understanding> tags emitted]\n" + s["raw"].strip()
            L.append("```\n" + shown + "\n```")

    L.append("\n---\n\n## What is going on\n")
    L.append(
        "Unlike the CosmosQA panel, this is **in-distribution** (LSAT understander evaluated on LSAT-AR),\n"
        "and the picture is correspondingly milder: all three models can produce a long, structured\n"
        "rule-extraction, because LSAT setups are close to the pretraining distribution of formal/logical\n"
        "text. The differences are about **reliability**, not the collapse we saw OOD.\n")
    L.append(
        f"\n**Understander — most reliable.** Emits a well-formed structural understanding "
        f"{su['closed']:.0f}% of the time, never empty ({su['fails']:.0f}% fails), averaging "
        f"{su['full']:.0f} tokens of clean bulleted rule/deduction extraction (entities, explicit rules,\n"
        "contrapositives, forced deductions — the categories the LSAT prompt asks for).\n")
    L.append(
        f"\n**Base — competent when it complies, but unreliable.** Fails to produce a valid understanding "
        f"{sb['fails']:.0f}% of the time and only closes the tag {sb['closed']:.0f}% of the time. When it\n"
        "*does* comply it is often good (Example 3), but it also emits junk (Example 1: a single token\n"
        "\"and\") or drifts into an ad-hoc XML schema of its own (Example 2) instead of the requested format.\n")
    L.append(
        f"\n**Reasoner (cot16) — in between.** Fails {sc['fails']:.0f}% and closes {sc['closed']:.0f}%. On\n"
        "LSAT text it does NOT degenerate the way it did on OOD CosmosQA — the repetition collapse is an\n"
        "out-of-distribution phenomenon, and LSAT is in-distribution for this model.\n")
    L.append(
        "\n### Relation to the RACE story\n"
        "This is the honest contrast the paper should show: the understander's edge in *understanding-\n"
        "generation reliability* is real on both datasets, but it is **large on OOD comprehension**\n"
        "(CosmosQA: base hallucinates, cot16 degenerates) and **modest in-distribution on LSAT** (everyone\n"
        "can extract the rules; the understander is just the most consistent). It does not manufacture a\n"
        "collapse that isn't there.\n")
    L.append(
        "\n### Honest caveats\n"
        f"- **The degeneration metric is unreliable on LSAT.** The understander's {su['degen']:.0f}% "
        "\"degenerates\" is almost entirely the >30%-repeated-5-gram heuristic misfiring on legitimately\n"
        "  repetitive rule lists (\"Kevin < Hakim\" / \"Kevin < Juanita\"; \"types of pop are on sale\"), not true\n"
        "  looping — the sampled understandings above are clearly non-degenerate. Read that column with care.\n"
        "- **Format is a confound.** Base and cot16 were never trained to emit `<understanding>` tags, so\n"
        "  part of their fails/closed gap is format non-compliance, not inability to extract the structure.\n"
        "- **n = 1 seed, 50 shared passages, 2 samples each.** Enough for the reliability ordering, not a\n"
        "  precise measurement.\n")
    L.append(
        "\n### Provenance\n"
        "`evaluations/qualitative_understandings/lsat-{8b,BASE,COT16}_s123.jsonl`, 8B checkpoints\n"
        "(understander step 316, reasoner step 152, base untrained), LSAT understanding prompt, LSAT-AR\n"
        "test passages, temp 0.6 / top-p 0.95 / seed 42.\n")
    open("experiments/qualitative_outputs_lsat.md", "w").write("\n".join(L))
    print("wrote experiments/qualitative_outputs_lsat.md")
    return su, sb, sc


# ---------------- CosmosQA understanding + answer ----------------
def build_ua():
    und = load_ua(f"{QD}/ua-cosmosqa-UND_s123.jsonl")
    base = load_ua(f"{QD}/ua-cosmosqa-BASE_s123.jsonl")
    cot = load_ua(f"{QD}/ua-cosmosqa-COT16_s123.jsonl")
    su, sb, sc = ua_stats(und), ua_stats(base), ua_stats(cot)
    amap = article_map("data/cosmosqa/test_42_all.jsonl", "cosmosqa")

    def idx(rows):
        return {r["example_id"]: r for r in rows}
    iu, ib, ic = idx(und), idx(base), idx(cot)
    common = [r["example_id"] for r in und if r["example_id"] in ib and r["example_id"] in ic]
    common.sort(key=lambda e: len(amap[e].article) if e in amap else 1e9)

    L = []
    L.append("# CosmosQA — the full understanding + answer pipeline (Base / Reasoner / Understander)\n")
    L.append("The deployed two-step run end to end on **CosmosQA** (an unseen RACE-generalization target): each\n"
             "model first writes an understanding, then answers each question **conditioned only on that\n"
             "understanding** (`qa_eval_understanding_only_prompt`, passage withheld — the self-contained\n"
             "understander setting). Understanding generated greedily (as in the real eval); answer greedy.\n"
             "Same 8B family, same v3 understanding prompt. Understander = RACE flatsimpl (step 188),\n"
             "Reasoner = RACE cot16 (step 204), Base = Qwen3-8B-Base.\n")
    L.append("Generated hands-free by `src/qualitative/aggregate_qual.py`. Answers/understandings verbatim.\n")
    L.append(f"\n## Aggregate over {su['npass']} CosmosQA passages\n")
    L.append("| model | mean understanding toks | closed-tag | fails to produce | **answer acc (from understanding)** |")
    L.append("|---|---|---|---|---|")
    L.append(f"| Understander | {su['ubody']:.0f} | {su['closed']:.0f}% | {su['fails']:.0f}% | **{su['acc']:.0f}%** |")
    L.append(f"| Base | {sb['ubody']:.0f} | {sb['closed']:.0f}% | {sb['fails']:.0f}% | {sb['acc']:.0f}% |")
    L.append(f"| Reasoner (cot16) | {sc['ubody']:.0f} | {sc['closed']:.0f}% | {sc['fails']:.0f}% | {sc['acc']:.0f}% |")
    L.append("\n*Accuracy is over all questions on all 50 passages, answered from the understanding alone. "
             "This is the two-step path, NOT the deployed direct-answer path — it isolates how well each "
             "model's understanding SUPPORTS answering.*\n")

    for n, eid in enumerate(common[:3], 1):
        art = amap[eid].article if eid in amap else "(passage unavailable)"
        nwords = len(re.findall(r"\S+", art))
        q0 = iu[eid]["questions"][0]
        L.append("\n---\n")
        L.append(f"## Example {n}  ({nwords}-word passage)\n")
        L.append("> **Passage.** " + art.replace("\n", " ") + "\n")
        opts = " · ".join(f"{chr(65+i)}. {o}" for i, o in enumerate(q0["options"]))
        L.append(f"> **Q (gold {q0['gold']}).** {q0['question']}  \n> {opts}\n")
        for label, r in (("Understander", iu[eid]), ("Base", ib[eid]), ("Reasoner (cot16)", ic[eid])):
            body = (r["understanding"] or "").strip()
            wf = bool(body)
            q = r["questions"][0]
            mark = "CORRECT" if q["correct"] else "WRONG"
            L.append(f"\n**{label}** — understanding ({r['n_tokens_understanding']} tok, well-formed={wf}):")
            shown = body if wf else "[no <understanding> tags emitted]\n" + (r["understanding_raw"] or "").strip()
            L.append("```\n" + shown + "\n```")
            L.append(f"**→ answer** (pred={q['pred'] or '(none)'} · gold={q['gold']} · {mark} · {q['n_tokens_answer']} tok):")
            L.append("```\n" + (q["answer_text"] or "").strip() + "\n```")

    L.append("\n---\n\n## What is going on\n")
    L.append(
        "This runs the SImpL pipeline the way it is *designed* — understanding first, then answer from the\n"
        "understanding alone — on unseen CosmosQA, and it separates cleanly by model.\n")
    L.append(
        f"\n**Understander — the pipeline works.** It reliably writes a faithful ~{su['ubody']:.0f}-token\n"
        f"understanding ({su['fails']:.0f}% fails, {su['closed']:.0f}% closed-tag) and then answers from it,\n"
        f"usually in a single boxed letter (~5 tokens), reaching **{su['acc']:.0f}%**. Examples 1–3 all show\n"
        "the same shape: clean understanding → `\\boxed{X}` → correct.\n")
    L.append(
        f"\n**Base and Reasoner — the intermediate mostly doesn't form.** Base fails to produce a usable\n"
        f"understanding {sb['fails']:.0f}% of the time and cot16 {sc['fails']:.0f}% of the time (they\n"
        "degenerate — Example 2/3 show the \"I was so mad I wanted to kill them\" and \"many places in the\n"
        "world\" loops — or drop the tags). When the understanding is empty the answer step runs with **no\n"
        "passage and no understanding**, i.e. it answers blind from the question and options.\n")
    L.append(
        f"\nThat is why base/cot16 still score {sb['acc']:.0f}%/{sc['acc']:.0f}%: CosmosQA options are\n"
        "answerable from priors well above chance even blind. The understander's "
        f"**+{su['acc']-max(sb['acc'],sc['acc']):.0f}pp** over the better of them is the value the reliably-\n"
        "formed understanding adds on top of those priors.\n")
    L.append(
        "\n### Two things worth noting\n"
        "- **Example 1 is the main finding in miniature.** Base *does* produce a faithful understanding here,\n"
        "  then **reasons its way out of the right answer** (95-token CoT → wrong C); the understander answers\n"
        "  directly (5 tokens → correct A). Direct answering beating visible CoT on comprehension is exactly\n"
        "  the accuracy story from the results tables.\n"
        "- **Example 3 shows base can hallucinate yet still be right** (it invents \"business consultant\",\n"
        "  \"Salt Lake City\" but the question is coarse enough that C still follows) — hallucinated\n"
        "  understandings are not always penalized by the answer, which is why the artifact-quality axis and\n"
        "  the accuracy axis are separate.\n")
    L.append(
        "\n### Honest caveats\n"
        "- **This is NOT a like-for-like understanding-quality contest.** Base/cot16 lose mostly by failing\n"
        "  to emit the intermediate at all (a format + OOD-robustness effect), after which they answer blind.\n"
        "  The defensible claim is: *the understander is the only one of the three that can actually run the\n"
        "  two-step pipeline on unseen text*, and doing so it answers directly and accurately.\n"
        "- **Greedy decode, n = 1 seed, 50 passages, first question per passage shown.** The aggregate\n"
        "  accuracy is over every question on all 50 passages; only the first question is displayed per example.\n"
        "- Degeneration rates are decode-sensitive (greedy here); the direction is robust, the exact rate is not.\n")
    L.append(
        "\n### Provenance\n"
        "`evaluations/qualitative_understandings/ua-cosmosqa-{UND,BASE,COT16}_s123.jsonl`, generated by\n"
        "`src/qualitative/understanding_answer_probe.py` (understanding greedy, answer greedy,\n"
        "`understanding_with_passage=False`), 8B checkpoints (understander step 188, reasoner step 204,\n"
        "base untrained), race-c v3 understanding prompt, CosmosQA passages, seed 42.\n")
    open("experiments/qualitative_understanding_answer.md", "w").write("\n".join(L))
    print("wrote experiments/qualitative_understanding_answer.md")
    return su, sb, sc


if __name__ == "__main__":
    lu, lb, lc = build_lsat()
    print(f"  LSAT   und full={lu['full']:.0f} closed={lu['closed']:.0f}% fails={lu['fails']:.0f}% degen={lu['degen']:.0f}%")
    print(f"  LSAT  base full={lb['full']:.0f} closed={lb['closed']:.0f}% fails={lb['fails']:.0f}% degen={lb['degen']:.0f}%")
    print(f"  LSAT cot16 full={lc['full']:.0f} closed={lc['closed']:.0f}% fails={lc['fails']:.0f}% degen={lc['degen']:.0f}%")
    au, ab, ac = build_ua()
    print(f"  UA    und ubody={au['ubody']:.0f} acc={au['acc']:.0f}% fails={au['fails']:.0f}%")
    print(f"  UA   base ubody={ab['ubody']:.0f} acc={ab['acc']:.0f}% fails={ab['fails']:.0f}%")
    print(f"  UA  cot16 ubody={ac['ubody']:.0f} acc={ac['acc']:.0f}% fails={ac['fails']:.0f}%")
