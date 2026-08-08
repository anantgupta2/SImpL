from __future__ import annotations

import logging
import re
import string
from typing import Any, List


def _mcq_block(question_text: str, options: List[str]) -> str:
    # Full alphabet, not a hardcoded "ABCDEFGH": CLUTRR-18 renders 18 options and anything past
    # H raised IndexError, which the eval's broad except turned into a silent empty CSV (exit 0).
    # parsing_utils uses string.ascii_uppercase[:num_options], so this keeps the two in sync.
    letters = string.ascii_uppercase
    if len(options) > len(letters):
        raise ValueError(f"_mcq_block supports at most {len(letters)} options, got {len(options)}")

    rendered = "\n".join(
        f"{letters[i]}. {opt}" for i, opt in enumerate(options)
    )

    return (
        f"Question:\n{question_text}\n\n"
        f"Options:\n{rendered}\n"
    )

# def qa_cot_prompt(article: str, question_text: str, options: List[str]) -> str:
#     letters = "ABCDEFGH"
#     valid_letters = [f"\\boxed{{{letters[i]}}}" for i in range(len(options))]
#     options_str = ", ".join(valid_letters[:-1]) + f", or {valid_letters[-1]}" if len(valid_letters) > 1 else valid_letters[0]
#     return (
#         "Solve the multiple-choice question using the passage.\n"
#         "Think step by step, then output exactly one final boxed letter.\n\n"
#         "Passage:\n"
#         f"{article}\n\n"
#         f"{_mcq_block(question_text, options)}\n\n"
#         f"Output format requirement: final line contains one of {options_str}."
#     )

# Shared rubric for ProofWriter-style deductive entailment (True/False/Unknown). The
# critical bit is defining "Unknown" — without it a model almost never selects it.
_PROOFWRITER_RUBRIC = (
    "You are given a set of facts and rules. Using ONLY the given facts and rules "
    "(no outside knowledge), decide the status of the statement:\n"
    "- True: the statement provably follows from the facts and rules.\n"
    "- False: the statement provably contradicts the facts and rules.\n"
    "- Unknown: its truth cannot be determined from the given facts and rules.\n\n"
)


def proofwriter_cot_prompt(article: str, question_text: str, options: List[str]) -> str:
    return (
        _PROOFWRITER_RUBRIC
        + "Facts and rules:\n"
        + f"{article}\n\n"
        + f"{_mcq_block(question_text, options)}\n\n"
        + "Think step-by-step, applying the rules to the facts, then return your final answer "
          "as exactly one boxed letter (e.g., \\boxed{A}, \\boxed{B}, or \\boxed{C}).\n\n"
    )


def qa_cot_prompt(article: str, question_text: str, options: List[str], dataset_name: str | None = None) -> str:
    if (dataset_name or "").strip().lower().startswith("proofwriter"):
        return proofwriter_cot_prompt(article, question_text, options)
    return (
        "Solve the multiple-choice question using the passage.\n"
        "Passage:\n"
        f"{article}\n\n"
        f"{_mcq_block(question_text, options)}\n\n"
        "Think step-by-step and return your final answer as exactly one boxed letter (e.g., \\boxed{A}, \\boxed{B}, etc.).\n\n"
    )

def race_understanding_prompt_v1(article: str) -> str:
    # ORIGINAL RACE understanding prompt (pre-2026-06-26). Kept so we can switch back:
    # point UNDERSTANDING_PROMPT_REGISTRY["race-c"/"race-high"] at this instead of
    # race_understanding_prompt. The v2 (current) is taxonomy-oriented + self-contained,
    # paired with use_understanding_passage=False.
    return (
        "Read the passage and work out, in advance, everything needed to answer hard reading-"
        "comprehension questions about it. Do the interpretive work NOW, before any question is asked -- "
        "commit to conclusions rather than merely restating the text.\n\n"
        "Produce a precise, structured analysis that states:\n"
        "- the central idea / main point, and the author's purpose in writing\n"
        "- the function of each paragraph (what it establishes, argues, or shifts to)\n"
        "- key facts, definitions, names, and numerical details likely to be tested\n"
        "- the author's stance, tone, and attitude toward the subject, and the textual evidence for it\n"
        "- important INFERENCES and implications that are NOT stated literally but follow from the text "
        "(what the passage suggests must be true)\n"
        "- contrasts, comparisons, cause-and-effect links, exceptions, and qualifications\n"
        "- the referents of any ambiguous pronouns or vague phrases\n\n"
        "Make implicit meaning explicit and resolve interpretation now; do not hedge or summarize wording. "
        "Extract the reasoning a reader needs to answer questions correctly. "
        "Think first, then wrap your structured analysis strictly inside <understanding> and </understanding> tags.\n\n"
        f"Passage:\n{article}\n"
    )


def race_understanding_prompt(article: str) -> str:
    return (
        "Read the passage and do ALL the interpretive work now, before seeing any question, so that "
        "someone who has NOT read the passage could answer any reading-comprehension question about it "
        "using ONLY what you write. Your understanding must stand alone -- assume the reader cannot see the "
        "passage. Commit to conclusions; do not merely restate or summarize the wording.\n\n"
        "RACE-style questions test a few recurring things -- cover EACH explicitly and concretely:\n"
        "- MAIN IDEA & PURPOSE: the central point, the author's purpose, and the single best title\n"
        "- KEY DETAILS: the specific facts, names, numbers, dates, definitions, and sequences likely to be "
        "asked -- state the actual values, not just that they exist\n"
        "- INFERENCES: what the passage implies but does not say literally -- spell out what MUST be true, "
        "what the author would agree or disagree with, and the consequence of each key claim\n"
        "- AUTHOR'S ATTITUDE & TONE: the author's stance toward the subject and the evidence for it\n"
        "- VOCABULARY-IN-CONTEXT: the intended meaning of any unusual, figurative, or pivotal word or phrase "
        "as used here, and the referent of every ambiguous pronoun\n"
        "- STRUCTURE & RELATIONS: the role of each paragraph and the contrasts, comparisons, cause-and-effect "
        "links, exceptions, and qualifications the passage sets up\n\n"
        "Be specific and self-contained: resolve every interpretation NOW, make implicit meaning explicit, and "
        "write down the actual conclusions and answers -- not merely the topics to consider. "
        "Think first, then wrap your worked-out understanding strictly inside <understanding> and "
        "</understanding> tags.\n\n"
        f"Passage:\n{article}\n"
    )


def race_understanding_prompt_v3(article: str) -> str:
    # SHORT, no-summarization RACE prompt (2026-07-05). The v1 prompt produced long
    # passage summaries (main idea / paragraph-by-paragraph / fact restatement) that were
    # lossy vs the passage and often hit the token cap. This one forbids summarizing and
    # asks ONLY for the interpretive conclusions the passage does not state, kept brief.
    return (
        "Read the passage and write ONLY the interpretive conclusions a reader needs that are NOT stated "
        "outright. Do NOT summarize, restate, or describe the passage or its structure -- the reader "
        "already has the passage in front of them. Be brief: a few sharp lines, not prose, no headers.\n\n"
        "State only:\n"
        "- the key INFERENCES the passage forces but never says (what must be true; what the author would "
        "agree or disagree with)\n"
        "- the author's ATTITUDE/stance and the specific evidence that fixes it\n"
        "- the intended meaning of any pivotal/figurative/unusual word AS USED HERE, and the referent of "
        "any genuinely ambiguous pronoun\n"
        "- any single decisive contrast, cause-effect link, exception, or qualification a question could hinge on\n\n"
        "Commit to conclusions; skip anything already explicit in the text. Wrap your understanding strictly "
        "inside <understanding> and </understanding> tags.\n\n"
        f"Passage:\n{article}\n"
    )


def race_understanding_prompt_v4(article: str) -> str:
    # v4 (2026-07-22): identical to v3 in WHAT it asks for, but DROPS the brevity constraint
    # ("Be brief: a few sharp lines...") and instead asks the model to SELECT the important
    # content -- to test whether v3's terseness at deploy time is caused by the brevity demand.
    # Same categories, same no-summarization rule, no length cap.
    return (
        "Read the passage and write the interpretive conclusions a reader needs that are NOT stated "
        "outright. Do NOT summarize, restate, or describe the passage or its structure -- the reader "
        "already has the passage in front of them. Focus on the content that matters: pick out and work "
        "through the points a question is most likely to hinge on, in as much depth as each needs.\n\n"
        "Cover:\n"
        "- the key INFERENCES the passage forces but never says (what must be true; what the author would "
        "agree or disagree with)\n"
        "- the author's ATTITUDE/stance and the specific evidence that fixes it\n"
        "- the intended meaning of any pivotal/figurative/unusual word AS USED HERE, and the referent of "
        "any genuinely ambiguous pronoun\n"
        "- any decisive contrast, cause-effect link, exception, or qualification a question could hinge on\n\n"
        "Commit to conclusions; skip anything already explicit in the text. Wrap your understanding strictly "
        "inside <understanding> and </understanding> tags.\n\n"
        f"Passage:\n{article}\n"
    )


def quail_understanding_prompt(article: str) -> str:
    # QuAIL-specific: narrative comprehension whose questions probe TYPED reasoning
    # (causality, temporal order, event duration, subsequent state, belief states,
    # character identity/properties) AND include "not enough information" answers, so the
    # understanding must also track what the passage does NOT establish.
    return (
        "Read the passage and DERIVE, in advance, the reasoning needed to answer hard questions about it. "
        "Do the reasoning NOW and commit to conclusions -- do not merely summarize the plot or restate events.\n\n"
        "Work out and write down, precisely, for this passage:\n"
        "- CHARACTERS & ENTITIES: who/what appears, their identities, roles, relationships, and properties\n"
        "- EVENT ORDER: reconstruct the actual sequence of events -- what happened, in what order (before/after)\n"
        "- DURATION & TIMING: how long things take or last, and when they occur (absolute or relative)\n"
        "- CAUSALITY: for each key event, WHY it happened -- the cause, motive, or trigger -- and its effects\n"
        "- SUBSEQUENT STATES: what becomes true AFTER each key event, and the most likely state of things at "
        "and after the END of the passage\n"
        "- BELIEF & MENTAL STATES: what each character knows, believes, wants, or feels, and how that differs "
        "from reality or from what others believe\n\n"
        "For every fact and inference, mark its STATUS: what the passage establishes as DEFINITELY true, what "
        "it makes PROBABLE (the single best-supported reading, even if not stated outright), and what it does "
        "NOT determine (so 'not enough information' is recognizable when that is the correct answer). Resolve "
        "every inference to the most supported conclusion; commit, do not hedge.\n\n"
        "Be exhaustive and unambiguous; do not summarize in prose. Think first, then wrap your worked-out "
        "understanding strictly inside <understanding> and </understanding> tags.\n\n"
        f"Passage:\n{article}\n"
    )


def lsat_ar_understanding_prompt(article: str) -> str:
    return (
        "Read the setup of this analytical-reasoning problem and extract all logical structure "
        "in a precise, structured form that would let you solve questions about it.\n\n"
        "Identify and list:\n"
        "- the entities/elements being arranged and the slots/positions/groups they go into\n"
        "- every explicit rule or constraint, expressed precisely (e.g. 'If A then not B', 'C before D')\n"
        "- the contrapositive of every conditional rule\n"
        "- fixed assignments and absolute restrictions\n"
        "- ordering, grouping, and adjacency relationships\n"
        "- any forced deductions that necessarily follow from combining the rules "
        "(make implicit consequences explicit; state what must, cannot, or might be true)\n\n"
        "Be exhaustive and unambiguous; do not summarize in prose. "
        "Think first, then wrap your structured extraction strictly inside <understanding> and </understanding> tags.\n\n"
        f"Passage:\n{article}\n"
    )


def proofwriter_understanding_prompt(article: str) -> str:
    return (
        "Read the following facts and rules and extract their full logical structure so that you "
        "can answer True/False/Unknown entailment questions about them.\n\n"
        "Identify and list:\n"
        "- every base fact (what is explicitly stated, including negations)\n"
        "- every rule as a precise conditional (e.g. 'If X and Y then Z'), with its contrapositive\n"
        "- each entity and its known properties\n"
        "- every NEW fact derivable by applying the rules to the facts: forward-chain repeatedly to "
        "closure and make all implicit consequences explicit\n"
        "- briefly note what cannot be derived (remains Unknown)\n\n"
        "Be exhaustive and precise; do not summarize in prose. Think first, then wrap your structured "
        "extraction strictly inside <understanding> and </understanding> tags.\n\n"
        f"Facts and rules:\n{article}\n"
    )


# Registry mapping dataset name -> understanding prompt builder. Defaults to the
# RACE-style reading-comprehension prompt for any unknown dataset.
UNDERSTANDING_PROMPT_REGISTRY = {
    # race-c/race-high: v3 (2026-07-05) -- short, no-summarization, inference-only. Replaces
    # v1 (long lossy summaries that hit the token cap). v1 and v2 kept above for reference.
    "race-c": race_understanding_prompt_v3,
    "race-high": race_understanding_prompt_v3,
    "lsat-ar": lsat_ar_understanding_prompt,
    "quail": quail_understanding_prompt,
}


# opt-in prompt variants selected by config (understanding_prompt_version); default None keeps the
# registry. Keyed (dataset_key, version) so existing runs are byte-identical.
UNDERSTANDING_PROMPT_VERSIONS = {
    ("race-c", "v4"): race_understanding_prompt_v4,
    ("race-high", "v4"): race_understanding_prompt_v4,
}


def understanding_prompt(article: str, dataset_name: str | None = None, tagged: bool = True,
                         answer_ready: bool = False, version: str | None = None) -> str:
    key = (dataset_name or "").strip().lower()
    if version and (key, version) in UNDERSTANDING_PROMPT_VERSIONS:
        text = UNDERSTANDING_PROMPT_VERSIONS[(key, version)](article)
    elif key in UNDERSTANDING_PROMPT_REGISTRY:
        text = UNDERSTANDING_PROMPT_REGISTRY[key](article)
    elif key.startswith("proofwriter"):  # prefix match for per-difficulty variants (proofwriter-d3)
        text = proofwriter_understanding_prompt(article)
    else:
        text = race_understanding_prompt(article)
    if answer_ready:
        # Pair with direct (no-CoT) QA: the understanding must do ALL the reasoning so
        # the answer is a lookup. Replace the "extract/think-first" instruction with a
        # directive to work the problem out to answerable conclusions.
        text = re.sub(
            r"Think first.*?</understanding>\s*tags\.\s*",
            "Then REASON THROUGH the passage and WORK OUT every conclusion that could be asked about: "
            "state the resulting deductions, consequences, and answers explicitly and completely, so that "
            "any question can be answered directly from your understanding with NO further reasoning. "
            "Wrap your worked-out understanding strictly inside <understanding> and </understanding> tags.\n\n",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    if not tagged:
        # use_full_understanding_output mode: the WHOLE generation is the understanding, so
        # the <understanding>-tag instruction is counterproductive (wastes tokens on formatting
        # and pollutes the artifact). Replace it with a plain "produce the extraction" instruction.
        text = re.sub(
            r"Think first.*?</understanding>\s*tags\.\s*",
            "Think step by step and produce a thorough, structured extraction of the passage's "
            "reasoning-relevant content.\n\n",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return text

def _proofwriter_eval_understanding_prompt(article: str, understanding: str, question_text: str, options: List[str]) -> str:
    facts = f"Facts and rules:\n{article}\n\n" if article.strip() else ""
    return (
        _PROOFWRITER_RUBRIC
        + "You are given a structured understanding (derived facts) of the rules to help you.\n\n"
        + facts
        + "Understanding:\n"
        + f"{understanding}\n\n"
        + f"{_mcq_block(question_text, options)}\n"
        + "Using the understanding (and the facts/rules if given), return your final answer as exactly "
          "one boxed letter (\\boxed{A}, \\boxed{B}, or \\boxed{C}). After answering, terminate the response.\n"
    )


def qa_eval_understanding_only_prompt(article: str, understanding: str, question_text: str, options: List[str], dataset_name: str | None = None, direct: bool = False) -> str:
    if (dataset_name or "").strip().lower().startswith("proofwriter"):
        return _proofwriter_eval_understanding_prompt(article, understanding, question_text, options)
    # ``direct``: answer straight from the understanding with no chain-of-thought.
    # This forces the understanding (not the QA step's own reasoning) to carry the
    # work, making the reward a sharper signal of understanding quality.
    answer_line = (
        "Using only the information above, output your final choice as exactly one boxed letter "
        "(e.g., \\boxed{A}, \\boxed{B}, etc.). Do not explain or show any reasoning. "
        "After the boxed letter, terminate the response.\n"
        if direct else
        "Think step-by-step, then output your final choice as exactly one boxed letter "
        "(e.g., \\boxed{A}, \\boxed{B}, etc.). After answering, terminate the response.\n"
    )
    if article.strip() == "":
        return (
            "Use the provided understanding of a passage to solve the multiple-choice question.\n\n"
            "Understanding:\n"
            f"{understanding}\n\n"
            f"{_mcq_block(question_text, options)}\n"
            + answer_line
        )
    return (
        "Use the provided passage and understanding to solve the multiple-choice question.\n\n"
        "Passage:\n"
        f"{article}\n\n"
        "Understanding:\n"
        f"{understanding}\n\n"
        f"{_mcq_block(question_text, options)}\n"
        + answer_line
    )

def qa_eval_understanding_with_passage_prompt(article: str, understanding: str, question_text: str, options: List[str]) -> str:
    return (
        "Use the provided passage and understanding to solve the multiple-choice question.\n"
        "Reason briefly and then output your final choice as exactly one boxed letter (e.g., \\boxed{A}, \\boxed{B}, etc.).\n\n"
        "Passage:\n"
        f"{article}\n\n"
        "Understanding:\n"
        f"{understanding}\n\n"
        f"{_mcq_block(question_text, options)}\n\n"
    )

def understand_and_answer_prompt(
    article: str,
    question_text: str,
    options: List[str],
    dataset_name: str | None = None,
) -> str:
    if (dataset_name or "").strip().lower().startswith("proofwriter"):
        return (
            _PROOFWRITER_RUBRIC
            + "First work out the consequences of the rules (forward-chain the facts), then answer.\n\n"
            + f"Facts and rules:\n{article}\n\n"
            + f"{_mcq_block(question_text, options)}\n\n"
            + "Think step-by-step, then output exactly one boxed letter (\\boxed{A}, \\boxed{B}, or \\boxed{C}).\n\n"
        )
    return (
        "Reason the passage and create an understanding of the passage before solving.\n"
        "Then answer the question and output exactly one boxed letter (e.g., \\boxed{A}, \\boxed{B}, etc.).\n\n"
        f"Passage:\n{article}\n\n"
        f"{_mcq_block(question_text, options)}\n\n"
    )


def reflection_prompt(understanding: str, question_text: str, options: List[str], chosen_answer: str) -> str:
    return (
        "You were given the understanding below to help answer a multiple-choice question, "
        "and you chose an answer.\n\n"
        "Understanding:\n"
        f"{understanding}\n\n"
        f"{_mcq_block(question_text, options)}\n"
        f"Your chosen answer: {chosen_answer}\n\n"
        "Did the understanding provide information that was actually necessary to arrive at this answer "
        "(i.e. you could not have answered as confidently from the passage/question alone)?\n"
        "Answer with exactly one boxed word: \\boxed{Yes} or \\boxed{No}.\n"
    )


def maybe_apply_chat_template(tokenizer: Any, prompt_text: str, is_instruct: bool) -> str:
    if is_instruct and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return  prompt_text