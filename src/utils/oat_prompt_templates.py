from __future__ import annotations

import logging
from typing import Any, List


def _mcq_block(question_text: str, options: List[str]) -> str:
    letters = "ABCDEFGH"

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

def qa_cot_prompt(article: str, question_text: str, options: List[str]) -> str:
    return (
        "Solve the multiple-choice question using the passage.\n"
        "Passage:\n"
        f"{article}\n\n"
        f"{_mcq_block(question_text, options)}\n\n"
        "Think step-by-step and return your final answer as exactly one boxed letter (e.g., \\boxed{A}, \\boxed{B}, etc.).\n\n"
    )

def race_understanding_prompt(article: str) -> str:
    return (
        "Read the passage and extract concise reasoning-relevant conclusions, relationships, and implications that would help answer difficult questions about it.\n\n"
        "Preserve:\n"
        "- key entities and relationships\n"
        "- causal and temporal structure\n"
        "- important distinctions and caveats\n"
        "- claims, viewpoints, and numerical details\n\n"
        "Focus on information useful for reasoning and question answering, not stylistic summarization.\n"
        "Think first then wrap your extraction strictly inside <understanding> and </understanding> tags.\n\n"
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


# Registry mapping dataset name -> understanding prompt builder. Defaults to the
# RACE-style reading-comprehension prompt for any unknown dataset.
UNDERSTANDING_PROMPT_REGISTRY = {
    "race-c": race_understanding_prompt,
    "race-high": race_understanding_prompt,
    "lsat-ar": lsat_ar_understanding_prompt,
}


def understanding_prompt(article: str, dataset_name: str | None = None) -> str:
    key = (dataset_name or "").strip().lower()
    builder = UNDERSTANDING_PROMPT_REGISTRY.get(key, race_understanding_prompt)
    return builder(article)

def qa_eval_understanding_only_prompt(article: str, understanding: str, question_text: str, options: List[str]) -> str:
    if article.strip() == "":
        return (
            "Use the provided understanding of a passage to solve the multiple-choice question.\n\n"
            "Understanding:\n"
            f"{understanding}\n\n"
            f"{_mcq_block(question_text, options)}\n"
            "Think step-by-step, then output your final choice as exactly one boxed letter (e.g., \\boxed{A}, \\boxed{B}, etc.). After answering, terminate the response.\n"
        )
    return (
        "Use the provided passage and understanding to solve the multiple-choice question.\n\n"
        "Passage:\n"
        f"{article}\n\n"
        "Understanding:\n"
        f"{understanding}\n\n"
        f"{_mcq_block(question_text, options)}\n"
        "Think step-by-step, then output your final choice as exactly one boxed letter (e.g., \\boxed{A}, \\boxed{B}, etc.). After answering, terminate the response.\n"
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
) -> str:
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