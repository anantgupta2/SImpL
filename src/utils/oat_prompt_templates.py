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


def implication_prompt(article: str, max_implications: int) -> str:
    return (
        "You are given a reading passage. Infer non-trivial implications that are likely to help solve reading-comprehension questions.\n\n"
        "Passage:\n"
        f"{article}\n\n"
        "Instructions:\n"
        f"- Produce up to {max_implications} concise implications.\n"
        "- Each implication must be grounded in the passage and useful for answering questions.\n"
        "- Prefer implications that connect multiple facts.\n"
        "- Avoid copying long spans from the passage.\n"
        "- Output ONLY the implications as numbered lines.\n"
    )


def understanding_prompt(article: str) -> str:
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

# def understanding_prompt(article: str) -> str:
#     return (
#         "You are preparing a compact reasoning memory for future question answering.\n\n"
#         "Your goal is NOT to summarize the passage for readability.\n",
#         "Your goal is to preserve information that could later be necessary to answer difficult or tricky questions.\n"
#         "Pay particular attention to: - entity relationships - chronology - causal structure - comparisons and contrasts "
#         "- quantities and numerical details - rare or unusual facts - exceptions and caveats - claims and opinions - "
#         "information that distinguishes similar answer choices - details that are easy to overlook. Preserve uncertainty and ambiguity when present.\n"
#         "Write the memory inside <understanding> tags.\n\n",
#         f"Passage: {article}\n"
#     )

# def understanding_prompt(article: str) -> str:
#     return (
#         "Read the passage and build a concise internal understanding.\n\n"
#         f"Passage:\n{article}\n\n"
#         "Reason about key entities, relationships, "
#         "causes, contrasts, and the author's viewpoint.\n\n"
#         "Then summarize the important information inside "
#         "<understanding> tags."
#     )
# def understanding_prompt(article: str) -> str:
#     return (
#         "You are given a reading passage. Your task is to build a complete "
#         "representation of the passage.\n\n"
#         "Passage:\n"
#         f"{article}\n\n"
#         "Reason freely about the passage inside <think> </think> tags. "
#         "Consider the main thesis, key entities and what is true of them, "
#         "causal relationships, important contrasts, and the author's implicit stance.\n\n"
#         "Then write your understanding inside <understanding> </understanding> tags. "
#         "Your understanding should be thorough enough that someone who hasn't read "
#         "the passage could answer questions about it."
#     )

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


# def qa_train_prompt(article: str, best_implications: str, question_text: str, options: List[str]) -> str:
#     return (
#         "Solve the multiple-choice question with step-by-step reasoning.\n"
#         "Use the provided implications as intermediate guidance.\n"
#         "Return final answer as exactly one boxed letter.\n\n"
#         "Passage:\n"
#         f"{article}\n\n"
#         "Best implications:\n"
#         f"{best_implications}\n\n"
#         f"{_mcq_block(question_text, options)}\n\n"
#         "Output format requirement: final line contains exactly one boxed letter (e.g., \\boxed{A}, \\boxed{B}, etc.)."
#     )


def maybe_apply_chat_template(tokenizer: Any, prompt_text: str, is_instruct: bool) -> str:
    if not is_instruct:
        return prompt_text

    if tokenizer is None:
        return prompt_text

    template_tokenizer = tokenizer
    if not hasattr(template_tokenizer, "apply_chat_template"):
        nested = getattr(template_tokenizer, "tokenizer", None)
        if nested is not None:
            template_tokenizer = nested

    apply_chat_template = getattr(template_tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return prompt_text

    system_message = (
        "You are a helpful assistant for reading comprehension. "
        "Follow the user's instructions exactly and keep answers grounded in the provided passage."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt_text},
    ]

    try:
        return apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        pass
    except Exception as exc:
        logging.warning("Failed system+user chat template call, trying user-only: %s", exc)

    user_only_messages = [{"role": "user", "content": prompt_text}]
    try:
        return apply_chat_template(
            user_only_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        try:
            return apply_chat_template(user_only_messages, tokenize=False)
        except Exception as exc:
            logging.warning("Failed to apply chat template, falling back to raw prompt: %s", exc)
            return prompt_text
    except Exception as exc:
        logging.warning("Failed to apply chat template, falling back to raw prompt: %s", exc)
        return prompt_text
