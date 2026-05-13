from __future__ import annotations

import logging
from typing import Any, List


def _mcq_block(question_text: str, options: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    rendered = []
    for i, opt in enumerate(options[:4]):
        rendered.append(f"{letters[i]}. {opt}")
    return f"Question: {question_text}\nOptions:\n" + "\n".join(rendered)


def qa_cot_prompt(article: str, question_text: str, options: List[str]) -> str:
    return (
        "Solve the multiple-choice question using the passage.\n"
        "Think step by step, then output exactly one final boxed letter.\n\n"
        "Passage:\n"
        f"{article}\n\n"
        f"{_mcq_block(question_text, options)}\n\n"
        "Output format requirement: final line contains one of \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}."
    )


def implication_prompt(article: str, max_implications: int) -> str:
    return (
        "You are given a reading passage. Infer non-trivial implications that are likely to help solve reading-comprehension questions.\n\n"
        "Passage:\n"
        f"{article}\n\n"
        "Instructions:\n"
        f"1) Produce up to {max_implications} concise implications.\n"
        "2) Each implication must be grounded in the passage and useful for answering questions.\n"
        "3) Prefer implications that connect multiple facts.\n"
        "4) Avoid copying long spans from the passage.\n"
        "5) Output ONLY the implications as numbered lines.\n"
    )


def qa_eval_prompt(article: str, implications: str, question_text: str, options: List[str]) -> str:
    return (
        "Use the passage and derived implications to answer the multiple-choice question.\n"
        "Reason briefly and then output the final choice as \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}.\n\n"
        "Passage:\n"
        f"{article}\n\n"
        "Derived implications:\n"
        f"{implications}\n\n"
        f"{_mcq_block(question_text, options)}\n\n"
        "Final answer must be boxed with one letter only."
    )


def understanding_prompt(article: str) -> str:
    return (
        "You are given a reading passage. Your task is to build a complete "
        "representation of the passage.\n\n"
        "Passage:\n"
        f"{article}\n\n"
        "Reason freely about the passage inside <think> </think> tags. "
        "Consider the main thesis, key entities and what is true of them, "
        "causal relationships, important contrasts, and the author's implicit stance.\n\n"
        "Then write your understanding inside <understanding> </understanding> tags. "
        "Your understanding should be thorough enough that someone who hasn't read "
        "the passage could answer questions about it."
    )

def qa_eval_understanding_only_prompt(understanding: str, question_text: str, options: List[str]) -> str:
    return (
        "Use the provided understanding of a passage to answer the multiple-choice question.\n"
        "Reason briefly and then output the final choice as \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}.\n\n"
        "Understanding:\n"
        f"{understanding}\n\n"
        f"{_mcq_block(question_text, options)}\n\n"
        "Final answer must be boxed with one letter only."
    )

def qa_train_prompt(article: str, best_implications: str, question_text: str, options: List[str]) -> str:
    return (
        "Solve the multiple-choice question with step-by-step reasoning.\n"
        "Use the provided implications as intermediate guidance.\n"
        "Return final answer as exactly one boxed letter.\n\n"
        "Passage:\n"
        f"{article}\n\n"
        "Best implications:\n"
        f"{best_implications}\n\n"
        f"{_mcq_block(question_text, options)}\n\n"
        "Output format requirement: final line contains one of \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}."
    )


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
