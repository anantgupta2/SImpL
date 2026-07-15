"""SImpL-A (experimental): difficulty-weighted, complementary understanding reward.

Motivation (see experiments/FINDINGS.md, the 2026-06-05 "drawing board" pivot):
the stock SImpL understanding reward is REDUNDANT with CoT. Mechanically, under
GRPO the understanding task only learns when the N understanding samples of a
passage earn DIFFERENT rewards. With the passage shown at QA time the model
answers correctly regardless of the understanding -> rewards ~equal -> ~no
gradient. We attack this two ways, stacked:

  (a) PASSAGE-FREE answering: score each understanding by answering the passage's
      questions from the UNDERSTANDING ALONE (set use_understanding_passage=False
      in the config). Now a good understanding -> correct, a bad one -> wrong, so
      the rewards spread out and there is real gradient. (This is "B".)

  (c) DIFFICULTY WEIGHTING: weight each question by w(q) = 1 - acc_baseline(q),
      where acc_baseline(q) is how well the model answers q DIRECTLY from the
      passage (empty understanding). Questions direct-CoT already nails get ~0
      weight, so the understanding is paid ONLY for the lift it adds over CoT.
      This is the "marginal value over CoT" idea done correctly for GRPO: a
      per-question WEIGHT (which changes the within-group ranking of samples and
      therefore survives the group-mean advantage) rather than a per-group
      CONSTANT subtraction (which cancels and trains identically to stock SImpL).

Everything else (data layout, cot rollouts, learner, advantages) is inherited
unchanged from SImpL_oat so this file stays a thin, reviewable experiment.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from oat.actors.base import ActorBase
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import TransitionData

from src.utils.oat_prompt_templates import (
    understanding_prompt,
    maybe_apply_chat_template,
    qa_cot_prompt,
    qa_eval_understanding_only_prompt,
    reflection_prompt,
)
from src.utils.parsing_utils import extract_boxed_letter, extract_yes_no

from src.algorithm.SImpL_oat import (
    SImpLArgs,
    SImpLActor,
    SImpLLearner,
    configure_simpl_args,
)
from src.utils.fail_fast import fail_fast


@dataclass
class SImpLMarginalArgs(SImpLArgs):
    # Weight each question by (1 - acc_baseline(q)) so the understanding is rewarded
    # only for lift over direct CoT. If False, falls back to plain mean accuracy
    # (i.e. behaves like stock SImpL but in this subclass).
    difficulty_weighting: bool = True
    # The baseline answers each question DIRECTLY from the passage (empty
    # understanding) -> measures what CoT already gets. This is the natural
    # "direct competence" reference for the weighting.
    baseline_with_passage: bool = True
    # Samples for the baseline QA pass (0 -> reuse qa_num_samples).
    baseline_num_samples: int = 0
    # Floor on a question's weight so every question keeps a little signal even if
    # the baseline already solves it (avoids zero-gradient passages).
    min_question_weight: float = 0.0
    # Extra multiplier on the UNDERSTANDING reward only (on top of reward_scale).
    # Under drgrpo (no per-group std normalization) the reward MAGNITUDE sets how
    # much the understanding groups contribute to the gradient relative to the cot
    # groups -- so this is the direct knob to balance cot vs understanding. 1.0 =
    # same scale as cot (current behaviour); >1 up-weights understanding gradient.
    understanding_reward_scale: float = 1.0


def configure_simpl_marginal_args(args: SImpLMarginalArgs) -> SImpLMarginalArgs:
    return configure_simpl_args(args)


class SImpLMarginalActor(SImpLActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.difficulty_weighting = bool(getattr(self.args, "difficulty_weighting", True))
        self.baseline_with_passage = bool(getattr(self.args, "baseline_with_passage", True))
        bns = int(getattr(self.args, "baseline_num_samples", 0))
        self.baseline_num_samples = bns if bns > 0 else self.qa_num_samples
        self.min_question_weight = float(getattr(self.args, "min_question_weight", 0.0))
        self.understanding_reward_scale = float(getattr(self.args, "understanding_reward_scale", 1.0))

    @fail_fast("SImpLMarginalActor.step")
    def step(self, prompts, formatted_prompts, references=None) -> List[TransitionData]:
        del formatted_prompts
        assert not self.eval_mode

        t0 = time.time()
        if not prompts:
            return self.ipc_client.serialize_ipc([])
        if references is None:
            references = [None] * len(prompts)

        # --- Phase A: build reasoning rollouts (cot or understanding) ---------
        generation_prompts: List[str] = []
        generation_meta: List[Dict] = []
        for doc_idx, ref_str in enumerate(references):
            try:
                parsed = json.loads(ref_str) if ref_str else {"task_type": "cot", "questions": []}
            except json.JSONDecodeError:
                parsed = {"task_type": "cot", "questions": []}

            task_type = parsed.get("task_type", "cot")
            valid_questions = parsed.get("questions", [])

            if task_type == "cot" and valid_questions:
                q = valid_questions[0]
                prompt_text = qa_cot_prompt(prompts[doc_idx], q["question"], q["options"], self.dataset_name)
                for _ in range(self.reasoning_num_samples):
                    generation_prompts.append(
                        maybe_apply_chat_template(self.tokenizer, prompt_text, self.is_instruct)
                    )
                    generation_meta.append({
                        "doc_idx": doc_idx,
                        "task_type": "cot",
                        "gold": q.get("answer", ""),
                        "num_opt": len(q.get("options", [])),
                        "valid_questions": valid_questions,
                    })
            else:
                prompt_text = understanding_prompt(prompts[doc_idx], self.dataset_name, tagged=not self.use_full_understanding_output, answer_ready=self.qa_direct_answer)
                for _ in range(self.reasoning_num_samples):
                    generation_prompts.append(
                        maybe_apply_chat_template(self.tokenizer, prompt_text, self.is_instruct)
                    )
                    generation_meta.append({
                        "doc_idx": doc_idx,
                        "task_type": "understanding",
                        "valid_questions": valid_questions,
                    })

        outputs = self.generate(
            generation_prompts,
            self._build_sampling_params(
                temperature=self.sampling_params.temperature,
                max_tokens=self.reasoning_max_tokens,
                with_logprobs=True,
            ),
        )

        extracted_data = []
        for out in outputs:
            prompt_ids, text, token_ids, logprobs, is_truncated = self._extract_output(out)
            extracted_data.append({
                "prompt_ids": prompt_ids,
                "text": text,
                "token_ids": token_ids,
                "logprobs": logprobs,
                "is_truncated": is_truncated,
            })

        # --- Phase B: answer questions using each understanding ---------------
        # eval_meta carries q_index + doc_idx so Phase D can group per question
        # and look up the matching baseline difficulty weight.
        eval_prompts: List[str] = []
        eval_meta: List[Dict] = []
        for i, meta in enumerate(generation_meta):
            if meta["task_type"] != "understanding" or not meta["valid_questions"]:
                continue
            understanding = self._extract_understanding_from_tags(extracted_data[i]["text"])
            if not understanding:
                continue
            article = prompts[meta["doc_idx"]] if self.use_understanding_passage else ""
            for q_index, q in enumerate(meta["valid_questions"]):
                eval_prompts.append(
                    maybe_apply_chat_template(
                        self.tokenizer,
                        qa_eval_understanding_only_prompt(article, understanding, q.get("question", ""), q.get("options", []), self.dataset_name, direct=self.qa_direct_answer),
                        self.is_instruct,
                    )
                )
                eval_meta.append({
                    "parent_idx": i,
                    "doc_idx": meta["doc_idx"],
                    "q_index": q_index,
                    "understanding": understanding,
                    "question": q.get("question", ""),
                    "options": q.get("options", []),
                    "gold": q.get("answer", ""),
                    "num_opt": len(q.get("options", [])),
                })

        eval_records = []
        total_eval_q = 0
        total_eval_correct = 0
        if eval_prompts:
            eval_outputs = self.generate(
                eval_prompts,
                self._build_sampling_params(
                    temperature=self.sampling_params.temperature if self.qa_num_samples > 1 else 0.0,
                    max_tokens=self.qa_eval_max_tokens,
                    with_logprobs=False,
                    n=self.qa_num_samples,
                ),
            )
            for out, e_meta in zip(eval_outputs, eval_meta):
                for completion in out.outputs:
                    answer_text = completion.text or ""
                    pred = extract_boxed_letter(answer_text, e_meta["num_opt"])
                    is_correct = int(pred == e_meta["gold"])
                    num_toks = len(completion.token_ids) if completion.token_ids else 0
                    eval_records.append({
                        "parent_idx": e_meta["parent_idx"],
                        "q_index": e_meta["q_index"],
                        "is_correct": is_correct,
                        "num_toks": num_toks,
                        "pred": pred,
                        "meta": e_meta,
                    })
                    total_eval_q += 1
                    total_eval_correct += is_correct

        # --- Phase B2: per-passage DIFFICULTY baseline (answer DIRECTLY) -------
        # For each passage with at least one valid understanding, answer every
        # question from the passage alone (empty understanding) once, and turn the
        # accuracy into a question weight w(q) = 1 - acc_baseline(q). This is the
        # "what does direct CoT already get" reference; reused across that
        # passage's N understanding samples.
        baseline_weight: Dict[tuple, float] = {}        # (doc_idx, q_index) -> weight
        baseline_acc_mean = 0.0
        if self.difficulty_weighting and eval_meta:
            # unique (doc_idx -> valid_questions) among understanding rollouts
            doc_questions: Dict[int, List[Dict]] = {}
            for i, meta in enumerate(generation_meta):
                if meta["task_type"] == "understanding" and meta["valid_questions"]:
                    doc_questions.setdefault(meta["doc_idx"], meta["valid_questions"])

            base_prompts: List[str] = []
            base_meta: List[Dict] = []
            for doc_idx, qs in doc_questions.items():
                article = prompts[doc_idx] if self.baseline_with_passage else ""
                for q_index, q in enumerate(qs):
                    base_prompts.append(
                        maybe_apply_chat_template(
                            self.tokenizer,
                            qa_eval_understanding_only_prompt(article, "", q.get("question", ""), q.get("options", []), self.dataset_name, direct=self.qa_direct_answer),
                            self.is_instruct,
                        )
                    )
                    base_meta.append({
                        "doc_idx": doc_idx,
                        "q_index": q_index,
                        "gold": q.get("answer", ""),
                        "num_opt": len(q.get("options", [])),
                    })

            if base_prompts:
                base_outputs = self.generate(
                    base_prompts,
                    self._build_sampling_params(
                        temperature=self.sampling_params.temperature if self.baseline_num_samples > 1 else 0.0,
                        max_tokens=self.qa_eval_max_tokens,
                        with_logprobs=False,
                        n=self.baseline_num_samples,
                    ),
                )
                accs = []
                for out, b_meta in zip(base_outputs, base_meta):
                    correct = 0
                    total = 0
                    for completion in out.outputs:
                        pred = extract_boxed_letter(completion.text or "", b_meta["num_opt"])
                        correct += int(pred == b_meta["gold"])
                        total += 1
                    acc = correct / max(total, 1)
                    w = max(self.min_question_weight, 1.0 - acc)
                    baseline_weight[(b_meta["doc_idx"], b_meta["q_index"])] = w
                    accs.append(acc)
                baseline_acc_mean = float(np.mean(accs)) if accs else 0.0

        # --- Phase C (optional): self-reported reflection ---------------------
        reflection_by_parent: Dict[int, List[float]] = defaultdict(list)
        reflection_yes_rate = 0.0
        if self.use_reflection_reward and eval_records:
            reflection_prompts = [
                maybe_apply_chat_template(
                    self.tokenizer,
                    reflection_prompt(
                        rec["meta"]["understanding"],
                        rec["meta"]["question"],
                        rec["meta"]["options"],
                        rec["pred"] or "(no answer)",
                    ),
                    self.is_instruct,
                )
                for rec in eval_records
            ]
            reflection_outputs = self.generate(
                reflection_prompts,
                self._build_sampling_params(
                    temperature=0.0, max_tokens=self.reflection_max_tokens,
                    with_logprobs=False, n=1,
                ),
            )
            yes_count = 0
            for rec, out in zip(eval_records, reflection_outputs):
                said_yes = extract_yes_no(out.outputs[0].text or "") == "Yes"
                yes_count += int(said_yes)
                reflection_by_parent[rec["parent_idx"]].append(
                    self._reflection_signal(rec["is_correct"], said_yes)
                )
            reflection_yes_rate = yes_count / max(len(eval_records), 1)

        # --- Phase D: assemble rewards & transitions --------------------------
        # Group eval records per (understanding sample, question) so we can take a
        # per-question accuracy and combine with the difficulty weights.
        eval_by_parent_q: Dict[int, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        for rec in eval_records:
            eval_by_parent_q[rec["parent_idx"]][rec["q_index"]].append(rec)

        info_rewards_cot: List[float] = []
        info_rewards_und: List[float] = []
        correct_count_cot = 0
        valid_count_cot = 0

        trajectory_data: List[TransitionData] = []
        rewards_per_meta: List[float] = []
        loss_masks: List[bool] = []
        for i, (meta, ext_data) in enumerate(zip(generation_meta, extracted_data)):
            reward = 0.0
            has_valid_q = len(meta["valid_questions"]) > 0

            if meta["task_type"] == "cot":
                pred = extract_boxed_letter(ext_data["text"], meta["num_opt"])
                gold = meta["gold"]
                num_opt = meta["num_opt"]
                if pred == gold:
                    reward = self.correct_reward
                elif pred in [chr(ord("A") + k) for k in range(num_opt)]:
                    reward = 0.1
                else:
                    reward = self.incorrect_reward
                reward *= self.scale_reward
                if has_valid_q and not ext_data["is_truncated"]:
                    valid_count_cot += 1
                    correct_count_cot += int(pred == gold)
                info_rewards_cot.append(reward)
            else:
                per_q = eval_by_parent_q.get(i, {})
                if not per_q and has_valid_q:
                    reward = -0.1 * self.scale_reward
                elif per_q:
                    doc_idx = meta["doc_idx"]
                    num = 0.0
                    den = 0.0
                    correct_toks = []  # (num_toks) of correct answers for conciseness bonus
                    for q_index, recs in per_q.items():
                        q_correct = sum(r["is_correct"] for r in recs)
                        q_total = len(recs)
                        q_acc = q_correct / max(q_total, 1)
                        if self.difficulty_weighting:
                            w = baseline_weight.get((doc_idx, q_index), 1.0)
                        else:
                            w = 1.0
                        num += w * q_acc
                        den += w
                        correct_toks.extend(r["num_toks"] for r in recs if r["is_correct"])
                    weighted_acc = (num / den) if den > 1e-8 else 0.0
                    base_reward = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * weighted_acc
                    if correct_toks:
                        bonus = sum(
                            self.conciseness_penalty_k * (1.0 - nt / self.qa_eval_max_tokens)
                            for nt in correct_toks
                        )
                        base_reward += bonus / len(correct_toks)
                    if self.use_reflection_reward and reflection_by_parent.get(i):
                        sig = reflection_by_parent[i]
                        base_reward += self.reflection_bonus_k * (sum(sig) / len(sig))
                    reward = base_reward * self.scale_reward * self.understanding_reward_scale
                info_rewards_und.append(reward)

            loss_mask = has_valid_q
            if self.args.ignore_no_eos and ext_data["is_truncated"]:
                loss_mask = False
            rewards_per_meta.append(reward)
            loss_masks.append(loss_mask)

        info = {
            "actor/num_documents": float(len(prompts)),
            "actor/num_samples": float(self.reasoning_num_samples),
            "actor/cot_reward_mean": float(np.mean(info_rewards_cot)) if info_rewards_cot else 0.0,
            "actor/cot_accuracy": float(correct_count_cot / max(valid_count_cot, 1)),
            "actor/understanding_reward_mean": float(np.mean(info_rewards_und)) if info_rewards_und else 0.0,
            "actor/eval_question_accuracy": float(total_eval_correct / max(total_eval_q, 1)),
            "actor/baseline_direct_accuracy": float(baseline_acc_mean),
            "actor/reflection_yes_rate": float(reflection_yes_rate),
            "actor/step_time": float(time.time() - t0),
        }

        for i, ext_data in enumerate(extracted_data):
            trajectory_data.append(
                TransitionData(
                    prompt=generation_prompts[i],
                    prompt_ids=ext_data["prompt_ids"],
                    response=ext_data["text"],
                    response_ids=ext_data["token_ids"],
                    response_logprobs=ext_data["logprobs"],
                    rewards=self._terminal_reward(len(ext_data["token_ids"]), rewards_per_meta[i]),
                    loss_mask=loss_masks[i],
                    info=info,
                )
            )

        expected = len(prompts) * self.reasoning_num_samples
        assert len(trajectory_data) == expected
        logging.info(
            "SImpL-marginal actor done: docs=%d transitions=%d baseline_acc=%.3f",
            len(prompts), len(trajectory_data), baseline_acc_mean,
        )
        return self.ipc_client.serialize_ipc(trajectory_data)


def run_simpl_marginal_oat(args: SImpLMarginalArgs):
    args = configure_simpl_marginal_args(args)
    args = default_args_validation(args)

    program, local_resources = get_program(
        args, learner_cls=SImpLLearner, actor_cls=SImpLMarginalActor,
    )
    lp.launch(
        program, launch_type=args.launch_type,
        local_resources=local_resources, terminal="current_terminal",
    )


if __name__ == "__main__":
    cli_args: SImpLMarginalArgs = get_default_args(SImpLMarginalArgs)
    run_simpl_marginal_oat(cli_args)
