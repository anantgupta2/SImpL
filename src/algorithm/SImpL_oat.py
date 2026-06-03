"""SImpL: the "ideal" combined trainer.

Goal: extract more learning signal from the same data than plain CoT GRPO by
co-training two behaviours on shared weights:

  * ``cot``           - answer a question directly from the passage (this is the
                        behaviour the headline eval metric measures, and the one
                        we hope the understanding signal transfers into).
  * ``understanding`` - extract a structured, reasoning-relevant understanding of
                        the passage, rewarded by how well that understanding lets
                        the *same* model answer the passage's questions.

Improvements over ``cot_and_understanding_oat``:
  1. Dataset-specific understanding prompts (via ``understanding_prompt`` registry).
  2. The understanding is scored with the SAME prompt the evaluator uses
     (``qa_eval_understanding_only_prompt``); ``use_understanding_passage`` controls
     whether the passage is shown alongside the understanding, and must match eval.
  3. Per-step metrics are aggregated once at the end (the old code built the info
     dict mid-loop over partially-filled accumulators).
  4. Optional self-reported "did the understanding help?" reflection reward,
     grounded against correctness so it is harder to hack (off by default).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import numpy as np
import vllm
from datasets import load_dataset

from oat.actors.base import ActorBase
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import TransitionData
from oat.utils.data import PromptDataset, load_data_from_disk_or_hf
from oat.utils.ops import masked_mean, masked_sum

from src.utils.oat_prompt_templates import (
    understanding_prompt,
    maybe_apply_chat_template,
    qa_cot_prompt,
    qa_eval_understanding_only_prompt,
    reflection_prompt,
)
from src.utils.parsing_utils import (
    extract_boxed_letter,
    extract_yes_no,
    normalize_gold_letter,
    parse_questions,
)


def collate_prompt_batch(batch):
    processed_prompts = [item[0] for item in batch]
    raw_prompts = [item[1] for item in batch]
    references = [item[2] for item in batch]
    return processed_prompts, raw_prompts, references


@dataclass
class SImpLArgs(PPOArgs):
    prompt_data: str = ""
    input_key: str = "article"
    output_key: str = "questions"
    train_split: str = "train"
    max_train: int = 999999
    seed: int = 42
    dataset_name: str = "race-c"

    reasoning_num_samples: int = 8
    reasoning_max_tokens: int = 512
    qa_eval_max_tokens: int = 128
    qa_num_samples: int = 1

    incorrect_reward: float = 0.0
    correct_reward: float = 1.0
    conciseness_penalty_k: float = 0.3
    reward_scale: float = 2.0

    # Whether to show the passage alongside the understanding when scoring it.
    # Must match the evaluator's --understanding_with_passage flag.
    use_understanding_passage: bool = True

    # Self-reported "did the understanding help?" reflection reward (grounded
    # against answer correctness). Off by default; see _reflection_signal.
    use_reflection_reward: bool = False
    reflection_bonus_k: float = 0.2
    reflection_max_tokens: int = 16

    critic_type: Literal["ppo", "grpo", "drgrpo"] = field(default="drgrpo")
    remove_len_bias: bool = False
    remove_std_bias: bool = False

    # Standardize each (homogeneous) GRPO group's advantage by its own std.
    # WARNING: this re-introduces the GRPO std/"difficulty" bias that drgrpo
    # deliberately removes -- it amplifies near-zero-variance groups (i.e. noise)
    # and empirically DEGRADED LSAT-AR below the random floor. Keep OFF with
    # drgrpo. Proper CoT-vs-understanding balancing should use a per-task scale,
    # not per-group std. Default off.
    standardize_group_advantage: bool = False

    eval_steps: int = -1
    online_evaluation: bool = True
    apply_chat_template: bool = False
    is_instruct: bool = False
    beta: float = 0.0


def configure_simpl_args(args: SImpLArgs) -> SImpLArgs:
    if int(args.reasoning_num_samples) < 1:
        raise ValueError("reasoning_num_samples must be >= 1")

    args.algo = "PPO"
    args.oracle = ""
    args.oracle_type = "reward"
    args.is_instruct = bool(args.is_instruct)
    args.apply_chat_template = False
    args.online_evaluation = True
    args.eval_steps = -1
    args.beta = 0.04
    if "qwen" in args.pretrain.lower():
        args.use_fused_lm_head = False

    args.num_samples = int(args.reasoning_num_samples)
    minimum_buffer = int(args.rollout_batch_size_per_device) * int(args.num_samples)
    if args.pi_buffer_maxlen_per_device < minimum_buffer:
        args.pi_buffer_maxlen_per_device = minimum_buffer
    return args


class SImpLActor(PPOActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.reasoning_num_samples = int(self.args.reasoning_num_samples)
        self.reasoning_max_tokens = int(self.args.reasoning_max_tokens)
        self.qa_eval_max_tokens = int(self.args.qa_eval_max_tokens)
        self.qa_num_samples = int(getattr(self.args, "qa_num_samples", 1))
        self.correct_reward = float(self.args.correct_reward)
        self.incorrect_reward = float(self.args.incorrect_reward)
        self.conciseness_penalty_k = float(self.args.conciseness_penalty_k)
        self.is_instruct = bool(getattr(self.args, "is_instruct", False))
        self.scale_reward = float(getattr(self.args, "reward_scale", 2.0))
        self.dataset_name = str(getattr(self.args, "dataset_name", "race-c"))
        self.use_understanding_passage = bool(getattr(self.args, "use_understanding_passage", True))
        self.use_reflection_reward = bool(getattr(self.args, "use_reflection_reward", False))
        self.reflection_bonus_k = float(getattr(self.args, "reflection_bonus_k", 0.2))
        self.reflection_max_tokens = int(getattr(self.args, "reflection_max_tokens", 16))

        self.sampling_params.stop = None
        self.sampling_params.stop_token_ids = None

    # ---- helpers -------------------------------------------------------------
    def _build_sampling_params(self, *, temperature, max_tokens, with_logprobs, n=1):
        return vllm.SamplingParams(
            temperature=temperature,
            top_p=self.sampling_params.top_p,
            top_k=self.sampling_params.top_k,
            max_tokens=max_tokens,
            n=n,
            logprobs=1 if with_logprobs else None,
        )

    def _fallback_token_id(self) -> int:
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        return int(eos_id) if eos_id is not None else 0

    def _terminal_reward(self, token_count: int, reward: float) -> List[float]:
        n = max(1, int(token_count))
        dense = [0.0] * n
        dense[-1] = float(reward)
        return dense

    def _extract_output(self, output) -> Tuple[List[int], str, List[int], List[float], bool]:
        completion = output.outputs[0]
        prompt_ids = list(output.prompt_token_ids or [])
        text = completion.text or ""
        token_ids = list(completion.token_ids or [])
        if not token_ids:
            token_ids = [self._fallback_token_id()]

        logprobs = []
        if completion.logprobs:
            for i, token_id in enumerate(token_ids):
                if i >= len(completion.logprobs):
                    break
                token_map = completion.logprobs[i]
                if token_map and token_id in token_map:
                    logprobs.append(token_map[token_id].logprob)
                else:
                    logprobs.append(0.0)
        if len(logprobs) != len(token_ids):
            logprobs = [0.0] * len(token_ids)

        is_truncated = completion.finish_reason == "length"
        return prompt_ids, text, token_ids, logprobs, is_truncated

    def _extract_understanding_from_tags(self, text: str) -> str:
        match = re.search(r"<understanding>\s*(.*?)\s*</understanding>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match2 = re.search(r"<understanding>\s*(.*?)\s*$", text, re.DOTALL | re.IGNORECASE)
        if match2:
            return match2.group(1).strip()
        return ""

    @staticmethod
    def _reflection_signal(is_correct: int, said_yes: bool) -> float:
        """Couple the self-report to correctness so the model cannot earn the
        bonus by blindly claiming the understanding helped.

          correct & "yes" -> +1  (understanding plausibly contributed)
          wrong   & "yes" -> -1  (claimed help but failed -> discourage)
          otherwise       ->  0
        """
        if is_correct and said_yes:
            return 1.0
        if (not is_correct) and said_yes:
            return -1.0
        return 0.0

    # ---- main rollout --------------------------------------------------------
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
                prompt_text = qa_cot_prompt(prompts[doc_idx], q["question"], q["options"])
                for sample_idx in range(self.reasoning_num_samples):
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
                prompt_text = understanding_prompt(prompts[doc_idx], self.dataset_name)
                for sample_idx in range(self.reasoning_num_samples):
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
        eval_prompts: List[str] = []
        eval_meta: List[Dict] = []
        for i, meta in enumerate(generation_meta):
            if meta["task_type"] != "understanding" or not meta["valid_questions"]:
                continue
            understanding = self._extract_understanding_from_tags(extracted_data[i]["text"])
            if not understanding:
                continue
            article = prompts[meta["doc_idx"]] if self.use_understanding_passage else ""
            for q in meta["valid_questions"]:
                eval_prompts.append(
                    maybe_apply_chat_template(
                        self.tokenizer,
                        qa_eval_understanding_only_prompt(article, understanding, q.get("question", ""), q.get("options", [])),
                        self.is_instruct,
                    )
                )
                eval_meta.append({
                    "parent_idx": i,
                    "understanding": understanding,
                    "question": q.get("question", ""),
                    "options": q.get("options", []),
                    "gold": q.get("answer", ""),
                    "num_opt": len(q.get("options", [])),
                })

        eval_records = []  # one per (eval prompt, completion)
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
                        "is_correct": is_correct,
                        "num_toks": num_toks,
                        "pred": pred,
                        "meta": e_meta,
                    })
                    total_eval_q += 1
                    total_eval_correct += is_correct

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
                    temperature=0.0,
                    max_tokens=self.reflection_max_tokens,
                    with_logprobs=False,
                    n=1,
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
        eval_grouped: Dict[int, List[Dict]] = defaultdict(list)
        for rec in eval_records:
            eval_grouped[rec["parent_idx"]].append(rec)

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
                results = eval_grouped.get(i, [])
                if not results and has_valid_q:
                    # Produced no usable understanding -> penalise.
                    reward = -0.2 * self.scale_reward
                elif results:
                    correct = sum(r["is_correct"] for r in results)
                    total = len(results)
                    base_reward = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * (correct / total)
                    if correct > 0:
                        bonus = sum(
                            self.conciseness_penalty_k * (1.0 - r["num_toks"] / self.qa_eval_max_tokens)
                            for r in results if r["is_correct"]
                        )
                        base_reward += bonus / correct
                    if self.use_reflection_reward and reflection_by_parent.get(i):
                        sig = reflection_by_parent[i]
                        base_reward += self.reflection_bonus_k * (sum(sig) / len(sig))
                    reward = base_reward * self.scale_reward
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
            "SImpL actor done: docs=%d transitions=%d per_doc=%d",
            len(prompts), len(trajectory_data), self.reasoning_num_samples,
        )
        return self.ipc_client.serialize_ipc(trajectory_data)


class SImpLLearner(PPOLearner):
    def _init(self, args: SImpLArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.args = args
        self.args.max_queries = np.inf
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.reasoning_max_tokens)
            if args.critic_type == "drgrpo"
            else masked_mean
        )
        if args.critic_type in ["grpo", "ppo"] and args.remove_len_bias:
            self.masked_aggregator = functools.partial(
                masked_sum, constant_normalizer=args.reasoning_max_tokens,
            )

    def compute_monte_carlo_advantages(self, rewards, response_masks):
        del response_masks
        rewards = rewards.sum(-1)
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        # Standardize per group when using GRPO's std-bias term, OR whenever
        # standardize_group_advantage is set (the SImpL default), so that CoT and
        # understanding groups contribute comparable-magnitude gradients.
        std_normalize = (
            (self.args.critic_type in ["grpo"]) and (not self.args.remove_std_bias)
        ) or bool(getattr(self.args, "standardize_group_advantage", False))
        if std_normalize:
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.args.num_samples, dim=0)
            advantages = advantages / (std_grouped_rewards + 1e-8)
        return advantages

    def _load_prompt_data(self):
        prompt_data = self.args.prompt_data
        if os.path.isfile(prompt_data):
            ext = os.path.splitext(prompt_data)[1].lower()
            if ext in {".jsonl", ".json"}:
                logging.info("Loading prompt data from JSON file: %s", prompt_data)
                return load_dataset("json", data_files=prompt_data, split="train")
        return load_data_from_disk_or_hf(prompt_data)

    def prepare_data(self, strategy, tokenizer):
        data_obj = self._load_prompt_data()
        if hasattr(data_obj, "keys") and self.args.train_split in data_obj:
            train_dataset = data_obj[self.args.train_split]
        else:
            train_dataset = data_obj

        input_key = self.args.input_key
        if input_key not in train_dataset.column_names:
            input_key = "article"
        output_key = self.args.output_key
        if output_key not in train_dataset.column_names:
            output_key = "questions"

        def flatten_and_mix_questions(batch):
            new_articles = []
            new_questions = []
            for article, qs_json in zip(batch[input_key], batch[output_key]):
                qs = parse_questions(qs_json)
                valid_qs = []
                for q in qs:
                    q_text = q.get("question", "")
                    opts = q.get("options", [])
                    gold = normalize_gold_letter(q.get("answer", ""), len(opts) if isinstance(opts, list) else 4)
                    if isinstance(opts, list) and len(opts) >= 2 and q_text and gold:
                        valid_qs.append({"question": q_text, "options": opts, "answer": gold})
                if not valid_qs:
                    continue
                # One understanding task (covers all questions of the doc)...
                new_articles.append(article)
                new_questions.append(json.dumps({"task_type": "understanding", "questions": valid_qs}))
                # ...and one cot task per question.
                for vq in valid_qs:
                    new_articles.append(article)
                    new_questions.append(json.dumps({"task_type": "cot", "questions": [vq]}))
            return {input_key: new_articles, output_key: new_questions}

        train_dataset = train_dataset.map(
            flatten_and_mix_questions, batched=True, remove_columns=train_dataset.column_names
        )
        max_train = min(int(self.args.max_train), len(train_dataset))
        train_dataset = train_dataset.select(range(max_train))
        train_dataset = train_dataset.select_columns([input_key, output_key])

        self.prompts_dataset = PromptDataset(
            train_dataset, tokenizer, strategy,
            input_key=input_key, output_key=output_key,
            apply_chat_template=False, get_reference=True,
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset, strategy.args.rollout_batch_size_per_device,
            pin_memory=True, shuffle=True, collate_fn=collate_prompt_batch,
        )
        self.eval_prompts_dataset = None
        self.eval_prompts_dataloader = None

    def evaluate(self, dataloader, steps):
        del dataloader, steps
        return {}


def run_simpl_oat(args: SImpLArgs):
    args = configure_simpl_args(args)
    args = default_args_validation(args)

    program, local_resources = get_program(
        args, learner_cls=SImpLLearner, actor_cls=SImpLActor,
    )
    lp.launch(
        program, launch_type=args.launch_type,
        local_resources=local_resources, terminal="current_terminal",
    )


if __name__ == "__main__":
    cli_args: SImpLArgs = get_default_args(SImpLArgs)
    run_simpl_oat(cli_args)
