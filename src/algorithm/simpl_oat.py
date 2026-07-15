"""SImpL (rotate): co-train an *understanding* objective alongside cot, on shared weights.

This is the single, consolidated SImpL implementation (the old SImpL/marginal/spice
variants are archived under src/algorithm/archive/). Per passage, in one step:

  1. generate N *understanding* rollouts from the passage alone;
  2. score each understanding by answering ALL of the passage's questions
     (difficulty-weighted marginal reward -- harder questions weigh more);
  3. select ONE question (rotate = round-robin across epochs, or random) and
     generate N *cot* rollouts answering it (rewarded by correctness).

So per passage we emit 1 understanding GRPO group (N samples) + 1 cot group (N
samples), contiguous, and Dr.GRPO recovers them via rewards.view(-1, num_samples).

The cot-only baseline (cot_only_oat.py) trains cot on the *flattened* (one row per
question) dataset -- the standard way these datasets are trained -- so "SImpL vs
cot" isolates the understanding objective.
"""

from __future__ import annotations

import functools
import json
import logging
import random
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
)
from src.utils.parsing_utils import extract_boxed_letter, normalize_gold_letter, parse_questions
from src.utils.fail_fast import fail_fast


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
    reasoning_max_tokens: int = 768
    qa_eval_max_tokens: int = 16
    qa_num_samples: int = 4

    incorrect_reward: float = 0.0
    correct_reward: float = 1.0
    conciseness_penalty_k: float = 0.5
    reward_scale: float = 1.0
    understanding_reward_scale: float = 1.0

    # Difficulty-weighted understanding reward: weight each question by (1 - direct
    # passage-only accuracy), so the understanding is rewarded most for enabling the
    # questions the model can't already answer cold. Needs the direct-baseline pass.
    difficulty_weighting: bool = True
    baseline_with_passage: bool = True   # direct baseline sees the passage (matches cot)
    baseline_num_samples: int = 4
    min_question_weight: float = 0.0

    # Whether to show the passage alongside the understanding when scoring it.
    # Must match the evaluator's --understanding_with_passage flag.
    use_understanding_passage: bool = True
    # Answer DIRECTLY from the understanding (no cot, just the boxed letter) -> forces
    # the understanding to carry the reasoning. Pair with a small qa_eval_max_tokens.
    qa_direct_answer: bool = True
    # Treat the whole understanding generation as the understanding (base models often
    # omit the <understanding> tags, so strict extraction silently drops good ones).
    use_full_understanding_output: bool = False

    # Which question to train cot on per passage:
    #   "rotate" -> deterministic round-robin (question[i] on the i-th time the passage
    #               is seen == epoch i); per-passage counter lives in the actor, so it
    #               is only authoritative with a SINGLE actor process (guarded below);
    #   "random" -> a fresh uniform draw each step.
    selection_mode: str = "rotate"

    critic_type: Literal["ppo", "grpo", "drgrpo"] = field(default="drgrpo")
    remove_len_bias: bool = False
    remove_std_bias: bool = False

    eval_steps: int = -1
    online_evaluation: bool = True
    apply_chat_template: bool = False
    is_instruct: bool = False
    beta: float = -1.0  # -1 => use default 0.04; set in config to sweep


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
    if float(args.beta) < 0:
        args.beta = 0.04
    if "qwen" in args.pretrain.lower():
        args.use_fused_lm_head = False

    args.num_samples = int(args.reasoning_num_samples)
    # Each passage emits 1 understanding group + 1 cot group of N samples.
    need = int(args.rollout_batch_size_per_device) * int(args.num_samples) * 2
    if args.pi_buffer_maxlen_per_device < need:
        args.pi_buffer_maxlen_per_device = need

    # selection_mode=="rotate" keeps its per-passage round-robin counter in the actor
    # PROCESS, so it is only authoritative with exactly ONE actor. Replicate oat's
    # launch-time actor count (interface.py) and fail fast otherwise.
    if str(getattr(args, "selection_mode", "rotate")).lower() == "rotate":
        actor_gpus = int(args.gpus) if args.collocate else (
            args.gpus // 2 if args.gpus % 2 == 0 else args.gpus // 2 + 1)
        total_actors = (actor_gpus // int(args.num_gpus_per_actor)) * int(getattr(args, "num_groups", 1))
        assert total_actors == 1, (
            f"selection_mode='rotate' needs a single actor for its per-passage round-robin "
            f"to be authoritative, but this config implies {total_actors} actors. Use 1 actor "
            f"or selection_mode='random'."
        )
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
        self.scale_reward = float(getattr(self.args, "reward_scale", 2.0))
        self.understanding_reward_scale = float(getattr(self.args, "understanding_reward_scale", 1.0))
        self.is_instruct = bool(getattr(self.args, "is_instruct", False))
        self.dataset_name = str(getattr(self.args, "dataset_name", "race-c"))
        self.use_understanding_passage = bool(getattr(self.args, "use_understanding_passage", True))
        self.qa_direct_answer = bool(getattr(self.args, "qa_direct_answer", True))
        self.use_full_understanding_output = bool(getattr(self.args, "use_full_understanding_output", False))
        self.difficulty_weighting = bool(getattr(self.args, "difficulty_weighting", True))
        self.baseline_with_passage = bool(getattr(self.args, "baseline_with_passage", True))
        self.baseline_num_samples = max(1, int(getattr(self.args, "baseline_num_samples", 4)))
        self.min_question_weight = float(getattr(self.args, "min_question_weight", 0.0))
        self.selection_mode = str(getattr(self.args, "selection_mode", "rotate")).lower()

        base_seed = int(getattr(self.args, "seed", 0))
        self.rng = np.random.default_rng(base_seed + int(actor_id))
        self._rotate_pos: Dict[str, int] = {}

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

    def _extract_understanding(self, text: str) -> str:
        if self.use_full_understanding_output:
            return text.strip()
        m = re.search(r"<understanding>\s*(.*?)\s*</understanding>", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"<understanding>\s*(.*?)\s*$", text, re.DOTALL | re.IGNORECASE)
        if m2:
            return m2.group(1).strip()
        return ""

    def _rotate_pick(self, passage: str, num_q: int) -> int:
        # Deterministic per-passage shuffled order (seeded by passage hash), advanced
        # one question each time the passage recurs (== once per epoch).
        order = list(range(num_q))
        random.Random(hash(passage) & 0xFFFFFFFF).shuffle(order)
        pos = self._rotate_pos.get(passage, 0)
        self._rotate_pos[passage] = pos + 1
        return order[pos % num_q]

    def _gen_reasoning(self, prompts):
        outs = self.generate(prompts, self._build_sampling_params(
            temperature=self.sampling_params.temperature, max_tokens=self.reasoning_max_tokens, with_logprobs=True))
        data = []
        for o in outs:
            pid, text, tok, lp_, trunc = self._extract_output(o)
            data.append({"prompt_ids": pid, "text": text, "token_ids": tok, "logprobs": lp_, "is_truncated": trunc})
        return data

    # ---- main rollout --------------------------------------------------------
    @fail_fast("SImpLActor.step")
    def step(self, prompts, formatted_prompts, references=None) -> List[TransitionData]:
        del formatted_prompts
        assert not self.eval_mode
        t0 = time.time()
        if not prompts:
            return self.ipc_client.serialize_ipc([])
        if references is None:
            references = [None] * len(prompts)

        doc_qs: List[List[Dict]] = []
        for ref in references:
            try:
                parsed = json.loads(ref) if ref else {"questions": []}
            except json.JSONDecodeError:
                parsed = {"questions": []}
            doc_qs.append(parsed.get("questions", []))

        # ---- Phase A: understanding rollouts (N per passage) ----
        u_prompts: List[str] = []
        u_doc: List[int] = []
        for d, qs in enumerate(doc_qs):
            if not qs:
                continue
            pt = understanding_prompt(prompts[d], self.dataset_name,
                                      tagged=not self.use_full_understanding_output, answer_ready=self.qa_direct_answer)
            for _ in range(self.reasoning_num_samples):
                u_prompts.append(maybe_apply_chat_template(self.tokenizer, pt, self.is_instruct))
                u_doc.append(d)
        u_data = self._gen_reasoning(u_prompts) if u_prompts else []

        # ---- Phase B: understanding-conditioned QA on ALL questions ----
        eval_prompts: List[str] = []
        eval_meta: List[Dict] = []
        for i, d in enumerate(u_doc):
            understanding = self._extract_understanding(u_data[i]["text"])
            if not understanding:
                continue
            article = prompts[d] if self.use_understanding_passage else ""
            for qi, q in enumerate(doc_qs[d]):
                eval_prompts.append(maybe_apply_chat_template(self.tokenizer,
                    qa_eval_understanding_only_prompt(article, understanding, q.get("question", ""), q.get("options", []), self.dataset_name, direct=self.qa_direct_answer),
                    self.is_instruct))
                eval_meta.append({"u_idx": i, "q_index": qi, "gold": q.get("answer", ""), "num_opt": len(q.get("options", []))})
        eval_by_u_q: Dict[int, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        if eval_prompts:
            eo = self.generate(eval_prompts, self._build_sampling_params(
                temperature=self.sampling_params.temperature if self.qa_num_samples > 1 else 0.0,
                max_tokens=self.qa_eval_max_tokens, with_logprobs=False, n=self.qa_num_samples))
            for out, m in zip(eo, eval_meta):
                for comp in out.outputs:
                    pred = extract_boxed_letter(comp.text or "", m["num_opt"])
                    eval_by_u_q[m["u_idx"]][m["q_index"]].append({
                        "is_correct": int(pred == m["gold"]),
                        "num_toks": len(comp.token_ids) if comp.token_ids else 0,
                    })

        # ---- Phase B2: direct passage-only baseline (difficulty weight) ----
        base_acc: Dict[tuple, float] = {}
        if self.difficulty_weighting:
            bprompts: List[str] = []
            bmeta: List[Dict] = []
            for d, qs in enumerate(doc_qs):
                for qi, q in enumerate(qs):
                    article = prompts[d] if self.baseline_with_passage else ""
                    bprompts.append(maybe_apply_chat_template(self.tokenizer,
                        qa_eval_understanding_only_prompt(article, "", q.get("question", ""), q.get("options", []), self.dataset_name, direct=self.qa_direct_answer),
                        self.is_instruct))
                    bmeta.append({"doc": d, "qi": qi, "gold": q.get("answer", ""), "num_opt": len(q.get("options", []))})
            if bprompts:
                bo = self.generate(bprompts, self._build_sampling_params(
                    temperature=self.sampling_params.temperature, max_tokens=self.qa_eval_max_tokens,
                    with_logprobs=False, n=self.baseline_num_samples))
                for out, m in zip(bo, bmeta):
                    cor = tot = 0
                    for comp in out.outputs:
                        cor += int(extract_boxed_letter(comp.text or "", m["num_opt"]) == m["gold"])
                        tot += 1
                    base_acc[(m["doc"], m["qi"])] = cor / max(tot, 1)

        # ---- Phase Select: ONE cot question per passage (rotate or random) ----
        sel: Dict[int, int] = {}
        for d, qs in enumerate(doc_qs):
            if not qs:
                continue
            if self.selection_mode == "random":
                sel[d] = int(self.rng.integers(0, len(qs)))
            else:
                sel[d] = self._rotate_pick(prompts[d], len(qs))

        # ---- Phase A2: cot rollouts on the selected question (N per passage) ----
        c_prompts: List[str] = []
        c_doc: List[int] = []
        c_gold: List[str] = []
        c_nopt: List[int] = []
        for d, qs in enumerate(doc_qs):
            if not qs:
                continue
            q = qs[sel[d]]
            pt = qa_cot_prompt(prompts[d], q["question"], q["options"], self.dataset_name)
            for _ in range(self.reasoning_num_samples):
                c_prompts.append(maybe_apply_chat_template(self.tokenizer, pt, self.is_instruct))
                c_doc.append(d)
                c_gold.append(q.get("answer", ""))
                c_nopt.append(len(q.get("options", [])))
        c_data = self._gen_reasoning(c_prompts) if c_prompts else []

        # ---- understanding rewards (difficulty-weighted marginal) ----
        u_reward = [0.0] * len(u_doc)
        und_rewards = []
        for i, d in enumerate(u_doc):
            per_q = eval_by_u_q.get(i, {})
            if not per_q:
                u_reward[i] = -0.1 * self.scale_reward
                und_rewards.append(u_reward[i])
                continue
            num = den = 0.0
            correct_toks = []
            for qi, recs in per_q.items():
                qacc = sum(x["is_correct"] for x in recs) / max(len(recs), 1)
                w = max(self.min_question_weight, 1.0 - base_acc.get((d, qi), 0.0)) if self.difficulty_weighting else 1.0
                num += w * qacc
                den += w
                correct_toks.extend(x["num_toks"] for x in recs if x["is_correct"])
            wacc = (num / den) if den > 1e-8 else 0.0
            br = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * wacc
            if correct_toks:
                br += sum(self.conciseness_penalty_k * (1.0 - nt / self.qa_eval_max_tokens) for nt in correct_toks) / len(correct_toks)
            u_reward[i] = br * self.scale_reward * self.understanding_reward_scale
            und_rewards.append(u_reward[i])

        # ---- cot rewards (correctness on the selected question) ----
        c_reward = [0.0] * len(c_doc)
        cot_correct = cot_valid = 0
        for j in range(len(c_doc)):
            pred = extract_boxed_letter(c_data[j]["text"], c_nopt[j])
            if pred == c_gold[j]:
                r = self.correct_reward
            elif pred in [chr(ord("A") + k) for k in range(c_nopt[j])]:
                r = 0.1
            else:
                r = self.incorrect_reward
            c_reward[j] = r * self.scale_reward
            if not c_data[j]["is_truncated"]:
                cot_valid += 1
                cot_correct += int(pred == c_gold[j])

        info = {
            "actor/num_documents": float(len(prompts)),
            "actor/num_samples": float(self.reasoning_num_samples),
            "actor/cot_accuracy": float(cot_correct / max(cot_valid, 1)),
            "actor/cot_reward_mean": float(np.mean(c_reward)) if c_reward else 0.0,
            "actor/understanding_reward_mean": float(np.mean(und_rewards)) if und_rewards else 0.0,
            "actor/baseline_direct_accuracy": float(np.mean(list(base_acc.values()))) if base_acc else 0.0,
            "actor/step_time": float(time.time() - t0),
        }

        # ---- assemble: per passage, N understanding then N cot (contiguous groups) ----
        u_by_doc: Dict[int, List[int]] = defaultdict(list)
        for i, d in enumerate(u_doc):
            u_by_doc[d].append(i)
        c_by_doc: Dict[int, List[int]] = defaultdict(list)
        for j, d in enumerate(c_doc):
            c_by_doc[d].append(j)

        traj: List[TransitionData] = []
        def _emit(prompt, ext, reward):
            traj.append(TransitionData(
                prompt=prompt, prompt_ids=ext["prompt_ids"], response=ext["text"],
                response_ids=ext["token_ids"], response_logprobs=ext["logprobs"],
                rewards=self._terminal_reward(len(ext["token_ids"]), reward),
                loss_mask=not (self.args.ignore_no_eos and ext["is_truncated"]),
                info=info,
            ))
        for d in sorted(set(u_doc) | set(c_doc)):
            for i in u_by_doc.get(d, []):
                _emit(u_prompts[i], u_data[i], u_reward[i])
            for j in c_by_doc.get(d, []):
                _emit(c_prompts[j], c_data[j], c_reward[j])

        logging.info("SImpL actor: docs=%d transitions=%d (u=%d c=%d) sel=%s",
                     len(prompts), len(traj), len(u_doc), len(c_doc), self.selection_mode)
        return self.ipc_client.serialize_ipc(traj)


class SImpLLearner(PPOLearner):
    @fail_fast("SImpLLearner.run")
    def run(self):
        return super().run()

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
            self.masked_aggregator = functools.partial(masked_sum, constant_normalizer=args.reasoning_max_tokens)

    def compute_monte_carlo_advantages(self, rewards, response_masks):
        del response_masks
        rewards = rewards.sum(-1)
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        if (self.args.critic_type in ["grpo"]) and (not self.args.remove_std_bias):
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.args.num_samples, dim=0)
            advantages = advantages / (std_grouped_rewards + 1e-8)
        return advantages

    def _load_prompt_data(self):
        prompt_data = self.args.prompt_data
        if isinstance(prompt_data, str) and prompt_data and __import__("os").path.isfile(prompt_data):
            ext = __import__("os").path.splitext(prompt_data)[1].lower()
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
        input_key = self.args.input_key if self.args.input_key in train_dataset.column_names else "article"
        output_key = self.args.output_key if self.args.output_key in train_dataset.column_names else "questions"

        def to_passage_rows(batch):
            # One row per passage, carrying all its valid questions (understanding is
            # scored on all of them; cot trains on one selected per step).
            arts, qss = [], []
            for article, qs_json in zip(batch[input_key], batch[output_key]):
                valid = []
                for q in parse_questions(qs_json):
                    opts = q.get("options", [])
                    gold = normalize_gold_letter(q.get("answer", ""), len(opts) if isinstance(opts, list) else 4)
                    if isinstance(opts, list) and len(opts) >= 2 and q.get("question", "") and gold:
                        valid.append({"question": q["question"], "options": opts, "answer": gold})
                if not valid:
                    continue
                arts.append(article)
                qss.append(json.dumps({"questions": valid}))
            return {input_key: arts, output_key: qss}

        train_dataset = train_dataset.map(to_passage_rows, batched=True, remove_columns=train_dataset.column_names)
        max_train = min(int(self.args.max_train), len(train_dataset))
        train_dataset = train_dataset.select(range(max_train)).select_columns([input_key, output_key])

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
    program, local_resources = get_program(args, learner_cls=SImpLLearner, actor_cls=SImpLActor)
    lp.launch(program, launch_type=args.launch_type, local_resources=local_resources, terminal="current_terminal")


if __name__ == "__main__":
    cli_args: SImpLArgs = get_default_args(SImpLArgs)
    run_simpl_oat(cli_args)
