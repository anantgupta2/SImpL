"""SImpL-no-bias: SImpL with the difficulty weighting removed.

Identical to simpl_oat.py EXCEPT the understanding reward is a *uniform* average over the
passage's questions instead of a difficulty-weighted one, and the direct passage-only
baseline pass is dropped entirely (that pass only existed to compute the difficulty
weights, so removing it is also the speedup). Concretely, per passage in one step:

  1. generate N *understanding* rollouts from the passage alone;
  2. score each understanding by answering ALL of the passage's questions, and reward it
     with the PLAIN (unweighted) mean accuracy over those questions -- every question
     counts equally (w_q = 1), and there is NO baseline pass;
  3. select ONE question (rotate = round-robin across epochs, or random) and generate N
     *cot* rollouts answering it (rewarded by correctness).

So no_bias differs from full SImpL by exactly one thing: difficulty weighting on vs off.
`full SImpL - no_bias` isolates the difficulty weighting; everything else (per-passage
marginal structure over all questions, rotate selection, scales, beta) is unchanged.
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
class SImpLNoBiasArgs(PPOArgs):
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
    # When True, an understanding whose generation consumed the entire reasoning_max_tokens
    # budget (finish_reason == "length" -> never closed </understanding>, degraded) gets its
    # reward zeroed out. False keeps the old behaviour (default).
    zero_understanding_on_truncation: bool = False

    # Whether to show the passage alongside the understanding when scoring it.
    # Must match the evaluator's --understanding_with_passage flag.
    use_understanding_passage: bool = True
    # Answer DIRECTLY from the understanding (no cot, just the boxed letter) -> forces
    # the understanding to carry the reasoning. Pair with a small qa_eval_max_tokens.
    qa_direct_answer: bool = True
    # Treat the whole understanding generation as the understanding (base models often
    # omit the <understanding> tags, so strict extraction silently drops good ones).
    use_full_understanding_output: bool = False

    # cot question selection per passage (same as SImpL): "rotate" round-robin / "random".
    selection_mode: str = "rotate"
    # If True, train cot on a FLATTENED dataset: one row per (passage, question) with the
    # cot target FIXED to that row's question (no rotate), so cot sees every question as a
    # direct training example -- while the understanding is STILL scored on ALL the
    # passage's questions (uniform). If False (default): per-passage rows + rotate cot.
    flatten_cot: bool = False

    critic_type: Literal["ppo", "grpo", "drgrpo"] = field(default="drgrpo")
    remove_len_bias: bool = False
    remove_std_bias: bool = False

    eval_steps: int = -1
    online_evaluation: bool = True
    apply_chat_template: bool = False
    is_instruct: bool = False
    beta: float = -1.0  # -1 => use default 0.04; set in config to sweep


def configure_simpl_no_bias_args(args: SImpLNoBiasArgs) -> SImpLNoBiasArgs:
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

    # rotate keeps its per-passage round-robin counter in the actor PROCESS, so it is only
    # authoritative with exactly ONE actor (mirror simpl_oat's guard). In flatten_cot mode
    # the cot target is fixed per row, so rotate is unused -> no single-actor requirement.
    if (str(getattr(args, "selection_mode", "rotate")).lower() == "rotate"
            and not bool(getattr(args, "flatten_cot", False))):
        actor_gpus = int(args.gpus) if args.collocate else (
            args.gpus // 2 if args.gpus % 2 == 0 else args.gpus // 2 + 1)
        total_actors = (actor_gpus // int(args.num_gpus_per_actor)) * int(getattr(args, "num_groups", 1))
        assert total_actors == 1, (
            f"selection_mode='rotate' needs a single actor, but this config implies "
            f"{total_actors}. Use 1 actor or selection_mode='random'."
        )
    return args


class SImpLNoBiasActor(PPOActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.reasoning_num_samples = int(self.args.reasoning_num_samples)
        self.reasoning_max_tokens = int(self.args.reasoning_max_tokens)
        self.qa_eval_max_tokens = int(self.args.qa_eval_max_tokens)
        self.qa_num_samples = int(getattr(self.args, "qa_num_samples", 1))
        self.correct_reward = float(self.args.correct_reward)
        self.incorrect_reward = float(self.args.incorrect_reward)
        self.conciseness_penalty_k = float(self.args.conciseness_penalty_k)
        self.zero_understanding_on_truncation = bool(getattr(self.args, "zero_understanding_on_truncation", False))
        self.scale_reward = float(getattr(self.args, "reward_scale", 1.0))
        self.understanding_reward_scale = float(getattr(self.args, "understanding_reward_scale", 1.0))
        self.is_instruct = bool(getattr(self.args, "is_instruct", False))
        self.dataset_name = str(getattr(self.args, "dataset_name", "race-c"))
        self.use_understanding_passage = bool(getattr(self.args, "use_understanding_passage", True))
        self.qa_direct_answer = bool(getattr(self.args, "qa_direct_answer", True))
        self.use_full_understanding_output = bool(getattr(self.args, "use_full_understanding_output", False))
        self.selection_mode = str(getattr(self.args, "selection_mode", "rotate")).lower()
        self.flatten_cot = bool(getattr(self.args, "flatten_cot", False))

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
    @fail_fast("SImpLNoBiasActor.step")
    def step(self, prompts, formatted_prompts, references=None) -> List[TransitionData]:
        del formatted_prompts
        assert not self.eval_mode
        t0 = time.time()
        if not prompts:
            return self.ipc_client.serialize_ipc([])
        if references is None:
            references = [None] * len(prompts)

        # Each row carries ALL the passage's questions (understanding is scored on all of
        # them). In flatten_cot mode the row also carries "cot_index" -> the fixed cot
        # target for that row; otherwise cot_index is None and cot uses rotate/random.
        doc_qs: List[List[Dict]] = []
        cot_idx: List[int] = []
        for ref in references:
            try:
                parsed = json.loads(ref) if ref else {"questions": []}
            except json.JSONDecodeError:
                parsed = {"questions": []}
            doc_qs.append(parsed.get("questions", []))
            ci = parsed.get("cot_index", None)
            cot_idx.append(int(ci) if ci is not None else None)

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
        # NOTE: no difficulty-weighting baseline pass here (that's the whole point of no_bias).

        # ---- Phase Select: ONE cot question per passage (rotate or random) ----
        sel: Dict[int, int] = {}
        for d, qs in enumerate(doc_qs):
            if not qs:
                continue
            if cot_idx[d] is not None:                       # flatten_cot: fixed target
                sel[d] = min(int(cot_idx[d]), len(qs) - 1)
            elif self.selection_mode == "random":
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

        # ---- understanding rewards (UNIFORM marginal over all questions; w_q = 1) ----
        u_reward = [0.0] * len(u_doc)
        und_rewards = []
        n_trunc = 0
        for i in range(len(u_doc)):
            per_q = eval_by_u_q.get(i, {})
            if not per_q:
                u_reward[i] = -0.1 * self.scale_reward
            else:
                accs = []
                correct_toks = []
                for qi, recs in per_q.items():
                    accs.append(sum(x["is_correct"] for x in recs) / max(len(recs), 1))
                    correct_toks.extend(x["num_toks"] for x in recs if x["is_correct"])
                wacc = sum(accs) / max(len(accs), 1)  # plain mean over questions
                br = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * wacc
                if correct_toks:
                    br += sum(self.conciseness_penalty_k * (1.0 - nt / self.qa_eval_max_tokens) for nt in correct_toks) / len(correct_toks)
                u_reward[i] = br * self.scale_reward * self.understanding_reward_scale
            # Truncated understanding (consumed the whole token budget, never closed
            # </understanding>) is degraded -> zero its reward.
            if u_data[i].get("is_truncated"):
                n_trunc += 1
                if self.zero_understanding_on_truncation:
                    u_reward[i] = 0.0
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
            "actor/understanding_trunc_rate": float(n_trunc / max(len(u_doc), 1)),
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

        logging.info("SImpL-no-bias actor: docs=%d transitions=%d (u=%d c=%d) sel=%s",
                     len(prompts), len(traj), len(u_doc), len(c_doc), self.selection_mode)
        return self.ipc_client.serialize_ipc(traj)


class SImpLNoBiasLearner(PPOLearner):
    @fail_fast("SImpLNoBiasLearner.run")
    def run(self):
        return super().run()

    def _init(self, args: SImpLNoBiasArgs, actors: List[ActorBase]) -> None:
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

        flatten_cot = bool(getattr(self.args, "flatten_cot", False))

        # JOINT training (e.g. 25 LSAT + 25 RACE in one file): a per-row "source_dataset" column
        # rides along in the reference blob so the actor can pick the right UNDERSTANDING prompt
        # per example -- LSAT-AR and RACE-C need different ones. Absent column => unchanged
        # behaviour (the actor falls back to self.dataset_name).
        has_src = "source_dataset" in train_dataset.column_names

        def to_rows(batch):
            # Validate questions per passage. Default: ONE row per passage carrying all
            # valid questions. flatten_cot: ONE row per question -- each row still carries
            # ALL the passage's questions (for the understanding marginal) plus "cot_index"
            # = the row's fixed cot target.
            arts, qss = [], []
            srcs = batch["source_dataset"] if has_src else [None] * len(batch[input_key])
            for article, qs_json, src in zip(batch[input_key], batch[output_key], srcs):
                valid = []
                for q in parse_questions(qs_json):
                    opts = q.get("options", [])
                    gold = normalize_gold_letter(q.get("answer", ""), len(opts) if isinstance(opts, list) else 4)
                    if isinstance(opts, list) and len(opts) >= 2 and q.get("question", "") and gold:
                        valid.append({"question": q["question"], "options": opts, "answer": gold})
                if not valid:
                    continue
                extra = {"source_dataset": src} if src else {}
                if flatten_cot:
                    for i in range(len(valid)):
                        arts.append(article)
                        qss.append(json.dumps({"questions": valid, "cot_index": i, **extra}))
                else:
                    arts.append(article)
                    qss.append(json.dumps({"questions": valid, **extra}))
            return {input_key: arts, output_key: qss}

        train_dataset = train_dataset.map(to_rows, batched=True, remove_columns=train_dataset.column_names)
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


def run_simpl_no_bias_oat(args: SImpLNoBiasArgs):
    args = configure_simpl_no_bias_args(args)
    args = default_args_validation(args)
    program, local_resources = get_program(args, learner_cls=SImpLNoBiasLearner, actor_cls=SImpLNoBiasActor)
    lp.launch(program, launch_type=args.launch_type, local_resources=local_resources, terminal="current_terminal")


if __name__ == "__main__":
    cli_args: SImpLNoBiasArgs = get_default_args(SImpLNoBiasArgs)
    run_simpl_no_bias_oat(cli_args)
