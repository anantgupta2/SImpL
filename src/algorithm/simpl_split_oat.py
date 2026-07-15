"""SImpL-split: simpl_no_bias with a decoupled understanding/cot rollout budget.

Same objective as simpl_no_bias (uniform per-passage understanding marginal + cot on a selected
question, flatten_cot etc. all inherited unchanged). The ONLY additions:

  * `num_understanding_rollouts` (n_u) and `num_cot_rollouts` (n_c) -- the number of understanding
    vs cot rollouts generated per passage, independently. Both default to `reasoning_num_samples`,
    so leaving them unset reproduces simpl_no_bias EXACTLY (8/8). Set e.g. n_u=4, n_c=12.
  * `understanding_every_k` -- only run the understanding role on rounds where round % k == 0
    (k=1 => every round, the default/current behaviour). On the skipped rounds the step is pure
    cot (no understanding rollouts, no understanding reward).

CORRECTNESS -- advantage grouping (the part that must stay right):
  Dr.GRPO/GRPO gives each rollout an advantage = its reward minus the MEAN reward of its *group*,
  where a group = the rollouts sharing one prompt. Here, per passage, the understanding samples
  are one group and the cot samples are another. oat's stock implementation assumes every
  contiguous block of `num_samples` rows is a group (`rewards.view(-1, num_samples)`), which is
  only valid when n_u == n_c. To support unequal splits (and understanding-off rounds) we tag each
  emitted transition with `info["grp_start"]` (1.0 on the FIRST row of each group, else 0.0) and
  the learner rebuilds per-group baselines from those flags. This reduces EXACTLY to the stock
  grouping when n_u == n_c (verified). We pin `num_samples = 1` so oat's divisibility-based metric
  lines (`final_rewards.view(-1, num_samples)`) can never assume a fixed group size and crash.

  The learner reads the flags from `self.pi_buffer`, which is aligned 1:1 and in-order with the
  `rewards` tensor: the GRPO dataloader does not shuffle and loads the whole buffer as one batch,
  and TransitionDataset maps buffer[i] -> row i. (info is dropped from the tensor but pi_buffer[i]
  stays readable.)
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import TransitionData

from src.algorithm.simpl_no_bias_oat import (
    SImpLNoBiasArgs,
    SImpLNoBiasActor,
    SImpLNoBiasLearner,
    configure_simpl_no_bias_args,
)
from src.utils.oat_prompt_templates import (
    understanding_prompt,
    maybe_apply_chat_template,
    qa_cot_prompt,
    qa_eval_understanding_only_prompt,
)
from src.utils.parsing_utils import extract_boxed_letter
from src.utils.fail_fast import fail_fast


@dataclass
class SImpLSplitArgs(SImpLNoBiasArgs):
    # Independent rollout budgets. -1 => fall back to reasoning_num_samples (=> reproduces 8/8).
    num_understanding_rollouts: int = -1
    num_cot_rollouts: int = -1
    # Run the understanding role only on rounds where (round % k == 0). 1 => every round.
    understanding_every_k: int = 1


def configure_simpl_split_args(args: SImpLSplitArgs) -> SImpLSplitArgs:
    # Reuse all of no_bias's setup (beta default, rotate single-actor guard, fused head, etc.).
    args = configure_simpl_no_bias_args(args)

    n_u = int(args.num_understanding_rollouts)
    n_c = int(args.num_cot_rollouts)
    # -1 (the default) means "unset -> fall back to reasoning_num_samples". An EXPLICIT 0 is a
    # valid, honored value: it turns that role off entirely. So ANY split is allowed -- pure cot
    # (n_u=0), pure understanding (n_c=0, note: no cot training, cot_accuracy still measured at
    # eval), or any mix -- as long as at least one rollout happens per passage.
    if n_u < 0:
        n_u = int(args.reasoning_num_samples)
    if n_c < 0:
        n_c = int(args.reasoning_num_samples)
    if n_u < 0 or n_c < 0 or (n_u + n_c) < 1:
        raise ValueError("num_understanding_rollouts/num_cot_rollouts must be >=0 and sum to >=1")
    if int(args.understanding_every_k) < 1:
        raise ValueError("understanding_every_k must be >= 1")
    args.num_understanding_rollouts = n_u
    args.num_cot_rollouts = n_c

    # Advantage grouping is done from info["grp_start"], NOT from num_samples. Pin it to 1 so the
    # base metric lines `final_rewards.view(-1, num_samples)` are always divisible (never crash).
    args.num_samples = 1
    # Size the per-device rollout buffer to hold a full round: n_u + n_c rows per passage.
    need = int(args.rollout_batch_size_per_device) * (n_u + n_c)
    if args.pi_buffer_maxlen_per_device < need:
        args.pi_buffer_maxlen_per_device = need
    return args


class SImpLSplitActor(SImpLNoBiasActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.n_und = int(getattr(self.args, "num_understanding_rollouts"))
        self.n_cot = int(getattr(self.args, "num_cot_rollouts"))
        self.understanding_every_k = int(getattr(self.args, "understanding_every_k", 1))
        self._round = 0  # rollout-round counter (this actor process)

    @fail_fast("SImpLSplitActor.step")
    def step(self, prompts, formatted_prompts, references=None) -> List[TransitionData]:
        del formatted_prompts
        assert not self.eval_mode
        t0 = time.time()
        if not prompts:
            return self.ipc_client.serialize_ipc([])
        if references is None:
            references = [None] * len(prompts)

        do_und = (self._round % self.understanding_every_k == 0)
        self._round += 1

        # Parse per-passage questions (+ optional flatten_cot fixed target).
        doc_qs: List[List[Dict]] = []
        cot_idx: List[int] = []
        # Per-example source dataset for JOINT training (25 LSAT + 25 RACE in one file): LSAT-AR
        # and RACE-C need different UNDERSTANDING prompts, so a single self.dataset_name would
        # give half the passages the wrong one. None => fall back to self.dataset_name.
        doc_src: List[str] = []
        for ref in references:
            try:
                parsed = json.loads(ref) if ref else {"questions": []}
            except json.JSONDecodeError:
                parsed = {"questions": []}
            doc_qs.append(parsed.get("questions", []))
            ci = parsed.get("cot_index", None)
            cot_idx.append(int(ci) if ci is not None else None)
            doc_src.append(parsed.get("source_dataset") or self.dataset_name)

        # ---- Phase A: understanding rollouts (n_und per passage) -- skipped on non-und rounds ----
        u_prompts: List[str] = []
        u_doc: List[int] = []
        if do_und:
            for d, qs in enumerate(doc_qs):
                if not qs:
                    continue
                pt = understanding_prompt(prompts[d], doc_src[d],
                                          tagged=not self.use_full_understanding_output, answer_ready=self.qa_direct_answer)
                for _ in range(self.n_und):
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
                    qa_eval_understanding_only_prompt(article, understanding, q.get("question", ""), q.get("options", []), doc_src[d], direct=self.qa_direct_answer),
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

        # ---- Phase Select: ONE cot question per passage (rotate or random) ----
        sel: Dict[int, int] = {}
        for d, qs in enumerate(doc_qs):
            if not qs:
                continue
            if cot_idx[d] is not None:
                sel[d] = min(int(cot_idx[d]), len(qs) - 1)
            elif self.selection_mode == "random":
                sel[d] = int(self.rng.integers(0, len(qs)))
            else:
                sel[d] = self._rotate_pick(prompts[d], len(qs))

        # ---- Phase A2: cot rollouts on the selected question (n_cot per passage) ----
        c_prompts: List[str] = []
        c_doc: List[int] = []
        c_gold: List[str] = []
        c_nopt: List[int] = []
        for d, qs in enumerate(doc_qs):
            if not qs:
                continue
            q = qs[sel[d]]
            pt = qa_cot_prompt(prompts[d], q["question"], q["options"], doc_src[d])
            for _ in range(self.n_cot):
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
                wacc = sum(accs) / max(len(accs), 1)
                br = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * wacc
                if correct_toks:
                    br += sum(self.conciseness_penalty_k * (1.0 - nt / self.qa_eval_max_tokens) for nt in correct_toks) / len(correct_toks)
                u_reward[i] = br * self.scale_reward * self.understanding_reward_scale
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
            "actor/num_samples": float(self.n_cot),
            "actor/n_understanding_rollouts": float(self.n_und),
            "actor/n_cot_rollouts": float(self.n_cot),
            "actor/do_understanding": float(1.0 if do_und else 0.0),
            "actor/cot_accuracy": float(cot_correct / max(cot_valid, 1)),
            "actor/cot_reward_mean": float(np.mean(c_reward)) if c_reward else 0.0,
            "actor/understanding_reward_mean": float(np.mean(und_rewards)) if und_rewards else 0.0,
            "actor/understanding_trunc_rate": float(n_trunc / max(len(u_doc), 1)),
            "actor/step_time": float(time.time() - t0),
        }

        # ---- assemble: per passage, [n_und understanding][n_cot cot]; flag each group's 1st row ----
        u_by_doc: Dict[int, List[int]] = defaultdict(list)
        for i, d in enumerate(u_doc):
            u_by_doc[d].append(i)
        c_by_doc: Dict[int, List[int]] = defaultdict(list)
        for j, d in enumerate(c_doc):
            c_by_doc[d].append(j)

        traj: List[TransitionData] = []
        def _emit(prompt, ext, reward, grp_start):
            tinfo = dict(info)
            tinfo["grp_start"] = 1.0 if grp_start else 0.0
            traj.append(TransitionData(
                prompt=prompt, prompt_ids=ext["prompt_ids"], response=ext["text"],
                response_ids=ext["token_ids"], response_logprobs=ext["logprobs"],
                rewards=self._terminal_reward(len(ext["token_ids"]), reward),
                loss_mask=not (self.args.ignore_no_eos and ext["is_truncated"]),
                info=tinfo,
            ))
        for d in sorted(set(u_doc) | set(c_doc)):
            for k, i in enumerate(u_by_doc.get(d, [])):
                _emit(u_prompts[i], u_data[i], u_reward[i], grp_start=(k == 0))
            for k, j in enumerate(c_by_doc.get(d, [])):
                _emit(c_prompts[j], c_data[j], c_reward[j], grp_start=(k == 0))

        logging.info("SImpL-split actor: docs=%d transitions=%d (u=%d c=%d) do_und=%s round=%d n_u=%d n_c=%d",
                     len(prompts), len(traj), len(u_doc), len(c_doc), do_und, self._round, self.n_und, self.n_cot)
        return self.ipc_client.serialize_ipc(traj)


class SImpLSplitLearner(SImpLNoBiasLearner):
    def compute_monte_carlo_advantages(self, rewards, response_masks):
        """Per-GROUP baseline advantages, where groups are delimited by info['grp_start'] flags in
        self.pi_buffer (aligned 1:1 with `rewards`). Reduces exactly to base GRPO when n_u == n_c.
        drgrpo: no std normalization; grpo (unless remove_std_bias): divide by the group's std."""
        r = rewards.sum(-1)  # [T]
        T = int(r.shape[0])
        starts = None
        if len(self.pi_buffer) == T:
            try:
                starts = [float(t.info.get("grp_start", -1.0)) for t in self.pi_buffer]
            except Exception:
                starts = None
        if (starts is None) or any(s < 0.0 for s in starts) or (starts and starts[0] < 0.5):
            # grp_start must be present and the first row must open a group. Fail loud rather than
            # silently mis-group (num_samples is pinned to 1, so the base fallback would zero out).
            raise RuntimeError(
                f"SImpLSplitLearner: cannot recover group boundaries (len(pi_buffer)={len(self.pi_buffer)}, "
                f"T={T}, first_start={starts[0] if starts else None}). Advantage grouping aborted.")

        gids = np.cumsum([1 if s >= 0.5 else 0 for s in starts]) - 1  # contiguous 0-indexed group ids
        gids_t = torch.as_tensor(gids, device=r.device, dtype=torch.long)
        adv = torch.zeros_like(r)
        use_std = (self.args.critic_type == "grpo") and (not self.args.remove_std_bias)
        for g in range(int(gids_t.max().item()) + 1):
            mask = gids_t == g
            vals = r[mask]
            a = vals - vals.mean()
            if use_std:
                a = a / (vals.std() + 1e-8)
            adv[mask] = a
        return adv


def run_simpl_split_oat(args: SImpLSplitArgs):
    args = configure_simpl_split_args(args)
    args = default_args_validation(args)
    program, local_resources = get_program(args, learner_cls=SImpLSplitLearner, actor_cls=SImpLSplitActor)
    lp.launch(program, launch_type=args.launch_type, local_resources=local_resources, terminal="current_terminal")


if __name__ == "__main__":
    cli_args: SImpLSplitArgs = get_default_args(SImpLSplitArgs)
    run_simpl_split_oat(cli_args)
