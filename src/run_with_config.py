from __future__ import annotations

import argparse
import json
import os
import random
import numpy as np
import torch
from typing import Any, Callable

from src.utils.preprocess_data import create_or_load_preprocessed_data
from src.algorithm.simpl_no_bias_oat import SImpLNoBiasArgs, run_simpl_no_bias_oat
from src.algorithm.simpl_split_oat import SImpLSplitArgs, run_simpl_split_oat
from src.algorithm.cot_only_oat import CoTOnlyArgs, run_cot_only_oat


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a JSON object")
    return cfg


def _sanitize_oat_args(
    oat_args: dict[str, Any],
    *,
    cot_only: bool,
    simpl_no_bias: bool,
    simpl_split: bool = False,
) -> tuple[dict[str, Any], str, Callable[[Any], None], str]:
    if cot_only:
        valid_fields = set(CoTOnlyArgs.__dataclass_fields__.keys())
        args_cls_name = "CoTOnlyArgs"
        run_fn = run_cot_only_oat
        run_name = "CoT-only"
    elif simpl_split:
        valid_fields = set(SImpLSplitArgs.__dataclass_fields__.keys())
        args_cls_name = "SImpLSplitArgs"
        run_fn = run_simpl_split_oat
        run_name = "SImpL-split"
    else:  # simpl_no_bias
        valid_fields = set(SImpLNoBiasArgs.__dataclass_fields__.keys())
        args_cls_name = "SImpLNoBiasArgs"
        run_fn = run_simpl_no_bias_oat
        run_name = "SImpL-no-bias"
    filtered = {k: v for k, v in oat_args.items() if k in valid_fields}
    dropped = sorted([k for k in oat_args.keys() if k not in valid_fields])
    if dropped:
        print(
            f"[run_with_config] Dropping unsupported fields for {args_cls_name}: {dropped}"
        )

    filtered.setdefault("critic_type", "drgrpo")
    return filtered, args_cls_name, run_fn, run_name


def _maybe_prepare_data(config: dict[str, Any]) -> str | None:
    preprocess_cfg = config.get("preprocess", {})
    if not isinstance(preprocess_cfg, dict):
        raise ValueError("preprocess must be a JSON object when provided")

    enabled = bool(preprocess_cfg.get("enabled", False))
    if not enabled:
        return None

    output_dir = str(preprocess_cfg.get("output_dir", "data"))
    dataset_name = str(preprocess_cfg.get("dataset_name", "race-c"))
    split = str(preprocess_cfg.get("split", "train"))
    subset_raw = preprocess_cfg.get("subset", None)
    subset = None if subset_raw is None else str(subset_raw)
    seed = int(preprocess_cfg.get("seed", 42))
    num_samples_raw = preprocess_cfg.get("num_samples", None)
    num_samples = None if num_samples_raw is None else int(num_samples_raw)

    _, output_path = create_or_load_preprocessed_data(
        num_samples=num_samples,
        split=split,
        subset=subset,
        seed=seed,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )
    return os.path.abspath(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SImpL/OAT variants from a JSON config")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument(
        "--cot_only",
        action="store_true",
        help="Run CoT-only training (no implication stage)",
    )
    parser.add_argument(
        "--simpl_no_bias",
        action="store_true",
        help="Run SImpL-no-bias (flattened data, plain single-question understanding "
             "reward; no difficulty weighting / baseline pass / rotate)",
    )
    parser.add_argument(
        "--simpl_split",
        action="store_true",
        help="Run SImpL-split (simpl_no_bias with decoupled num_understanding_rollouts / "
             "num_cot_rollouts and optional understanding_every_k)",
    )
    parser.add_argument(
        "--wb_run_name",
        "--wandb_run_name",
        dest="wb_run_name",
        default=None,
        help="Override Weights & Biases run name (oat_args.wb_run_name)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    cli_args, unknown_args = parser.parse_known_args()

    set_seed(cli_args.seed)

    config = _load_config(cli_args.config)
    
    if "oat_args" not in config:
        config["oat_args"] = {}
    config["oat_args"]["seed"] = cli_args.seed

    dataset_name = config.get("preprocess", {}).get("dataset_name", "race-c")
    config["oat_args"]["save_path"] = config.get("oat_args", {}).get("save_path", "oat-output") + f"/{dataset_name}"
    # Propagate the dataset name so actors can select dataset-specific prompts
    # (e.g. the understanding prompt). An explicit oat_args.dataset_name wins.
    config["oat_args"].setdefault("dataset_name", dataset_name)
    # Process unknown arguments as overrides for config
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]
            val = True
            if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith("--"):
                val_str = unknown_args[i+1]
                if val_str.lower() == 'true':
                    val = True
                elif val_str.lower() == 'false':
                    val = False
                else:
                    try:
                        val = int(val_str)
                    except ValueError:
                        try:
                            val = float(val_str)
                        except ValueError:
                            val = val_str
                i += 2
            else:
                i += 1
            
            # Apply to preprocess if key exists there, otherwise default to oat_args
            if "preprocess" in config and key in config["preprocess"]:
                config["preprocess"][key] = val
                print(f"[run_with_config] Overriding config.preprocess.{key} = {val}")
            else:
                if "oat_args" not in config:
                    config["oat_args"] = {}
                config["oat_args"][key] = val
                print(f"[run_with_config] Overriding config.oat_args.{key} = {val}")
        else:
            i += 1

    cot_only = bool(config.get("cot_only", False) or cli_args.cot_only)
    simpl_no_bias = bool(config.get("simpl_no_bias", False) or cli_args.simpl_no_bias)
    simpl_split = bool(config.get("simpl_split", False) or cli_args.simpl_split)

    modes_enabled = sum([cot_only, simpl_no_bias, simpl_split])
    if modes_enabled > 1:
        raise ValueError("Choose only one mode: cot_only, simpl_no_bias, or simpl_split")
    if modes_enabled == 0:
        raise ValueError("Choose a mode: --cot_only, --simpl_no_bias, or --simpl_split")

    prepared_data_path = _maybe_prepare_data(config)

    oat_args = config.get("oat_args", {})
    # oat_args["train_batch_size"] //= 2  # Reduce batch size by half to save memory, since we're doing in-process training
    if not isinstance(oat_args, dict):
        raise ValueError("oat_args must be a JSON object")

    if cli_args.wb_run_name is not None:
        oat_args["wb_run_name"] += "_" + str(cli_args.wb_run_name)
        print(f"[run_with_config] Overriding wb_run_name from CLI: {oat_args['wb_run_name']}")

    if prepared_data_path and not oat_args.get("prompt_data"):
        oat_args["prompt_data"] = prepared_data_path

    prompt_data = oat_args.get("prompt_data")
    if isinstance(prompt_data, str) and os.path.exists(prompt_data):
        oat_args["prompt_data"] = os.path.abspath(prompt_data)

    if not oat_args.get("prompt_data"):
        raise ValueError("prompt_data is required (either in oat_args or via preprocess.enabled)")

    sanitized_oat_args, args_cls_name, run_fn, run_name = _sanitize_oat_args(
        oat_args,
        cot_only=cot_only,
        simpl_no_bias=simpl_no_bias,
        simpl_split=simpl_split,
    )
    if args_cls_name == "CoTOnlyArgs":
        run_args = CoTOnlyArgs(**sanitized_oat_args)
    elif args_cls_name == "SImpLSplitArgs":
        run_args = SImpLSplitArgs(**sanitized_oat_args)
    else:
        run_args = SImpLNoBiasArgs(**sanitized_oat_args)

    print(f"[run_with_config] Launching {run_name} run in-process")
    print(
        f"[run_with_config] mode={run_name}, pretrain={run_args.pretrain}, critic_type={run_args.critic_type}"
    )
    run_fn(run_args)


if __name__ == "__main__":
    main()
