from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable

from src.utils.preprocess_data import create_or_load_preprocessed_data
from src.algorithm.SImpL_oat import SImpLArgs, run_simpl_oat
from src.algorithm.cot_only_oat import CoTOnlyArgs, run_cot_only_oat
from src.algorithm.implications_only_oat import ImplicationsOnlyArgs, run_implications_only_oat
from src.algorithm.understanding_only_oat import UnderstandingOnlyArgs, run_understandings_only_oat


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
    implications_only: bool,
    understanding_only: bool,
) -> tuple[dict[str, Any], str, Callable[[Any], None], str]:
    if cot_only:
        valid_fields = set(CoTOnlyArgs.__dataclass_fields__.keys())
        args_cls_name = "CoTOnlyArgs"
        run_fn = run_cot_only_oat
        run_name = "CoT-only"
    elif implications_only:
        valid_fields = set(ImplicationsOnlyArgs.__dataclass_fields__.keys())
        args_cls_name = "ImplicationsOnlyArgs"
        run_fn = run_implications_only_oat
        run_name = "Implications-only"
    elif understanding_only:
        valid_fields = set(UnderstandingOnlyArgs.__dataclass_fields__.keys())
        args_cls_name = "UnderstandingOnlyArgs"
        run_fn = run_understandings_only_oat
        run_name = "Understanding-only"
    else:
        valid_fields = set(SImpLArgs.__dataclass_fields__.keys())
        args_cls_name = "SImpLArgs"
        run_fn = run_simpl_oat
        run_name = "SImpL"

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
    split = str(preprocess_cfg.get("split", "train"))
    subset = str(preprocess_cfg.get("subset", "high"))
    seed = int(preprocess_cfg.get("seed", 42))
    num_samples_raw = preprocess_cfg.get("num_samples", None)
    num_samples = None if num_samples_raw is None else int(num_samples_raw)

    _, output_path = create_or_load_preprocessed_data(
        num_samples=num_samples,
        split=split,
        subset=subset,
        seed=seed,
        output_dir=output_dir,
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
        "--implications_only",
        action="store_true",
        help="Run implications-only training (no CoT stage)",
    )
    parser.add_argument(
        "--understanding_only",
        action="store_true",
        help="Run understanding-only training",
    )
    parser.add_argument(
        "--wb_run_name",
        "--wandb_run_name",
        dest="wb_run_name",
        default=None,
        help="Override Weights & Biases run name (oat_args.wb_run_name)",
    )
    cli_args, unknown_args = parser.parse_known_args()

    config = _load_config(cli_args.config)
    
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
    implications_only = bool(
        config.get("implications_only", False) or cli_args.implications_only
    )
    understanding_only = bool(
        config.get("understanding_only", False) or cli_args.understanding_only
    )
    
    modes_enabled = sum([cot_only, implications_only, understanding_only])
    if modes_enabled > 1:
        raise ValueError("Choose only one mode: cot_only, implications_only, or understanding_only")

    prepared_data_path = _maybe_prepare_data(config)

    oat_args = config.get("oat_args", {})
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
        implications_only=implications_only,
        understanding_only=understanding_only,
    )
    if args_cls_name == "CoTOnlyArgs":
        run_args = CoTOnlyArgs(**sanitized_oat_args)
    elif args_cls_name == "ImplicationsOnlyArgs":
        run_args = ImplicationsOnlyArgs(**sanitized_oat_args)
    elif args_cls_name == "UnderstandingOnlyArgs":
        run_args = UnderstandingOnlyArgs(**sanitized_oat_args)
    else:
        run_args = SImpLArgs(**sanitized_oat_args)

    print(f"[run_with_config] Launching {run_name} run in-process")
    print(
        f"[run_with_config] mode={run_name}, pretrain={run_args.pretrain}, critic_type={run_args.critic_type}"
    )
    run_fn(run_args)


if __name__ == "__main__":
    main()
