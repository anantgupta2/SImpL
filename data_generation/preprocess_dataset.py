from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.preprocess_data import create_or_load_preprocessed_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize supported QA datasets into SImpL's article/questions JSONL format."
    )
    parser.add_argument(
        "--dataset_name",
        default="race-c",
        help="Dataset alias or HF dataset name. Supported aliases include race-high, race-c, and lsat-ar.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--subset",
        default=None,
        help="HF subset for non-registered datasets. Ignored for registered aliases.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, output_path = create_or_load_preprocessed_data(
        num_samples=args.num_samples,
        split=args.split,
        subset=args.subset,
        seed=args.seed + args.num_samples if args.num_samples is not None else args.seed,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )
    print(f"Dataset: {args.dataset_name}")
    print(f"Records: {len(dataset)}")
    print(f"JSONL: {output_path}")


if __name__ == "__main__":
    main()
