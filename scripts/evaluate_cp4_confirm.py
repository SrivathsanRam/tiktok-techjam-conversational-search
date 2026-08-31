"""One-shot confirmation runs on the untouched holdout partitions.

Replays the real evaluator loop and prints aggregate metrics only; holdout
sessions are never inspected individually (aggregate-only protocol).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/cp2_split.json"))
    parser.add_argument(
        "--public-holdout", action="store_true",
        help="restrict --dataset to the manifest holdout_ids",
    )
    parser.add_argument(
        "--fresh-dominance", action="store_true",
        help="enable dominance ordering on fresh-disclosure turns",
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.public_holdout:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        holdout_ids = {str(value) for value in manifest["holdout_ids"]}
        samples = [
            sample for sample in samples
            if str(sample["sample_id"]) in holdout_ids
        ]
        if len(samples) != len(holdout_ids):
            raise ValueError("holdout manifest and dataset do not match")

    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog, use_fresh_tier_dominance=args.fresh_dominance)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    print(json.dumps(
        {key: value for key, value in result.items() if key != "sessions"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
