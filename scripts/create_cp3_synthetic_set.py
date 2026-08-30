from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from pathlib import Path

from evaluator.local_evaluator import load_jsonl


SCENARIO_PROPORTIONS = (
    ("buying", 0.40),
    ("browsing", 0.40),
    ("intent_override", 0.15),
    ("boundary", 0.05),
)


def weighted_sample_without_replacement(
    products: list[tuple[str, float]], count: int, seed: int
) -> list[str]:
    """Deterministic PPS sample using exponential-race keys."""
    rng = random.Random(seed)
    keyed = (
        (-math.log(max(rng.random(), 1e-15)) / weight, parent_asin)
        for parent_asin, weight in products
        if weight > 0.0
    )
    return [parent_asin for _, parent_asin in heapq.nsmallest(count, keyed)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a catalog-derived target-disjoint cp3 validation set"
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-set", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    public_targets = {
        str(sample["ground_truth"]["parent_asin"])
        for sample in load_jsonl(args.public_set)
    }
    eligible: list[tuple[str, float]] = []
    with args.catalog.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            if parent_asin in public_targets:
                continue
            features = product.get("features")
            details = product.get("details")
            if not features and not details:
                continue
            try:
                rating_number = max(0.0, float(product.get("rating_number") or 0.0))
            except (TypeError, ValueError):
                rating_number = 0.0
            if rating_number < 5.0:
                continue
            # Direct rating-count PPS approximates the purchase-derived 5-core
            # target prior while retaining non-head products probabilistically.
            eligible.append((parent_asin, rating_number))

    if args.count > len(eligible):
        raise ValueError("requested more samples than eligible catalog products")
    selected = weighted_sample_without_replacement(eligible, args.count, args.seed)
    scenarios: list[str] = []
    remaining = args.count
    for index, (scenario, proportion) in enumerate(SCENARIO_PROPORTIONS):
        scenario_count = remaining if index == len(SCENARIO_PROPORTIONS) - 1 else round(
            args.count * proportion
        )
        scenarios.extend([scenario] * scenario_count)
        remaining -= scenario_count
    rng = random.Random(args.seed + 1)
    rng.shuffle(scenarios)

    rows = [
        {
            "category_bucket": "clothing",
            "difficulty_bucket": "synthetic",
            "ground_truth": {"parent_asin": parent_asin},
            "sample_id": f"cp3_synthetic_{index:04d}",
            "scenario_type": scenarios[index],
            "user_profile": {
                "purchase_frequency": "3-4 prior purchases",
                "average_prior_rating": None,
                "rating_style": "mixed",
                "preference_tags": [],
                "summary": "Synthetic target-disjoint validation profile.",
            },
        }
        for index, parent_asin in enumerate(selected)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "sample_count": len(rows),
        "public_target_overlap": 0,
        "eligible_products": len(eligible),
        "seed": args.seed,
        "scenario_counts": {
            scenario: scenarios.count(scenario) for scenario, _ in SCENARIO_PROPORTIONS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
