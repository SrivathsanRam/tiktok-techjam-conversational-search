"""Generate the frozen CP4 synthetic session sets in one deterministic run.

Produces 5000 target-disjoint sessions with the official scenario mix and
splits them into synthetic_train.jsonl (3000), synthetic_dev.jsonl (1000),
and synthetic_holdout.jsonl (1000, aggregate-only reporting). Targets are
disjoint from every public target and from each other, so the three splits
never share a product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from evaluator.local_evaluator import load_jsonl
from scripts.create_cp3_synthetic_set import (
    SCENARIO_PROPORTIONS,
    weighted_sample_without_replacement,
)


SPLITS = (("train", 3000), ("dev", 1000), ("holdout", 1000))


def scenario_assignment(count: int, seed: int) -> list[str]:
    scenarios: list[str] = []
    remaining = count
    for index, (scenario, proportion) in enumerate(SCENARIO_PROPORTIONS):
        scenario_count = remaining if index == len(SCENARIO_PROPORTIONS) - 1 else round(
            count * proportion
        )
        scenarios.extend([scenario] * scenario_count)
        remaining -= scenario_count
    random.Random(seed).shuffle(scenarios)
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-set", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/releases/cp4"))
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    total = sum(count for _, count in SPLITS)
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
            if not product.get("features") and not product.get("details"):
                continue
            try:
                rating_number = max(0.0, float(product.get("rating_number") or 0.0))
            except (TypeError, ValueError):
                rating_number = 0.0
            if rating_number < 5.0:
                continue
            eligible.append((parent_asin, rating_number))

    if total > len(eligible):
        raise ValueError("requested more samples than eligible catalog products")
    selected = weighted_sample_without_replacement(eligible, total, args.seed)
    # nsmallest returns a weight-ordered list; shuffle before slicing so the
    # train/dev/holdout splits are exchangeable rather than popularity-tiered.
    random.Random(args.seed + 2).shuffle(selected)
    scenarios = scenario_assignment(total, args.seed + 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    offset = 0
    for split_name, count in SPLITS:
        rows = [
            {
                "category_bucket": "clothing",
                "difficulty_bucket": "synthetic",
                "ground_truth": {"parent_asin": selected[offset + index]},
                "sample_id": f"cp4_{split_name}_{index:04d}",
                "scenario_type": scenarios[offset + index],
                "user_profile": {
                    "purchase_frequency": "3-4 prior purchases",
                    "average_prior_rating": None,
                    "rating_style": "mixed",
                    "preference_tags": [],
                    "summary": "Synthetic target-disjoint validation profile.",
                },
            }
            for index in range(count)
        ]
        offset += count
        path = args.output_dir / f"synthetic_{split_name}.jsonl"
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        path.write_text(payload, encoding="utf-8")
        summary[split_name] = {
            "path": str(path),
            "sample_count": len(rows),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "scenario_counts": {
                scenario: sum(1 for row in rows if row["scenario_type"] == scenario)
                for scenario, _ in SCENARIO_PROPORTIONS
            },
        }

    print(json.dumps({
        "seed": args.seed,
        "eligible_products": len(eligible),
        "public_target_overlap": 0,
        "splits": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
