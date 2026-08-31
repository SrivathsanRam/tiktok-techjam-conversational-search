"""Build the CP4 hard-case session set.

Selects catalog targets that are hard for the exact-evidence funnel:

- a hard constraint that is a common attribute (``cotton`` or ``imported``),
- more than 100 neighbors satisfying every hard constraint simultaneously,
- plus at least two of: a long feature/detail string (>= 100 characters),
  a missing description, and low popularity (rating_number <= 50).

Targets are disjoint from every public target. The scenario mix follows the
official 40/40/15/5 proportions. The evaluator derives intent cards from the
products at run time, so rows carry only public fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from evaluator.local_evaluator import intent_card, load_jsonl
from scripts.create_cp3_synthetic_set import SCENARIO_PROPORTIONS
from starter.agent import Agent, _normalized_value


COMMON_ATTRIBUTES = {"cotton", "imported"}


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
    parser.add_argument(
        "--output", type=Path, default=Path("data/releases/cp4/hard_cases.jsonl")
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--neighbor-threshold", type=int, default=100)
    args = parser.parse_args()

    public_targets = {
        str(sample["ground_truth"]["parent_asin"])
        for sample in load_jsonl(args.public_set)
    }
    agent = Agent(args.catalog)

    def neighbor_count(constraints: list[str]) -> int:
        postings = [
            set(agent._evidence_postings(constraint)) for constraint in constraints
        ]
        postings = [p for p in postings if p]
        if not postings:
            return 0
        joint = set.intersection(*postings)
        return len(joint)

    eligible: list[str] = []
    criteria_counts = {"common_attribute": 0, "neighbors": 0, "long_string": 0,
                       "missing_description": 0, "low_popularity": 0}
    with args.catalog.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            if parent_asin in public_targets:
                continue
            card = intent_card(product)
            hard = [str(value) for value in card["hard_constraints"]]
            if not hard:
                continue
            normalized_hard = {_normalized_value(value) for value in hard}
            common_attribute = bool(normalized_hard & COMMON_ATTRIBUTES)
            neighbors = neighbor_count(hard) > args.neighbor_threshold
            raw_values: list[str] = []
            for field in (product.get("features"), product.get("details")):
                if isinstance(field, dict):
                    raw_values.extend(f"{k}: {v}" for k, v in field.items())
                elif isinstance(field, list):
                    raw_values.extend(str(v) for v in field)
                elif field not in (None, ""):
                    raw_values.append(str(field))
            long_string = any(len(value) >= 100 for value in raw_values)
            missing_description = not product.get("description")
            try:
                rating_number = float(product.get("rating_number") or 0.0)
            except (TypeError, ValueError):
                rating_number = 0.0
            low_popularity = rating_number <= 50.0
            if common_attribute:
                criteria_counts["common_attribute"] += 1
            if neighbors:
                criteria_counts["neighbors"] += 1
            if long_string:
                criteria_counts["long_string"] += 1
            if missing_description:
                criteria_counts["missing_description"] += 1
            if low_popularity:
                criteria_counts["low_popularity"] += 1
            secondary = sum((long_string, missing_description, low_popularity))
            if common_attribute and neighbors and secondary >= 2:
                eligible.append(parent_asin)

    rng = random.Random(args.seed)
    if len(eligible) > args.count:
        selected = sorted(rng.sample(eligible, args.count))
    else:
        selected = sorted(eligible)
    rng.shuffle(selected)
    scenarios = scenario_assignment(len(selected), args.seed + 1)

    rows = [
        {
            "category_bucket": "clothing",
            "difficulty_bucket": "hard_case",
            "ground_truth": {"parent_asin": parent_asin},
            "sample_id": f"cp4_hard_{index:04d}",
            "scenario_type": scenarios[index],
            "user_profile": {
                "purchase_frequency": "3-4 prior purchases",
                "average_prior_rating": None,
                "rating_style": "mixed",
                "preference_tags": [],
                "summary": "Synthetic hard-case profile.",
            },
        }
        for index, parent_asin in enumerate(selected)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "seed": args.seed,
        "eligible_hard_cases": len(eligible),
        "sample_count": len(rows),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "criteria_prevalence": criteria_counts,
        "scenario_counts": {
            scenario: sum(1 for row in rows if row["scenario_type"] == scenario)
            for scenario, _ in SCENARIO_PROPORTIONS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
