from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent


def product_document(product: dict) -> str:
    def flattened(field: str) -> str:
        value = product.get(field)
        if isinstance(value, dict):
            return "; ".join(f"{key}: {item}" for key, item in value.items())
        if isinstance(value, list):
            return "; ".join(str(item) for item in value)
        return "" if value is None else str(value)

    return " ".join(
        part for part in (
            f"title: {flattened('title')}",
            f"category: {flattened('categories')}",
            f"features: {flattened('features')}",
            f"details: {flattened('details')}",
            f"brand: {flattened('store')}",
            f"description: {flattened('description')}",
        ) if not part.endswith(": ")
    )


def structured_query(state: dict[str, object]) -> str:
    messages = state.get("messages")
    constraints = Agent._constraint_phrases(messages)
    profile = state.get("user_profile")
    tags = profile.get("preference_tags", []) if isinstance(profile, dict) else []
    if constraints:
        mode = "constrained exploratory" if state.get("exploratory") else "specific buying"
    else:
        mode = "exploratory browsing" if state.get("exploratory") else "specific shopping"
    return " ".join(
        part for part in (
            f"intent: {mode}.",
            f"request: {state.get('base_message', '')}.",
            f"requirements: {'; '.join(constraints)}." if constraints else "",
            f"profile priorities: {'; '.join(str(tag) for tag in tags)}." if tags else "",
        ) if part
    )


def split_name(sample_id: str, validation_fraction: float) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if value < validation_fraction else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-disjoint CP4 pairwise groups")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--hard-negatives", type=int, default=5)
    parser.add_argument("--tail-negatives", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    _, categories, products = catalog_index(args.catalog)
    # Mine against the frozen CP3 funnel.  Keeping the neural stage disabled
    # prevents circular negatives and makes regeneration independent of CP4.
    agent = Agent(
        args.catalog,
        use_coverage_rotation=False,
        use_cross_encoder=False,
    )
    outputs: dict[str, list[dict]] = {"train": [], "validation": []}
    coverage_states = 0
    total_states = 0

    for sample in samples:
        sample_id = str(sample["sample_id"])
        target = str(sample["ground_truth"]["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": intent_card, "behavior": behavior}
        session_id = f"cp4_pairs_{sample_id}"
        agent.reset(session_id, sample["user_profile"])
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(
            effective, coarse_category(categories.get(target, [])), disclosed
        )
        seen_signatures: set[tuple[str, ...]] = set()

        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, message, turn, 10)
            state = agent._sessions[session_id]
            terms = tuple(str(value) for value in state.get("last_query_terms", []))
            if override_applied and terms not in seen_signatures:
                seen_signatures.add(terms)
                total_states += 1
                candidates = [str(value) for value in state.get("last_candidates", [])]
                ranked = [str(value) for value in state.get("last_ranking", candidates)]
                if target in ranked:
                    coverage_states += 1
                    negative_ids = [asin for asin in ranked if asin != target]
                    hard_ids = negative_ids[:args.hard_negatives]
                    tail_pool = negative_ids[args.hard_negatives:]
                    rng = random.Random(f"{args.seed}:{sample_id}:{turn}")
                    tail_ids = rng.sample(
                        tail_pool, min(args.tail_negatives, len(tail_pool))
                    )
                    outputs[split_name(sample_id, args.validation_fraction)].append({
                        "sample_id": sample_id,
                        "scenario_type": sample["scenario_type"],
                        "turn": turn,
                        "query": structured_query(state),
                        "positive": product_document(products[target]),
                        "negatives": [
                            product_document(products[parent_asin])
                            for parent_asin in [*hard_ids, *tail_ids]
                        ],
                    })

            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(override.get("message", "Actually, use the new preference."))
            else:
                message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )
                if "don't have an additional preference" in message.lower():
                    break

    for name, path in (
        ("train", args.train_output), ("validation", args.validation_output)
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in outputs[name]),
            encoding="utf-8",
        )
    print(json.dumps({
        "samples": len(samples),
        "train_groups": len(outputs["train"]),
        "validation_groups": len(outputs["validation"]),
        "candidate_coverage": round(coverage_states / max(1, total_states), 6),
        "total_states": total_states,
    }, indent=2))


if __name__ == "__main__":
    main()
