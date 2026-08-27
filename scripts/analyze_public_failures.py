from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent, _terms


def route_ranks(
    agent: Agent,
    terms: list[str],
    target: str,
    limit: int,
    disjunctive_weight: float,
    popularity_weight: float,
) -> dict[str, int | None]:
    quoted = [f'"{term}"' for term in terms]
    expressions = {
        "conjunctive": " AND ".join(quoted),
        "phrase": " OR ".join(
            f'"{terms[index]} {terms[index + 1]}"'
            for index in range(len(terms) - 1)
        ),
        "disjunctive": " OR ".join(quoted),
    }
    result: dict[str, int | None] = {}
    for name, expression in expressions.items():
        ranking = agent._ranked_asins(expression, limit) if expression else []
        result[name] = ranking.index(target) + 1 if target in ranking else None
    fused = agent._fused_search(
        terms,
        limit,
        disjunctive_weight=disjunctive_weight,
        popularity_weight=popularity_weight,
    )
    result["fused"] = fused.index(target) + 1 if target in fused else None
    return result


def analyze(
    catalog_path: Path,
    dataset_path: Path,
    results_path: Path,
    route_limit: int,
) -> dict:
    catalog_ids, categories, products = catalog_index(catalog_path)
    del catalog_ids
    samples = {sample["sample_id"]: sample for sample in load_jsonl(dataset_path)}
    prior_result = json.loads(results_path.read_text(encoding="utf-8"))
    misses = [session for session in prior_result["sessions"] if not session["hit"]]
    agent = Agent(catalog_path)
    report: list[dict] = []
    rank_cache: dict[tuple[str, ...], dict[str, int | None]] = {}

    for miss in misses:
        sample = samples[miss["sample_id"]]
        target = str(sample["ground_truth"]["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}
        session_id = f'diagnostic_{sample["sample_id"]}'
        agent.reset(session_id, sample["user_profile"])
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective_sample,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        turns: list[dict] = []

        for turn in range(1, 11):
            response = agent.respond(session_id, user_message, turn, 10)
            state = agent._sessions[session_id]
            query_text = " ".join(str(item) for item in state["messages"])
            terms = list(dict.fromkeys(_terms(query_text)))[:80]
            rank_key = (target, *terms)
            ranks = rank_cache.get(rank_key)
            if ranks is None:
                ranks = route_ranks(
                    agent,
                    terms,
                    target,
                    route_limit,
                    disjunctive_weight=1.0 if state["exploratory"] else 2.0,
                    popularity_weight=0.0 if state["exploratory"] else 1.0,
                )
                rank_cache[rank_key] = ranks
            turns.append({
                "turn": turn,
                "scorable": override_applied,
                "user_message": user_message,
                "terms": terms,
                "target_route_ranks": ranks,
            })
            if turn == 10:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get("message", "Actually, please ignore my earlier preference.")
                )
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )

        product = products[target]
        report.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "target": target,
            "target_title": product.get("title"),
            "target_categories": product.get("categories"),
            "intent_card": intent_card,
            "behavior": behavior,
            "turns": turns,
        })

    return {
        "source_results": str(results_path),
        "route_limit": route_limit,
        "miss_count": len(report),
        "misses": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze public misses without changing the evaluator")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("results.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-limit", type=int, default=1000)
    args = parser.parse_args()
    report = analyze(args.catalog, args.dataset, args.results, args.route_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"miss_count": report["miss_count"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
