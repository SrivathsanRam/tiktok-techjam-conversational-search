from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent


def rank_of(values: object, target: str) -> int | None:
    if not isinstance(values, list) or target not in values:
        return None
    return values.index(target) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure CP5 ranking headroom by turn")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--ambiguous-output-k", type=int, default=10)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.sample_id:
        samples = [
            sample for sample in samples if str(sample["sample_id"]) == args.sample_id
        ]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(
        args.catalog,
        use_dialogue_cards=True,
        dialogue_tiebreak="popularity",
        opening_output_k=1,
        ambiguous_output_k=args.ambiguous_output_k,
    )
    traces: list[dict] = []
    sessions: list[dict] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        session_id = f"cp5_diagnostic_{sample_id}"
        agent.reset(session_id, sample["user_profile"])
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(
            effective, coarse_category(categories.get(target, [])), disclosed
        )
        hit_turn: int | None = None
        hit_rank: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, message, turn, TOP_K)
            state = agent._sessions[session_id]
            candidates = state.get("last_candidates")
            linear = state.get("last_linear_ranking")
            final = state.get("last_ranking")
            output = normalize_recommendations(response["recommendations"], catalog_ids)
            constraints = agent._constraint_phrases(state.get("messages"))
            prefix_matches = agent._dialogue_index.matching_prefix(
                str(state.get("category_query", "")), constraints
            ) if agent._dialogue_index is not None else ()
            popularity_order = sorted(
                prefix_matches,
                key=lambda asin: (-agent._popularity.get(asin, 0.0), asin),
            )
            if override_applied:
                traces.append({
                    "sample_id": sample_id,
                    "scenario_type": sample["scenario_type"],
                    "turn": turn,
                    "constraint_count": len(constraints),
                    "candidate_rank": rank_of(candidates, target),
                    "linear_rank": rank_of(linear, target),
                    "final_rank": rank_of(final, target),
                    "output_rank": rank_of(output, target),
                    "prefix_target_match": target in prefix_matches,
                    "prefix_group_size": len(prefix_matches),
                    "prefix_popularity_rank": rank_of(popularity_order, target),
                })
                if target in output:
                    hit_turn = turn
                    hit_rank = output.index(target) + 1
                    break
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
        sessions.append({
            "sample_id": sample_id,
            "scenario_type": sample["scenario_type"],
            "hit_turn": hit_turn,
            "hit_rank": hit_rank,
        })

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for trace in traces:
        grouped[(trace["scenario_type"], trace["turn"])].append(trace)
    diagnostics: list[dict] = []
    for (scenario, turn), rows in sorted(grouped.items()):
        def recall(field: str, limit: int) -> float:
            return sum(
                int(isinstance(row[field], int) and row[field] <= limit) for row in rows
            ) / len(rows)

        group_sizes = [row["prefix_group_size"] for row in rows if row["prefix_group_size"]]
        diagnostics.append({
            "scenario_type": scenario,
            "turn": turn,
            "states": len(rows),
            "candidate_recall_at_20": round(recall("candidate_rank", 20), 6),
            "candidate_recall_at_80": round(recall("candidate_rank", 80), 6),
            "oracle_mrr_at_20": round(recall("candidate_rank", 20), 6),
            "linear_mrr_at_20": round(sum(
                1.0 / row["linear_rank"]
                if isinstance(row["linear_rank"], int) and row["linear_rank"] <= 20
                else 0.0 for row in rows
            ) / len(rows), 6),
            "dialogue_mrr": round(sum(
                1.0 / row["final_rank"] if isinstance(row["final_rank"], int) else 0.0
                for row in rows
            ) / len(rows), 6),
            "prefix_target_coverage": round(
                sum(int(row["prefix_target_match"]) for row in rows) / len(rows), 6
            ),
            "unique_prefix_fraction": round(
                sum(int(row["prefix_group_size"] == 1) for row in rows) / len(rows), 6
            ),
            "median_prefix_group_size": (
                statistics.median(group_sizes) if group_sizes else None
            ),
        })
    payload = {"diagnostics": diagnostics, "sessions": sessions, "traces": traces}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
