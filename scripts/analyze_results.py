from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from starter.agent import Agent


def _mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(statistics.fmean(clean), 6)


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "miss"
    if rank == 1:
        return "rank_1"
    if rank <= 3:
        return "rank_2_3"
    return "rank_4_10"


def summarize_result_file(path: str | Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    sessions = result.get("sessions", [])
    buckets: dict[str, int] = defaultdict(int)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        buckets[_rank_bucket(session.get("best_rank"))] += 1
        grouped[str(session.get("scenario_type", "unknown"))].append(session)
    return {
        "source": str(path),
        "overall": {key: result.get(key) for key in (
            "sample_count",
            "hit_rate_at_10",
            "mrr",
            "mttc",
            "efficiency",
            "recommended_technical_score",
        )},
        "rank_buckets": dict(sorted(buckets.items())),
        "scenario_metrics": {
            name: metric_summary(items)
            for name, items in sorted(grouped.items())
        },
    }


def run_diagnostics(catalog_path: str | Path, dataset_path: str | Path, candidate_limit: int) -> dict:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path)
    sessions: list[dict] = []

    for sample in samples:
        session_id = f"diag_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
        hit_turn: int | None = None
        best_rank: int | None = None
        candidate_seen_turn: int | None = None
        best_candidate_rank: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}

            state = agent._sessions.get(session_id)
            if state is not None:
                candidates = agent._blended_candidates(state, candidate_limit)
                candidate_ids = [candidate.parent_asin for candidate in candidates]
                if target in candidate_ids:
                    rank = candidate_ids.index(target) + 1
                    candidate_seen_turn = candidate_seen_turn or turn
                    best_candidate_rank = rank if best_candidate_rank is None else min(best_candidate_rank, rank)

            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break

            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )

        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            "target_seen_in_candidates": candidate_seen_turn is not None,
            "candidate_seen_turn": candidate_seen_turn,
            "best_candidate_rank": best_candidate_rank,
        })

    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    overall = metric_summary(sessions)
    return {
        "sample_count": len(sessions),
        "metrics": overall,
        "candidate_limit": candidate_limit,
        "candidate_recall_at_limit": round(
            sum(int(item["target_seen_in_candidates"]) for item in sessions) / len(sessions),
            6,
        ) if sessions else 0.0,
        "recalled_but_missed": sum(
            int(item["target_seen_in_candidates"] and not item["hit"])
            for item in sessions
        ),
        "hit_but_not_rank_1": sum(
            int(item["hit"] and item["best_rank"] != 1)
            for item in sessions
        ),
        "average_best_candidate_rank": _mean(item["best_candidate_rank"] for item in sessions),
        "scenario_diagnostics": {
            name: {
                **metric_summary(items),
                "candidate_recall_at_limit": round(
                    sum(int(item["target_seen_in_candidates"]) for item in items) / len(items),
                    6,
                ) if items else 0.0,
                "average_best_candidate_rank": _mean(item["best_candidate_rank"] for item in items),
                "recalled_but_missed": sum(
                    int(item["target_seen_in_candidates"] and not item["hit"])
                    for item in items
                ),
                "hit_but_not_rank_1": sum(
                    int(item["hit"] and item["best_rank"] != 1)
                    for item in items
                ),
            }
            for name, items in sorted(grouped.items())
        },
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TechJam evaluator results and ranking diagnostics")
    parser.add_argument("--results", default="results.json", help="Evaluator results JSON to summarize")
    parser.add_argument("--catalog", default="data/catalog.jsonl", help="Catalog for diagnostic simulation")
    parser.add_argument("--dataset", default="data/public_set.jsonl", help="Public dataset for diagnostic simulation")
    parser.add_argument("--candidate-limit", type=int, default=350)
    parser.add_argument("--diagnostics-output", default="diagnostics.json")
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()

    random.seed(0)
    if Path(args.results).exists():
        print(json.dumps(summarize_result_file(args.results), indent=2))
    else:
        print(json.dumps({"warning": f"{args.results} does not exist yet"}, indent=2))

    if args.skip_diagnostics:
        return
    if not Path(args.catalog).exists():
        print(json.dumps({"warning": f"{args.catalog} does not exist; skipping diagnostics"}, indent=2))
        return

    diagnostics = run_diagnostics(args.catalog, args.dataset, args.candidate_limit)
    Path(args.diagnostics_output).write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    printable = {key: value for key, value in diagnostics.items() if key != "sessions"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
