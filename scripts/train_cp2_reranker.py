from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
except ImportError as error:  # pragma: no cover - optional training dependency
    raise SystemExit(
        "Training requires numpy and scikit-learn. Runtime inference does not."
    ) from error

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
)
from starter.agent import Agent, RERANK_FEATURE_NAMES


C_GRID = (0.01, 0.05, 0.1, 0.5, 1.0)


def generate_groups(
    agent: Agent,
    samples: list[dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> tuple[list[dict], dict[str, str]]:
    groups: list[dict] = []
    scenarios: dict[str, str] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        scenarios[sample_id] = str(sample["scenario_type"])
        target = str(sample["ground_truth"]["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": intent_card, "behavior": behavior}
        session_id = f"cp2_train_{sample_id}"
        agent.reset(session_id, sample["user_profile"])
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        seen_states: set[tuple[str, ...]] = set()

        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, 10)
            state = agent._sessions[session_id]
            query_terms = [str(value) for value in state.get("last_query_terms", [])]
            candidates = [str(value) for value in state.get("last_candidates", [])]
            signature = tuple(query_terms)
            if override_applied and signature not in seen_states:
                seen_states.add(signature)
                messages = state["messages"]
                query_text = (
                    " ".join(str(item) for item in messages)
                    if isinstance(messages, list) else ""
                )
                constraints = agent._constraint_phrases(messages)
                features = [
                    agent._feature_vector(
                        parent_asin,
                        rank,
                        state,
                        query_terms,
                        constraints,
                        query_text,
                    )
                    for rank, parent_asin in enumerate(candidates, start=1)
                ]
                groups.append({
                    "sample_id": sample_id,
                    "turn": turn,
                    "target_index": candidates.index(target) if target in candidates else -1,
                    "features": features,
                })

            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
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
                    effective,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
    return groups, scenarios


def pairwise_examples(groups: list[dict], sample_ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    examples: list[np.ndarray] = []
    labels: list[int] = []
    for group in groups:
        if group["sample_id"] not in sample_ids or group["target_index"] < 0:
            continue
        matrix = np.asarray(group["features"], dtype=np.float64)
        positive = matrix[group["target_index"]]
        negative_indices = [
            index for index in range(min(30, len(matrix)))
            if index != group["target_index"]
        ]
        for index in negative_indices:
            difference = positive - matrix[index]
            examples.extend((difference, -difference))
            labels.extend((1, 0))
    if not examples:
        raise ValueError("no pairwise examples generated")
    return np.vstack(examples), np.asarray(labels, dtype=np.int8)


def fit_weights(groups: list[dict], sample_ids: set[str], regularization: float) -> np.ndarray:
    examples, labels = pairwise_examples(groups, sample_ids)
    model = LogisticRegression(
        C=regularization,
        fit_intercept=False,
        solver="liblinear",
        max_iter=2000,
        random_state=0,
    )
    model.fit(examples, labels)
    return np.asarray(model.coef_[0], dtype=np.float64)


def evaluate_weights(
    groups: list[dict],
    sample_ids: set[str],
    scenarios: dict[str, str],
    weights: np.ndarray,
) -> tuple[dict, list[dict]]:
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for group in groups:
        if group["sample_id"] in sample_ids:
            by_sample[group["sample_id"]].append(group)
    sessions: list[dict] = []
    for sample_id in sorted(sample_ids):
        hit_turn: int | None = None
        best_rank: int | None = None
        for group in sorted(by_sample.get(sample_id, []), key=lambda value: value["turn"]):
            target_index = int(group["target_index"])
            if target_index < 0:
                continue
            matrix = np.asarray(group["features"], dtype=np.float64)
            scores = matrix @ weights
            order = np.lexsort((np.arange(len(scores)), -scores))
            target_rank = int(np.where(order == target_index)[0][0]) + 1
            if target_rank <= 10:
                hit_turn = int(group["turn"])
                best_rank = target_rank
                break
        sessions.append({
            "sample_id": sample_id,
            "scenario_type": scenarios[sample_id],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    summary = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(summary["mttc"])) / 10.0))
    score = 0.5 * summary["hit_rate_at_10"] + 0.3 * summary["mrr"] + 0.2 * efficiency
    return {
        **summary,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
    }, sessions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cp2 dev-only linear pairwise reranker")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/cp2_split.json"))
    parser.add_argument("--weights-output", type=Path, default=Path("starter/reranker_weights.json"))
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dev_ids = set(str(value) for value in manifest["dev_ids"])
    samples = [
        sample for sample in load_jsonl(args.dataset)
        if str(sample["sample_id"]) in dev_ids
    ]
    if len(samples) != len(dev_ids):
        raise ValueError("dev manifest and dataset do not match")

    _, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog, load_reranker=False)
    groups, scenarios = generate_groups(agent, samples, categories, products)

    candidates: list[dict] = []
    fold_sets = [set(str(value) for value in fold) for fold in manifest["folds"]]
    for regularization in C_GRID:
        out_of_fold_sessions: list[dict] = []
        fold_metrics: list[dict] = []
        for fold_index, validation_ids in enumerate(fold_sets):
            training_ids = dev_ids - validation_ids
            weights = fit_weights(groups, training_ids, regularization)
            metrics, sessions = evaluate_weights(
                groups,
                validation_ids,
                scenarios,
                weights,
            )
            fold_metrics.append({"fold": fold_index, **metrics})
            out_of_fold_sessions.extend(sessions)
        pooled = metric_summary(out_of_fold_sessions)
        efficiency = max(0.0, min(1.0, (11.0 - float(pooled["mttc"])) / 10.0))
        technical_score = (
            0.5 * pooled["hit_rate_at_10"]
            + 0.3 * pooled["mrr"]
            + 0.2 * efficiency
        )
        candidates.append({
            "C": regularization,
            "pooled": {
                **pooled,
                "efficiency": round(efficiency, 6),
                "recommended_technical_score": round(technical_score, 6),
            },
            "folds": fold_metrics,
        })

    selected = max(
        candidates,
        key=lambda item: (
            item["pooled"]["recommended_technical_score"],
            item["pooled"]["hit_rate_at_10"],
            item["pooled"]["mrr"],
            -item["C"],
        ),
    )
    final_weights = fit_weights(groups, dev_ids, float(selected["C"]))
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    weights_payload = {
        "model": "pairwise_logistic_regression",
        "training_partition": "cp2_dev_150",
        "manifest_sha256": manifest_hash,
        "regularization_C": selected["C"],
        "features": list(RERANK_FEATURE_NAMES),
        "weights": {
            name: round(float(weight), 12)
            for name, weight in zip(RERANK_FEATURE_NAMES, final_weights, strict=True)
        },
    }
    report = {
        "protocol": "cp2-dev-only-five-fold-v1",
        "manifest_sha256": manifest_hash,
        "sample_count": len(dev_ids),
        "group_count": len(groups),
        "candidate_coverage": round(
            sum(group["target_index"] >= 0 for group in groups) / max(1, len(groups)),
            6,
        ),
        "regularization_candidates": candidates,
        "selected": selected,
        "final_weights": weights_payload["weights"],
    }
    args.weights_output.parent.mkdir(parents=True, exist_ok=True)
    args.weights_output.write_text(json.dumps(weights_payload, indent=2) + "\n", encoding="utf-8")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_count": report["sample_count"],
        "group_count": report["group_count"],
        "candidate_coverage": report["candidate_coverage"],
        "selected": report["selected"],
        "weights_output": str(args.weights_output),
    }, indent=2))


if __name__ == "__main__":
    main()
