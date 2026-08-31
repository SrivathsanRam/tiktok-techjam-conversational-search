"""UNOFFICIAL paraphrase-robustness harness — NOT the official evaluator.

This is a local copy of the evaluator's session loop that rewrites the
simulator's *wording* while keeping every catalog-derived value verbatim.
Scores produced here are diagnostic only: they are not comparable to
`evaluator.local_evaluator` output and must never be reported as official
results. The official evaluator is never imported for message generation and
is never modified.

What is paraphrased:

- constraint template markers are dropped or replaced ("A key requirement
  is:" -> "it must have", "For that, what matters is:" -> "I care about"),
- framing verbs and browsing/override/no-preference/rejection wording are
  reworded,
- list delimiters change (";" -> "," / " and " / "|"),
- the opening sentence boundary changes ("." -> "—" / ";").

What is never paraphrased: the catalog-derived constraint values, the
category string, and the target identifiers. Hits stay exact code matches.

Usage:
    python3 -m tests.paraphrase_harness [--dataset ...] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import uuid
from pathlib import Path

from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    MAX_TURNS,
    TOP_K,
    catalog_index,
    classify_constraint,
    coarse_category,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent


OPENING_BUYING = (
    "I'm looking for {category}. A key requirement is: {value}.",
    "I want {category} — it must have {value}.",
    "Shopping for {category}; needs to be {value}.",
    "After {category}, and {value} is essential.",
)
OPENING_OVERRIDE = (
    "I'm looking for {category}. {value}",
    "I want {category} — {value}",
    "Shopping for {category}; {value}",
    "After {category}. {value}",
)
OPENING_BROWSING = (
    "I'm looking for {category}, but I'm still exploring.",
    "Browsing {category} for now, nothing decided.",
    "Checking out {category} — just looking at the moment.",
    "{category} please, though I'm undecided so far.",
)
DISCLOSURE = (
    "For that, what matters is: {values}.",
    "I care about {values}.",
    "It must have {values}.",
    "What I need is {values}.",
)
DELIMITERS = ("; ", ", ", " and ", " | ")
NO_PREFERENCE = (
    "I don't have an additional preference for {attribute}.",
    "No strong preference on {attribute}, honestly.",
    "{attribute} doesn't matter to me.",
    "I'm easy about {attribute} — anything is fine.",
)
BOUNDARY = (
    "I don't have a preference for {attribute}; please use your judgment.",
    "Whatever you think is best for {attribute}.",
    "Your call on {attribute}.",
    "Up to you regarding {attribute}.",
)
REJECTION = (
    "Those options are not quite right yet. Ask me about one specific attribute.",
    "None of those work for me. Ask about one attribute.",
    "These aren't right — try asking about a single attribute.",
    "That's off the mark. Ask me about one attribute.",
)
OVERRIDE = (
    "Actually, ignore my earlier preference. What I need is: {value}.",
    "Scratch that — I changed my mind. I need {value}.",
    "Never mind my earlier note; make that {value}.",
    "On second thought, disregard that. It must have {value}.",
)


def _pick(options: tuple[str, ...], rng: random.Random, paraphrase: bool) -> str:
    """Clean runs always take the official template (index 0)."""
    return options[rng.randrange(1, len(options))] if paraphrase else options[0]


def initial_message(
    sample: dict, category: str, disclosed: set[str],
    rng: random.Random, paraphrase: bool,
) -> str:
    scenario = sample["scenario_type"]
    if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
        constraint = str(sample["intent_card"]["hard_constraints"][0])
        disclosed.add(constraint)
        return _pick(OPENING_BUYING, rng, paraphrase).format(
            category=category, value=constraint
        )
    if scenario == "intent_override":
        old_value = str(sample["behavior"]["override"]["old_value"])
        return _pick(OPENING_OVERRIDE, rng, paraphrase).format(
            category=category, value=old_value
        )
    return _pick(OPENING_BROWSING, rng, paraphrase).format(category=category)


def customer_reply(
    sample: dict, ask_attribute: object, disclosed: set[str], boundary_used: bool,
    rng: random.Random, paraphrase: bool,
) -> tuple[str, bool]:
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
        return _pick(BOUNDARY, rng, paraphrase).format(attribute=attribute), True
    if not attribute:
        return _pick(REJECTION, rng, paraphrase), boundary_used
    if attribute not in ALLOWED_ATTRIBUTES:
        attribute = "other"
    constraints = [
        *[str(value) for value in sample["intent_card"].get("hard_constraints", [])],
        *[str(value) for value in sample["intent_card"].get("soft_preferences", [])],
    ]
    matches = [
        value for value in constraints
        if value not in disclosed
        and (attribute == "other" or classify_constraint(value) == attribute)
    ][:2]
    if not matches:
        return _pick(NO_PREFERENCE, rng, paraphrase).format(attribute=attribute), boundary_used
    disclosed.update(matches)
    delimiter = "; " if not paraphrase else DELIMITERS[rng.randrange(len(DELIMITERS))]
    return (
        _pick(DISCLOSURE, rng, paraphrase).format(values=delimiter.join(matches)),
        boundary_used,
    )


def override_message(override: dict, rng: random.Random, paraphrase: bool) -> str:
    value = str(override.get("new_value", ""))
    if not value:
        return "Actually, please ignore my earlier preference."
    return _pick(OVERRIDE, rng, paraphrase).format(value=value)


def run(
    agent: Agent, samples: list[dict], catalog_ids: set[str],
    categories: dict[str, list[str]], products: dict[str, dict], paraphrase: bool,
) -> dict:
    sessions: list[dict] = []
    for sample in samples:
        # One stream per sample keeps clean and paraphrased runs aligned.
        rng = random.Random(f"{sample['sample_id']}\0paraphrase")
        session_id = f"harness_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective, coarse_category(categories.get(target, [])), disclosed,
            rng, paraphrase,
        )
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = override_message(override, rng, paraphrase)
            else:
                user_message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed,
                    boundary_used, rng, paraphrase,
                )
        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            "dialogue_compatible": bool(
                agent._sessions[session_id].get("dialogue_compatible")
            ),
            "dialogue_ever_active": bool(
                agent._sessions[session_id].get("dialogue_ever_active")
            ),
        })
    hit_rate = sum(int(item["hit"]) for item in sessions) / len(sessions)
    mrr = statistics.fmean(item["reciprocal_rank"] for item in sessions)
    mttc = statistics.fmean(
        item["first_hit_turn"] if item["first_hit_turn"] is not None else MAX_TURNS + 1
        for item in sessions
    )
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(0.5 * hit_rate + 0.3 * mrr + 0.2 * efficiency, 6),
        "dialogue_compatible_rate": round(
            sum(int(item["dialogue_compatible"]) for item in sessions) / len(sessions),
            6,
        ),
        "dialogue_active_rate": round(
            sum(int(item["dialogue_ever_active"]) for item in sessions) / len(sessions),
            6,
        ),
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="UNOFFICIAL paraphrase harness")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[:args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)
    clean = run(agent, samples, catalog_ids, categories, products, paraphrase=False)
    paraphrased = run(agent, samples, catalog_ids, categories, products, paraphrase=True)
    gap = round(clean["technical_score"] - paraphrased["technical_score"], 6)
    report = {
        "harness": "UNOFFICIAL — diagnostic only, not the official evaluator",
        "dataset": str(args.dataset),
        "clean": {key: value for key, value in clean.items() if key != "sessions"},
        "paraphrased": {key: value for key, value in paraphrased.items() if key != "sessions"},
        "technical_score_gap": gap,
        "target_gap": 0.1,
        "meets_target": gap < 0.1,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            **report,
            "clean_sessions": clean["sessions"],
            "paraphrased_sessions": paraphrased["sessions"],
        }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
