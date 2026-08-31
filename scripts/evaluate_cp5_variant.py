"""Evaluate a configurable Arjo CP5 dialogue-card variant."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", choices=("dev", "holdout", "full"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/cp2_split.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-dialogue-cards", action="store_true")
    parser.add_argument(
        "--dialogue-tiebreak", choices=("popularity", "linear", "hybrid"),
        default="popularity",
    )
    parser.add_argument("--dialogue-candidate-limit", type=int, default=80)
    parser.add_argument("--opening-output-k", type=int, default=1)
    parser.add_argument("--ambiguous-output-k", type=int, default=1)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.partition != "full":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        selected = {str(value) for value in manifest[f"{args.partition}_ids"]}
        samples = [
            sample for sample in samples
            if str(sample["sample_id"]) in selected
        ]

    started = time.perf_counter()
    catalog_ids, categories, products = catalog_index(args.catalog)
    catalog_load_seconds = time.perf_counter() - started
    agent_started = time.perf_counter()
    agent = Agent(
        args.catalog,
        use_dialogue_cards=not args.no_dialogue_cards,
        dialogue_tiebreak=args.dialogue_tiebreak,
        dialogue_candidate_limit=args.dialogue_candidate_limit,
        opening_output_k=args.opening_output_k,
        ambiguous_output_k=args.ambiguous_output_k,
    )
    agent_init_seconds = time.perf_counter() - agent_started
    evaluation_started = time.perf_counter()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    evaluation_seconds = time.perf_counter() - evaluation_started
    result["cp5_variant"] = {
        "partition": args.partition,
        "use_dialogue_cards": not args.no_dialogue_cards,
        "dialogue_tiebreak": args.dialogue_tiebreak,
        "dialogue_candidate_limit": args.dialogue_candidate_limit,
        "opening_output_k": args.opening_output_k,
        "ambiguous_output_k": args.ambiguous_output_k,
        "catalog_load_seconds": round(catalog_load_seconds, 6),
        "agent_init_seconds": round(agent_init_seconds, 6),
        "evaluation_seconds": round(evaluation_seconds, 6),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {key: value for key, value in result.items() if key != "sessions"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
