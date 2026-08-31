from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


def comma_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune a CP5 mode-specific tie head")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=("all", "browsing", "buying", "intent_override", "boundary"),
        required=True,
    )
    parser.add_argument("--weights", type=comma_floats, default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(
        args.catalog,
        use_dialogue_cards=True,
        dialogue_tiebreak="blend",
        opening_output_k=1,
        # Isolate the rank head from the later ambiguity-abstention experiment.
        ambiguous_output_k=10,
    )
    attributes = {
        "browsing": "_dialogue_browsing_linear_weight",
        "buying": "_dialogue_buying_linear_weight",
        "intent_override": "_dialogue_override_linear_weight",
        "boundary": "_dialogue_boundary_linear_weight",
    }
    scenarios = list(attributes) if args.scenario == "all" else [args.scenario]
    results: list[dict] = []
    for scenario in scenarios:
        samples = [
            sample for sample in all_samples if sample["scenario_type"] == scenario
        ]
        for weight in args.weights:
            setattr(agent, attributes[scenario], weight)
            result = evaluate(agent, samples, catalog_ids, categories, products)
            record = {
                "scenario": scenario,
                "linear_weight": weight,
                **{key: value for key, value in result.items() if key != "sessions"},
            }
            results.append(record)
            print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
