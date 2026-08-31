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


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def comma_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep CP5 dialogue ranking controls")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/cp2_split.json"))
    parser.add_argument("--partition", choices=("dev", "full"), default="dev")
    parser.add_argument("--opening-k", type=comma_ints, default=[1, 3, 5, 10])
    parser.add_argument(
        "--tiebreaks", type=comma_strings, default=["popularity", "linear", "hybrid"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.partition == "dev":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        selected = set(str(value) for value in manifest["dev_ids"])
        samples = [sample for sample in samples if str(sample["sample_id"]) in selected]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(
        args.catalog,
        use_dialogue_cards=True,
        # Isolate opening width from the later ambiguity-abstention experiment.
        ambiguous_output_k=10,
    )
    results: list[dict] = []
    for tiebreak in args.tiebreaks:
        agent._dialogue_tiebreak = tiebreak
        for opening_k in args.opening_k:
            agent._opening_output_k = opening_k
            result = evaluate(agent, samples, catalog_ids, categories, products)
            record = {
                "dialogue_tiebreak": tiebreak,
                "opening_output_k": opening_k,
                **{key: value for key, value in result.items() if key != "sessions"},
            }
            results.append(record)
            print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
