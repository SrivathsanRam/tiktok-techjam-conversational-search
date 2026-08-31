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


VARIANTS: dict[str, dict[str, object]] = {
    "cp5": {},
    "category": {"_use_category_filter": True},
    "exhaustion": {"_use_exhaustion_release": True},
    "release3": {"_ambiguity_release_turn": 3},
    "release4": {"_ambiguity_release_turn": 4},
    "release5": {"_ambiguity_release_turn": 5},
    "rating002": {"_dialogue_rating_weight": 0.02},
    "rating005": {"_dialogue_rating_weight": 0.05},
    "rating010": {"_dialogue_rating_weight": 0.10},
    "category_exhaustion": {
        "_use_category_filter": True,
        "_use_exhaustion_release": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep isolated CP6 candidates")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/cp2_split.json"))
    parser.add_argument("--partition", choices=("dev", "full"), default="dev")
    parser.add_argument(
        "--variants", default=",".join(VARIANTS),
        help="Comma-separated variant names",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = [value for value in requested if value not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    samples = load_jsonl(args.dataset)
    if args.partition == "dev":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        selected = set(map(str, manifest["dev_ids"]))
        samples = [sample for sample in samples if str(sample["sample_id"]) in selected]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)
    defaults = {
        "_use_category_filter": False,
        "_use_exhaustion_release": False,
        "_ambiguity_release_turn": 10,
        "_dialogue_rating_weight": 0.0,
    }
    results: list[dict[str, object]] = []
    for name in requested:
        for attribute, value in defaults.items():
            setattr(agent, attribute, value)
        for attribute, value in VARIANTS[name].items():
            setattr(agent, attribute, value)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        record = {
            "variant": name,
            "settings": {**defaults, **VARIANTS[name]},
            **{key: value for key, value in result.items() if key != "sessions"},
        }
        results.append(record)
        print(json.dumps(record), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
