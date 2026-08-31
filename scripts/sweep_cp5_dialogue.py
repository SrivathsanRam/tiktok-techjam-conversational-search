"""Sweep Arjo CP5 output widths and dialogue-prefix tie-breaks on public dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


CONFIGS = (
    ("cards_top10", "popularity", 10, 10),
    ("opening_top1", "popularity", 1, 10),
    ("ambiguity_top1", "popularity", 1, 1),
    ("ambiguity_top1_linear", "linear", 1, 1),
    ("ambiguity_top1_hybrid", "hybrid", 1, 1),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/cp2_split.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dev_ids = {str(value) for value in manifest["dev_ids"]}
    samples = [
        sample for sample in load_jsonl(args.dataset)
        if str(sample["sample_id"]) in dev_ids
    ]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)
    results: list[dict] = []
    for name, tiebreak, opening_k, ambiguous_k in CONFIGS:
        agent._dialogue_tiebreak = tiebreak
        agent._opening_output_k = opening_k
        agent._ambiguous_output_k = ambiguous_k
        result = evaluate(agent, samples, catalog_ids, categories, products)
        record = {
            "name": name,
            "dialogue_tiebreak": tiebreak,
            "opening_output_k": opening_k,
            "ambiguous_output_k": ambiguous_k,
            **{key: value for key, value in result.items() if key != "sessions"},
        }
        results.append(record)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
