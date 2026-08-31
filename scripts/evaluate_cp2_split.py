from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


def selected_ids(manifest: dict, partition: str) -> set[str]:
    if partition == "dev":
        return set(manifest["dev_ids"])
    if partition == "holdout":
        return set(manifest["holdout_ids"])
    if partition.startswith("fold-"):
        index = int(partition.removeprefix("fold-"))
        return set(manifest["folds"][index])
    if partition.startswith("train-"):
        index = int(partition.removeprefix("train-"))
        validation = set(manifest["folds"][index])
        return set(manifest["dev_ids"]) - validation
    raise ValueError(f"unknown partition: {partition}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a target-blind cp2 partition")
    parser.add_argument("partition", choices=[
        "dev", "holdout",
        "fold-0", "fold-1", "fold-2", "fold-3", "fold-4",
        "train-0", "train-1", "train-2", "train-3", "train-4",
    ])
    parser.add_argument("--manifest", type=Path, default=Path("data/cp2_split.json"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    identifiers = selected_ids(manifest, args.partition)
    samples = [
        sample for sample in load_jsonl(args.dataset)
        if str(sample["sample_id"]) in identifiers
    ]
    if len(samples) != len(identifiers):
        raise ValueError("split manifest and public dataset do not match")

    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    # Holdout artifacts are aggregate-only by construction. This prevents target-
    # level or sample-level inspection during cp2 development.
    output = {key: value for key, value in result.items() if key != "sessions"}
    if args.partition != "holdout":
        output["sessions"] = result["sessions"]
    output["cp2_partition"] = args.partition
    output["cp2_manifest_sha256"] = __import__("hashlib").sha256(
        args.manifest.read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
