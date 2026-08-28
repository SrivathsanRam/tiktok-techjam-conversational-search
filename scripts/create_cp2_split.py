from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_SEED = "techjam-cp2-v1"
HOLDOUT_COUNTS = {
    "buying": 20,
    "browsing": 20,
    "intent_override": 8,
    "boundary": 2,
}


def stable_key(seed: str, purpose: str, scenario: str, sample_id: str) -> str:
    value = f"{seed}\0{purpose}\0{scenario}\0{sample_id}".encode()
    return hashlib.sha256(value).hexdigest()


def build_manifest(dataset_path: Path, seed: str) -> dict:
    # Deliberately retain only target-blind metadata. Ground-truth fields are never
    # read or copied into the split manifest.
    grouped: dict[str, list[str]] = defaultdict(list)
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[str(row["scenario_type"])].append(str(row["sample_id"]))

    if set(grouped) != set(HOLDOUT_COUNTS):
        raise ValueError(f"unexpected scenarios: {sorted(grouped)}")

    dev_ids: list[str] = []
    holdout_ids: list[str] = []
    scenarios: dict[str, str] = {}
    for scenario, sample_ids in sorted(grouped.items()):
        ordered = sorted(
            sample_ids,
            key=lambda sample_id: stable_key(seed, "holdout", scenario, sample_id),
        )
        holdout_count = HOLDOUT_COUNTS[scenario]
        holdout_ids.extend(ordered[:holdout_count])
        dev_ids.extend(ordered[holdout_count:])
        scenarios.update({sample_id: scenario for sample_id in ordered})

    folds: list[list[str]] = [[] for _ in range(5)]
    dev_by_scenario: dict[str, list[str]] = defaultdict(list)
    for sample_id in dev_ids:
        dev_by_scenario[scenarios[sample_id]].append(sample_id)
    for scenario, sample_ids in sorted(dev_by_scenario.items()):
        ordered = sorted(
            sample_ids,
            key=lambda sample_id: stable_key(seed, "fold", scenario, sample_id),
        )
        for index, sample_id in enumerate(ordered):
            folds[index % 5].append(sample_id)

    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    return {
        "protocol": "cp2-target-blind-split-v1",
        "seed": seed,
        "dataset": str(dataset_path).replace("\\", "/"),
        "dataset_sha256": dataset_sha256,
        "dev_ids": sorted(dev_ids),
        "holdout_ids": sorted(holdout_ids),
        "folds": [sorted(fold) for fold in folds],
        "counts": {
            "dev": len(dev_ids),
            "holdout": len(holdout_ids),
            "folds": [len(fold) for fold in folds],
            "holdout_by_scenario": dict(HOLDOUT_COUNTS),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the target-blind cp2 split manifest")
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/cp2_split.json"))
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = build_manifest(args.dataset, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
