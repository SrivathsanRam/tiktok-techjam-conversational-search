from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def split_name(sample_id: str, validation_fraction: float) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if value < validation_fraction else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize session-level CP4 train/validation splits")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()

    outputs: dict[str, list[str]] = {"train": [], "validation": []}
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            outputs[split_name(str(row["sample_id"]), args.validation_fraction)].append(line)
    for name, path in (
        ("train", args.train_output), ("validation", args.validation_output)
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(outputs[name]), encoding="utf-8")
    print(json.dumps({name: len(rows) for name, rows in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
