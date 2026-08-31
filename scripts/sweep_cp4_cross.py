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


def parse_values(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep CP4 intent-specific neural RRF weights")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/cp2_split.json"))
    parser.add_argument("--partition", choices=("dev", "full"), default="dev")
    parser.add_argument("--buying-weights", type=parse_values, default=[0.0, 0.05, 0.1, 0.2, 0.35])
    parser.add_argument("--browsing-weight", type=float, default=0.0)
    parser.add_argument("--browsing-weights", type=parse_values)
    parser.add_argument("--constrained-browsing-weight", type=float)
    parser.add_argument("--cross-candidates", type=int, default=20)
    parser.add_argument("--min-constraints", type=parse_values, default=[1])
    parser.add_argument("--min-margins", type=parse_values, default=[0.0])
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
        cross_encoder_candidates=args.cross_candidates,
        cross_encoder_browsing_weight=args.browsing_weight,
    )
    if agent._cross_encoder is None:
        raise SystemExit("CP4 cross encoder could not be loaded")

    results: list[dict] = []
    browsing_weights = args.browsing_weights or [args.browsing_weight]
    for browsing_weight in browsing_weights:
        agent._cross_encoder_browsing_weight = browsing_weight
        for minimum_constraints in args.min_constraints:
            agent._cross_encoder_min_constraints = int(minimum_constraints)
            for minimum_margin in args.min_margins:
                agent._cross_encoder_min_margin = minimum_margin
                for weight in args.buying_weights:
                    agent._cross_encoder_buying_weight = weight
                    agent._cross_encoder_constrained_browsing_weight = (
                        weight
                        if args.constrained_browsing_weight is None
                        else args.constrained_browsing_weight
                    )
                    result = evaluate(agent, samples, catalog_ids, categories, products)
                    results.append({
                        "buying_weight": weight,
                        "browsing_weight": browsing_weight,
                        "constrained_browsing_weight": agent._cross_encoder_constrained_browsing_weight,
                        "override_weight": agent._cross_encoder_override_weight,
                        "minimum_constraints": agent._cross_encoder_min_constraints,
                        "minimum_margin": agent._cross_encoder_min_margin,
                        **{key: value for key, value in result.items() if key != "sessions"},
                    })
                    print(json.dumps(results[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
