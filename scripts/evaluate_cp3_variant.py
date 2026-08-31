from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a configurable cp3 agent variant")
    parser.add_argument("partition", choices=("dev", "holdout", "full"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/cp2_split.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-exact", action="store_true")
    parser.add_argument("--exact-limit", type=int, default=60)
    parser.add_argument("--exact-single-max-df", type=int, default=50000)
    parser.add_argument("--rerank-limit", type=int, default=80)
    parser.add_argument("--sparse-limit", type=int, default=60)
    parser.add_argument("--route-limit", type=int, default=150)
    parser.add_argument("--no-rotation", action="store_true")
    parser.add_argument("--coverage-head", type=int, default=0)
    parser.add_argument("--no-cross-encoder", action="store_true")
    parser.add_argument("--cross-candidates", type=int, default=20)
    parser.add_argument("--cross-buying-weight", type=float, default=0.0)
    parser.add_argument("--cross-browsing-weight", type=float, default=0.0)
    parser.add_argument("--cross-constrained-browsing-weight", type=float)
    parser.add_argument("--cross-override-weight", type=float, default=0.0)
    parser.add_argument("--cross-min-constraints", type=int, default=1)
    parser.add_argument("--cross-min-margin", type=float, default=0.0)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.partition != "full":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        selected = set(str(value) for value in manifest[f"{args.partition}_ids"])
        samples = [sample for sample in samples if str(sample["sample_id"]) in selected]

    started = time.perf_counter()
    catalog_ids, categories, products = catalog_index(args.catalog)
    catalog_load_seconds = time.perf_counter() - started
    agent_started = time.perf_counter()
    agent = Agent(
        args.catalog,
        use_exact_evidence=not args.no_exact,
        exact_candidate_limit=args.exact_limit,
        exact_single_max_df=args.exact_single_max_df,
        rerank_candidate_limit=args.rerank_limit,
        sparse_candidate_limit=args.sparse_limit,
        route_candidate_limit=args.route_limit,
        use_coverage_rotation=not args.no_rotation,
        coverage_head=args.coverage_head,
        use_cross_encoder=not args.no_cross_encoder,
        cross_encoder_candidates=args.cross_candidates,
        cross_encoder_buying_weight=args.cross_buying_weight,
        cross_encoder_browsing_weight=args.cross_browsing_weight,
        cross_encoder_constrained_browsing_weight=args.cross_constrained_browsing_weight,
        cross_encoder_override_weight=args.cross_override_weight,
        cross_encoder_min_constraints=args.cross_min_constraints,
        cross_encoder_min_margin=args.cross_min_margin,
    )
    agent_init_seconds = time.perf_counter() - agent_started
    evaluation_started = time.perf_counter()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    evaluation_seconds = time.perf_counter() - evaluation_started
    result["cp3_variant"] = {
        "partition": args.partition,
        "use_exact_evidence": not args.no_exact,
        "exact_candidate_limit": args.exact_limit,
        "exact_single_max_df": args.exact_single_max_df,
        "rerank_candidate_limit": args.rerank_limit,
        "sparse_candidate_limit": args.sparse_limit,
        "route_candidate_limit": args.route_limit,
        "use_coverage_rotation": not args.no_rotation,
        "coverage_head": args.coverage_head,
        "cross_encoder_loaded": agent._cross_encoder is not None,
        "cross_encoder_candidates": args.cross_candidates,
        "cross_encoder_buying_weight": args.cross_buying_weight,
        "cross_encoder_browsing_weight": args.cross_browsing_weight,
        "cross_encoder_constrained_browsing_weight": (
            args.cross_buying_weight
            if args.cross_constrained_browsing_weight is None
            else args.cross_constrained_browsing_weight
        ),
        "cross_encoder_override_weight": args.cross_override_weight,
        "cross_encoder_min_constraints": args.cross_min_constraints,
        "cross_encoder_min_margin": args.cross_min_margin,
        "catalog_load_seconds": round(catalog_load_seconds, 6),
        "agent_init_seconds": round(agent_init_seconds, 6),
        "evaluation_seconds": round(evaluation_seconds, 6),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
