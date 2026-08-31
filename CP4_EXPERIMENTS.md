# CP4 experiments

Every row is a full `python3 -m evaluator.local_evaluator` run on the public
200 set unless the partition column says otherwise. The controlled baseline is
the CP3 selected configuration at TechnicalScore `0.932620`.

| # | Change | Partition | HitRate@10 | MRR | MTTC | TechnicalScore | Decision |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | CP3 baseline (reproduced) | Public 200 | 0.9950 | 0.845734 | 1.930 | 0.932620 | Baseline |
| 1.1 | Dominance tiers everywhere, top tier uncapped (cap 1000) | Public 200 | 0.9950 | 0.804984 | 1.865 | 0.921695 | Revert: CP3 weights cannot order flooded tiers; six turn-1 buying sessions fell to rank 4-6 |
| 1.2 | As 1.1 with tier cap 60 / 150 / 300 | Public 200 | 0.9950 | 0.814290 / 0.811234 / 0.804984 | 1.880 / 1.875 / 1.865 | 0.924187 / 0.923370 / 0.921695 | Revert: dominance-with-old-weights loss is not a cap problem |
| 1.3 | As 1.1 with sparse-relevance-first pool order (retrieval feature keeps BM25 rank) | Public 200 | 0.9950 | 0.804234 | 1.860 | 0.921570 | Revert: within-tier loss is scoring, not the pool-rank feature |
| 1.4 | Dominance only when top tier ≤ 1000 (broad tiers keep CP3 precision slice) | Public 200 | 0.9950 | 0.842901 | 1.925 | 0.931870 | Revert: remaining −0.00075 is public_0184 hitting one turn earlier at rank 3 instead of rank 1 |
| 1.5 | As 1.4 with popularity features zeroed inside dominance tiers | Public 200 | 0.9950 | 0.832290 | 1.930 | 0.928587 | Revert: popularity carries real within-tier signal (targets are popularity-biased) |
| 1.6 | **Task 1 final: tiered dominance + uncapped tier paging on rotation turns; fresh-disclosure turns keep the CP3 pool until the reranker is retrained within tiers (Task 3)** | Public 200 | 0.9950 | 0.845734 | 1.930 | **0.932620** | **Keep: no regression; rotation now pages the whole ≤1000 top tier unseen-first and never reintroduces shown products while unseen tier members remain** |

## Task 1 notes

- `_exact_evidence_pool()` now returns the complete top dominance tier (every
  product satisfying the maximum number of exact constraints, safety cap 1000)
  plus per-product satisfied-constraint counts. A tier wider than the cap is
  not a discriminative dominance signal (for example every cotton product), so
  it falls back to the CP3 60-product precision slice.
- `_rerank()` accepts tier counts and sorts by `(-satisfied_count, -score,
  pool_rank, asin)`: more-satisfied always outranks fewer-satisfied; the
  learned score orders only within a tier.
- Rotation turns (unchanged query signature, which includes boundary,
  no-preference, and invalid-question replies — all verified by unit tests)
  rerank the full tier pool and select the first `top_k` unseen products in
  tiered order. The refill that may reintroduce shown products runs only after
  every unseen ranked candidate is exhausted.
- public_0020 is not yet fixed: after full disclosure its tier holds 463
  products sharing identical fabric boilerplate and the CP3 weights place the
  target at within-tier rank 335, beyond the ~100 products rotation can page
  by turn 10. Fixing it requires the Task 3 within-tier retraining (and the
  Task 3d coarse-category / rarity features); the Task 1 machinery is the
  prerequisite. Full dominance ordering on fresh-disclosure turns (experiment
  1.4) should also be re-tested after Task 3 retraining.
- Baseline-equality sanity check: constructing `Agent` with
  `exact_candidate_limit=0` forces the legacy path everywhere and reproduces
  CP3's 0.932620 exactly, confirming the refactor (including the new
  per-product feature caches) is value-identical.
