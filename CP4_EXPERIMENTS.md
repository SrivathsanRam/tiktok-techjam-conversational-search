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

| 2.1 | Frozen CP4 synthetic splits generated (no runtime change) | Public 200 | 0.9950 | 0.845734 | 1.930 | 0.932620 | Keep: data-only task, agent untouched |
| 2.2 | Current agent on synthetic_dev (1000) | Synthetic dev | 1.0000 | 0.696668 | 2.040 | 0.888200 | Recorded: Task 3 selection baseline |
| 2.3 | Current agent on hard-case set (500) | Hard cases | 0.8680 | 0.462138 | 4.060 | 0.711441 | Recorded: confirms the funnel struggles on common-attribute / dense-neighborhood targets |

| 3.1 | Retrained within-tier MRR-weighted reranker (18 features, C=0.1), full dominance ON | Public 200 | **1.0000** | **0.853929** | **1.880** | **0.938579** | **Keep: new best; public_0020 fixed (turn 2, rank 7); baseline +0.005959** |
| 3.2 | Same weights, dominance on rotation turns only | Public 200 | 1.0000 | 0.842208 | 1.900 | 0.934662 | Revert: full dominance is better with within-tier-trained weights |
| 3.3 | Selected config, one-shot confirm | Public holdout 50 | 1.0000 | 0.875000 | 1.920 | 0.944100 | Confirmed (untouched during selection) |
| 3.4 | Selected config, one-shot confirm | Synthetic holdout 1000 | 1.0000 | 0.763188 | 2.040 | 0.908156 | Confirmed (aggregate-only) |
| 3.5 | Selected config re-run | Synthetic dev 1000 | 1.0000 | 0.749971 | 1.981 | 0.905371 | Recorded: +0.017171 over row 2.2 |
| 3.6 | Selected config re-run | Hard cases 500 | 0.9060 | 0.518359 | 3.546 | 0.757588 | Recorded: +0.046147 over row 2.3 |

| 4.1 | **Per-constraint retrieval routes with df-based generic downweighting (df threshold 12000)** | Public 200 | **1.0000** | **0.856365** | **1.885** | **0.939210** | **Keep: +0.000631 over row 3.1** |
| 4.2 | Same, generic df threshold 6000 / 9000 / 15000 / 20000 | Public 200 | 1.0000 | 0.856276 / 0.856276 / 0.856365 / 0.856365 | 1.885 | 0.939183 / 0.939183 / 0.939210 / 0.939210 | Keep 12000: flat plateau, chosen mid-range so `imported` (df 15300) and bare `100` (df 17397) stay generic while `polyester` (10884) and `sole` (10441) stay specific |
| 4.3 | Selected routes | Synthetic dev 1000 | 1.0000 | 0.766978 | 2.016 | 0.909773 | Keep: +0.004402 over row 3.5 |
| 4.4 | Selected routes | Hard cases 500 | 0.9060 | 0.543895 | 3.578 | 0.764609 | Keep: +0.007021 over row 3.6 |

## Task 4 notes

- `_fused_search()` replaces the single concatenated-bag conjunctive query
  with per-constraint routes, all fused by the existing RRF (`weight /
  (20 + rank)`, rank constant 20):

  | Route | FTS expression | Weight |
  |---|---|---:|
  | Category-only | turn-1 category terms joined by `AND` | 2.0 |
  | Category + material | category terms + requested materials, `AND` | 2.5 |
  | Color + category | category terms + requested colors, `AND` | 2.0 |
  | Exact constraint phrases | each non-generic constraint as a quoted phrase, joined by `OR` | 2.5 |
  | Concatenated bag (fallback only) | every term `AND`-joined, used when no category or usable constraint exists | 2.5 |
  | Adjacent phrase | unchanged | 1.25 |
  | Disjunctive | unchanged | 1.0 exploratory / 2.0 otherwise |

- Generic tokens are identified by catalog document frequency computed at
  index time (`_token_df`, one count per product), never by a hardcoded list.
  A constraint whose every token has `df > 12000` contributes no phrase route:
  measured df values are `imported` 15300, `100` 17397, `closure` 19303,
  `wash` 16134, versus `polyester` 10884, `cotton` 9775, `leather` 7503,
  `zipper` 4166, `grey` 2017.
- The category phrase comes from the turn-1 message via
  `_requested_category()`, the same parse the Task 3 coarse-category features
  use, so no new wording dependency is introduced.

## Task 3 notes

- Protocol `cp4-within-tier-mrr-weighted-v1` (`scripts/train_cp4_reranker.py`,
  full report in `docs/cp4_cv_report.json`). Groups replay public dev 150 +
  synthetic_train 3000 + synthetic_dev 1000 with full tier dominance; 11,502
  groups, 79.18% train-group target coverage, 168,413 weighted pairs.
- Pairs exist only within the target's dominance tier (3a). Each pair is
  weighted by |1/r_target − 1/r_other| under the CP3 reference ranking, ×3
  when the pair crosses the top-10 boundary, ×20 for public sessions (3b/3c).
- Selection = mean of public five-fold out-of-fold TechnicalScore and
  synthetic_dev TechnicalScore, computed statically with the dominance rank
  rule (global rank = higher-tier count + within-tier rank). The untouched
  public holdout 50 and synthetic_holdout were evaluated exactly once, with
  the real evaluator, after freezing the configuration (rows 3.3/3.4).
- Candidate-feature outcomes (3d, selection score vs base16 0.907441 at the
  same C): coarse_category_equality +0.011244 (kept),
  coarse_category_overlap +0.006716 (kept), satisfied_constraints −0.000445
  (dropped: within a tier the count is constant, so pairwise differences are
  zero in dominance mode and it adds nothing beyond feature 15 elsewhere),
  max_matched_rarity −0.000007 (dropped), unmatched_rarity −0.000007
  (dropped), price_presence_when_budget ±0.000000 (dropped),
  preference_tag_overlap −0.001235 (dropped: synthetic profiles carry no
  tags and public tags like "comfort" match most apparel).
- Runtime consequence: `use_fresh_tier_dominance` now defaults to True —
  dominance ordering applies on every turn, which is the ordering the
  training pairs assume (3a). The 18 shipped weights live in
  `starter/reranker_weights.json`; the loader zero-fills the five unselected
  candidate features, keeping the runtime feature vector stable at 23.

## Task 2 notes

- `scripts/create_cp4_synthetic_sets.py` (seed 20260831) generates all 5000
  sessions in one deterministic run from the 34,447 eligible catalog products
  (features/details present, rating_number >= 5, disjoint from all public
  targets), shuffles, and slices `synthetic_train.jsonl` (3000),
  `synthetic_dev.jsonl` (1000), and `synthetic_holdout.jsonl` (1000,
  aggregate-only reporting). `data/releases/` is gitignored, so the freeze is
  the committed script + seed + these SHA256 digests:
  train `768dc75ce302e1b901002298053cd046f19d6d14e37720765ad483fa3e8d33bb`,
  dev `211f6ea169d9017c6a506080ee0568fd278f1b3ac889df3cd168b4d0a5108c01`,
  holdout `d6ae30889da20bd764ac1dcffa197cd632f2a028a201a4597a3a551542c32614`.
- `scripts/create_cp4_hardcase_set.py` selects 500 of 4303 eligible hard
  targets (hard constraint in {cotton, imported}, >100 neighbors satisfying
  every hard constraint jointly, plus >= 2 of: feature/detail string >= 100
  chars, missing description, rating_number <= 50); SHA256
  `5f0fef9772f83206508b314e8894cb8b9863106de0f92257848ef55965f01bcb`.
- Hard-case scenario aggregates (row 2.3): boundary 0.800/0.427/5.48,
  browsing 0.870/0.440/4.16, buying 0.870/0.447/3.45, intent_override
  0.880/0.573/4.96 (HitRate/MRR/MTTC). The synthetic holdout is not evaluated
  until the Task 3 configuration is frozen.

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
