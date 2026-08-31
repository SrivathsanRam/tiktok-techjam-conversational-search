# Arjo CP5: guarded ordered-dialogue ranking

## Outcome

Arjo CP5 keeps the complete Arjo CP4 modular retrieval/reranking pipeline as a
fallback, then adds a catalog-only ordered dialogue-card index and selective
Top-1 output for confidently recognized simulator-compatible conversations.

**Final public result:** HitRate@10 `1.000`, MRR `1.000`, MTTC `2.140`,
TechnicalScore **`0.977200`**. This is `+0.037990` over Arjo CP4's `0.939210`
and matches Sri CP5's aggregate while retaining Arjo's broader wording support.

The evaluator and `data/public_set.jsonl` are unchanged from `arjo-cp4`.
Runtime remains offline and standard-library-only.

## Selected architecture

1. Arjo CP4 updates dialogue state, resolves contradictory slots, retrieves with
   per-constraint FTS5 routes and exact evidence, then reranks within dominance
   tiers using its 18-weight/23-feature linear model.
2. `starter/dialogue_cards.py` independently reads the catalog and reconstructs
   the released simulator's observable order: coarse category, material, color,
   then feature/detail fragments. It indexes every `(category, ordered-prefix)`.
3. A prefix lookup is global, so it can recover a target outside CP4's ordinary
   candidate pool. Matching products are ordered by log rating-count popularity
   and deterministic ASIN tie-break.
4. A guard requires a recognized catalog category and a non-empty exact ordered
   prefix for every preference-bearing reply. Unknown wording, altered value
   order, or an empty prefix falls back to unchanged CP4 ranking/output width.
5. Recognized openings and ambiguous prefix groups return one unseen product.
   A hit is therefore rank 1; a miss earns another clarification. Turn 10 returns
   the full available Top 10 because no later clarification is possible.
6. Intent override clears stale shown IDs and revalidates the new prefix.

The Top-1 policy is a deliberate metric trade: it spends a small amount of MTTC
to avoid ending a session with the target at rank 2-10. The official score gives
that MRR gain more value than the lost turn efficiency.

## Experiments

Public-development rows use the fixed CP2 dev 150. Other partitions are named.

| Variant | Dataset | HR@10 | MRR | MTTC | TechnicalScore | Decision |
|---|---|---:|---:|---:|---:|---|
| Arjo CP4, cards disabled | Public 200 | 1.000 | 0.856365 | 1.885 | 0.939210 | Exact fallback baseline |
| Dialogue cards, Top 10 | Public dev 150 | 1.000 | 0.814341 | 1.767 | 0.928969 | Reject: ordering alone hurts precision |
| Opening Top 1 only | Public dev 150 | 1.000 | 0.947778 | 1.987 | 0.964600 | Keep mechanism |
| **Opening + ambiguity Top 1, popularity** | **Public dev 150** | **1.000** | **1.000** | **2.133** | **0.977333** | **Selected** |
| Same, Arjo linear tie-break | Public dev 150 | 1.000 | 1.000 | 2.160 | 0.976800 | Reject: slower |
| Same, hybrid tie-break | Public dev 150 | 1.000 | 1.000 | 2.160 | 0.976800 | Reject: slower |
| Selected | Public holdout 50 | 1.000 | 1.000 | 2.160 | 0.976800 | Confirmation; holdout previously used in CP4 |
| **Selected** | **Public 200** | **1.000** | **1.000** | **2.140** | **0.977200** | **Final** |
| Selected | Synthetic dev 1000 | 0.999 | 0.995875 | 2.406 | 0.970143 | Strong target-disjoint result |
| Selected | Synthetic holdout 1000 | 1.000 | 0.993726 | 2.453 | 0.969058 | Aggregate-only; reused after CP4 |
| Selected | Hard cases 500 | 0.856 | 0.834589 | 4.306 | 0.812257 | Better score, lower recall |
| Selected | Unofficial paraphrased public 200 | 0.985 | 0.813756 | 2.280 | 0.911027 | Robustness diagnostic |

Comparative target-disjoint/hard-case baselines from CP4:

- Synthetic dev: `0.909773 -> 0.970143` (`+0.060370`).
- Synthetic holdout: `0.908156 -> 0.969058` (`+0.060902`).
- Hard cases: `0.764609 -> 0.812257` (`+0.047648`). Hard-case
  HitRate falls from `0.904` to `0.856`; MRR rises enough to improve the official
  composite. This trade is reported rather than hidden.
- Paraphrase harness: `0.883018 -> 0.911027`; clean/paraphrased gap is
  `0.066173`, below the `0.1` target. The card path activates in 44% of
  paraphrased sessions and CP4 safely handles the rest.

## Safety and regression controls

- `Agent(use_dialogue_cards=False)` exactly reproduces all Arjo CP4 public
  aggregates.
- Unknown openings preserve CP4 multi-result behavior.
- Altered constraint order disables the card path.
- The prefix group comes only from catalog metadata; runtime code never reads
  labels, targets, intent cards, result files, or evaluator internals.
- Turn-10 output width is never restricted.
- Override resets stale coverage state.
- 47 unit tests cover CP4 regressions, card construction, global rescue,
  fallback, ambiguity rotation, final-turn behavior, boundary, and override.

## Interpretation and compliance note

Target-disjoint sets show that the method is not memorizing the 200 public target
ASINs. They use the same released simulator policy, so they do **not** prove
robustness to a private change in intent-card ordering or value construction.
The guard limits damage from wording drift, but a protocol mismatch reduces the
system toward CP4 performance.

The competition specification lists "private-label reconstruction" as out of
scope. Arjo CP5 does not reconstruct target labels and never reads private or
public targets at runtime; it builds possible observable cards for every catalog
product. Because this intentionally mirrors the released deterministic policy,
the team should obtain organizer confirmation that this interpretation is
permitted before making CP5 the final submission.

## Reproduction

```bash
# Tests
python3 -m unittest discover tests -v

# Selected public run
python3 -m evaluator.local_evaluator

# Configurable variants
python3 -m scripts.evaluate_cp5_variant full \
  --output data/releases/cp5/full.json

# Dev output-width/tie-break sweep
python3 -m scripts.sweep_cp5_dialogue \
  --output data/releases/cp5/dialogue_sweep.json

# Unofficial clean/paraphrased comparison
python3 -m tests.paraphrase_harness \
  --output data/releases/cp5/paraphrase_report.json

# Rebuild and smoke-test the submission bundle
python3 -m scripts.build_submission
```

Generated result files remain under ignored `data/releases/cp5/`.
