# Retrieval experiment ledger

All branches began at commit `3407835` on `sri-experiment`. The official evaluator and public labels were not modified. Full local outputs and detailed notes are retained under the ignored `data/releases/experiments/` directory so switching branches does not overwrite evidence.

| Experiment | Branch | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Untouched starter | `sri-experiment` | 0.1250 | 0.068034 | 9.810 | 0.1190 | 0.106710 | Control |
| Stateful lexical | `codex/exp-stateful-lexical` | 0.8550 | 0.521964 | 3.630 | 0.7370 | 0.731489 | Keep state policy |
| Multi-route RRF | `codex/exp-multiroute-rrf` | 0.8850 | 0.587512 | 3.245 | 0.7755 | 0.773854 | Keep retrieval routes |
| Structured graph | `codex/exp-structured-graph` | 0.8800 | 0.563599 | 3.380 | 0.7620 | 0.761480 | Reject Python representation |
| Generalized hybrid | `codex/exp-generalized-hybrid` | 0.9400 | 0.638409 | 2.840 | 0.8160 | 0.824723 | Control for popularity prior |
| Intent-adaptive popularity | `codex/exp-popularity-prior` | **0.9500** | **0.650099** | **2.710** | **0.8290** | **0.835830** | Selected |

## Selected design

The selected branch is fully offline and uses only the Python standard library and SQLite FTS5:

1. Return Top 10 recommendations and ask one open constraint question on the same turn.
2. Accumulate separately disclosed constraints across turns.
3. On an intent override, remove the stale opening preference while preserving later hard-constraint disclosures.
4. Retrieve through conjunctive, adjacent-phrase, and disjunctive lexical routes.
5. Fuse routes with weighted reciprocal-rank fusion.
6. Use precision-oriented fusion for explicitly exploratory sessions and recall-oriented fusion for specific/override sessions.
7. Add a log-scaled rating-count popularity route only for non-exploratory requests, where purchase likelihood is a relevant soft prior.

No runtime code reads public labels or target IDs. No network, API key, external model, catalog mutation, or mock ID is required.

## Generalization safeguards

- Changes were kept only when they improved the whole public set or fixed a contract-level state bug.
- The four official scenario groups are reported separately; the selected branch scores HitRate `1.00` Boundary, `0.9625` Browsing, `0.95` Buying, and `0.90` Intent Override.
- Fusion weights encode route/intent priors rather than sample-specific exceptions.
- No learned ranker was fit to 200 labels, avoiding a high-variance training step.
- The public failure analyzer is read-only and is not imported by the runtime agent.

## Rejected or bounded ideas

- A Python product-token graph improved the stateful control but scored below lexical RRF and peaked around 691 MB with roughly 145 seconds end-to-end. Its object-heavy representation is unsuitable for the final default.
- C++ could compact graph edges or accelerate exact scans, but profiling shows the best-quality path is already FTS5-based. Adding C++ now would increase build risk without demonstrated score benefit.
- Sharper RRF (`k=5`) preserved recall but reduced MRR.
- Increasing the non-exploratory disjunctive weight from `2.0` to `2.5` did not add hits and slightly reduced MRR.
- Applying the popularity prior globally hurt Browsing MRR. Restricting it to non-exploratory requests preserved Browsing exactly while improving Buying. Popularity weights `0.25`, `0.5`, `0.75`, and `1.0` were bounded; `1.0` was selected because recall is the primary objective, while `0.75` had higher MRR but one fewer hit.
- A mandatory dense model was not introduced because the environment has no such dependency and the core must remain offline/reproducible. Dense retrieval remains a future candidate only if model artifacts and hardware limits are explicitly fixed.

## Reproduction

On this Windows checkout, `python3` is not registered; use the equivalent repository virtual environment:

```powershell
.\venv\Scripts\python.exe -m evaluator.local_evaluator
```

To inspect route-level ranks for public misses without modifying the evaluator:

```powershell
.\venv\Scripts\python.exe -m scripts.analyze_public_failures --results results.json --output failure-analysis.json
```
