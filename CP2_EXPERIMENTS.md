# CP2 reranking protocol

The cp2 experiments start from commit `07e5f1c` on `sri-experiment`.

## Leakage boundary

The runtime `Agent` may build indexes only from `data/catalog.jsonl`. Runtime decisions may use only the `user_profile` supplied to `reset` and messages supplied to `respond`.

The deterministic split manifest is generated from `sample_id` and `scenario_type` only. It contains no targets, intent cards, or simulator-derived constraints. New cp2 tuning uses only the 150 development IDs. The 50 holdout targets must not be inspected; holdout evaluation artifacts are aggregate-only.

The inherited agent was previously selected using the full 200-session public metric. Therefore the cp2 holdout is clean for comparing new reranking changes against the frozen inherited baseline, but it is not a historically pristine estimate of the inherited system itself.

## Reproduction

```powershell
.\venv\Scripts\python.exe -m scripts.create_cp2_split
.\venv\Scripts\python.exe -m scripts.evaluate_cp2_split dev --output data\releases\cp2\baseline-dev\results.json
.\venv\Scripts\python.exe -m scripts.evaluate_cp2_split holdout --output data\releases\cp2\baseline-holdout\aggregate.json
```

Five development folds are available as `fold-0` through `fold-4`. Their complementary training partitions are `train-0` through `train-4`.

## Results

| System | Partition | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Frozen inherited agent | Dev 150 | 0.946667 | 0.654402 | 2.720000 | 0.828000 | 0.835254 | Baseline |
| Frozen inherited agent | Holdout 50 | 0.960000 | 0.637190 | 2.680000 | 0.832000 | 0.837557 | Aggregate-only baseline |
| Structured compatibility | Dev 150 | 0.946667 | 0.654476 | 2.720000 | 0.828000 | 0.835276 | Rejected: neutral |
| Learned pairwise reranker | 5-fold out-of-fold | 0.986667 | 0.781638 | 2.000000 | 0.900000 | 0.907825 | Selected on dev |
| Learned pairwise reranker | Dev 150, final weights | 0.986667 | 0.779497 | 1.986667 | 0.901333 | 0.907449 | Finalist |
| Learned pairwise reranker | Holdout 50 | 1.000000 | 0.822048 | 1.980000 | 0.902000 | 0.927014 | Selected; single holdout run |
| TinyBERT cross-encoder | Dev 150 | 0.986667 | 0.775267 | 1.986667 | 0.901333 | 0.906180 | Rejected: slower and lower MRR |
| **Merged cp2 agent** | **Full public 200** | **0.990000** | **0.790135** | **1.985000** | **0.901500** | **0.912340** | **Merged** |

The five learned-reranker fold TechnicalScores were `0.936866`, `0.915000`, `0.899167`, `0.900776`, and `0.885115`. The gain was therefore not confined to one development fold.

## Merged reranker

The merged runtime reranks the inherited top-60 candidate union using a 14-feature linear model. Features are computed jointly from accumulated conversation state and catalog product data:

- inherited retrieval rank;
- overall, title, category, attribute, and description token coverage;
- distilled constraint coverage and exact-phrase compatibility;
- material, color, and explicit budget compatibility;
- log-scaled catalog popularity with intent interactions.

Training uses pairwise target-versus-hard-negative differences from the 150 development sessions. Five-fold cross-validation selects the regularization strength, after which the final 14 weights are fitted on all 150 development IDs. Only numeric weights are committed; labeled examples, targets, and candidate groups are never persisted.

Runtime inference uses only the standard library and SQLite. NumPy and scikit-learn are training-only dependencies.

## Cross-encoder result

The separate `codex/cp2-cross-encoder` branch vendors the official 4.52 MB AVX2-quantized ONNX export of `cross-encoder/ms-marco-TinyBERT-L2-v2`, a minimal WordPiece tokenizer, model checksum tests, and a learned-ranker fallback. Neural fusion weights `0.35` and `0.10` both reduced dev MRR, and full dev evaluation was roughly 2.5-3 times slower. It was therefore not merged.

## Resource bottlenecks

- Final full-set evaluation peaked near 431 MB working set on this machine. The main cost is retaining normalized catalog fields for top-60 feature extraction in addition to the evaluator's catalog objects and the FTS5 index.
- Offline model training peaked near 562 MB and took several minutes. This does not affect runtime submission latency.
- A future systems pass should replace the Python `parent_asin -> tuple[str, ...]` product-view dictionary with compact row IDs plus batched SQLite/array-backed field access. This should be benchmarked carefully because quality is now dominated by reranking, not retrieval throughput.

## Final reproduction

```powershell
# Standard-library runtime evaluation
.\venv\Scripts\python.exe -m evaluator.local_evaluator

# Optional retraining (requires NumPy and scikit-learn)
& 'C:\Program Files\Python313\python.exe' -m scripts.train_cp2_reranker `
  --report-output data\releases\cp2\02-learned\training\cv_report.json
```
