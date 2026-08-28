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
