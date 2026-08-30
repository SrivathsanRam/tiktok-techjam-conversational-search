# CP3 evidence-funnel experiments

The `sri-experiment-cp3` branch starts directly from `main` at `b8403e6` and
imports the reproducible CP2 commits as its controlled baseline. Runtime code
uses only Python's standard library and SQLite FTS5. NumPy and scikit-learn are
training-only dependencies.

## Leakage and validation policy

- The inherited CP2 `150 dev / 50 holdout` manifest is retained.
- CP3 values were selected on the 150 development IDs and five-fold
  out-of-fold metrics.
- A deterministic 1,000-session supplementary set samples catalog products
  disjoint from every public target, proportional to `rating_number`, and uses
  the official 40/40/15/5 scenario mix. It is generated, not committed.
- Runtime indexes use catalog content only. Runtime code never reads targets,
  labels, the split manifest, or generated sessions.
- The selected configuration was evaluated on the 50-ID holdout only after
  candidate-pool and rotation choices were frozen. An initial auxiliary runner
  invocation accidentally retained an earlier `df=100` default; both its
  `0.934357` aggregate and the corrected selected-config `0.936757` aggregate
  are retained under the ignored experiment directory.

## Selected architecture

CP3 keeps CP2's stateful BM25/RRF and learned reranking, then adds:

1. An in-memory SQLite exact-value index over complete feature/detail values,
   semicolon-delimited fragments, material aliases, and color aliases.
2. Exact evidence candidate ranking by number of matched values, value rarity
   (IDF), catalog popularity, and deterministic ASIN tie-breaking.
3. Two learned features: exact-evidence coverage and normalized rarity.
4. A multi-lane candidate pool: up to 60 exact candidates plus enough CP2
   sparse candidates to rerank 80 unique products.
5. Coverage rotation when a reply adds no evidence. Since the preceding Top 10
   is then known to be wrong, CP3 presents the next unseen ranked products.
6. A 16-feature pairwise logistic reranker fitted on target-versus-top-80 hard
   negatives, with five-fold selection over `C={0.01,0.05,0.1,0.5,1.0}`.

## Results

| System / ablation | Partition | HitRate@10 | MRR | MTTC | TechnicalScore | Decision |
|---|---|---:|---:|---:|---:|---|
| Main BM25 starter | Public 200 | 0.1250 | 0.068034 | 9.810 | 0.106710 | Reference |
| CP2 learned reranker | Public 200 | 0.9900 | 0.790135 | 1.985 | 0.912340 | Controlled baseline |
| CP2 learned reranker | Synthetic 1000 | 0.9750 | 0.705398 | 2.184 | 0.875439 | Controlled generalization baseline |
| Raw exact lane + CP2 weights | Dev 150 | 0.9933 | 0.822841 | 2.033 | 0.922852 | Keep exact lane |
| Exact lane + 16-feature ranker, pool 60 | Dev 150 | 0.9933 | 0.851222 | 2.033 | 0.931366 | Strong precision control |
| Exact lane + ranker, pool 60 | Synthetic 1000 | 0.9620 | 0.783037 | 2.444 | 0.887031 | Higher MRR, lower recall |
| Multi-lane pool 120 | Synthetic 1000 | 0.9850 | 0.712489 | 2.059 | 0.885067 | Reject: recall gain did not offset MRR |
| **Multi-lane pool 80** | **Dev 150** | **0.9933** | **0.843693** | **1.927** | **0.931241** | **Select pool size** |
| Multi-lane pool 80 | Synthetic 1000 | 0.9850 | 0.749098 | 2.130 | 0.894629 | Keep |
| Pool 80 + rotation head 3 | Synthetic 1000 | 0.9950 | 0.750357 | 2.065 | 0.901307 | Keep rotation |
| **Pool 80 + full rotation** | **Synthetic 1000** | **0.9950** | **0.751482** | **2.063** | **0.901685** | **Selected** |
| Selected CP3 | Holdout 50 | 1.0000 | 0.851857 | 1.940 | 0.936757 | Final aggregate |
| **Selected CP3** | **Public 200** | **0.9950** | **0.845734** | **1.930** | **0.932620** | **Final** |

The selected five-fold out-of-fold score is `0.931017` with `C=0.5`; its fold
scores are `0.953548`, `0.965807`, `0.908667`, `0.937586`, and `0.886293`.

## Value sweeps and rejected ideas

### Exact evidence document-frequency cutoff

Single-value cutoffs of `10`, `100`, `500`, `2000`, and `50000` were tested.
The corresponding early exact-intersection dev scores were `0.920160`,
`0.921494`, `0.919893`, `0.920760`, and `0.922738`. The fixed catalog has
50,000 products, so `50000` is selected: learned ranking can use broad material
evidence without treating it as a hard filter.

### Exact candidate limit and total rerank pool

Exact limits `5`, `10`, `20`, `40`, and `60` were tested. Small limits excluded
targets that shared common apparel composition strings; `60` performed best.
Total pools `60`, `80`, and `120` exposed the precision/recall tradeoff. Pool
`80` gave the best five-fold and target-disjoint balance.

### Category sharding

A hard maximum category-overlap gate scored `0.916486` on dev and was rejected.
Catalog category strings are coarse and inconsistent; BM25 category weighting
remains useful, but category is not a safe hard filter.

### Dense retrieval and neural reranking

Dense retrieval is technically feasible: a 50,000 x 384 float16 matrix is about
38.4 MB and direct dot products avoid a hosted vector database. It was not made
the CP3 default because official messages and constraints are deterministically
derived from exact catalog text, the runtime currently has no ONNX/PyTorch
dependency, and CP2's already implemented 4.52 MB quantized TinyBERT experiment
was 2.5-3x slower while reducing dev score to `0.906180`. Exact evidence raised
the final score more while preserving an offline standard-library runtime.

No neural fine-tuning is required, so no `fine-tune/` directory is created. The
only fitted artifact is the small JSON file of 16 linear weights, trained
locally on CPU.

## Reproduction

```powershell
# Tests
.\venv\Scripts\python.exe -m unittest discover -v

# Generate target-disjoint supplementary validation sessions
.\venv\Scripts\python.exe -m scripts.create_cp3_synthetic_set `
  --count 1000 `
  --output data\releases\cp3\synthetic-1000.jsonl

# Evaluate the selected runtime
.\venv\Scripts\python.exe -m evaluator.local_evaluator

# Optional CPU retraining (NumPy and scikit-learn are training-only)
& 'C:\Program Files\Python313\python.exe' -m scripts.train_cp3_reranker `
  --report-output data\releases\cp3\final\cv_report.json
```
