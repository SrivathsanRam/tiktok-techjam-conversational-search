# CP4 experiments: intent-gated fine-tuned cross-encoder

## Outcome

CP4 keeps CP3's high-recall evidence funnel and adds semantic pairwise ranking
only where it generalizes: specific-buying queries. The selected full-public
result is Hit Rate@10 `0.995`, MRR `0.848067`, MTTC `1.93`, and TechnicalScore
`0.933320`. CP3 scored `0.995`, `0.845734`, `1.93`, and `0.932620` respectively.

The gain is deliberately small and low-risk. Browsing, boundary, and override
queries stay on CP3 because neural fusion reduced their development scores.

## Intuitive technical workflow

Think of the system as a funnel, not one large model:

1. **Conversation state and intent classification.** A deterministic classifier
   labels the current state as specific buying, exploratory browsing,
   constrained browsing, or intent override. It builds a query-derived profile
   containing the active base request, accumulated hard constraints, weak static
   profile tags, and whether an override occurred. Override handling removes the
   stale opener while retaining later valid disclosures.
2. **Catalog preprocessing.** At startup the 50,000 products are loaded once
   into an in-memory SQLite FTS5 index. A second inverted index maps normalized
   feature/detail values and semicolon fragments to product IDs. Product field
   views, prices, and log-popularity are also materialized. WordPiece encodings
   are cached lazily for candidates that reach neural reranking.
3. **Initial recall.** Three field-weighted BM25 routes—conjunctive terms,
   adjacent phrases, and disjunctive terms—are fused with reciprocal-rank fusion
   (RRF). Exact catalog-evidence matches form a precision lane. Their union is
   capped at 80 candidates.
4. **Fast structured reranking.** CP3's 16-feature linear pairwise model scores
   retrieval rank, per-field coverage, exact evidence coverage/rarity,
   material/color/budget compatibility, and popularity.
5. **Fine-tuned cross-encoder.** Only for specific-buying intent, the top 20
   candidates are serialized as field-labelled documents. A 2-layer TinyBERT
   jointly attends to each query-document pair and emits a relevance logit.
   Neural and linear ranks are combined with RRF at neural weight `0.15`.
6. **Multi-turn coverage.** When no new evidence arrives, the already precise
   ordering is retained but previously shown candidates rotate out, exposing
   more of the 80-item pool. Every turn still returns valid catalog IDs.
7. **Failure-safe execution.** Inference is offline through a 4.49 MB QUInt8
   ONNX model. If the asset or dependency is unavailable, model loading returns
   `None` and the unchanged CP3 rank is used.

The cross-encoder is more expressive than cosine dense retrieval at this stage:
instead of encoding query and product independently, it performs token-level
cross-attention over the pair. That is expensive, so it is used only after
lexical recall has reduced 50,000 products to 20 serious candidates.

## Leakage controls and training

- Synthetic targets are sampled from catalog products excluding all 200 public
  target ASINs (SHA-256 source manifest records zero overlap).
- A stable SHA-256 split assigns complete sessions/targets: 787 training and
  213 validation. No target appears in both.
- The actual CP3 funnel mines five hard negatives plus two tail negatives per
  query state. The result is 1,740 training groups / 12,180 pairs and 465
  validation groups.
- Pairwise logistic loss is `softplus(score_negative - score_positive)`.
- TinyBERT was trained for three epochs on CPU with AdamW, LR `3e-5`, batch 32,
  10% warmup, linear decay, gradient clipping 1.0, and max length 160.

| Epoch | Validation group MRR | Pairwise accuracy |
|---:|---:|---:|
| 0 (pretrained) | 0.698262 | 0.827650 |
| 1 | 0.757115 | 0.861751 |
| 2 | 0.769875 | 0.868510 |
| 3 (selected) | 0.774857 | 0.871275 |

A `1e-4` training run was stopped after it failed to complete an epoch within
the available CPU budget; it did not replace the validated checkpoint.

## Ablations and selected values

All public tuning used the fixed 150-session CP2 development partition. The
50-session CP2 holdout was inspected only after freezing the configuration.

| Variant | HR@10 | MRR | MTTC | TechnicalScore | Decision |
|---|---:|---:|---:|---:|---|
| CP3 dev baseline | 0.993333 | 0.843693 | 1.926667 | 0.931241 | reference |
| Cross weight 0.10, buying only | 0.993333 | 0.847360 | 1.926667 | 0.932341 | good |
| Cross weight 0.12, buying only | 0.993333 | 0.846249 | 1.926667 | 0.932008 | reject |
| Cross weight 0.15, buying only | 0.993333 | 0.847915 | 1.926667 | 0.932508 | selected |
| Cross candidates 10 | 0.993333 | 0.846804 | 1.926667 | 0.932174 | reject |
| Cross candidates 30 | 0.993333 | 0.844026 | 1.926667 | 0.931341 | reject |
| Confidence margin 0.25 | 0.993333 | 0.852360 | 1.926667 | 0.933841 | reject: validation overfit |
| Recall pool 120 | 0.993333 | 0.811138 | 1.873333 | 0.922541 | reject |
| Recall pool 240 | 0.993333 | 0.776008 | 1.800000 | 0.913469 | reject |

Browsing neural weights `0.10` and `0.15` reduced browsing MRR from `0.706891`
to `0.704723` and `0.702400` on target-disjoint validation. Constrained browsing
and override fusion also caused regressions, so all three weights are zero.

Increasing initial recall improved MTTC but sharply reduced top-rank precision.
At an already-saturated `0.995` HR@10, this is a bad exchange under the official
score. The cross-encoder cannot repair candidates far below its top-20 input,
and running it over 120–240 items would multiply latency.

A neural confidence gate requiring a `0.25` logit gap looked best on public dev,
but reduced target-disjoint validation from `0.904637` to `0.904003`. Larger
margins were worse. The runtime retains the ablation control, but the selected
default is the ungated `0.0`; this prevents a public-only improvement from being
mistaken for generalization.

## Generalization checks

| Dataset | CP3 score | CP4 score | Delta |
|---|---:|---:|---:|
| Public dev 150 | 0.931241 | 0.932508 | +0.001267 |
| Synthetic target-disjoint validation 213 | 0.902414 | 0.904637 | +0.002223 |
| Public holdout 50 | 0.936757 | 0.935757 | -0.001000 |
| Full public 200 | 0.932620 | 0.933320 | +0.000700 |

The holdout decline is reported rather than tuned away. Its 20 buying cases are
small enough that a single rank swap moves the aggregate visibly. The positive
target-disjoint validation and full-public result support shipping CP4 as a
conservative optional stage, while the CP3 fallback limits operational risk.

On the full 200 cases, specific-buying MRR improves from `0.844583` to
`0.850417`. Browsing remains exactly `0.811627`, intent override remains
`0.888333`, and boundary remains `1.0`; this is the intended effect of the
classifier gate rather than an accidental aggregate trade-off.

## Runtime and feasibility decisions

The selected full evaluation loaded the catalog in `0.57 s`, built the FTS5 and
exact indexes plus ONNX session in `12.75 s`, and evaluated 200 sessions in
`42.19 s` on this Windows CPU. Timings varied with host load, so they are
engineering observations rather than hard guarantees.

A custom C++ SIMD retrieval engine was not justified: the complete selected run
is already under a minute after catalog loading, while neural inference—not
50,000-item Python scanning—is the incremental cost. ONNX Runtime supplies its
own optimized CPU kernels, and candidate/token caches avoid repeated work.

A full dense catalog matrix was also not selected. Exact values originate in
the catalog and BM25 already reaches `0.995` HR@10; adding a dense lane mainly
increases artifact size and cold-start work. The pairwise cross-encoder targets
the demonstrated weakness—ordering semantically close candidates—at 4.49 MB.

## Reproduction

Install runtime dependencies and run the selected full evaluator:

```powershell
python -m pip install -r requirements-cp4.txt
python -m scripts.evaluate_cp4_variant full `
  --output data\releases\cp4\final\full.json
```

The model training pipeline and exact commands are in `fine-tune/README.md`.
Generated datasets/checkpoints are ignored; their counts and hashes are frozen
in `fine-tune/data/manifest.json`. The shipped model hash and provenance are in
`models/cp4-tinybert-reranker/manifest.json`.
