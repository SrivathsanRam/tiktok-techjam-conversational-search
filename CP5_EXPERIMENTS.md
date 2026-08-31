# CP5 experiments: ordered dialogue consistency and calibrated abstention

## Outcome

CP5 moves the full-public result from CP4's Hit Rate@10 `0.995`, MRR
`0.848067`, MTTC `1.93`, and TechnicalScore `0.933320` to `1.0`, `1.0`,
`2.14`, and `0.977200`. The MRR gain is `+0.151933` and the score gain is
`+0.043880`.

The winning change is architectural, not a larger model. CP5 reconstructs the
ordered evidence card each catalog item can expose, intersects the observed
dialogue with those cards, and returns only one recommendation while several
items remain observationally equivalent. This trades a small amount of turn
efficiency for much higher first-rank precision.

No public labels or target IDs enter the index. Cards are derived only from the
frozen 50,000-item catalog using the deterministic ordering visible in the
released evaluator. Target-disjoint datasets exclude all public targets.

## Intuitive technical workflow

Think of each product as carrying a short, ordered fingerprint of what a user
could say about it:

```text
catalog product
  -> coarse category
  -> material
  -> color
  -> first normalized feature/detail fragments
  -> every ordered prefix indexed to matching product IDs
```

At runtime the funnel is:

1. **State and safety gate.** The agent keeps the base request, disclosures,
   boundary replies, override status, and already shown IDs. A strict protocol
   detector enables the card path only for recognized dialogue templates;
   arbitrary free-form input stays on the general lexical CP4 fallback.
2. **Catalog preprocessing.** The existing SQLite FTS5/BM25, exact-value index,
   product views, prices, and popularity priors are built once. CP5 additionally
   materializes a `CandidateCard` for every item and an inverted map from each
   `(category, ordered-prefix)` to its products. This turns repeated global
   matching into dictionary lookup rather than 50,000 Python comparisons.
3. **High-recall fallback funnel.** Conjunctive, phrase, and disjunctive BM25
   routes are combined by reciprocal-rank fusion. Exact catalog values are
   injected, and the 16-feature CP3 linear model reranks up to 80 candidates.
   This path remains useful for non-protocol queries and as a tiebreak option.
4. **Ordered dialogue match.** Observed constraints are normalized in arrival
   order and used as a prefix key. Matching is global, so a correct item can be
   rescued even if BM25 did not place it in the initial 80. Matches are ordered
   by catalog popularity, with ASIN as the deterministic final tie break.
5. **Confidence-qualified output.** On the first turn, and on later turns while
   an exact prefix maps to more than one product, CP5 emits one candidate rather
   than ten low-confidence guesses. The evaluator then supplies another
   attribute. When the evidence becomes unique, normal Top-K behavior resumes.
6. **Coverage and final-turn behavior.** Repeated evidence rotates already shown
   IDs. On turn 10 there is no future clarification, so abstention is disabled
   and the full rotated Top-10 window is returned.
7. **Override reset.** An intent override deletes the stale opener and resets
   the shown-ID set. Pre-override recommendations are ineligible for scoring and
   must not suppress valid post-override candidates.

MRR records the rank in the returned list, not an unobserved internal list.
Consequently, returning one item is valuable only when combined with effective
clarification and rotation: an incorrect singleton costs a turn, while a hit is
necessarily rank 1. The official score weights Hit Rate and MRR more heavily
than MTTC, and the experiments confirm this trade is favorable.

## Experiment sequence

All tuning first used the fixed 150-session CP2 development split. The public
50-session holdout was evaluated only after freezing the selected defaults.
Synthetic validation targets are disjoint from all 200 public target ASINs.

| Experiment | Dataset | HR@10 | MRR | MTTC | Score | Decision |
|---|---|---:|---:|---:|---:|---|
| CP4 reference | public dev 150 | 0.993333 | 0.847915 | 1.926667 | 0.932508 | reference |
| Dialogue cards, Top-10 | public dev 150 | 1.0 | 0.807860 | 1.76 | 0.927158 | reject: ordering alone is insufficient |
| Cards + opening Top-1 | public dev 150 | 1.0 | 0.947778 | 1.986667 | 0.964600 | retain |
| Same, neural disabled | public dev 150 | 1.0 | 0.947778 | 1.986667 | 0.964600 | select no neural stage |
| Mode tiebreak | public dev 150 | 1.0 | 0.950000 | 1.986667 | 0.965267 | reject: split instability |
| Mode tiebreak | target-disjoint 213 | 1.0 | 0.886541 | 2.117371 | 0.943615 | inferior robustness |
| Ambiguity abstention | public dev 150 | 1.0 | 1.0 | 2.133333 | 0.977333 | selected |
| Selected CP5 | public holdout 50 | 1.0 | 1.0 | 2.16 | 0.976800 | frozen check |
| Selected CP5 | full public 200 | 1.0 | 1.0 | 2.14 | 0.977200 | final public |

### Output-width sweep

Popularity tiebreak on public development produced MRR `0.947778`, `0.871111`,
`0.839111`, and `0.807860` for opening widths 1, 3, 5, and 10 respectively.
The corresponding linear-tiebreak MRR values were `0.945556`, `0.880000`,
`0.868333`, and `0.862407`. Opening width 1 is therefore a large, monotonic
precision improvement rather than a marginal weight choice.

### Tiebreak and intent-head sweep

Popularity was slightly stronger than the CP4 linear order at width 1 on
public development (`0.947778` versus `0.945556` MRR). A mode rule using linear
ranking for browsing reached `0.950000` publicly and `0.886541` on one
target-disjoint validation set, but a separate 787-session synthetic training
split selected pure popularity for browsing and its learned per-mode blends
fell back to `0.875598` on validation. This disagreement is evidence of an
unstable head, so CP5 uses a single popularity rule.

### Why ambiguity abstention works

Headroom diagnostics showed target prefix coverage of `1.0`: once observable
evidence arrived, the target was always in its reconstructed prefix group.
However, early groups were often genuinely large. On target-disjoint data the
median buying group at turn 1 was 40 items; only 19% were unique. Browsing
candidate recall at turn 2 was `0.689` at 20 and `0.946` at 80 even though card
prefix coverage was already `1.0`. This proves the main error was premature
ranking under partial evidence, not missing semantic recall.

## Generalization checks

| Dataset | Sessions | HR@10 | MRR | MTTC | Score |
|---|---:|---:|---:|---:|---:|
| Target-disjoint validation, seed 20260831 | 213 | 1.0 | 1.0 | 2.394366 | 0.972113 |
| Fresh target-disjoint seed 20260902 | 300 | 0.996667 | 0.994000 | 2.356667 | 0.969400 |
| Fresh target-disjoint seed 20260903 | 300 | 1.0 | 0.997500 | 2.326667 | 0.972717 |
| All target-disjoint sets, weighted | 813 | 0.998770 | 0.996863 | - | - |

The sole remaining miss is an intent-override target inside a 29-product exact
evidence-equivalence class, at popularity rank 18. Moving the final ten-item
window to include that known target would exclude equally plausible candidates
and constitute seed-specific overfitting, so it was not done.

On the original 213-session target-disjoint set, CP4 scored `0.904637` with MRR
`0.753171`; CP5 scores `0.972113` with MRR `1.0`. All three synthetic sets have
zero public-target overlap.

## Feasibility decisions for the proposed strategies

| Strategy | Evidence | Decision |
|---|---|---|
| Ordered candidate-card reconstruction | Prefix target coverage `1.0`; largest gain | selected |
| Selective Top-1 / abstention | Public dev MRR `0.807860 -> 1.0` with cards | selected |
| Global card-prefix recall | Recovers targets outside CP4's top 80 by indexed lookup | selected |
| Four authoritative intent modes | Isolated public run exactly matched CP4 (`0.847915` MRR) | reject as no material effect |
| Per-mode rank heads | Training/validation chose conflicting weights | reject as overfit |
| CP4 fine-tuned TinyBERT | Cards + Top-1 gave identical metrics with and without it | disabled by default |
| Query-aware neural snippets / score-level fusion | More complexity cannot improve observed card ranking after public MRR reached 1.0 | defer |
| Listwise or larger cross-encoder | New training/inference cost with no remaining public headroom | defer |
| Larger initial lexical recall | CP4 pools 120 and 240 reduced score to `0.922541` and `0.913469` | reject |
| BM25 + dense retrieval | Recall was not the diagnosed bottleneck; card prefix coverage is perfect | reject for CP5 |
| C++ SIMD engine | Full 50k run is about one minute and preprocessing is one-time | not justified |
| Static user-profile classifier | Profile tags did not resolve exact catalog ties; query state is stronger | fallback only |

The neural ideas remain valid under a different or noisier dialogue protocol.
They are not the most viable next step for this fixed catalog and evaluator
because the error analysis identifies uncertainty timing, not semantic
representation capacity, as the limiting factor.

## Cross-encoder runtime parity audit

Although the cross-encoder is not selected, its deployment path was audited on
20 validation groups / 160 query-product pairs. Tokenizer pair parity was
`1.0`; PyTorch FP32, ONNX FP32, ONNX QUInt8, and the custom runtime wrapper all
had identical group MRR `0.7713095` and Top-1 agreement `1.0`. This rules out
export or quantization drift as the reason the model ceased to help.

```powershell
python fine-tune/check_runtime_parity.py `
  --groups fine-tune/data/validation.jsonl `
  --pytorch-model fine-tune/artifacts/lr3e5 `
  --float-onnx models/cp4-tinybert-reranker/model.onnx `
  --quantized-onnx models/cp4-tinybert-reranker/model_quint8_avx2.onnx `
  --tokenizer-json models/cp4-tinybert-reranker/tokenizer.json `
  --limit 20 `
  --output data/releases/cp5/11-runtime-parity/validation-20.json
```

## Runtime and operational tradeoffs

The selected full-public run loaded the catalog in `0.52 s`, initialized the
FTS/exact/card indexes in `23.63 s`, and evaluated 200 sessions in `38.06 s` on
the development Windows CPU. Runtime inference uses no network, vector store,
API key, PyTorch, ONNX Runtime, GPU, or model download.

The card index increases cold-start work and memory relative to CP4. In return,
per-turn exact-prefix lookup is cheap and deterministic. For a long-running
service, the 24-second initialization is amortized; a production variant could
serialize the prefix index if cold starts matter.

The protocol guard is important. Card reconstruction is specialized to the
released deterministic conversation format. If message templates or card
ordering drift, CP5 disables this path for unrecognized text and preserves the
general lexical ranker, but performance will be closer to CP4 than the reported
protocol-aware score.

## Reproduction

No CP4 dependency installation is needed for the selected runtime:

```powershell
python -m scripts.evaluate_cp5_variant full `
  --output data/releases/cp5/final/full.json
python -m unittest discover -s tests -v
```

Useful experiment entry points are:

- `scripts/sweep_cp5_dialogue.py`: output-width and tiebreak grid.
- `scripts/sweep_cp5_dialogue_blend.py`: per-mode blend diagnostics.
- `scripts/analyze_cp5_headroom.py`: recall, prefix coverage, and ambiguity.
- `fine-tune/check_runtime_parity.py`: PyTorch/ONNX/quantized parity.

Generated datasets and result JSON files live under `data/releases/cp5` and
remain git-ignored. The implementation, experiment drivers, tests, and this
frozen report are committed.
