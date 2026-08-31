# Experiment Ledger

This file consolidates every experiment checkpoint (CP1–CP6) that produced the
current `main` agent, plus the parallel work on the unmerged `arjo-cp4`,
`arjo-cp5`, and `cody-branch` branches. The official evaluator and public
labels were never modified. Full local outputs are retained under the ignored
`data/releases/` directory so switching branches does not overwrite evidence.

All scores use the official formula:

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency     = clip((11 − MTTC) / 10, 0, 1)
```

## Score progression (full public 200)

| Checkpoint | Branch | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---:|---:|---:|---:|
| Weak BM25 starter | upstream `main` | 0.1250 | 0.068034 | 9.810 | 0.106710 |
| CP1: intent-adaptive hybrid retrieval | `sri-experiment` | 0.9500 | 0.650099 | 2.710 | 0.835830 |
| CP2: learned pairwise reranker | `sri-experiement-cp2` | 0.9900 | 0.790135 | 1.985 | 0.912340 |
| CP3: exact-evidence funnel + rotation | `sri-experiment-cp3` | 0.9950 | 0.845734 | 1.930 | 0.932620 |
| CP4: intent-gated TinyBERT cross-encoder | `sri-experiment-cp4` | 0.9950 | 0.848067 | 1.930 | 0.933320 |
| CP5: ordered dialogue cards + abstention | `sri-experiment-cp5` | 1.0000 | 1.000000 | 2.140 | 0.977200 |
| CP6: exact category-scoped retrieval | `sri-experiment-cp6` → `main` | 1.0000 | 1.000000 | 2.100 | **0.978000** |

Parallel unmerged results (details in [Unmerged branch experiments](#unmerged-branch-experiments)):

| Line | Branch | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---:|---:|---:|---:|
| Arjo CP4: dominance tiers + retrained reranker | `arjo-cp4` | 1.0000 | 0.856365 | 1.885 | 0.939210 |
| Arjo CP5: guarded dialogue cards over Arjo CP4 | `arjo-cp5` | 1.0000 | 1.000000 | 2.140 | 0.977200 |
| Cody: synonym-expansion reranker | `cody-branch` | — | — | — | reverted from `main` |

## Validation protocol (all checkpoints)

- The runtime `Agent` builds indexes only from `data/catalog.jsonl`. Runtime
  decisions use only the `user_profile` supplied to `reset` and the messages
  supplied to `respond`. Runtime code never reads targets, labels, intent
  cards, the split manifest, or evaluator internals.
- From CP2 onward, a deterministic target-blind manifest (`data/cp2_split.json`,
  generated from `sample_id` and `scenario_type` only) splits the public set
  into 150 development and 50 holdout sessions. All tuning used the 150
  development IDs and five-fold out-of-fold metrics; the holdout was evaluated
  once per checkpoint, aggregate-only, after freezing each configuration.
- From CP3 onward, deterministic **target-disjoint** synthetic session sets
  (seeded, excluding all 200 public target ASINs, official 40/40/15/5 scenario
  mix) provided generalization checks that cannot leak public labels.
- Changes were kept only when they improved the whole development partition or
  fixed a contract-level state bug; anything that lowered the score was
  reverted and recorded below.

---

## CP1: retrieval experiments (`sri-experiment`)

All CP1 branches began at commit `3407835`. Each experiment was a full public
200-session evaluation.

| Experiment | Branch | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Untouched starter | `sri-experiment` | 0.1250 | 0.068034 | 9.810 | 0.1190 | 0.106710 | Control |
| Stateful lexical | `codex/exp-stateful-lexical` | 0.8550 | 0.521964 | 3.630 | 0.7370 | 0.731489 | Keep state policy |
| Multi-route RRF | `codex/exp-multiroute-rrf` | 0.8850 | 0.587512 | 3.245 | 0.7755 | 0.773854 | Keep retrieval routes |
| Structured graph | `codex/exp-structured-graph` | 0.8800 | 0.563599 | 3.380 | 0.7620 | 0.761480 | Reject Python representation |
| Generalized hybrid | `codex/exp-generalized-hybrid` | 0.9400 | 0.638409 | 2.840 | 0.8160 | 0.824723 | Control for popularity prior |
| Intent-adaptive popularity | `codex/exp-popularity-prior` | **0.9500** | **0.650099** | **2.710** | **0.8290** | **0.835830** | Selected |

### Selected design

Fully offline, Python standard library plus SQLite FTS5:

1. Return Top 10 recommendations and ask one open constraint question on the
   same turn.
2. Accumulate separately disclosed constraints across turns; on an intent
   override, remove the stale opening preference while preserving later
   hard-constraint disclosures.
3. Retrieve through conjunctive, adjacent-phrase, and disjunctive lexical
   routes, fused with weighted reciprocal-rank fusion (RRF).
4. Use precision-oriented fusion for explicitly exploratory sessions and
   recall-oriented fusion for specific/override sessions.
5. Add a log-scaled rating-count popularity route only for non-exploratory
   requests, where purchase likelihood is a relevant soft prior.

Scenario breakdown of the selected branch: HitRate `1.00` Boundary, `0.9625`
Browsing, `0.95` Buying, `0.90` Intent Override. Fusion weights encode
route/intent priors rather than sample-specific exceptions; no learned ranker
was fit to 200 labels at this stage.

### Rejected or bounded ideas

- A Python product-token graph improved the stateful control but scored below
  lexical RRF and peaked around 691 MB / ~145 s end-to-end. Rejected.
- Sharper RRF (`k=5`) preserved recall but reduced MRR.
- Raising the non-exploratory disjunctive weight from `2.0` to `2.5` added no
  hits and slightly reduced MRR.
- Applying the popularity prior globally hurt Browsing MRR; restricting it to
  non-exploratory requests preserved Browsing exactly while improving Buying.
  Popularity weights `0.25`–`1.0` were swept; `1.0` was selected because recall
  is the primary objective (`0.75` had higher MRR but one fewer hit).
- A mandatory dense model was rejected to keep the core offline/reproducible.

---

## CP2: learned pairwise reranking

CP2 started from commit `07e5f1c`. The inherited agent had been selected using
the full 200-session public metric, so the new 150/50 split is clean for
comparing CP2 changes against the frozen baseline, but not a historically
pristine estimate of the inherited system itself.

| System | Partition | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Frozen inherited agent | Dev 150 | 0.946667 | 0.654402 | 2.720000 | 0.828000 | 0.835254 | Baseline |
| Frozen inherited agent | Holdout 50 | 0.960000 | 0.637190 | 2.680000 | 0.832000 | 0.837557 | Aggregate-only baseline |
| Structured compatibility | Dev 150 | 0.946667 | 0.654476 | 2.720000 | 0.828000 | 0.835276 | Rejected: neutral |
| Learned pairwise reranker | 5-fold out-of-fold | 0.986667 | 0.781638 | 2.000000 | 0.900000 | 0.907825 | Selected on dev |
| Learned pairwise reranker | Dev 150, final weights | 0.986667 | 0.779497 | 1.986667 | 0.901333 | 0.907449 | Finalist |
| Learned pairwise reranker | Holdout 50 | 1.000000 | 0.822048 | 1.980000 | 0.902000 | 0.927014 | Selected; single holdout run |
| TinyBERT cross-encoder | Dev 150 | 0.986667 | 0.775267 | 1.986667 | 0.901333 | 0.906180 | Rejected: slower and lower MRR |
| **Merged CP2 agent** | **Full public 200** | **0.990000** | **0.790135** | **1.985000** | **0.901500** | **0.912340** | **Merged** |

Fold TechnicalScores were `0.936866`, `0.915000`, `0.899167`, `0.900776`, and
`0.885115`, so the gain was not confined to one fold.

### Merged reranker

The runtime reranks the inherited top-60 candidate union with a 14-feature
linear model over conversation state and catalog data: inherited retrieval
rank; overall/title/category/attribute/description token coverage; distilled
constraint coverage and exact-phrase compatibility; material, color, and
explicit budget compatibility; and log-scaled popularity with intent
interactions. Training uses pairwise target-versus-hard-negative differences
from the 150 development sessions; five-fold CV selects the regularization
strength before final fitting. Only numeric weights are committed. NumPy and
scikit-learn are training-only; runtime inference is standard library plus
SQLite.

### Rejected: vendored cross-encoder

`codex/cp2-cross-encoder` vendored the official 4.52 MB AVX2-quantized ONNX
export of `cross-encoder/ms-marco-TinyBERT-L2-v2` with a minimal WordPiece
tokenizer and learned-ranker fallback. Neural fusion weights `0.35` and `0.10`
both reduced dev MRR, and full dev evaluation was 2.5–3× slower. Not merged.

### Resource notes

Full-set evaluation peaked near 431 MB (normalized catalog fields for top-60
feature extraction plus the evaluator's catalog objects and the FTS5 index).
Offline training peaked near 562 MB over several minutes; this does not affect
runtime latency.

---

## CP3: exact-evidence funnel and coverage rotation

`sri-experiment-cp3` started from `main` at `b8403e6` and imported the CP2
commits as its controlled baseline. CP3 added a deterministic 1,000-session
supplementary set sampling catalog products disjoint from every public target,
proportional to `rating_number`, with the official scenario mix.

CP3 keeps CP2's stateful BM25/RRF and learned reranking, then adds:

1. An in-memory SQLite exact-value index over complete feature/detail values,
   semicolon-delimited fragments, material aliases, and color aliases.
2. Exact-evidence candidate ranking by matched-value count, value rarity (IDF),
   catalog popularity, and deterministic ASIN tie-breaking.
3. Two new learned features: exact-evidence coverage and normalized rarity.
4. A multi-lane candidate pool: up to 60 exact candidates plus enough CP2
   sparse candidates to rerank 80 unique products.
5. Coverage rotation when a reply adds no evidence: since the preceding Top 10
   is then known to be wrong, present the next unseen ranked products.
6. A 16-feature pairwise logistic reranker fitted on target-versus-top-80 hard
   negatives, with five-fold selection over `C={0.01,0.05,0.1,0.5,1.0}`.

| System / ablation | Partition | HitRate@10 | MRR | MTTC | TechnicalScore | Decision |
|---|---|---:|---:|---:|---:|---|
| Main BM25 starter | Public 200 | 0.1250 | 0.068034 | 9.810 | 0.106710 | Reference |
| CP2 learned reranker | Public 200 | 0.9900 | 0.790135 | 1.985 | 0.912340 | Controlled baseline |
| CP2 learned reranker | Synthetic 1000 | 0.9750 | 0.705398 | 2.184 | 0.875439 | Generalization baseline |
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

The selected five-fold out-of-fold score was `0.931017` with `C=0.5` (fold
scores `0.953548`, `0.965807`, `0.908667`, `0.937586`, `0.886293`).

### Value sweeps and rejected ideas

- **Document-frequency cutoff:** single-value cutoffs `10`/`100`/`500`/`2000`/
  `50000` gave early exact-intersection dev scores `0.920160`/`0.921494`/
  `0.919893`/`0.920760`/`0.922738`. `50000` (the catalog size) was selected so
  learned ranking can use broad material evidence without a hard filter.
- **Exact candidate limit:** limits `5`–`60` were swept; small limits excluded
  targets sharing common apparel composition strings, so `60` won. Pools `60`,
  `80`, `120` exposed the precision/recall trade; `80` balanced best.
- **Category sharding:** a hard category-overlap gate scored `0.916486` on dev
  and was rejected — catalog category strings are coarse and inconsistent, so
  category is useful as BM25 weighting but unsafe as a hard filter (CP6 later
  found the safe structural form of this idea).
- **Dense retrieval:** technically feasible (a 50,000×384 float16 matrix is
  ~38.4 MB), but official messages are deterministically derived from exact
  catalog text, the runtime had no ONNX/PyTorch dependency, and CP2's TinyBERT
  experiment was 2.5–3× slower at a lower score. Exact evidence raised the
  score more while preserving an offline standard-library runtime.

One process incident is retained for honesty: an auxiliary holdout run
accidentally used an earlier `df=100` default (`0.934357`); the corrected
selected-config aggregate is `0.936757`. Both artifacts were kept.

---

## CP4: intent-gated fine-tuned cross-encoder

CP4 keeps CP3's high-recall funnel and adds semantic pairwise ranking only
where it generalizes: specific-buying queries. A deterministic intent
classifier labels the state as specific buying, exploratory browsing,
constrained browsing, or intent override. Only for specific buying, the top 20
candidates are serialized as field-labelled documents and scored by a
fine-tuned 2-layer TinyBERT cross-encoder (4.49 MB QUInt8 ONNX, offline CPU
inference); neural and linear ranks combine via RRF at weight `0.15`. If the
model asset or ONNX Runtime is unavailable, loading returns `None` and the
unchanged CP3 ranking is used.

### Training (leakage-controlled)

- Synthetic targets sampled from catalog products excluding all 200 public
  target ASINs (SHA-256 manifest records zero overlap).
- A stable SHA-256 split assigns complete sessions/targets: 787 training and
  213 validation, no target in both.
- The actual CP3 funnel mines five hard negatives plus two tail negatives per
  query state: 1,740 training groups / 12,180 pairs, 465 validation groups.
- Pairwise logistic loss `softplus(score_neg − score_pos)`; three epochs on
  CPU with AdamW, LR `3e-5`, batch 32, 10% warmup, linear decay, clip 1.0,
  max length 160. (A `1e-4` run failed to finish an epoch in the CPU budget.)

| Epoch | Validation group MRR | Pairwise accuracy |
|---:|---:|---:|
| 0 (pretrained) | 0.698262 | 0.827650 |
| 1 | 0.757115 | 0.861751 |
| 2 | 0.769875 | 0.868510 |
| 3 (selected) | 0.774857 | 0.871275 |

### Ablations (dev 150)

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

Browsing neural weights `0.10`/`0.15` reduced browsing MRR from `0.706891` to
`0.704723`/`0.702400` on target-disjoint validation; constrained-browsing and
override fusion also regressed, so all three weights are zero. Larger recall
pools improved MTTC but sharply reduced top-rank precision — a bad exchange at
an already-saturated `0.995` HR@10. The `0.25` confidence gate looked best on
public dev but reduced target-disjoint validation (`0.904637 → 0.904003`), so
the ungated default was shipped.

### Generalization

| Dataset | CP3 score | CP4 score | Delta |
|---|---:|---:|---:|
| Public dev 150 | 0.931241 | 0.932508 | +0.001267 |
| Synthetic target-disjoint 213 | 0.902414 | 0.904637 | +0.002223 |
| Public holdout 50 | 0.936757 | 0.935757 | −0.001000 |
| Full public 200 | 0.932620 | 0.933320 | +0.000700 |

The holdout decline is reported rather than tuned away (20 buying cases; one
rank swap moves the aggregate visibly). On the full 200, specific-buying MRR
improved `0.844583 → 0.850417` while browsing, override, and boundary stayed
exactly unchanged — the intended effect of the classifier gate.

### Feasibility

Selected full evaluation on the dev machine: catalog load `0.57 s`, index and
ONNX session build `12.75 s`, 200 sessions in `42.19 s`. A C++ SIMD engine was
not justified (neural inference, not Python scanning, was the incremental
cost). A dense catalog matrix was rejected: BM25 already reached `0.995` HR@10
and the cross-encoder targets the demonstrated weakness — ordering semantically
close candidates — at 4.49 MB. Model provenance and hashes are frozen in
`models/cp4-tinybert-reranker/manifest.json` and `fine-tune/data/manifest.json`.

---

## CP5: ordered dialogue cards and calibrated abstention

CP5 moved the full-public result from `0.933320` to `0.977200` (+0.043880),
reaching HR@10 `1.0` and MRR `1.0`. The winning change is architectural, not a
larger model: reconstruct the ordered evidence card each catalog item can
expose, intersect the observed dialogue with those cards, and return only one
recommendation while several items remain observationally equivalent. This
trades a little turn efficiency for much higher first-rank precision.

Each product carries a short ordered fingerprint — coarse category, material,
color, first normalized feature/detail fragments — and every ordered prefix is
indexed to matching product IDs. The runtime funnel:

1. **State and safety gate.** A strict protocol detector enables the card path
   only for recognized dialogue templates; free-form input stays on the
   general lexical CP4 fallback.
2. **Preprocessing.** Existing FTS5/BM25, exact-value index, prices, and
   popularity priors, plus a `CandidateCard` per item and an inverted
   `(category, ordered-prefix) → products` map.
3. **High-recall fallback funnel.** BM25 routes + RRF + exact values + the
   16-feature CP3 reranker over 80 candidates, for non-protocol queries.
4. **Ordered dialogue match.** Observed constraints in arrival order form a
   prefix key; matching is global, so a correct item is rescued even if BM25
   missed it. Matches order by popularity, then ASIN.
5. **Confidence-qualified output.** While an exact prefix maps to more than
   one product, emit one candidate rather than ten low-confidence guesses;
   resume Top-K when the evidence becomes unique.
6. **Coverage and final turn.** Repeated evidence rotates shown IDs; turn 10
   disables abstention and returns the full rotated Top-10 window.
7. **Override reset.** An override deletes the stale opener and resets the
   shown-ID set.

MRR records the rank in the returned list, so returning one item is valuable
only combined with effective clarification and rotation: an incorrect
singleton costs a turn, while a hit is necessarily rank 1.

| Experiment | Dataset | HR@10 | MRR | MTTC | Score | Decision |
|---|---|---:|---:|---:|---:|---|
| CP4 reference | public dev 150 | 0.993333 | 0.847915 | 1.926667 | 0.932508 | reference |
| Dialogue cards, Top-10 | public dev 150 | 1.0 | 0.807860 | 1.76 | 0.927158 | reject: ordering alone insufficient |
| Cards + opening Top-1 | public dev 150 | 1.0 | 0.947778 | 1.986667 | 0.964600 | retain |
| Same, neural disabled | public dev 150 | 1.0 | 0.947778 | 1.986667 | 0.964600 | select no neural stage |
| Mode tiebreak | public dev 150 | 1.0 | 0.950000 | 1.986667 | 0.965267 | reject: split instability |
| Mode tiebreak | target-disjoint 213 | 1.0 | 0.886541 | 2.117371 | 0.943615 | inferior robustness |
| Ambiguity abstention | public dev 150 | 1.0 | 1.0 | 2.133333 | 0.977333 | selected |
| Selected CP5 | public holdout 50 | 1.0 | 1.0 | 2.16 | 0.976800 | frozen check |
| **Selected CP5** | **full public 200** | **1.0** | **1.0** | **2.14** | **0.977200** | **final public** |

- **Output-width sweep:** popularity-tiebreak dev MRR was `0.947778`,
  `0.871111`, `0.839111`, `0.807860` for opening widths 1, 3, 5, 10 (linear
  tiebreak: `0.945556`, `0.880000`, `0.868333`, `0.862407`). Width 1 is a
  large, monotonic precision improvement.
- **Tiebreak sweep:** popularity beat the CP4 linear order at width 1
  (`0.947778` vs `0.945556`). A per-mode rule reached `0.950000` publicly but
  fell to `0.875598`–`0.886541` across disagreeing validation sets — an
  unstable head — so a single popularity rule was kept.
- **Why abstention works:** target prefix coverage was `1.0` (once observable
  evidence arrived, the target was always in its reconstructed prefix group),
  but early groups were genuinely large — the median buying group at turn 1
  was 40 items and only 19% were unique. The main error was premature ranking
  under partial evidence, not missing semantic recall.

### Generalization

| Dataset | Sessions | HR@10 | MRR | MTTC | Score |
|---|---:|---:|---:|---:|---:|
| Target-disjoint validation, seed 20260831 | 213 | 1.0 | 1.0 | 2.394366 | 0.972113 |
| Fresh target-disjoint seed 20260902 | 300 | 0.996667 | 0.994000 | 2.356667 | 0.969400 |
| Fresh target-disjoint seed 20260903 | 300 | 1.0 | 0.997500 | 2.326667 | 0.972717 |
| All target-disjoint sets, weighted | 813 | 0.998770 | 0.996863 | — | — |

The sole remaining miss is an intent-override target inside a 29-product exact
evidence-equivalence class at popularity rank 18; forcing the window to
include it would be seed-specific overfitting and was not done. On the
213-session set, CP4 scored `0.904637` (MRR `0.753171`) versus CP5's
`0.972113` (MRR `1.0`).

### Strategy decisions

| Strategy | Evidence | Decision |
|---|---|---|
| Ordered candidate-card reconstruction | Prefix target coverage `1.0`; largest gain | selected |
| Selective Top-1 / abstention | Dev MRR `0.807860 → 1.0` with cards | selected |
| Global card-prefix recall | Recovers targets outside CP4's top 80 by indexed lookup | selected |
| Four authoritative intent modes | Isolated run exactly matched CP4 | reject: no material effect |
| Per-mode rank heads | Training/validation chose conflicting weights | reject: overfit |
| CP4 fine-tuned TinyBERT | Identical metrics with and without it | disabled by default |
| Query-aware neural snippets / score fusion | No headroom after public MRR reached 1.0 | defer |
| Listwise or larger cross-encoder | New cost, no remaining public headroom | defer |
| Larger initial lexical recall | CP4 pools 120/240 scored 0.922541/0.913469 | reject |
| BM25 + dense retrieval | Recall was not the bottleneck; prefix coverage perfect | reject |
| C++ SIMD engine | Full 50k run ~1 min; preprocessing one-time | not justified |
| Static user-profile classifier | Profile tags did not resolve exact catalog ties | fallback only |

### Runtime parity audit

Although the cross-encoder was disabled, its deployment path was audited on 20
validation groups / 160 pairs: tokenizer pair parity `1.0`; PyTorch FP32, ONNX
FP32, ONNX QUInt8, and the custom runtime wrapper all produced identical group
MRR `0.7713095` and Top-1 agreement `1.0`. Export/quantization drift is ruled
out as the reason the model ceased to help.

### Operational notes

Selected full-public run: catalog load `0.52 s`, index initialization
`23.63 s`, 200 sessions in `38.06 s`. No network, vector store, API key,
PyTorch, ONNX Runtime, GPU, or model download at inference. The protocol guard
matters: if message templates or card ordering drift, CP5 disables the card
path for unrecognized text and degrades toward CP4 performance rather than
failing.

---

## CP6: cross-repository audit and exact-category retrieval

CP6 keeps CP5's perfect full-public HR@10 and MRR, reduces MTTC `2.14 → 2.10`,
and raises TechnicalScore `0.977200 → 0.978000`. The selected change is an
exact coarse-category constraint inside every FTS5 retrieval route, improving
all six evaluated splits/seeds without changing HR@10 or MRR on any of them.

The opening template contains a deterministic coarse category derived from the
target's catalog record. CP6 normalizes that text, validates it against the
50,000-item category vocabulary, and runs the conjunctive/phrase/disjunctive
FTS5 routes *inside* that category (`coarse_category` is an unindexed FTS5
metadata column, so the SQL evaluates `products MATCH ?` and
`coarse_category = ?` together rather than filtering a global Top 150
afterward). Loose exact-evidence injections are filtered to the same category.
If the parser fails, the category is absent, or the category-scoped search is
empty, retrieval fails open to the global CP5 route.

### Repository audit

All 15 public repositories were shallow-cloned on 2026-09-01 at pinned
commits. Metrics are self-reported upstream; the audit looked for mechanisms,
ablations, and transferability.

| Repository | Audited commit | Useful evidence | CP6 decision |
|---|---|---|---|
| johngao122 | [`88daecf`](https://github.com/johngao122/techjam-conversational-search/tree/88daecf6a1ff20096e8f9036573adb10d7811b00) | Early Top-1, category buckets, confidence/release policy, exhaustion handling | Top-1 already CP5; separately test exhaustion and fixed releases |
| Khanna-Aman | [`3b76e2a`](https://github.com/Khanna-Aman/techjam-2026-shopping-copilot/tree/3b76e2a38caed9eb4fe9191a35f3ccd51eaacc76) | Exact category lock, strongest Top-1 sweep, review-count popularity; dense/profile ablations | Select exact category; retain review count; do not revive dense/profile |
| Antelyuu | [`b7d553a`](https://github.com/Antelyuu/techjam-conversational-search/tree/b7d553a02d870dce52b5d4edd5183965cf022122) | E1–E9 ablations, category-scoped FTS, strict ownership, evidence-aware shortlist, fail-open | Category-scoped FTS selected; ownership/shortlist already CP5 |
| Kairon-2005 | [`dcfbd52`](https://github.com/Kairon-2005/techjam2026/tree/dcfbd52cc3a61154a638e926fc5f603895aaa85b) | Popularity and dynamic intent routing strongest; semantic/profile paths not demonstrated | Keep CP5 popularity and state classifier; no semantic expansion |
| algorathem | [`27d1afc`](https://github.com/algorathem/techjam2026-shopping-copilot/tree/27d1afc155ca877c6575c1dbbc55d545c346989d) | Early Top-1, category-tail parsing, `other`, override provenance; question-policy ablations | Existing CP5 equivalents retained; no information-gain policy switch |
| Creomeow | [`6f3b05b`](https://github.com/Creomeow/techjam-conversational-search/tree/6f3b05b52f984669d6994c734e9304a0a34b0bb2) | `other`-first gains, boundary/exhaustion distinction, rotation/backfill | Keep `other` and rotation; test true-exhaustion release separately |
| 13shreyansh | [`e9d3db9`](https://github.com/13shreyansh/shopping-copilot-techjam-2026/tree/e9d3db9dea05e4c454d858ec3b97b9d7725b900a) | Parser/Unicode/row-order audits and clean validation boundaries | Add narrow parser normalization and regression tests |
| Shaneeen | [`8d8822e`](https://github.com/Shaneeen/ShopCopilot/tree/8d8822e7d27dc510c78c9c1fbc5eb93bf0cf5fdc) | Broad hybrid retrieval and full-pool audits expose precision loss from deep candidates | Do not widen CP5 pools or add global dense fusion |
| kxphan05 | [`37b9fd4`](https://github.com/kxphan05/Spider-Rank/tree/37b9fd408b5cb20f9ba127c675f7cb092bc57950) | Lexical+BGE, PRF, optional cross-encoder, shown-result exclusion | Shown-result rotation already CP5; neural/PRF path lacks score evidence here |
| ImNuza | [`5fa7b39`](https://github.com/ImNuza/opoyo-tiktok/tree/5fa7b39b95b9411df079da46ebf47e02f70ebc4b) | BM25, category lexicon, clarification policy, optional MiniLM | Exact structural category is stronger; optional neural path not promoted |
| sci-m-wang | [`6b1aca6`](https://github.com/sci-m-wang/techjam-conversational-search-agent/tree/6b1aca69483d2d624757fd6aa7cc2ae131741799) | Measured LLM-agent token and serial-runtime cost with lower retrieval metrics | Reject API/LLM path for this fixed deterministic protocol |
| fatbolster | [`a021df3`](https://github.com/fatbolster/techjam-shopping-agent/tree/a021df30a56b2a346047dccfdd9e73883c664856) | Fitted ranking features, state, clarification and scripted ablations | CP3/CP5 already contain stronger measured versions; no new promotion |
| tristan1127 | [`3b273c3`](https://github.com/tristan1127/techjam2026-shopping-copilot/tree/3b273c36daf69908170b3e2042fef7571bcda207) | Deterministic FTS5 plus rule-based reranking | CP5 is a measured superset; no isolated candidate added |
| wayneenxz | [`20cdc7b`](https://github.com/wayneenxz/maihenduo/tree/20cdc7b7f0d12639478465c43a800c68357485f1) | Whole-token/category parsing, override state, light diversity; negative RRF/wider-pool/stemming results | Parser/state already covered; negative results reinforce current pool sizes |
| dngvmnh | [`2398a87`](https://github.com/dngvmnh/techjam-shopping-copilot/tree/2398a87c9511e2ced2407499e82a97ca55da96ae) | Exact protocol replay, category partition, candidate elimination, abstention, free-form fallback | Confirms CP5 architecture and CP6 category partition; zero-output/IG not adopted |

The leading repositories converge structurally: for this fixed protocol,
timing of exposure and exact catalog consistency matter more than a larger
semantic model. BM25+dense, global cross-encoder reranking, larger pools,
static personalization, and LLM agents were not rerun — CP3–CP5 had already
measured the same families negatively or the audited repositories supplied
stronger negative ablations.

### Isolated experiments (dev 150)

The first row reruns CP5 through the modified harness and reproduces its
metric exactly, ruling out an accidental baseline shift.

| Variant | HR@10 | MRR | MTTC | Score | Decision |
|---|---:|---:|---:|---:|---|
| CP5 control | 1.0 | 1.0 | 2.133333 | 0.977333 | exact baseline reproduction |
| Exact category-scoped retrieval | 1.0 | 1.0 | 2.086667 | 0.978267 | **selected** |
| Release on true exhaustion | 1.0 | 0.991333 | 2.100000 | 0.975400 | reject: premature lower-rank hit |
| Fixed release on turn 3 | 1.0 | 0.975000 | 2.060000 | 0.971300 | reject |
| Fixed release on turn 4 | 1.0 | 0.988000 | 2.093333 | 0.974533 | reject |
| Fixed release on turn 5 | 1.0 | 0.995000 | 2.113333 | 0.976233 | reject |
| Average-rating weight 0.02 | 1.0 | 1.0 | 2.140000 | 0.977200 | reject: slower override |
| Average-rating weight 0.05 | 1.0 | 1.0 | 2.140000 | 0.977200 | reject |
| Average-rating weight 0.10 | 1.0 | 1.0 | 2.140000 | 0.977200 | reject |
| Category + exhaustion | 1.0 | 0.991333 | 2.053333 | 0.976333 | reject |

The exhaustion result is unintuitive but decisive: "no additional preference"
does mean the card is exhausted, yet returning a full list then can lock in
rank 2+ before rotation has exposed the best candidate. Average rating also
loses to the established `log1p(rating_number)` ordering.

### Generalization and frozen checks

| Dataset | Sessions | CP5 HR / MRR / MTTC / score | CP6 HR / MRR / MTTC / score | Score delta |
|---|---:|---|---|---:|
| Public dev | 150 | 1 / 1 / 2.133333 / 0.977333 | 1 / 1 / 2.086667 / 0.978267 | +0.000934 |
| Public holdout | 50 | 1 / 1 / 2.160000 / 0.976800 | 1 / 1 / 2.140000 / 0.977200 | +0.000400 |
| Full public | 200 | 1 / 1 / 2.140000 / 0.977200 | 1 / 1 / 2.100000 / 0.978000 | +0.000800 |
| Target-disjoint 20260831 | 213 | 1 / 1 / 2.394366 / 0.972113 | 1 / 1 / 2.356808 / 0.972864 | +0.000751 |
| Target-disjoint 20260902 | 300 | 0.996667 / 0.994 / 2.356667 / 0.969400 | 0.996667 / 0.994 / 2.330000 / 0.969934 | +0.000534 |
| Target-disjoint 20260903 | 300 | 1 / 0.9975 / 2.326667 / 0.972717 | 1 / 0.9975 / 2.303333 / 0.973183 | +0.000466 |

CP6 does not repair or worsen CP5's sole 300-seed miss; it reaches the same
targets and ranks while using the category partition to reach them sooner.
The full public run initializes all indexes in about 44 seconds and evaluates
200 sessions in about 44 seconds on the development CPU; a SIMD engine, dense
model, or LLM cannot improve the remaining errors, which are exact-card ties.

---

## Unmerged branch experiments

### `arjo-cp4`: dominance tiers, retrained reranker, robustness hardening

A parallel CP4 line that improved the CP3 baseline through structured ranking
rather than a neural model. **Final public result: HitRate@10 `1.000`, MRR
`0.856365`, MTTC `1.885`, TechnicalScore `0.939210`** (+0.006590 over CP3,
higher than the merged CP4's 0.933320), with the only CP3 miss
(`public_0020`) fixed and no session regressed below baseline.

Key mechanisms, each isolated in the branch's full experiment grid:

- **Dominance tiers (Task 1).** The exact-evidence pool returns the complete
  top tier — every product satisfying the maximum number of exact constraints
  (cap 1000) — and reranking sorts by `(-satisfied_count, -score, pool_rank,
  asin)`, so more-satisfied always outranks fewer-satisfied. Rotation pages
  the whole tier unseen-first. With the original CP3 weights this *lost*
  score (0.921695–0.931870 across variants); it only pays off after
  retraining, which is why the tasks were sequenced.
- **Frozen synthetic splits (Task 2).** Seed 20260831 generated 5,000
  target-disjoint sessions (train 3000 / dev 1000 / holdout 1000, SHA-256
  frozen) plus a 500-target hard-case set (common attributes, dense
  neighborhoods, sparse products) on which the CP3-era agent scored only
  `0.711441`.
- **Within-tier MRR-weighted retraining (Task 3).** An 18-feature (of 23
  candidates) linear model trained on 11,502 groups / 168,413 weighted pairs
  from public dev + synthetic train + synthetic dev, with pairs restricted to
  the target's dominance tier and weighted by reciprocal-rank difference.
  Coarse-category equality (+0.011244) and overlap (+0.006716) were the
  winning new features — independently anticipating CP6's category result.
  Public 200: `0.938579`; public holdout: `0.944100`; synthetic holdout:
  `0.908156`.
- **Per-constraint retrieval routes (Task 4).** Replacing the concatenated
  conjunctive query with per-constraint FTS routes (category-only, category+
  material, color+category, exact-phrase, fallback bag) fused by RRF, with
  generic tokens identified by measured catalog document frequency
  (`df > 12000`) rather than a hardcoded list. Public 200: `0.939210`.
- **Paraphrase robustness (Task 5).** Wider marker/override/no-preference/
  budget vocabularies, an index-gated n-gram constraint fallback, and a slot
  store with contradiction-aware supersession. Identical on all 200 clean
  sessions; on the unofficial paraphrase harness the score rose from
  `0.852830` to `0.883018` (clean/paraphrased gap `0.056192`, under the `0.1`
  target).
- **Module split, question policy, LLM layer (Task 6).** `agent.py` became a
  thin orchestrator over eight modules, verified behavior-identical per
  session, not just in aggregate. An expected-tier-reduction question policy
  was measured exactly free of cost after a tie-break fix but ships disabled
  (its "stop when tier ≤ 10" rule cost −0.000976). An optional
  Anthropic-backed message-text layer (structurally unable to change
  rankings) ships off by default, keeping the runtime standard-library only.
  Reported feasibility: 31.7 ms mean per turn, 4.8 s index build, 12.0 s full
  200-session run, ~0.9 GB peak RSS, $0.00 API cost.

Notable rejected variants: broadened "looking for" constraint markers
(polluted every opener; `0.923581`), comma-splitting repairs for truncated
values (five variants, all below baseline), alphabetical question-policy
tie-breaking (−0.062, because `feature` beat the equally-scoring open
question).

### `arjo-cp5`: guarded ordered-dialogue ranking over Arjo CP4

Adds the catalog-only ordered dialogue-card index and selective Top-1 output
on top of the full Arjo CP4 pipeline as fallback. **Final public result:
HitRate@10 `1.000`, MRR `1.000`, MTTC `2.140`, TechnicalScore `0.977200`** —
matching the merged CP5 aggregate while retaining Arjo CP4's broader wording
support.

| Variant | Dataset | HR@10 | MRR | MTTC | TechnicalScore | Decision |
|---|---|---:|---:|---:|---:|---|
| Arjo CP4, cards disabled | Public 200 | 1.000 | 0.856365 | 1.885 | 0.939210 | Exact fallback baseline |
| Dialogue cards, Top 10 | Public dev 150 | 1.000 | 0.814341 | 1.767 | 0.928969 | Reject: ordering alone hurts precision |
| Opening Top 1 only | Public dev 150 | 1.000 | 0.947778 | 1.987 | 0.964600 | Keep mechanism |
| **Opening + ambiguity Top 1, popularity** | **Public dev 150** | **1.000** | **1.000** | **2.133** | **0.977333** | **Selected** |
| Same, Arjo linear tie-break | Public dev 150 | 1.000 | 1.000 | 2.160 | 0.976800 | Reject: slower |
| Same, hybrid tie-break | Public dev 150 | 1.000 | 1.000 | 2.160 | 0.976800 | Reject: slower |
| Selected | Public holdout 50 | 1.000 | 1.000 | 2.160 | 0.976800 | Confirmation |
| **Selected** | **Public 200** | **1.000** | **1.000** | **2.140** | **0.977200** | **Final** |
| Selected | Synthetic dev 1000 | 0.999 | 0.995875 | 2.406 | 0.970143 | Strong target-disjoint result |
| Selected | Synthetic holdout 1000 | 1.000 | 0.993726 | 2.453 | 0.969058 | Aggregate-only |
| Selected | Hard cases 500 | 0.856 | 0.834589 | 4.306 | 0.812257 | Better score, lower recall |
| Selected | Paraphrased public 200 (unofficial) | 0.985 | 0.813756 | 2.280 | 0.911027 | Robustness diagnostic |

Against Arjo CP4: synthetic dev `0.909773 → 0.970143`, synthetic holdout
`0.908156 → 0.969058`, hard cases `0.764609 → 0.812257` (hard-case HitRate
falls `0.904 → 0.856` while MRR rises enough to improve the composite — the
trade is reported rather than hidden). The card path activates in 44% of
paraphrased sessions; the guard falls back to CP4 for the rest, keeping the
clean/paraphrased gap at `0.066173`.

The branch's larger 1000-session target-disjoint validations are the strongest
generalization evidence for the dialogue-card approach in either line. It also
carries a compliance note: the specification lists "private-label
reconstruction" as out of scope; Arjo CP5 never reads targets and builds
possible observable cards for *every* catalog product, but because this
mirrors the released deterministic policy the team should confirm the
interpretation with the organizer before finalizing a card-based submission.

### `cody-branch`: synonym-expansion reranker (reverted)

An early dependency-free two-stage system: FTS5 retrieval with a curated
clothing synonym map (sneakers/shoes, tee/shirt, grey/gray, …), multi-route
candidate blending (latest message, active state, full history, profile tags),
and field-weighted reranking with material/color/budget boosts and light
rating tie-breaks. Its commit was reverted from `main` at `b8403e6` before the
CP1 experiments began; the CP1+ lines superseded it with measured route
weights, learned reranking, and no hand-curated synonym list. The branch's
submission report survives as documentation of the approach.

### `synth-cp6`: synthetic test tooling (merged)

Not an experiment line — it generalized the `arjo-cp4` synthetic-data and
paraphrase tooling so `scripts.evaluate_cp4_confirm` and
`tests.paraphrase_harness` test whichever agent is checked out. Merged into
`main`; usage is documented in `README_DEV.md`.

---

## Reproduction

From the repository root, with `data/catalog.jsonl` and
`data/public_set.jsonl` present (activate `.venv` if using the repository
virtual environment):

```bash
# Official public-set evaluation of the selected (CP6) agent
python3 -m evaluator.local_evaluator

# Unit tests
python3 -m unittest discover tests

# CP6 policy grid and final run
python3 -m scripts.sweep_cp6_variants --partition dev --output data/releases/cp6/01-policy-grid/dev.json
python3 -m scripts.evaluate_cp6_variant full --output data/releases/cp6/final/full.json

# Earlier checkpoint variant runners
python3 -m scripts.evaluate_cp5_variant full --output data/releases/cp5/final/full.json

# Target-disjoint synthetic sets and confirmation runs (see README_DEV.md)
python3 -m scripts.create_cp4_synthetic_sets
python3 -m scripts.create_cp4_hardcase_set
python3 -m scripts.evaluate_cp4_confirm --dataset data/releases/cp4/synthetic_dev.jsonl

# Diagnostics
python3 -m scripts.analyze_public_failures --results results.json --output failure-analysis.json
python3 -m scripts.analyze_cp5_headroom
python3 -m tests.paraphrase_harness --limit 25   # UNOFFICIAL wording diagnostic

# Optional retraining (NumPy/scikit-learn are training-only)
python3 -m scripts.train_cp2_reranker --report-output data/releases/cp2/02-learned/training/cv_report.json
python3 -m scripts.train_cp3_reranker --report-output data/releases/cp3/final/cv_report.json
```

The CP4 cross-encoder training pipeline (data building, splitting, training,
ONNX export, and the PyTorch/ONNX/quantized parity check), the quantized model
asset, its provenance manifests, and `scripts.evaluate_cp4_variant` were
removed from `main` in the post-CP6 cleanup below; they are preserved intact
on the `sri-experiment-cp4` branch (`fine-tune/README.md` and
`models/cp4-tinybert-reranker/manifest.json` there).

---

## Post-CP6 cleanup: neural-path removal

After CP6 froze, the disabled cross-encoder path was removed from `main` to
slim the submission: the 5.0 MB `models/` asset (about 90% of the tracked
repository size), `starter/cp4_cross_encoder.py`, the `fine-tune/` pipeline,
`requirements-cp4.txt`, the cross-encoder options in `Agent` and the variant
scripts, and the three model-specific unit tests. The model had been inert
since CP5's no-effect ablation (weights zero by default, `use_cross_encoder`
defaulting to `False`), so no scored behavior changed.

| Change | Partition | HitRate@10 | MRR | MTTC | TechnicalScore | Decision |
|---|---|---:|---:|---:|---:|---|
| Remove disabled cross-encoder path | Public 200 | 1.0 | 1.0 | 2.100 | 0.978000 | Keep: exact baseline reproduction, all scenario aggregates unchanged |

The remaining 20 unit tests pass. The runtime is now standard-library-only
with no optional third-party dependency at all.
