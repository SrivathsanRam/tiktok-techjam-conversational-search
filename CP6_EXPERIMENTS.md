# CP6 experiments: cross-repository audit and exact-category retrieval

## Outcome

CP6 keeps CP5's perfect full-public Hit Rate@10 and MRR, reduces MTTC from
`2.14` to `2.10`, and raises TechnicalScore from `0.977200` to `0.978000`.
The selected change is an exact coarse-category constraint inside every FTS5
retrieval route. It improves all six evaluated splits/seeds without changing
HR@10 or MRR on any of them.

This is deliberately a small release. CP5 had already implemented the largest
ideas shared by the leading public systems: ordered target-card replay,
other-first clarification, Top-1 exposure while evidence is ambiguous, shown-ID
rotation, override reset, popularity ordering, and a final-turn safety release.
CP6 tested the credible mechanisms that remained rather than combining every
public technique into an unidentifiable bundle.

## Architecture change

The opening template contains the same deterministic coarse category derived
from the target's catalog record. CP6 normalizes that text and uses it as a
structural partition:

```text
opening message
  -> parse exact coarse category
  -> validate against the 50,000-item category vocabulary
  -> run conjunctive / phrase / disjunctive FTS5 routes inside that category
  -> filter loose exact-evidence injections to the same category
  -> reciprocal-rank fusion and CP3 feature reranking
  -> CP5 ordered-card intersection and popularity ordering
  -> expose Top 1 while the card prefix is ambiguous
  -> rotate shown items; full Top K on unique evidence or the final turn
```

`coarse_category` is stored as an unindexed FTS5 metadata column. The SQL query
therefore evaluates both `products MATCH ?` and `coarse_category = ?`; it does
not retrieve a global Top 150 and filter afterward. If the parser is
unrecognized, the category is absent, or the complete category-scoped search is
empty, retrieval fails open to the global CP5 route. A single empty sub-route
does not leak global candidates into an otherwise valid category pool.

No target IDs, public labels, or session answers are used to build this index.
The new protocol normalizer only canonicalizes curly apostrophes and whitespace;
unrecognized free-form messages still use the general lexical fallback.

## Repository audit method

All 15 public repositories were shallow-cloned on 2026-09-01 and inspected at
the immutable commits below. Their reported metrics are self-reported upstream;
CP6 does not claim to have reproduced their complete environments. The audit
looked for mechanisms, ablations, failure analysis, and transferability to the
current CP5 architecture.

| Repository | Audited commit | Useful evidence | CP6 decision |
|---|---|---|---|
| johngao122 | [`88daecf`](https://github.com/johngao122/techjam-conversational-search/tree/88daecf6a1ff20096e8f9036573adb10d7811b00) | Early Top-1, category buckets, confidence/release policy, exhaustion handling | Top-1 already CP5; separately test exhaustion and fixed releases |
| Khanna-Aman | [`3b76e2a`](https://github.com/Khanna-Aman/techjam-2026-shopping-copilot/tree/3b76e2a38caed9eb4fe9191a35f3ccd51eaacc76) | Exact category lock, strongest Top-1 sweep, review-count popularity; dense/profile ablations | Select exact category; retain review count; do not revive dense/profile |
| Antelyuu | [`b7d553a`](https://github.com/Antelyuu/techjam-conversational-search/tree/b7d553a02d870dce52b5d4edd5183965cf022122) | E1-E9 ablations, category-scoped FTS, strict ownership, evidence-aware shortlist, fail-open | Category-scoped FTS selected; ownership/shortlist already CP5 |
| Kairon-2005 | [`dcfbd52`](https://github.com/Kairon-2005/techjam2026/tree/dcfbd52cc3a61154a638e926fc5f603895aaa85b) | Popularity and dynamic intent routing strongest; semantic/profile paths not demonstrated | Keep CP5 popularity and state classifier; no semantic expansion |
| algorathem | [`27d1afc`](https://github.com/algorathem/techjam2026-shopping-copilot/tree/27d1afc155ca877c6575c1dbbc55d545c346989d) | Early Top-1, category-tail parsing, `other`, override provenance; question-policy ablations | Existing CP5 equivalents retained; no information-gain policy switch |
| Creomeow | [`6f3b05b`](https://github.com/Creomeow/techjam-conversational-search/tree/6f3b05b52f984669d6994c734e9304a0a34b0bb2) | `other`-first gains, boundary/exhaustion distinction, rotation/backfill | Keep `other` and rotation; test true-exhaustion release separately |
| 13shreyansh | [`e9d3db9`](https://github.com/13shreyansh/shopping-copilot-techjam-2026/tree/e9d3db9dea05e4c454d858ec3b97b9d7725b900a) | Parser/Unicode/row-order audits and clean validation boundaries; profile/semantic skepticism | Add narrow parser normalization and regression tests |
| Shaneeen | [`8d8822e`](https://github.com/Shaneeen/ShopCopilot/tree/8d8822e7d27dc510c78c9c1fbc5eb93bf0cf5fdc) | Broad hybrid retrieval and full-pool audits expose precision loss from deep candidates | Do not widen CP5 pools or add global dense fusion |
| kxphan05 | [`37b9fd4`](https://github.com/kxphan05/Spider-Rank/tree/37b9fd408b5cb20f9ba127c675f7cb092bc57950) | Lexical+BGE, PRF, optional cross-encoder, shown-result exclusion, profile decision | Shown-result rotation already CP5; neural/PRF path lacks score evidence here |
| ImNuza | [`5fa7b39`](https://github.com/ImNuza/opoyo-tiktok/tree/5fa7b39b95b9411df079da46ebf47e02f70ebc4b) | BM25, category lexicon, clarification policy, optional MiniLM | Exact structural category is stronger; optional neural path not promoted |
| sci-m-wang | [`6b1aca6`](https://github.com/sci-m-wang/techjam-conversational-search-agent/tree/6b1aca69483d2d624757fd6aa7cc2ae131741799) | Measured LLM-agent token and serial-runtime cost with lower retrieval metrics | Reject API/LLM path for this fixed deterministic protocol |
| fatbolster | [`a021df3`](https://github.com/fatbolster/techjam-shopping-agent/tree/a021df30a56b2a346047dccfdd9e73883c664856) | Fitted ranking features, state, clarification and scripted ablations | CP3/CP5 already contain stronger measured versions; no new promotion |
| tristan1127 | [`3b273c3`](https://github.com/tristan1127/techjam2026-shopping-copilot/tree/3b273c36daf69908170b3e2042fef7571bcda207) | Deterministic FTS5 plus rule-based reranking | CP5 is a measured superset; no isolated candidate added |
| wayneenxz | [`20cdc7b`](https://github.com/wayneenxz/maihenduo/tree/20cdc7b7f0d12639478465c43a800c68357485f1) | Whole-token/category parsing, override state, light diversity; negative RRF/wider-pool/stemming results | Parser/state already covered; negative results reinforce current pool sizes |
| dngvmnh | [`2398a87`](https://github.com/dngvmnh/techjam-shopping-copilot/tree/2398a87c9511e2ced2407499e82a97ca55da96ae) | Exact protocol replay, category partition, candidate elimination, abstention, free-form fallback | Confirms CP5 architecture and CP6 category partition; zero-output/IG not adopted |

The leading repositories converge on a structural conclusion: for this fixed
protocol, timing of exposure and exact catalog consistency matter more than a
larger semantic model. BM25+dense, global cross-encoder reranking, a larger
initial pool, stronger static personalization, and an LLM agent were therefore
not rerun in CP6; CP3-CP5 had already measured the same families negatively or
the audited repositories supplied stronger negative ablations.

## Isolated development experiments

The first row reruns CP5 through the modified harness and reproduces its metric
exactly, ruling out an accidental baseline shift. Selection used only the
frozen 150-session development split.

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

The exhaustion result is unintuitive but decisive. “No additional preference”
does mean the card is exhausted, yet returning a full list at that point can
lock in rank 2+ before CP5 rotation has exposed the best candidate. CP6 tracks
exhaustion correctly for state semantics but does not use it as a release gate.

Average rating also loses to the established `log1p(rating_number)` ordering.
The tested weights preserve MRR but delay one override session; review count is
the more stable prior for otherwise observationally equivalent products.

## Generalization and frozen checks

| Dataset | Sessions | CP5 HR / MRR / MTTC / score | CP6 HR / MRR / MTTC / score | Score delta |
|---|---:|---|---|---:|
| Public dev | 150 | 1 / 1 / 2.133333 / 0.977333 | 1 / 1 / 2.086667 / 0.978267 | +0.000934 |
| Public holdout | 50 | 1 / 1 / 2.160000 / 0.976800 | 1 / 1 / 2.140000 / 0.977200 | +0.000400 |
| Full public | 200 | 1 / 1 / 2.140000 / 0.977200 | 1 / 1 / 2.100000 / 0.978000 | +0.000800 |
| Target-disjoint 20260831 | 213 | 1 / 1 / 2.394366 / 0.972113 | 1 / 1 / 2.356808 / 0.972864 | +0.000751 |
| Target-disjoint 20260902 | 300 | 0.996667 / 0.994 / 2.356667 / 0.969400 | 0.996667 / 0.994 / 2.330000 / 0.969934 | +0.000534 |
| Target-disjoint 20260903 | 300 | 1 / 0.9975 / 2.326667 / 0.972717 | 1 / 0.9975 / 2.303333 / 0.973183 | +0.000466 |

All three target-disjoint datasets exclude the 200 public target ASINs. CP6
does not repair or worsen CP5's sole 300-seed miss; it reaches the same targets
and ranks while using the exact category partition to reach them sooner.

## Why no new dense model, LLM, or SIMD engine

CP6's diagnosed opportunity is search space precision, not raw recall or CPU
throughput. The full public run initializes the 50,000-item FTS, exact-value,
card, and category indexes in about 44 seconds and evaluates 200 sessions in
about 44 seconds on the development CPU. In a service, initialization is
amortized; no per-request network or model latency exists.

A SIMD C++ engine would reduce latency but cannot improve MRR, and the current
runtime is already practical. A dense model or global cross-encoder adds model
size and failure modes while several audited implementations—and CP4/CP5's own
ablations—show that this benchmark's remaining errors are exact-card ties.
Those tools become viable only if the private protocol is substantially noisier
than the released one or profiling identifies a real service-level bottleneck.

## Reproduction

The selected runtime uses only the Python standard library:

```powershell
python -m scripts.sweep_cp6_variants `
  --partition dev `
  --output data/releases/cp6/01-policy-grid/dev.json

python -m scripts.evaluate_cp6_variant full `
  --output data/releases/cp6/final/full.json

python -m unittest discover -s tests -q
```

Generated repository clones, synthetic data, and result JSON files remain under
git-ignored `data/releases/cp6`. The implementation, experiment driver, tests,
and this report are committed.
