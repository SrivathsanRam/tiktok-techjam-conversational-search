# TechJam Conversational E-Commerce Search — Agent

An offline, standard-library conversational catalog-search agent. It retrieves
with SQLite FTS5 over multiple per-constraint routes, partitions candidates
into dominance tiers, and adds a guarded ordered-dialogue index that can recover
products globally from the sequence of catalog values disclosed by the customer.

**Public 200-session result:** HitRate@10 `1.000`, MRR `1.000`, MTTC `2.140`,
TechnicalScore **`0.977200`**.

## No network required

The agent performs no network calls, reads no credentials, and reports zero
token usage. Everything it needs is the catalog file and the 18 fitted weights
in `starter/reranker_weights.json`. An optional presentation layer can call an
LLM, but it is **off by default** and never affects ranking (see below).

## Requirements

- Python 3.10+ (uses PEP 604 unions, `:=`, `zip(strict=True)`)
- A CPython `sqlite3` built with FTS5:
  ```bash
  python3 -c "import sqlite3; sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)')"
  ```
- No third-party runtime packages (`requirements.txt` is intentionally empty of
  dependencies). Training extras are pinned separately in
  `requirements-train.txt`; the optional LLM extra is in `requirements-llm.txt`.

## One-command reproduction

```bash
python3 -m evaluator.local_evaluator          # writes results.json
```

The catalog defaults to `data/catalog.jsonl` and can be relocated with the
`TECHJAM_CATALOG` environment variable (or the `--catalog` flag):

```bash
TECHJAM_CATALOG=/data/catalog.jsonl python3 -m evaluator.local_evaluator
```

Other commands:

```bash
python3 -m unittest discover tests            # 38 tests
python3 -m scripts.build_submission           # assemble + smoke-test submission/
python3 -m tests.paraphrase_harness           # UNOFFICIAL robustness diagnostic

# Optional CPU retraining (numpy/scikit-learn are training-only)
pip install -r requirements-train.txt
python3 -m scripts.create_cp4_synthetic_sets
python3 -m scripts.train_cp4_reranker --report-output data/releases/cp4/cv_report.json
```

## Architecture

`starter/agent.py` is a thin orchestrator; each stage is its own module. When a
dialogue does not satisfy the ordered-card guard, the complete CP4 pipeline is
the unchanged fallback.

| Module | Responsibility |
|---|---|
| `catalog_index.py` | One pass over the catalog builds the FTS5 table, the exact-value evidence table, popularity, per-product field views, coarse categories, and catalog-wide token document frequencies. |
| `dialogue_cards.py` | Reconstructs catalog-only ordered evidence cards and indexes every category/prefix for global protocol-aware recall. |
| `intent_router.py` | Every wording rule in one place: override / browsing / no-preference / rejection / budget detection, and index-gated constraint extraction. |
| `dialog_state.py` | Session memory plus the slot store, with a write/erase log. A disclosed value retires a stored one only on genuine contradiction. |
| `sparse_retrieval.py` | Per-constraint FTS routes (category, category+material, color+category, exact constraint phrases, adjacent-phrase, disjunctive) fused by reciprocal rank. |
| `exact_evidence.py` | Looks constraints up by normalized exact equality and returns the top dominance tier plus per-product satisfied-constraint counts. |
| `reranker.py` | 23 features; sorts by `(satisfied_count, learned_score)` so a product satisfying more constraints always outranks one satisfying fewer. |
| `question_policy.py` | Estimates the expected tier reduction of each allowed question and picks the argmax. Off by default (see limitations). |
| `llm_layer.py` | Optional presentation text. Off by default. |

Per turn: update state → resolve slots → build the query → retrieve (sparse
routes + exact tier) → rerank within tiers → apply a guarded global dialogue
prefix → select one high-confidence unseen product while ambiguous, or use the
normal CP4 result width when the guard fails. Turn 10 always restores Top 10.

## Cost and latency

Measured on an Apple M-series CPU, 50,000-product catalog, single process:

| Metric | Value |
|---|---|
| Index + dialogue-card build (once per process) | ~8.3 s |
| Full 200-session public evaluation | ~12.5 s |
| Mean evaluation time per hit turn | ~29 ms |
| Peak resident memory | ~0.9 GB |
| Model API cost | **$0.00** — no API calls |
| Reported token usage | `0` prompt, `0` completion |
| Training cost | CPU only, ~35 min end to end, $0.00 |

## Optional LLM layer (off by default)

`starter/llm_layer.py` may rewrite the assistant's message and attach one-line
explanations. It is enabled only when `TECHJAM_LLM_ENABLED=1`:

```bash
export TECHJAM_LLM_ENABLED=1
export TECHJAM_LLM_MODEL=claude-opus-5      # optional
pip install -r requirements-llm.txt
```

Guarantees, enforced structurally and by test: it receives the already-ranked
list and returns text only, so it **cannot reorder, add, or drop a
recommendation**; it reports the API's **real** token counts in `usage`; and any
error, missing SDK, missing credential, or refusal falls back to the
deterministic template. With the flag unset the runtime is offline and
stdlib-only — the SDK is imported lazily inside the call.

## Limitations

- **Protocol coupling.** The large CP5 gain assumes the private evaluator keeps
  the released catalog-value ordering. Ordered-card matching is disabled when
  the category or constraint prefix cannot be verified, preserving CP4 as the
  fallback. Target-disjoint tests reuse the same simulator policy and therefore
  do not prove robustness to an ordering change. See `CP5_EXPERIMENTS.md` for
  the compliance note.
- **Selective output.** Recognized ambiguous sessions usually receive one
  recommendation so a later hit is rank 1. This improves the official score but
  can take more turns and may be less useful in a real shopping interface.
- **Exact-match evidence.** Constraint matching is normalized exact equality
  over catalog values. A paraphrase *around* a value is handled (token n-grams
  are matched against the index), but a value whose own tokens are altered —
  synonyms, reordering, number-word conversion — has no posting and falls back
  to lexical retrieval. Measured cost: TechnicalScore `0.939210` clean vs
  `0.883018` under `tests/paraphrase_harness.py`, a gap of `0.056`.
- **The question policy ships disabled.** Its estimator is sound — with the
  spec's "stop asking once the tier fits one page" rule removed it reproduces
  the baseline exactly — but that stop rule costs `0.000976` on the public set,
  because a small tier that does not contain the target can still be rescued by
  one more disclosure. Enable with `Agent(use_question_policy=True)`.
- **`ask_attribute="other"` is load-bearing.** The local simulator treats
  `other` as matching any undisclosed constraint, which makes the open question
  optimal here. A private policy with different `other` semantics would change
  the disclosure sequence.
- **Memory.** The index holds the catalog in memory (~0.9 GB peak for 50,000
  products); it is not designed for a catalog an order of magnitude larger.
- **Single process.** Sessions are held in a plain dict, so state does not
  survive a restart and is not shared across workers.

## Reports

- `CP5_EXPERIMENTS.md` — selected CP5 architecture, ablations, validation,
  safety gate, and compliance note.
- `CP4_EXPERIMENTS.md` — every experiment, kept or reverted, with metrics.
- `docs/cp4_cv_report.json` — cross-validation and candidate-feature outcomes.
- `docs/cp4_paraphrase_report.json` — clean vs paraphrased robustness.
- `CP3_STATE.md` — the inherited implementation state this work started from.

## Data attribution

See `DATA_ATTRIBUTION.md`. The evaluator and `data/public_set.jsonl` are
organizer artifacts and are not modified by this submission.
