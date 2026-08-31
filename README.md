# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## CP3 Agent

The `sri-experiment-cp3` implementation is an offline evidence-funnel agent:
stateful BM25/RRF retrieval, exact catalog-value matching, a 16-feature linear
reranker, and coverage-aware multi-turn rotation. Its recorded public result is
Hit Rate@10 `0.995`, MRR `0.845734`, MTTC `1.93`, and TechnicalScore `0.932620`.
See `CP3_EXPERIMENTS.md` for leakage controls, value sweeps, rejected variants,
target-disjoint validation, resource tradeoffs, and reproduction commands.

## CP4 Agent

The `sri-experiment-cp4` branch adds a genuinely fine-tuned 2-layer TinyBERT
cross-encoder. A deterministic intent classifier applies it only to the top 20
specific-buying candidates; browsing, boundary, and override traffic retains
the stronger CP3 lexical ranking. The 4.49 MB quantized ONNX model runs locally
on CPU and falls back automatically to CP3 if ONNX Runtime is unavailable.

```bash
python -m pip install -r requirements-cp4.txt
python -m evaluator.local_evaluator
```

No network, API key, vector service, PyTorch, or GPU is required at inference
time. See `CP4_EXPERIMENTS.md` for the complete workflow and ablations.

## CP5 Agent

The `sri-experiment-cp5` branch replaces neural rank fusion with a
protocol-aware, catalog-derived dialogue-card index and confidence-qualified
output. It matches ordered evidence prefixes globally across all 50,000 items,
returns one candidate while a prefix remains ambiguous, rotates alternatives
across turns, and restores a full window on the final turn. Free-form messages
automatically retain the CP4 lexical fallback.

The selected full-public result is Hit Rate@10 `1.0`, MRR `1.0`, MTTC `2.14`,
and TechnicalScore `0.977200`. It uses only the Python standard library at
runtime; the CP4 model is deliberately disabled after a no-effect ablation.

```bash
python -m scripts.evaluate_cp5_variant full --output data/releases/cp5/final/full.json
```

See `CP5_EXPERIMENTS.md` for the complete workflow, target-disjoint validation,
all accepted and rejected strategies, runtime-parity audit, and limitations.

## CP6 Agent

The `sri-experiment-cp6` branch audits 15 public solutions and promotes the one
remaining mechanism that improves CP5 consistently: exact coarse-category
scoping inside every FTS5 retrieval route. It preserves full-public Hit Rate@10
and MRR at `1.0`, reduces MTTC from `2.14` to `2.10`, and raises
TechnicalScore from `0.977200` to `0.978000`. The gain repeats on the public
development split, frozen holdout, and three target-disjoint validation sets.

```bash
python -m scripts.evaluate_cp6_variant full --output data/releases/cp6/final/full.json
```

See `CP6_EXPERIMENTS.md` for the pinned repository audit, isolated strategy
grid, rejected neural/release/profile alternatives, and reproduction commands.

## Synthetic Test Cases

Beyond the 200 labeled public sessions, the agent can be validated against
larger, **target-disjoint** synthetic session sets. Every synthetic target is
disjoint from all public targets (and, for the split sets, from every other
split), so scores never leak the public labels. Sessions carry only public
fields; the evaluator derives the hidden intent cards from the catalog at run
time, exactly as with the official set.

Generation is deterministic (fixed seeds). Regenerate the sets with:

```bash
# 5000 sessions split into synthetic_train (3000) / synthetic_dev (1000) /
# synthetic_holdout (1000), all target-disjoint, official 40/40/15/5 mix.
python3 -m scripts.create_cp4_synthetic_sets

# 500 sessions whose targets are hard for the exact-evidence funnel
# (common attributes, many neighbors, sparse/low-popularity products).
python3 -m scripts.create_cp4_hardcase_set
```

Both scripts write to `data/releases/cp4/` and print a JSON summary with a
SHA-256 per file and the scenario counts.

Run the agent against any generated set through the real evaluator loop
(aggregate metrics only — holdout sessions are never inspected individually):

```bash
# Confirmation run on the synthetic holdout partition.
python3 -m scripts.evaluate_cp4_confirm --dataset data/releases/cp4/synthetic_holdout.jsonl

# Or the dev split / hard-case set.
python3 -m scripts.evaluate_cp4_confirm --dataset data/releases/cp4/synthetic_dev.jsonl
python3 -m scripts.evaluate_cp4_confirm --dataset data/releases/cp4/hard_cases.jsonl

# Restrict a full dataset to the public holdout ids from the CP2 manifest.
python3 -m scripts.evaluate_cp4_confirm --dataset data/public_set.jsonl --public-holdout
```

An **UNOFFICIAL** paraphrase-robustness diagnostic reworks the simulator's
wording (framing verbs, list delimiters, sentence boundaries) while keeping all
catalog-derived values verbatim, then reports the clean-vs-paraphrased gap. Its
scores are diagnostic only and are never comparable to the official evaluator:

```bash
python3 -m tests.paraphrase_harness                 # full public set
python3 -m tests.paraphrase_harness --limit 25      # quick smoke run
python3 -m tests.paraphrase_harness --output docs/cp4_paraphrase_report.json
```

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
