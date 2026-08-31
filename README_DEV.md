# Developer Guide: Running the Synthetic Test Cases

Check out `synth-cp6`, then run the remaining commands from the repository
root:

```bash
git checkout synth-cp6
```

If you use the repository's virtual environment, activate it first:

```bash
source .venv/bin/activate
```

The commands below expect `data/catalog.jsonl` and `data/public_set.jsonl` to
be present.

## Generate the synthetic datasets

Generate the deterministic train, development, and holdout splits:

```bash
python3 -m scripts.create_cp4_synthetic_sets
```

This creates:

- `data/releases/cp4/synthetic_train.jsonl`: 3,000 sessions for tuning.
- `data/releases/cp4/synthetic_dev.jsonl`: 1,000 sessions for repeated
  evaluation.
- `data/releases/cp4/synthetic_holdout.jsonl`: 1,000 sessions for final
  confirmation.

The targets are unique across the three splits and disjoint from the public
set's targets. Generation uses a fixed seed, so repeated runs against the same
catalog produce the same files and SHA-256 checksums.

Generate the additional hard-case set:

```bash
python3 -m scripts.create_cp4_hardcase_set
```

This creates `data/releases/cp4/hard_cases.jsonl`, containing up to 500 targets
selected to be difficult for the exact-evidence retrieval funnel.

## Test CP6 against the generated datasets

Use the development split for normal, repeatable testing:

```bash
python3 -m scripts.evaluate_cp4_confirm \
  --dataset data/releases/cp4/synthetic_dev.jsonl
```

Run the hard-case evaluation with:

```bash
python3 -m scripts.evaluate_cp4_confirm \
  --dataset data/releases/cp4/hard_cases.jsonl
```

Use the synthetic holdout only for final confirmation and inspect only its
aggregate metrics:

```bash
python3 -m scripts.evaluate_cp4_confirm \
  --dataset data/releases/cp4/synthetic_holdout.jsonl
```

The evaluator prints aggregate metrics such as hit rate, MRR, MTTC, efficiency,
and the recommended technical score. It does not print individual holdout
sessions.

Although the script names and output directory contain `cp4`, they test CP6
when run from `synth-cp6`. The tooling originated on `arjo-cp4`, but
`scripts.evaluate_cp4_confirm` imports `starter.agent.Agent` from the currently
checked-out branch.

## Optional public-holdout confirmation

To evaluate only the public holdout IDs recorded in `data/cp2_split.json`:

```bash
python3 -m scripts.evaluate_cp4_confirm \
  --dataset data/public_set.jsonl \
  --public-holdout
```

## Optional paraphrase diagnostic

Run a quick 25-session smoke test:

```bash
python3 -m tests.paraphrase_harness --limit 25
```

Run the diagnostic against the full public set:

```bash
python3 -m tests.paraphrase_harness
```

Save a detailed report, including per-session results:

```bash
python3 -m tests.paraphrase_harness \
  --output docs/cp4_paraphrase_report.json
```

The paraphrase harness changes simulator wording while retaining the original
catalog-derived values. Its score is unofficial and must not be compared
directly with official evaluator scores.
