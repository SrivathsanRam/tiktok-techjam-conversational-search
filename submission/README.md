# TechJam Conversational Search Submission

## Contents

- `agent.py`: submission entry file exporting `Agent`
- `requirements.txt`: dependency manifest
- `REPORT.md`: architecture, model, cost, limitations, and contribution notes

## Requirements

- Python 3.10 or later
- No third-party Python packages
- No network access required at inference time
- Frozen competition catalog available as `data/catalog.jsonl` in the harness environment

## Setup

```bash
python3 -m pip install -r submission/requirements.txt
```

The requirements file is intentionally empty apart from a comment because the agent uses only the Python standard library.

## Local Evaluation

From the repository root, after placing the downloaded catalog at `data/catalog.jsonl`, run:

```bash
python3 -m evaluator.local_evaluator
```

The current `submission/agent.py` mirrors `starter/agent.py`, which is the module imported by the provided local evaluator.

## Official Harness

Point the official harness at:

```text
submission/agent.py
```

The file exports the required `Agent` class with:

```python
reset(session_id: str, user_profile: dict) -> None
respond(session_id: str, user_message: str, turn: int, top_k: int) -> dict
```

The response contains `message`, `ask_attribute`, `recommendations`, and zero token usage because no LLM is used.
