# Submission Report

## Architecture

The agent is a dependency-free two-stage retrieval system.

First, it builds an in-memory SQLite FTS5 index over the frozen catalog fields: title, categories, features, details, store, and description. Query terms are expanded with a small clothing-focused synonym map covering common variants such as sneakers/shoes, tee/shirt, hoodie/sweatshirt, grey/gray, and comfort/comfy.

Second, it blends candidates from several lexical routes: the latest user message, active conversation state, full conversation history, and anonymized profile tags. This preserves recall while letting recent clarifications carry more rank weight.

Third, it reranks candidates using accumulated conversation state. The reranker weights matches by field importance, with title and category matches strongest, followed by features, details, store, and description. It also boosts explicit material, color, and budget signals, including hard treatment for "under budget" constraints. Product rating and rating count are used only as light tie-breakers.

The agent tracks session-level dialogue history, so later clarifications and intent updates influence ranking. If the customer says they are changing or ignoring an earlier preference, older conversation terms are strongly downweighted. The agent also emits structured `ask_attribute` values in a fixed, non-repeating order, starting with a broad `other` clarification to reveal one additional requirement when the simulator supports it.

## Model Choice

No LLM or external model is used. The system relies on SQLite FTS5 and deterministic Python reranking.

## Network and Offline Behavior

The agent does not require network access, external services, credentials, API keys, or vector databases. It can run in an offline final-scoring environment as long as the official catalog file is available to the harness.

## Cost and Token Usage

- Estimated model cost: USD 0.00
- Prompt tokens: 0
- Completion tokens: 0
- External API calls: 0

## Latency

The agent builds an in-memory index at initialization. Per-turn inference is deterministic local retrieval plus reranking over a bounded candidate pool. Actual latency should be measured by the official harness on its hardware.

## Limitations

The ranker is lexical rather than semantic, so it may still miss paraphrases outside the curated synonym map. It does not train on private labels, use dense embeddings, or perform model-based query rewriting. The quality prior can help break ties but may not reflect a specific user's hidden target.

## Demonstrated Multi-Turn Behavior

Example flow:

1. User says they are looking for winter boots.
2. Agent returns candidate products and asks for material.
3. User says leather and black matter.
4. Agent reranks candidates that explicitly match leather and black above otherwise similar products.

## Team Contributions

Implementation and documentation in this bundle cover the deterministic retrieval/reranking agent, submission setup notes, and reproducibility disclosures. Replace this section with named individual contributions if the final submission portal requires them.
