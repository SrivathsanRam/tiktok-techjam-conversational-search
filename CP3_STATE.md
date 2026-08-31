# CP3 current implementation state

This report describes the checked-out implementation in `starter/agent.py` and
the checked-out `starter/reranker_weights.json`. The failure table is a replay
of the current 200-session public set. Its aggregate result is HitRate@10
`0.995`, MRR `0.845734`, MTTC `1.93`, and TechnicalScore `0.932620`.

## 1. Per-turn pipeline

`reset(session_id, user_profile)` must have been called first. `respond()`
raises `RuntimeError` if the session is absent. The ordered path from
`user_message` to the returned dictionary is as follows.

1. **Mutate conversation state (`respond`, `_base_intent`, `_is_override`,
   `_has_preference`).** Inputs are `session_id`, `user_message`, `turn`, and
   the existing session state.

   - On `turn == 1`, `_base_intent(user_message)` returns everything before the
     first `.` and stores it as `base_message`. `exploratory` becomes true if
     the lowercased message contains `still exploring`, `just browsing`, or
     `not sure yet`. `messages` is replaced by `[user_message]`.
   - On a later override, `_is_override()` recognizes any occurrence of
     `actually`, `instead of`, `forget `, or `ignore my earlier`. The stored
     initial message is replaced with `base_message`; every stored message
     after the opener is retained; the new override message is appended.
   - On another later turn, `_has_preference()` decides whether to append the
     message. It returns false for five literal substrings listed in section 5.
     A false result leaves `messages` unchanged.

2. **Build the lexical query (`_terms` inside `respond`).** All stored messages
   are joined with spaces. `_terms()` extracts `[a-z0-9]+` tokens
   case-insensitively, lowercases them, drops one-character tokens, and drops
   the fixed `STOPWORDS` set. `dict.fromkeys` removes duplicates while
   preserving first occurrence. Only the first **80** unique terms are kept.
   Output: `unique_terms`.

3. **Sparse multi-route retrieval (`_fused_search`, `_ranked_asins`).** Inputs
   are `unique_terms`, a requested output size of **60**
   (`RERANK_CANDIDATES`), a disjunctive weight determined by `exploratory`, and
   a popularity weight determined by `exploratory`.

   `_ranked_asins()` runs each nonempty FTS5 expression and keeps at most
   **150** rows. Its BM25 call uses column weights `0.0, 6.0, 4.0, 2.5, 2.5,
   1.5, 1.0` for, respectively, the unindexed ASIN, title, categories,
   features, details, store, and description. FTS5 uses `unicode61` with
   `remove_diacritics 2`.

   The three routes and reciprocal-rank-fusion contributions are:

   | Route | FTS expression | Weight |
   |---|---|---:|
   | Conjunctive | every term quoted and joined by `AND` | `2.5` |
   | Adjacent phrase | every adjacent two-term phrase quoted and joined by `OR` | `1.25` |
   | Disjunctive | every quoted term joined by `OR` | `1.0` when exploratory, otherwise `2.0` |

   A route result at one-based rank `r` adds `weight / (20 + r)`. The literal
   **20** is the RRF rank constant. For non-exploratory state only, products
   already present in the route union are sorted by catalog popularity and get
   an additional `1.0 / (20 + popularity_rank)`; exploratory state uses
   popularity weight `0.0`. Popularity is `log1p(max(0, rating_number))`, with
   missing or invalid values treated as zero. Final ties use best route rank,
   then ASIN. Output: at most 60 sparse candidates.

4. **Parse disclosed constraints (`_constraint_phrases`).** Inputs are all
   stored messages. The exact parsing and lookup rules are in section 2.
   Output: an ordered, de-duplicated list of constraint strings.

5. **Build the exact-evidence lane (`_exact_evidence_candidates`).** Each
   constraint is looked up by normalized exact equality. Constraints with no
   postings are omitted. For every posting list, a product gets one match and
   rarity `ln(50001 / (df + 1))`. Only products with the maximum number of
   matched posting lists survive. They are sorted by match count, summed
   rarity, `log1p(rating_number)` popularity, and ASIN, all descending except
   ASIN. At most **60** survive (`exact_candidate_limit`).

   There is a single-value document-frequency guard: if the maximum match
   count is one and the **first nonempty posting list** has more than
   `exact_single_max_df == 50000` rows, the exact lane is discarded. With the
   fixed 50,000-product catalog and one de-duplicated row per product/value,
   the strict `> 50000` test does not fire.

6. **Merge candidate lanes (`respond`).** Exact candidates come first, followed
   by sparse candidates not already present. The merged list is truncated to
   **80** (`rerank_candidate_limit`). If the exact lane is empty, no padding
   route is added, so the pool remains at most 60. The pool and lexical terms
   are saved as `last_candidates` and `last_query_terms`.

7. **Learned reranking (`_rerank`, `_feature_vector`).** Inputs are the merged
   pool, session state, query terms, and the whole merged-pool length as
   `top_k`. Each candidate's pre-rerank rank is its one-based position in the
   merged list. Sixteen features are computed and dotted with the 16 loaded
   weights. Sort order is descending score, then original pool rank, then ASIN.
   Section 4 gives every formula and weight. Output: the entire merged pool in
   learned order.

   With the checked-in weights file, the fallback scorer is inactive. If the
   file is absent or `load_reranker=False`, `_fallback_structured_score()` uses
   coefficients `3.0, 0.6, 0.9, 1.6, 0.3, 2.2, 0.8, 0.6, 0.5, 0.8` for
   coverage through budget match, divides by `11.3`, and blends that score with
   retrieval using structured weight `0.10` when exploratory and `0.15`
   otherwise.

8. **Select or rotate (`respond`).** The signature is the exact tuple of the
   up-to-80 unique query terms. On a rotation turn, section 3's exclusion logic
   is applied. Otherwise the first `top_k` reranked products are selected.
   Selected ASINs are added to `shown`, and the signature is stored as
   `last_signature`.

9. **Return (`respond`).** Every selected ASIN becomes
   `{"parent_asin": parent_asin}`. The exact returned object is:

   ```python
   {
       "message": "Here are the closest matches. What other requirement matters most?",
       "ask_attribute": "other",
       "recommendations": recommendations,
       "usage": {"prompt_tokens": 0, "completion_tokens": 0},
   }
   ```

   The public evaluator supplies `top_k == 10`, removes invalid and duplicate
   identifiers, and scores its first 10 valid unique ASINs.

## 2. Exact-evidence index

### Indexed strings

For each catalog product, `_build_index()` constructs raw evidence values from:

- `features` and `details` only: a dictionary contributes `"key: value"` for
  each nonempty item; a list contributes each nonempty element; a scalar
  contributes its string value;
- every material in `MATERIAL_TERMS` that appears as a `_terms()` token anywhere
  in title, categories, features, details, store, or description, contributed
  as the bare material word; and
- every color in `COLOR_TERMS` that appears as a `_terms()` token in those same
  searchable fields, contributed as `"color: colorword"`.

It then also indexes every nonempty semicolon-delimited fragment of every raw
value, provided the fragment is not identical to the complete value. Complete
values remain indexed alongside their fragments. A set de-duplicates
normalized evidence within one product. Titles, categories, stores, and
descriptions are not indexed as complete exact values; they are used only to
detect the material and color aliases.

`_normalized_value()` applies `TOKEN_RE = [a-z0-9]+` case-insensitively,
lowercases every match, retains one-character and low-information words, and
joins matches with one ASCII space. It performs no stopword removal, stemming,
synonym expansion, or token reordering. The SQLite table stores
`(normalized_value, parent_asin)` and has an index on `normalized_value`.

### Constraint-to-key parsing

`_constraint_phrases(messages)` lowercases each stored message and looks, in
this order, for one of:

1. `key requirement is:`
2. `what matters is:`
3. `what i need is:`

For the first marker found in a message, it takes everything after the first
occurrence, splits it on `;`, strips characters from `" .;,-"` at both ends,
and keeps nonempty fragments. It processes no second marker in that message.
The final list is de-duplicated in first-seen order. Each fragment is then
normalized with `_normalized_value()` and queried by exact SQL equality.

### Miss behavior and fallback

`_evidence_postings()` returns an empty tuple for an empty normalized key, a
disabled exact index, or a key with no exact row. Empty results are cached.
When some constraint keys miss, those constraints do not affect exact-lane
match counts, rarity, or the denominators of the two exact-evidence reranker
features. The keys that did hit still construct the exact lane. When all keys
miss (or there are no parsed constraints, or the document-frequency guard
discards the lane), `_exact_evidence_candidates()` returns `[]`; `respond()`
then uses the sparse top-60 pool unchanged. The ordinary lexical coverage,
constraint-coverage, and exact-phrase features still operate on the stored
message text, while both exact-evidence features are zero.

## 3. Rotation

Rotation runs exactly when all of the following are true:

- `use_coverage_rotation` is true (the default);
- the prior turn's `last_signature` equals the current tuple of unique query
  terms exactly; and
- `shown` is a set.

The default `coverage_head` is **0**. Therefore no ranked head is protected,
and selection takes the first `top_k` products in the current reranked list
that are not in `shown`. If fewer than `top_k` unseen products exist, a refill
pass walks the complete ranked list, excludes only products already in the
new selection, and may reintroduce previously shown products until `top_k` is
reached. The selected products are then added to the cumulative `shown` set.

Override handling does **not** clear `shown`, `last_signature`, or any product
history. An override normally changes query terms, so the override turn itself
does not rotate because its signature differs. On a later unchanged turn,
however, the exclusion set still includes products shown before the override.
The evaluator does not score target appearances before the configured override
turn. Consequently, those pre-override products are not established failures,
and the implementation's persistent exclusion can suppress products that were
shown before they became scorable.

## 4. Sixteen reranker features

Notation: `Q` is the set of query terms; `T`, `K`, `A`, `D`, `S`, and `X` are
the `_terms()` sets for title, categories, features, details, store, and
description; `U = T ∪ K ∪ A ∪ D ∪ S ∪ X`; and
`d = max(1, |Q|)`. `C` is the ordered parsed-constraint list. For each
constraint with a nonempty `_terms()` set `c`, `cov(c) = |c ∩ U| / |c|`.

| # | Feature | Implemented formula | Current weight |
|---:|---|---|---:|
| 1 | `retrieval` | `1 / log2(pool_rank + 1)` | `0.734030354577` |
| 2 | `coverage` | `|Q ∩ U| / d` | `7.317681445927` |
| 3 | `title_coverage` | `|Q ∩ T| / d` | `-0.030846761625` |
| 4 | `category_coverage` | `|Q ∩ K| / d` | `9.670272666721` |
| 5 | `attribute_coverage` | `|Q ∩ (A ∪ D)| / d` | `0.123704470935` |
| 6 | `description_coverage` | `|Q ∩ X| / d` | `-0.674750239114` |
| 7 | `constraint_coverage` | mean `cov(c)` over nonempty constraints; otherwise `coverage` | `-0.133940298004` |
| 8 | `exact_fraction` | fraction of nonempty constraints whose space-joined, stopword-filtered token string is a Python substring of the space-joined, stopword-filtered combined field text | `2.725928058744` |
| 9 | `material_match` | fraction of requested `MATERIAL_TERMS` present in `U`; `0` if none requested | `0.839673076301` |
| 10 | `color_match` | fraction of requested `COLOR_TERMS` present in `U`; `0` if none requested | `0.518131470744` |
| 11 | `budget_match` | budget function below | `0.005175134308` |
| 12 | `popularity` | `log1p(max(0, rating_number)) / max_catalog_log_popularity` | `8.665881148482` |
| 13 | `exploratory_popularity` | `popularity` if exploratory, else `0` | `2.978293166876` |
| 14 | `specific_constraint` | `constraint_coverage` if not exploratory, else `0` | `-0.451630532505` |
| 15 | `exact_evidence_coverage` **(new since CP2)** | matched indexed constraints / indexed constraints | `5.331523336219` |
| 16 | `exact_evidence_rarity` **(new since CP2)** | sum over matched indexed constraints of `[ln(50001/(df+1)) / ln(50001/2)]`, divided by indexed-constraint count | `2.716774718290` |

For features 15 and 16, an “indexed constraint” is one with a nonempty exact
posting list; the denominator is `max(1, indexed_constraint_count)`. Thus exact
lookup misses are excluded rather than counted as unmatched constraints.

`budget_match` is `0` when price is missing. The first recognized form is
`budget around $N`: with `scale = max(10, 0.25N)`, its value is
`max(0, 1 - |price-N|/scale)`. Otherwise `(under|below|up to|<=) $N` returns
`1` at or below the cap and `-1` above it. If neither regex matches, it is `0`.

## 5. Session state

`reset()` replaces the session entry with:

| Key | Reset value | Later use/mutation |
|---|---|---|
| `base_message` | `""` | Set on turn 1 from text before the first period; used when rebuilding an override conversation. |
| `exploratory` | `False` | Set only on turn 1 from the three browsing markers; never recalculated on override. |
| `messages` | `[]` | Replaced on turn 1, appended for preference-bearing replies, rebuilt on override, unchanged for recognized no-preference replies. |
| `user_profile` | the object passed to `reset` | Stored but never read anywhere in `respond()` or its called ranking functions. |
| `shown` | empty `set()` | Cumulatively updated with every selected ASIN; never reset inside a session. |
| `last_signature` | `None` | Replaced after every response with the current query-term tuple. |

`respond()` also creates or replaces `last_candidates` and `last_query_terms`
on every turn.

An override rebuilds `messages` as `[base_message, *old_messages[1:],
user_message]`: it removes only the stale opener, retains disclosures after the
opener, and appends the override. It does not change `exploratory`, `shown`, or
the preceding signature.

`_has_preference()` prevents a message from entering `messages` if its
lowercased form contains any of:

- `don't have a preference`
- `don't have an additional preference`
- `do not have a preference`
- `do not have an additional preference`
- `not quite right`

The local boundary reply (`I don't have a preference for other; please use
your judgment.`), ordinary exhausted-preference reply (`I don't have an
additional preference for other.`), and invalid-question reply containing
`not quite right` therefore do not alter `messages`. They still cause the
normal per-turn updates to candidate diagnostics, `shown`, and
`last_signature`; because the query signature stays equal, they normally
trigger rotation.

## 6. Current miss and ranks below the top three

“Pool position” below is the one-based position after exact-first lane merging
and before learned reranking. Constraints are the evaluator's catalog-derived
values disclosed by the start of the hit turn. The miss lists everything
disclosed by turn 10. “Reranked low” means the target was in the pool but the
learned score placed it below rank 3; it is not a retrieval miss.

| `sample_id` | Scenario | First hit | Best rank | Constraints disclosed by that turn | Stage that placed the target too low |
|---|---|---:|---:|---|---|
| `public_0002` | `intent_override` | 3 | 5 | `leather`; `100% Leather` | In pool (position 8), but reranked to 5. |
| `public_0011` | `browsing` | 1 | 4 | None. | In pool (position 34), but reranked to 4. |
| `public_0020` | `buying` | — | — | `cotton`; `color: grey`; `Solid colors: 100% Cotton; Heather Grey: 90% Cotton, 10% Polyester; All Other Heathers: 50% Cotton, 50% Polyester`; `Imported` | Retrieved but outside the pool. After full disclosure the target is exact-lane rank 463, while that lane keeps 60; it is absent from sparse top 60 and never reaches reranking. At turn 1 it was also below both cutoffs (sparse fused rank 77 and exact rank 9575). |
| `public_0068` | `intent_override` | 3 | 4 | `Imported`; `Rubber sole` | In pool (position 32), but reranked to 4. |
| `public_0076` | `browsing` | 3 | 6 | `cotton`; `color: grey`; `Solid colors: 80% Cotton, 20% Polyester; Heather Grey: 78% Cotton, 22% Poly; Dark Heather: 50% Cotton, 50% Polyester`; `Imported` | In pool (position 7), but reranked to 6. |
| `public_0092` | `browsing` | 2 | 7 | `polyester`; `95% Polyester, 5% Spandex` | In pool (position 78), but reranked to 7. |
| `public_0140` | `browsing` | 1 | 4 | None. | In pool (position 57), but reranked to 4. |
| `public_0141` | `browsing` | 1 | 6 | None. | In pool (position 13), but reranked to 6. |
| `public_0144` | `intent_override` | 4 | 5 | `polyester`; `100% Polyester`; `Imported`; `Zipper closure` | In pool (position 38), but reranked to 5. |
| `public_0154` | `buying` | 1 | 6 | `cotton` | In pool (position 78), but reranked to 6. |
| `public_0161` | `buying` | 2 | 5 | `cotton`; `cotton blend`; `Imported` | In pool (position 14), but reranked to 5. |
| `public_0164` | `browsing` | 2 | 7 | `leather`; `color: black` | In pool (position 63), but reranked to 7. |
| `public_0167` | `browsing` | 1 | 9 | None. | In pool (position 55), but reranked to 9. |
| `public_0172` | `browsing` | 2 | 5 | `cotton`; `100% Cotton` | In pool (position 65), but reranked to 5. |
| `public_0200` | `buying` | 1 | 5 | `Ethylene Vinyl Acetate sole` | In pool (position 7), but reranked to 5. |

The first-hit rows above include every current session with `best_rank > 3`;
`public_0020` is the only current miss.

## 7. Robustness audit: dependencies on simulator wording

| Wording dependency | Exact implementation dependency | Effect of a paraphrase |
|---|---|---|
| Constraint template markers | `_constraint_phrases()` recognizes only `key requirement is:`, `what matters is:`, and `what i need is:`. | A differently introduced constraint is absent from the exact lane and both exact features. It remains only as undistilled lexical query text. |
| Semicolon delimiter | After a recognized marker, every `;` starts a new lookup phrase. Index construction likewise adds semicolon fragments. | Another list delimiter leaves multiple constraints in one exact key, which normally misses. A semicolon used as ordinary prose also splits one value into separate keys. |
| Browsing language | Exploratory routing requires `still exploring`, `just browsing`, or `not sure yet`. | An unrecognized browsing paraphrase is treated as non-exploratory: disjunctive weight changes from 1 to 2, the popularity route turns on, exploratory popularity turns off, and `specific_constraint` turns on. |
| Override language | `_is_override()` requires `actually`, `instead of`, `forget `, or `ignore my earlier`. | An unrecognized override is appended as an ordinary message, so the stale initial preference remains in the query. The base-intent replacement does not occur. |
| Override constraint marker | Exact constraint extraction from the local override specifically depends on `what i need is:` even if `_is_override()` recognizes the turn for another reason. | State can reset correctly while the replacement constraint still fails to enter the exact lane. |
| Initial sentence boundary | `_base_intent()` keeps text before the first literal period. | Different punctuation can retain stale preference text in `base_message`, or an earlier period can truncate the intended category, when a later override rebuilds state. |
| No-preference language | `_has_preference()` has four literal `don't/do not have [an additional] preference` substrings. | An unrecognized equivalent is appended, adds framing terms to retrieval, changes the signature, and prevents the intended unchanged-query rotation. |
| Boundary template | The local boundary sentence is ignored only because it contains `don't have a preference`. | A boundary paraphrase outside the four substrings is treated as positive preference evidence and has the same query-pollution and no-rotation effect. |
| “Options are wrong” template | Only the substring `not quite right` is suppressed. | A paraphrased negative response is appended as if it were a constraint and changes retrieval/signature. |
| Fixed framing stopwords | `STOPWORDS` explicitly removes words from the public templates, including `looking`, `still`, `exploring`, `key`, `requirement`, `what`, `matters`, `actually`, `ignore`, `earlier`, `preference`, `additional`, `other`, and `judgment`. | New framing words not in the fixed set become query terms and can change all sparse routes and coverage features. |
| Verbatim catalog values | Evidence lookup is normalized exact equality: punctuation/case changes are tolerated, but token insertion, deletion, substitution, reordering, number-word conversion, and synonymy are not. | Even after a marker is recognized, paraphrased catalog evidence commonly has no posting and is omitted from the exact lane/features. Sparse lexical matching is the remaining path. |
| Budget language | `_budget_score()` recognizes only `budget around`, or `under`, `below`, `up to`, and `<=`, followed by a number and optional `$`. | Other budget paraphrases receive budget feature value zero. |
| Structured `other` replies | The agent always returns `ask_attribute="other"`; the local simulator treats `other` as matching any undisclosed constraint and returns at most two. | This is a policy/template coupling rather than prose parsing in the agent. If a private policy assigns different semantics to `other`, the disclosure sequence and all later state differ. |

The specification explicitly states: “If natural-language paraphrasing is
added by the organizer, it cannot decide correctness. Hits are always exact
code matches.” Thus ASIN scoring itself remains exact-match, but paraphrasing
can change which ASINs this implementation retrieves, reranks, and rotates.

## 8. Submission-rule audit of the current repository

- **Runtime dependencies.** `starter/agent.py` imports only standard-library
  modules (`json`, `math`, `re`, `sqlite3`, `collections`, and `pathlib`). It
  requires a Python `sqlite3` build with FTS5 support. There is no
  `requirements.txt`, `pyproject.toml`, or other dependency manifest.
  `scripts/train_cp2_reranker.py` (also the implementation behind
  `train_cp3_reranker.py`) imports NumPy and scikit-learn; these third-party
  training dependencies are named in prose and in its error message but have
  no pinned versions or installation manifest. The README says Python 3.10 or
  later is “recommended,” not an exact interpreter requirement.

- **Network and credentials.** No Python source in the repository performs a
  network call, imports an HTTP/API client, reads credentials, or requires a
  live external service. The runtime is offline and reports zero tokens. The
  only URLs found are documentation/schema links. No API key, `.env`, or other
  credential file is present in the working tree inspected for this report.

- **Paths.** Runtime defaults use the relative path `data/catalog.jsonl`; the
  weights path is resolved beside `starter/agent.py`. There are no absolute
  paths in runtime code. `CP2_EXPERIMENTS.md` and `CP3_EXPERIMENTS.md` contain a
  Windows-specific absolute training interpreter path,
  `C:\Program Files\Python313\python.exe`. The default catalog path also means
  construction depends on a working directory from which
  `data/catalog.jsonl` resolves.

- **Files present outside the tracked source set.** The working tree contains
  ignored `data/catalog.jsonl` (about 60.5 MB), ignored `results.json` with
  per-session public evaluation output, ignored `__pycache__` directories, and
  untracked root files `catalog.jsonl.gz` (about 19.2 MB) and `SHA256SUMS`.
  These would be included by a filesystem-wide archive even though Git does
  not track them. A filesystem-wide archive would also include the `.git`
  directory. The repository contains the labeled public development set, its
  split manifest, evaluator, tests, and training/analysis scripts; none is read
  by `Agent` at runtime. No `organizer/`, `secure/`, private evaluation data, or
  organizer-only file is present.

- **Submission layout and runtime artifacts.** The entry file is
  `starter/agent.py`, not a top-level `agent.py` or the documented recommended
  `submission/agent.py`. Its required fitted artifact is the small tracked
  `starter/reranker_weights.json`. The repository has no dedicated submission
  directory or package manifest. The agent does not modify evaluator files or
  catalog data.
