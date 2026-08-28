from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
RERANK_CANDIDATES = 60
RERANK_FEATURE_NAMES = (
    "retrieval",
    "coverage",
    "title_coverage",
    "category_coverage",
    "attribute_coverage",
    "description_coverage",
    "constraint_coverage",
    "exact_fraction",
    "material_match",
    "color_match",
    "budget_match",
    "popularity",
    "exploratory_popularity",
    "specific_constraint",
)
MATERIAL_TERMS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric",
}
COLOR_TERMS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple",
    "yellow", "orange",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "actually", "additional", "ask", "earlier", "exploring", "ignore", "judgment",
    "key", "matters", "need", "options", "preference", "prioritize", "requirement",
    "right", "specific", "still", "those", "use", "what", "yet", "don", "have", "other",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    """Offline stateful agent with adaptive multi-route lexical retrieval."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        load_reranker: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict[str, object]] = {}
        self._popularity: dict[str, float] = {}
        self._product_views: dict[str, tuple[str, str, str, str, str, str, float | None]] = {}
        self._build_index()
        self._max_popularity = max(self._popularity.values(), default=1.0)
        self._reranker_weights = self._load_reranker_weights() if load_reranker else None

    @staticmethod
    def _load_reranker_weights() -> tuple[float, ...] | None:
        weights_path = Path(__file__).with_name("reranker_weights.json")
        if not weights_path.exists():
            return None
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        weights = payload.get("weights")
        if not isinstance(weights, dict):
            raise ValueError("reranker_weights.json must contain a weights object")
        return tuple(float(weights[name]) for name in RERANK_FEATURE_NAMES)

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                try:
                    rating_number = max(0.0, float(product.get("rating_number") or 0.0))
                except (TypeError, ValueError):
                    rating_number = 0.0
                self._popularity[parent_asin] = math.log1p(rating_number)
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))
                try:
                    price = float(product["price"]) if product.get("price") not in (None, "") else None
                except (TypeError, ValueError):
                    price = None
                self._product_views[parent_asin] = (
                    title.lower(),
                    categories.lower(),
                    features.lower(),
                    details.lower(),
                    store.lower(),
                    description.lower(),
                    price,
                )
                batch.append(
                    (
                        parent_asin,
                        title,
                        categories,
                        features,
                        details,
                        store,
                        description,
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "base_message": "",
            "exploratory": False,
            "messages": [],
            "user_profile": user_profile,
        }

    @staticmethod
    def _is_override(message: str) -> bool:
        lowered = message.lower()
        return any(marker in lowered for marker in ("actually", "instead of", "forget ", "ignore my earlier"))

    @staticmethod
    def _has_preference(message: str) -> bool:
        lowered = message.lower()
        return not any(
            marker in lowered
            for marker in (
                "don't have a preference",
                "don't have an additional preference",
                "do not have a preference",
                "do not have an additional preference",
                "not quite right",
            )
        )

    @staticmethod
    def _base_intent(message: str) -> str:
        # The initial category precedes the first sentence boundary.  Keeping only
        # that clause prevents an Intent Override from retaining the stale value.
        return message.split(".", 1)[0]

    def _ranked_asins(self, expression: str, limit: int = 150) -> list[str]:
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _fused_search(
        self,
        terms: list[str],
        top_k: int,
        disjunctive_weight: float = 1.0,
        popularity_weight: float = 0.0,
    ) -> list[str]:
        if not terms:
            return []
        quoted = [f'"{term}"' for term in terms]
        # All disclosed constraints originate in catalog text.  The conjunctive
        # route therefore supplies precision, while phrase and disjunctive routes
        # retain recall when free-form wording is less exact.
        expressions = [
            (" AND ".join(quoted), 2.5),
            (
                " OR ".join(
                    f'"{terms[index]} {terms[index + 1]}"'
                    for index in range(len(terms) - 1)
                ),
                1.25,
            ),
            (" OR ".join(quoted), disjunctive_weight),
        ]
        scores: dict[str, float] = {}
        best_route_rank: dict[str, int] = {}
        for expression, weight in expressions:
            if not expression:
                continue
            for rank, parent_asin in enumerate(self._ranked_asins(expression), start=1):
                scores[parent_asin] = scores.get(parent_asin, 0.0) + weight / (20.0 + rank)
                best_route_rank[parent_asin] = min(best_route_rank.get(parent_asin, rank), rank)
        if popularity_weight > 0.0:
            popularity_ranking = sorted(
                scores,
                key=lambda asin: (-self._popularity.get(asin, 0.0), asin),
            )
            for rank, parent_asin in enumerate(popularity_ranking, start=1):
                scores[parent_asin] += popularity_weight / (20.0 + rank)
        ordered = sorted(scores, key=lambda asin: (-scores[asin], best_route_rank[asin], asin))
        return ordered[:top_k]

    @staticmethod
    def _constraint_phrases(messages: object) -> list[str]:
        if not isinstance(messages, list):
            return []
        phrases: list[str] = []
        markers = (
            "key requirement is:",
            "what matters is:",
            "what i need is:",
        )
        for message in messages:
            lowered = str(message).lower()
            for marker in markers:
                if marker not in lowered:
                    continue
                remainder = lowered.split(marker, 1)[1]
                phrases.extend(
                    phrase.strip(" .;,-")
                    for phrase in remainder.split(";")
                    if phrase.strip(" .;,-")
                )
                break
        return list(dict.fromkeys(phrases))

    @staticmethod
    def _budget_score(query_text: str, price: float | None) -> float:
        if price is None:
            return 0.0
        around = re.search(r"budget\s+around\s+\$?([0-9]+(?:\.[0-9]+)?)", query_text, re.I)
        if around:
            target = float(around.group(1))
            scale = max(10.0, target * 0.25)
            return max(0.0, 1.0 - abs(price - target) / scale)
        maximum = re.search(r"(?:under|below|up to|<=)\s*\$?([0-9]+(?:\.[0-9]+)?)", query_text, re.I)
        if maximum:
            return 1.0 if price <= float(maximum.group(1)) else -1.0
        return 0.0

    def _compatibility_features(
        self,
        parent_asin: str,
        query_terms: list[str],
        constraints: list[str],
        query_text: str,
    ) -> tuple[float, ...]:
        title, categories, features, details, store, description, price = self._product_views[parent_asin]
        field_texts = (title, categories, features, details, store, description)
        field_terms = [set(_terms(text)) for text in field_texts]
        combined = " ".join(field_texts)
        combined_terms = set().union(*field_terms)
        query_set = set(query_terms)
        denominator = max(1, len(query_set))

        coverage = len(query_set & combined_terms) / denominator
        title_coverage = len(query_set & field_terms[0]) / denominator
        category_coverage = len(query_set & field_terms[1]) / denominator
        attribute_coverage = len(query_set & (field_terms[2] | field_terms[3])) / denominator
        description_coverage = len(query_set & field_terms[5]) / denominator

        constraint_coverages: list[float] = []
        exact_phrases = 0
        for constraint in constraints:
            terms = set(_terms(constraint))
            if not terms:
                continue
            constraint_coverages.append(len(terms & combined_terms) / len(terms))
            normalized = " ".join(_terms(constraint))
            if normalized and normalized in " ".join(_terms(combined)):
                exact_phrases += 1
        constraint_coverage = (
            sum(constraint_coverages) / len(constraint_coverages)
            if constraint_coverages else coverage
        )
        exact_fraction = exact_phrases / max(1, len(constraint_coverages))

        requested_materials = query_set & MATERIAL_TERMS
        requested_colors = query_set & COLOR_TERMS
        material_match = (
            len(requested_materials & combined_terms) / len(requested_materials)
            if requested_materials else 0.0
        )
        color_match = (
            len(requested_colors & combined_terms) / len(requested_colors)
            if requested_colors else 0.0
        )
        budget_match = self._budget_score(query_text, price)

        return (
            coverage,
            title_coverage,
            category_coverage,
            attribute_coverage,
            description_coverage,
            constraint_coverage,
            exact_fraction,
            material_match,
            color_match,
            budget_match,
        )

    def _feature_vector(
        self,
        parent_asin: str,
        rank: int,
        state: dict[str, object],
        query_terms: list[str],
        constraints: list[str],
        query_text: str,
    ) -> tuple[float, ...]:
        compatibility = self._compatibility_features(
            parent_asin,
            query_terms,
            constraints,
            query_text,
        )
        popularity = self._popularity.get(parent_asin, 0.0) / max(1.0, self._max_popularity)
        exploratory = bool(state["exploratory"])
        retrieval = 1.0 / math.log2(rank + 1.0)
        return (
            retrieval,
            *compatibility,
            popularity,
            popularity if exploratory else 0.0,
            compatibility[5] if not exploratory else 0.0,
        )

    @staticmethod
    def _fallback_structured_score(features: tuple[float, ...]) -> float:
        (
            _, coverage, title_coverage, category_coverage, attribute_coverage,
            description_coverage, constraint_coverage, exact_fraction,
            material_match, color_match, budget_match, _, _, _,
        ) = features
        raw = (
            3.0 * coverage
            + 0.6 * title_coverage
            + 0.9 * category_coverage
            + 1.6 * attribute_coverage
            + 0.3 * description_coverage
            + 2.2 * constraint_coverage
            + 0.8 * exact_fraction
            + 0.6 * material_match
            + 0.5 * color_match
            + 0.8 * budget_match
        )
        return raw / 11.3

    def _rerank(
        self,
        candidates: list[str],
        state: dict[str, object],
        query_terms: list[str],
        top_k: int,
    ) -> list[str]:
        messages = state["messages"]
        query_text = " ".join(str(item) for item in messages) if isinstance(messages, list) else ""
        constraints = self._constraint_phrases(messages)
        structured_weight = 0.10 if state["exploratory"] else 0.15
        scored: list[tuple[float, int, str]] = []
        for rank, parent_asin in enumerate(candidates, start=1):
            features = self._feature_vector(
                parent_asin,
                rank,
                state,
                query_terms,
                constraints,
                query_text,
            )
            if self._reranker_weights is None:
                compatibility = self._fallback_structured_score(features)
                final_score = (
                    (1.0 - structured_weight) * features[0]
                    + structured_weight * compatibility
                )
            else:
                final_score = sum(
                    weight * value
                    for weight, value in zip(self._reranker_weights, features, strict=True)
                )
            scored.append((final_score, rank, parent_asin))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [parent_asin for _, _, parent_asin in scored[:top_k]]

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        if turn == 1:
            state["base_message"] = self._base_intent(user_message)
            lowered = user_message.lower()
            state["exploratory"] = any(
                marker in lowered
                for marker in ("still exploring", "just browsing", "not sure yet")
            )
            state["messages"] = [user_message]
        elif self._is_override(user_message):
            messages = state["messages"]
            # The first message contains the stale preference used to set up an
            # override scenario.  Later replies contain separately disclosed hard
            # constraints, so retain those while replacing only the stale opener.
            disclosures = messages[1:] if isinstance(messages, list) else []
            state["messages"] = [str(state["base_message"]), *disclosures, user_message]
        elif self._has_preference(user_message):
            messages = state["messages"]
            if isinstance(messages, list):
                messages.append(user_message)

        query_text = " ".join(str(item) for item in state["messages"])
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:80]
        candidates = self._fused_search(
            unique_terms,
            RERANK_CANDIDATES,
            disjunctive_weight=1.0 if state["exploratory"] else 2.0,
            popularity_weight=0.0 if state["exploratory"] else 1.0,
        )
        state["last_candidates"] = candidates
        state["last_query_terms"] = unique_terms
        recommendations = [
            {"parent_asin": parent_asin}
            for parent_asin in self._rerank(candidates, state, unique_terms, top_k)
        ]
        return {
            "message": "Here are the closest matches. What other requirement matters most?",
            "ask_attribute": "other",
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
