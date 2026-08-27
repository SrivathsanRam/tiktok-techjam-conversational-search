from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "actually", "additional", "ask", "earlier", "exploring", "ignore", "judgment",
    "key", "matters", "need", "options", "preference", "prioritize", "requirement",
    "right", "specific", "still", "those", "use", "what", "yet",
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
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict[str, object]] = {}
        self._build_index()

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
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
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
            for marker in ("don't have a preference", "do not have a preference", "not quite right")
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

    def _fused_search(self, terms: list[str], top_k: int) -> list[str]:
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
            (" OR ".join(quoted), 1.0),
        ]
        scores: dict[str, float] = {}
        best_route_rank: dict[str, int] = {}
        for expression, weight in expressions:
            if not expression:
                continue
            for rank, parent_asin in enumerate(self._ranked_asins(expression), start=1):
                scores[parent_asin] = scores.get(parent_asin, 0.0) + weight / (20.0 + rank)
                best_route_rank[parent_asin] = min(best_route_rank.get(parent_asin, rank), rank)
        ordered = sorted(scores, key=lambda asin: (-scores[asin], best_route_rank[asin], asin))
        return ordered[:top_k]

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
            state["messages"] = [user_message]
        elif self._is_override(user_message):
            state["messages"] = [str(state["base_message"]), user_message]
        elif self._has_preference(user_message):
            messages = state["messages"]
            if isinstance(messages, list):
                messages.append(user_message)

        query_text = " ".join(str(item) for item in state["messages"])
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:80]
        recommendations = [
            {"parent_asin": parent_asin}
            for parent_asin in self._fused_search(unique_terms, top_k)
        ]
        return {
            "message": "Here are the closest matches. What other requirement matters most?",
            "ask_attribute": "other",
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
