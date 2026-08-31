"""In-memory catalog index: FTS5 text search plus exact-value evidence.

Builds everything the runtime needs from `catalog.jsonl` in a single pass:
the FTS5 table, the exact-evidence value table, popularity, per-product field
views, coarse categories, and catalog-wide token document frequencies.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

from starter.text_utils import (
    COLOR_TERMS,
    MATERIAL_TERMS,
    coarse_category,
    normalized_value,
    terms,
    text_of,
)


BM25_COLUMN_WEIGHTS = "0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0"


class CatalogIndex:
    """Owns the SQLite connection and every per-product lookup table."""

    def __init__(
        self,
        catalog_path: str | Path,
        use_exact_evidence: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.use_exact_evidence = use_exact_evidence
        self.popularity: dict[str, float] = {}
        self.product_views: dict[
            str, tuple[str, str, str, str, str, str, float | None]
        ] = {}
        self.coarse_category_views: dict[str, tuple[str, frozenset[str]]] = {}
        self.token_df: Counter[str] = Counter()
        self._profile_cache: dict[
            str, tuple[tuple[frozenset[str], ...], frozenset[str], str]
        ] = {}
        self._terms_cache: dict[str, frozenset[str]] = {}
        self._build()
        self.max_popularity = max(self.popularity.values(), default=1.0)

    @property
    def product_count(self) -> int:
        return len(self.product_views)

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        if self.use_exact_evidence:
            cursor.execute(
                "CREATE TABLE evidence_values ("
                "normalized_value TEXT NOT NULL, parent_asin TEXT NOT NULL)"
            )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        evidence_batch: list[tuple[str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                try:
                    rating_number = max(0.0, float(product.get("rating_number") or 0.0))
                except (TypeError, ValueError):
                    rating_number = 0.0
                self.popularity[parent_asin] = math.log1p(rating_number)
                title = text_of(product.get("title"))
                categories = text_of(product.get("categories"))
                features = text_of(product.get("features"))
                details = text_of(product.get("details"))
                store = text_of(product.get("store"))
                description = text_of(product.get("description"))
                try:
                    price = (
                        float(product["price"])
                        if product.get("price") not in (None, "") else None
                    )
                except (TypeError, ValueError):
                    price = None
                self.product_views[parent_asin] = (
                    title.lower(),
                    categories.lower(),
                    features.lower(),
                    details.lower(),
                    store.lower(),
                    description.lower(),
                    price,
                )
                coarse = coarse_category(
                    [str(value) for value in product.get("categories") or []]
                )
                self.coarse_category_views[parent_asin] = (
                    normalized_value(coarse),
                    frozenset(terms(coarse)),
                )
                batch.append(
                    (parent_asin, title, categories, features, details, store, description)
                )
                searchable = " ".join(
                    (title, categories, features, details, store, description)
                ).lower()
                searchable_terms = set(terms(searchable))
                self.token_df.update(searchable_terms)
                if self.use_exact_evidence:
                    evidence_batch.extend(
                        (value, parent_asin)
                        for value in self._evidence_values(product, searchable_terms)
                    )
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch
                    )
                    batch.clear()
                    if self.use_exact_evidence:
                        cursor.executemany(
                            "INSERT INTO evidence_values VALUES (?, ?)", evidence_batch
                        )
                        evidence_batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        if self.use_exact_evidence and evidence_batch:
            cursor.executemany("INSERT INTO evidence_values VALUES (?, ?)", evidence_batch)
        if self.use_exact_evidence:
            cursor.execute(
                "CREATE INDEX evidence_value_idx ON evidence_values(normalized_value)"
            )
        self.connection.commit()

    @staticmethod
    def _evidence_values(product: dict, searchable_terms: set[str]) -> set[str]:
        """Normalized exact values indexed for one product."""
        raw_values: list[str] = []
        for field in (product.get("features"), product.get("details")):
            if isinstance(field, dict):
                raw_values.extend(
                    f"{key}: {value}"
                    for key, value in field.items()
                    if value not in (None, "", [])
                )
            elif isinstance(field, list):
                raw_values.extend(str(value) for value in field if value not in (None, ""))
            elif field not in (None, ""):
                raw_values.append(str(field))
        raw_values.extend(MATERIAL_TERMS & searchable_terms)
        raw_values.extend(f"color: {color}" for color in COLOR_TERMS & searchable_terms)
        # Simulator replies use semicolons as a multi-constraint delimiter even
        # when a single catalog feature contains them.  Index both the complete
        # value and each disclosed fragment.
        raw_values.extend(
            fragment.strip()
            for value in tuple(raw_values)
            for fragment in value.split(";")
            if fragment.strip() and fragment.strip() != value.strip()
        )
        return {
            normalized
            for normalized in (normalized_value(value) for value in raw_values)
            if normalized
        }

    def ranked_asins(self, expression: str, limit: int = 150) -> list[str]:
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, {BM25_COLUMN_WEIGHTS}) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def price_of(self, parent_asin: str) -> float | None:
        return self.product_views[parent_asin][6]

    def profile(
        self, parent_asin: str
    ) -> tuple[tuple[frozenset[str], ...], frozenset[str], str]:
        """Per-product term sets and token string, computed once and cached."""
        cached = self._profile_cache.get(parent_asin)
        if cached is not None:
            return cached
        title, categories, features, details, store, description, _ = (
            self.product_views[parent_asin]
        )
        field_texts = (title, categories, features, details, store, description)
        field_terms = tuple(frozenset(terms(text)) for text in field_texts)
        combined_terms = frozenset().union(*field_terms)
        combined_token_string = " ".join(terms(" ".join(field_texts)))
        cached = (field_terms, combined_terms, combined_token_string)
        self._profile_cache[parent_asin] = cached
        return cached

    def value_terms(self, value: str) -> frozenset[str]:
        cached = self._terms_cache.get(value)
        if cached is None:
            cached = frozenset(terms(value))
            self._terms_cache[value] = cached
        return cached
