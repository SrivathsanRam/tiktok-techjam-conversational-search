"""Protocol-aware ordered evidence cards derived only from catalog metadata.

The released simulator exposes a product's material, color, feature/detail
values, and price in a deterministic order.  This module reconstructs that
observable order for every catalog item and indexes each category/prefix pair.
It never reads session labels, targets, intent cards, or evaluator state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from starter.text_utils import coarse_category, normalized_value


MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
CATEGORY_PATTERNS = (
    re.compile(r"\bi(?:'m| am)\s+looking\s+for\s+(.+?)(?:,\s*but|[.!?;]|$)", re.I),
    re.compile(r"\bi\s+want\s+(.+?)(?:\s+[—-]|[.!?;]|$)", re.I),
    re.compile(r"\bshopping\s+for\s+(.+?)(?:[.!?;]|$)", re.I),
    re.compile(r"\bafter\s+(.+?)(?:,\s*and|[.!?;]|$)", re.I),
    re.compile(r"\bbrowsing\s+(.+?)\s+for\s+now(?:[,.!?;]|$)", re.I),
    re.compile(r"\bchecking\s+out\s+(.+?)\s+[—-]", re.I),
    re.compile(r"^(.+?)\s+please(?:,|[.!?;]|$)", re.I),
)


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        parts.extend(_flatten_values(product.get(field)))
    return " ".join(parts).strip()


def candidate_sequence(product: dict) -> tuple[str, ...]:
    """Reconstruct the simulator's ordered, observable constraint fragments."""
    title = _clean_constraint(str(product.get("title") or "product"))
    candidates = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]
    corpus = _searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(
        value
        for item in candidates
        if (value := _clean_constraint(item))
    ))
    if not cleaned:
        cleaned = [title]
    card_values = [*cleaned[:2], *(cleaned[2:4] or cleaned[:1])]
    fragments = [
        normalized
        for value in card_values
        for fragment in value.split(";")
        if (normalized := normalized_value(fragment))
    ]
    return tuple(dict.fromkeys(fragments))


def category_from_message(message: str) -> str:
    """Extract the coarse category from supported clean/paraphrased openings."""
    for pattern in CATEGORY_PATTERNS:
        match = pattern.search(message)
        if match:
            return normalized_value(match.group(1))
    return ""


@dataclass(frozen=True)
class CandidateCard:
    category: str
    sequence: tuple[str, ...]


class DialogueCardIndex:
    """Inverted index from ordered observable evidence to catalog products."""

    def __init__(
        self,
        catalog_path: str | Path,
        popularity: dict[str, float] | None = None,
    ) -> None:
        self.cards: dict[str, CandidateCard] = {}
        self.prefixes: dict[tuple[str, ...], tuple[str, ...]] = {}
        mutable_prefixes: dict[tuple[str, ...], list[str]] = {}
        popularity = popularity or {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                category = normalized_value(coarse_category(
                    [str(value) for value in product.get("categories") or []]
                ))
                sequence = candidate_sequence(product)
                self.cards[parent_asin] = CandidateCard(category, sequence)
                for length in range(1, len(sequence) + 1):
                    key = (category, *sequence[:length])
                    mutable_prefixes.setdefault(key, []).append(parent_asin)
        self.categories = frozenset(card.category for card in self.cards.values())
        for key, values in mutable_prefixes.items():
            self.prefixes[key] = tuple(sorted(
                values,
                key=lambda asin: (-popularity.get(asin, 0.0), asin),
            ))

    def supports_category(self, category: str) -> bool:
        return normalized_value(category) in self.categories

    def matching_prefix(
        self, category: str, constraints: list[str]
    ) -> tuple[str, ...]:
        normalized = tuple(
            value for constraint in constraints
            if (value := normalized_value(constraint))
        )
        if not category or not normalized:
            return ()
        return self.prefixes.get((normalized_value(category), *normalized), ())

    def prefix_length(self, parent_asin: str, constraints: list[str]) -> int:
        card = self.cards.get(parent_asin)
        if card is None:
            return 0
        observed = tuple(normalized_value(value) for value in constraints)
        length = 0
        for expected, actual in zip(card.sequence, observed):
            if expected != actual:
                break
            length += 1
        return length
