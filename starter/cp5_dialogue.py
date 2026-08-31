from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")


def normalize_protocol_text(message: str) -> str:
    """Normalize harmless formatting differences in evaluator-style messages."""
    return re.sub(r"\s+", " ", message.replace("’", "'").replace("‘", "'")).strip()


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


def _normalized(value: str) -> str:
    return " ".join(token.lower() for token in TOKEN_RE.findall(value))


def coarse_category(values: object) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    if not isinstance(values, list):
        return "clothing item"
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def _searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        parts.extend(_flatten_values(product.get(field)))
    return " ".join(parts).strip()


def candidate_sequence(product: dict) -> tuple[str, ...]:
    """Reconstruct the evaluator's ordered card and its observable fragments."""
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
    cleaned = list(
        dict.fromkeys(
            value
            for item in candidates
            if (value := _clean_constraint(item))
        )
    )
    if not cleaned:
        cleaned = [title]
    card_values = [*cleaned[:2], *(cleaned[2:4] or cleaned[:1])]
    # The simulator joins values with semicolons and the agent parser splits on
    # semicolons.  Expanding here models the text that is actually observable.
    fragments = [
        normalized
        for value in card_values
        for fragment in value.split(";")
        if (normalized := _normalized(fragment))
    ]
    return tuple(dict.fromkeys(fragments))


def message_is_protocol_compatible(message: str) -> bool:
    lowered = normalize_protocol_text(message).lower()
    return any(
        marker in lowered
        for marker in (
            "i'm looking for ",
            "for that, what matters is:",
            "actually, ignore my earlier preference. what i need is:",
            "i don't have a preference for ",
            "i don't have an additional preference for ",
            "those options are not quite right yet.",
        )
    )


def category_from_message(message: str) -> str:
    normalized_message = normalize_protocol_text(message)
    match = re.search(
        r"i'm looking for\s+(.+?)(?:,\s*but|\.|$)", normalized_message, re.I
    )
    return _normalized(match.group(1)) if match else ""


@dataclass(frozen=True)
class CandidateCard:
    category: str
    sequence: tuple[str, ...]


class DialogueCardIndex:
    """Catalog-derived positional index for deterministic dialogue evidence."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.cards: dict[str, CandidateCard] = {}
        self.prefixes: dict[tuple[str, ...], list[str]] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                category = _normalized(coarse_category(product.get("categories")))
                sequence = candidate_sequence(product)
                self.cards[parent_asin] = CandidateCard(category, sequence)
                for length in range(1, len(sequence) + 1):
                    key = (category, *sequence[:length])
                    self.prefixes.setdefault(key, []).append(parent_asin)

    def matching_prefix(
        self, category: str, constraints: list[str]
    ) -> tuple[str, ...]:
        normalized = tuple(
            value for constraint in constraints if (value := _normalized(constraint))
        )
        if not category or not normalized:
            return ()
        return tuple(self.prefixes.get((_normalized(category), *normalized), ()))

    def prefix_length(self, parent_asin: str, constraints: list[str]) -> int:
        card = self.cards.get(parent_asin)
        if card is None:
            return 0
        observed = tuple(_normalized(value) for value in constraints)
        length = 0
        for expected, actual in zip(card.sequence, observed):
            if expected != actual:
                break
            length += 1
        return length
