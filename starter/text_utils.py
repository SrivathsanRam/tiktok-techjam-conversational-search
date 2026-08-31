"""Tokenization primitives shared by every runtime module (stdlib only)."""

from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
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


def text_of(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def normalized_value(text: str) -> str:
    """Normalize a complete catalog value without discarding low-information words."""
    return " ".join(token.lower() for token in TOKEN_RE.findall(text))


def coarse_category(values: list[str]) -> str:
    """The generic-prefix-free tail of a category list, as the simulator words it."""
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"
