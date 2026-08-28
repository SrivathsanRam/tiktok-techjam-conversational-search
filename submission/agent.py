from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
PRICE_RE = re.compile(r"(?:\$|under|below|less than|around|about|budget)[a-z\s:]*\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
OVERRIDE_RE = re.compile(
    r"\b(ignore\s+(?:my\s+)?earlier|actually\b.*\b(?:ignore|instead|need|want)|"
    r"instead\b|rather\b|change\b.*\b(?:to|my)|not\b.*\banymore)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "look", "need", "needs", "those", "options", "quite", "right", "yet",
    "ask", "about", "one", "specific", "attribute", "matters", "key",
    "requirement", "prefer", "preference", "use", "judgment", "still",
    "exploring", "actually", "ignore", "earlier", "closest", "matches",
    "have", "has", "had", "additional", "don", "dont", "doesn", "doesnt",
}
FIELD_WEIGHTS = {
    "title": 6.0,
    "categories": 4.0,
    "features": 3.25,
    "details": 2.75,
    "store": 2.0,
    "description": 1.5,
}
SYNONYM_GROUPS = (
    ("shoe", "shoes", "sneaker", "sneakers", "trainer", "trainers", "footwear"),
    ("boot", "boots"),
    ("shirt", "shirts", "tee", "tees", "tshirt", "tshirts", "top", "tops"),
    ("hoodie", "hoodies", "sweatshirt", "sweatshirts", "pullover", "pullovers"),
    ("pants", "pant", "trouser", "trousers", "slacks"),
    ("jeans", "jean", "denim"),
    ("jacket", "jackets", "coat", "coats"),
    ("dress", "dresses", "gown"),
    ("women", "womens", "woman", "female", "ladies", "lady"),
    ("men", "mens", "man", "male"),
    ("kid", "kids", "child", "children", "boy", "boys", "girl", "girls"),
    ("gray", "grey"),
    ("xl", "xlarge"),
    ("running", "run", "runner", "runners"),
    ("hiking", "hike", "trail"),
    ("gym", "workout", "training"),
    ("winter", "cold", "warm"),
    ("waterproof", "water", "resistant"),
    ("comfortable", "comfort", "comfy"),
    ("fit", "fitted"),
    ("wide", "width"),
)
SYNONYMS = {
    token: set(group)
    for group in SYNONYM_GROUPS
    for token in group
}
MATERIALS = {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"}
COLORS = {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"}
ASK_SEQUENCE = ("other", "material", "color", "size", "style", "brand", "budget", "feature")
ATTRIBUTE_QUESTIONS = {
    "other": "What one additional requirement should I account for?",
    "material": "Do you have a material preference?",
    "color": "What color should I prioritize?",
    "size": "Any sizing or fit details I should use?",
    "style": "What style or cut are you leaning toward?",
    "brand": "Is there a brand you prefer?",
    "budget": "What budget should I stay close to?",
    "feature": "What feature matters most?",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _coerce_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def _term_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) > 4 and token.endswith("ies"):
        variants.add(token[:-3] + "y")
    if len(token) > 4 and token.endswith("es"):
        variants.add(token[:-2])
    if len(token) > 3 and token.endswith("s"):
        variants.add(token[:-1])
    if len(token) > 5 and token.endswith("ing"):
        variants.add(token[:-3])
    for variant in tuple(variants):
        variants.update(SYNONYMS.get(variant, set()))
    return variants


def _terms(text: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in TOKEN_RE.findall(text):
        token = raw.lower()
        if len(token) <= 1 or token in STOPWORDS:
            continue
        for variant in _term_variants(token):
            if variant not in STOPWORDS and variant not in seen:
                seen.add(variant)
                result.append(variant)
    return result


def _phrases(text: str, max_phrases: int = 24) -> list[str]:
    terms = [term for term in _terms(text) if not term.isdigit()]
    phrases: list[str] = []
    for width in (3, 2):
        for index in range(0, max(0, len(terms) - width + 1)):
            phrase = " ".join(terms[index:index + width])
            if phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) >= max_phrases:
                return phrases
    return phrases


@dataclass
class ProductDoc:
    parent_asin: str
    field_text: dict[str, str]
    field_terms: dict[str, Counter[str]]
    all_text: str
    price: float | None
    average_rating: float | None
    rating_number: float | None
    quality_score: float = 0.0


@dataclass
class Candidate:
    parent_asin: str
    bm25_score: float
    recall_score: float = 0.0


@dataclass
class SessionState:
    profile_terms: Counter[str] = field(default_factory=Counter)
    messages: list[str] = field(default_factory=list)
    asked_attributes: set[str] = field(default_factory=set)
    active_from: int = 0


class Agent:
    """Dependency-free two-stage retriever with conversational reranking."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._products: dict[str, ProductDoc] = {}
        self._popular: list[str] = []
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
                parent_asin = str(product["parent_asin"])
                field_text = {
                    "title": _text(product.get("title")).lower(),
                    "categories": _text(product.get("categories")).lower(),
                    "features": _text(product.get("features")).lower(),
                    "details": _text(product.get("details")).lower(),
                    "store": _text(product.get("store")).lower(),
                    "description": _text(product.get("description")).lower(),
                }
                all_text = " ".join(field_text.values())
                average_rating = _coerce_float(product.get("average_rating"))
                rating_number = _coerce_float(product.get("rating_number"))
                quality_score = 0.0
                if average_rating is not None:
                    quality_score += max(0.0, min(5.0, average_rating)) / 5.0
                if rating_number is not None:
                    quality_score += min(1.0, math.log1p(max(0.0, rating_number)) / math.log1p(5000.0))
                self._products[parent_asin] = ProductDoc(
                    parent_asin=parent_asin,
                    field_text=field_text,
                    field_terms={name: Counter(_terms(text)) for name, text in field_text.items()},
                    all_text=all_text,
                    price=_coerce_float(product.get("price")),
                    average_rating=average_rating,
                    rating_number=rating_number,
                    quality_score=quality_score,
                )
                batch.append(
                    (
                        parent_asin,
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
        self._popular = [
            item.parent_asin
            for item in sorted(self._products.values(), key=lambda doc: doc.quality_score, reverse=True)
        ]

    def reset(self, session_id: str, user_profile: dict) -> None:
        profile_text = " ".join([
            _text(user_profile.get("summary")),
            _text(user_profile.get("preference_tags")),
            _text(user_profile.get("rating_style")),
        ])
        self._sessions[session_id] = SessionState(profile_terms=Counter(_terms(profile_text)))

    def _candidate_ids(self, terms: list[str], limit: int, recall_weight: float = 1.0) -> list[Candidate]:
        if not terms:
            return [
                Candidate(asin, float(index), recall_weight / math.sqrt(index + 1.0))
                for index, asin in enumerate(self._popular[:limit])
            ]
        expression = " OR ".join(f'"{term}"' for term in list(dict.fromkeys(terms))[:64])
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS bm25_score "
            "FROM products WHERE products MATCH ? ORDER BY bm25_score LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [
            Candidate(str(parent_asin), float(bm25_score), recall_weight / math.sqrt(index + 1.0))
            for index, (parent_asin, bm25_score) in enumerate(rows)
        ]

    def _active_messages(self, state: SessionState) -> list[str]:
        return state.messages[state.active_from:]

    def _blended_candidates(self, state: SessionState, limit: int) -> list[Candidate]:
        active_terms = list(self._query_terms(state).keys())
        recent_terms = _terms(state.messages[-1]) if state.messages else []
        full_terms = _terms(" ".join(state.messages))
        profile_terms = list(state.profile_terms.keys())
        routes = [
            (recent_terms, 140, 3.0),
            (active_terms, limit, 2.4),
            (full_terms, 140, 1.0),
            (profile_terms, 80, 0.4),
        ]
        merged: dict[str, Candidate] = {}
        for terms, route_limit, recall_weight in routes:
            for candidate in self._candidate_ids(terms, min(limit, route_limit), recall_weight):
                current = merged.get(candidate.parent_asin)
                if current is None:
                    merged[candidate.parent_asin] = candidate
                    continue
                current.bm25_score = min(current.bm25_score, candidate.bm25_score)
                current.recall_score += candidate.recall_score
        return sorted(merged.values(), key=lambda item: item.recall_score, reverse=True)[:limit]

    def _query_terms(self, state: SessionState) -> Counter[str]:
        weighted: Counter[str] = Counter()
        if state.messages:
            for index, message in enumerate(state.messages):
                message_weight = 0.25 if index < state.active_from else 1.0
                if index == len(state.messages) - 1:
                    message_weight = 2.25
                for term in _terms(message):
                    weighted[term] += message_weight
        for term, count in state.profile_terms.items():
            weighted[term] += 0.25 * count
        return weighted

    def _attribute_score(self, doc: ProductDoc, conversation_text: str) -> float:
        conversation_terms = set(_terms(conversation_text))
        score = 0.0
        wanted_materials = conversation_terms & MATERIALS
        wanted_colors = conversation_terms & COLORS
        doc_terms = set(_terms(doc.all_text))
        if wanted_materials:
            score += 9.0 * len(wanted_materials & doc_terms)
            if not wanted_materials & doc_terms:
                score -= 2.0
        if wanted_colors:
            normalized_doc_terms = doc_terms | ({"gray"} if "grey" in doc_terms else set()) | ({"grey"} if "gray" in doc_terms else set())
            score += 8.0 * len(wanted_colors & normalized_doc_terms)
            if not wanted_colors & normalized_doc_terms:
                score -= 1.5
        price_match = PRICE_RE.search(conversation_text)
        if price_match and doc.price is not None:
            target_price = float(price_match.group(1))
            lowered = conversation_text.lower()
            if any(word in lowered for word in ("under", "below", "less than")):
                score += 14.0 if doc.price <= target_price else -16.0
            else:
                ratio = abs(doc.price - target_price) / max(target_price, 1.0)
                score += max(0.0, 5.0 * (1.0 - ratio))
        return score

    def _rerank(self, candidates: list[Candidate], state: SessionState, top_k: int) -> list[dict]:
        query_weights = self._query_terms(state)
        conversation_text = " ".join(self._active_messages(state)).lower()
        recent_text = state.messages[-1] if state.messages else ""
        phrases = [*_phrases(conversation_text), *_phrases(recent_text)]
        scored: list[tuple[float, str]] = []
        for rank, candidate in enumerate(candidates):
            doc = self._products.get(candidate.parent_asin)
            if doc is None:
                continue
            score = 10.0 / math.sqrt(rank + 1.0)
            score += candidate.recall_score * 6.0
            score += max(0.0, -candidate.bm25_score) * 2.0
            for field_name, terms in doc.field_terms.items():
                field_weight = FIELD_WEIGHTS[field_name]
                for term, query_weight in query_weights.items():
                    if term in terms:
                        score += field_weight * query_weight * min(2, terms[term])
            for phrase in phrases:
                if not phrase:
                    continue
                if phrase in doc.field_text["title"]:
                    score += 10.0
                elif phrase in doc.field_text["categories"]:
                    score += 5.5
                elif phrase in doc.all_text:
                    score += 3.0
            score += self._attribute_score(doc, conversation_text)
            score += 0.75 * doc.quality_score
            scored.append((score, candidate.parent_asin))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"parent_asin": parent_asin, "score": round(score, 6)} for score, parent_asin in scored[:top_k]]

    def _next_attribute(self, state: SessionState, turn: int) -> str | None:
        if turn >= 10:
            return None
        profile_order = [
            attribute
            for attribute in ASK_SEQUENCE
            if attribute in state.profile_terms and attribute not in state.asked_attributes
        ]
        fallback_order = [attribute for attribute in ASK_SEQUENCE if attribute not in state.asked_attributes]
        for attribute in [*profile_order, *fallback_order]:
            state.asked_attributes.add(attribute)
            return attribute
        return None

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        if OVERRIDE_RE.search(user_message):
            state.active_from = len(state.messages)
        state.messages.append(user_message)
        candidates = self._blended_candidates(state, max(350, top_k * 50))
        recommendations = self._rerank(candidates, state, top_k)
        ask_attribute = self._next_attribute(state, turn)
        message = ATTRIBUTE_QUESTIONS.get(ask_attribute, "Here are the closest matches I found.")
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
