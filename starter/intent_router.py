"""Intent router: classifies a customer message and extracts disclosed values.

Every wording rule the runtime depends on lives here, so the paraphrase
surface is one file. Constraint extraction is index-gated: a value can only
be recovered if the exact-evidence index already contains it, which is what
keeps unrecognized framing out of the exact lane.
"""

from __future__ import annotations

import re

from starter.exact_evidence import ExactEvidencePool
from starter.text_utils import (
    COLOR_TERMS,
    MATERIAL_TERMS,
    TOKEN_RE,
    normalized_value,
    terms,
)


# Constraint introductions.  The three public templates are listed first so
# their exact wording keeps its verbatim delimiter-splitting behaviour; the
# rest let a paraphrased introduction reach the same code path.
CONSTRAINT_MARKERS = (
    "key requirement is:",
    "what matters is:",
    "what i need is:",
    "key requirement is",
    "what matters is",
    "what i need is",
    "requirement is",
    "i need",
    "i want",
    "i require",
    "must have",
    "needs to be",
    "has to be",
    "should be",
    "important is",
    "matters most is",
)
# A real sentence break, never the decimal point inside a price.
SENTENCE_BREAK_RE = re.compile(r"[.!?](?=\s|$)")
# Any of these split a disclosed list into separate constraint values.
CONSTRAINT_DELIMITERS = re.compile(r"[;|]|,\s+and\s+|\s+and\s+|,")
NO_PREFERENCE_RE = re.compile(
    r"(?:do(?:n'?t| not)\s+(?:really\s+)?(?:have|mind|care)"
    r"|have\s+no\b|no\s+(?:strong\s+|particular\s+|specific\s+)?preference"
    r"|not\s+fussed|no\s+opinion|either\s+(?:one\s+)?is\s+fine|either\s+works"
    r"|up\s+to\s+you|your\s+(?:judgment|judgement|call|choice)|you\s+(?:decide|choose|pick)"
    r"|does(?:n'?t| not)\s+matter|no\s+real\s+view|i'?m\s+(?:easy|flexible|open)"
    r"|open\s+to\s+(?:anything|whatever|any)|whatever\s+you|surprise\s+me"
    r"|anything\s+(?:is\s+)?(?:fine|works)|no\s+additional)",
    re.I,
)
NEGATIVE_RE = re.compile(
    r"(?:not\s+quite\s+right|none\s+of\s+(?:these|those|them)"
    r"|not\s+(?:really\s+)?what\s+i|(?:aren'?t|are\s+not|isn'?t|is\s+not)\s+(?:really\s+)?"
    r"(?:what|right|it|working|close)|off\s+(?:the\s+)?(?:mark|base)|miss(?:ing|es)?\s+the\s+mark"
    r"|wrong\s+(?:results|options|items|products)|(?:these|those)\s+don'?t\s+work"
    r"|try\s+again|nothing\s+here\s+works)",
    re.I,
)
EXPLORATORY_RE = re.compile(
    r"(?:still\s+exploring|just\s+browsing|not\s+sure\s+yet|just\s+looking"
    r"|looking\s+around|window\s+shopping|browsing|exploring|undecided"
    r"|haven'?t\s+decided|have\s+not\s+decided|no\s+rush|in\s+no\s+hurry"
    r"|open\s+to\s+(?:ideas|options|suggestions)|getting\s+ideas"
    r"|see\s+what'?s\s+(?:out\s+there|available)|shopping\s+around"
    r"|not\s+sure\s+what)",
    re.I,
)
OVERRIDE_RE = re.compile(
    r"(?:actually|instead\s+of|forget\s|ignore\s+my\s+earlier|ignore\s+(?:that|what)"
    r"|changed?\s+my\s+mind|change\s+of\s+plan|scratch\s+that|never\s+mind"
    r"|nevermind|rather\s+than|on\s+second\s+thought|second\s+thoughts"
    r"|switch\s+to|make\s+that|no\s+longer|disregard|scrap\s+that"
    r"|let'?s\s+go\s+with\s+instead|i\s+changed)",
    re.I,
)
# Sentence boundary for the opening intent: any terminator, or the clause
# separator the browsing template uses.
BASE_INTENT_RE = re.compile(r"[.!?;]|,\s+(?:but|though|although)\s")
BUDGET_AROUND_RE = re.compile(
    r"(?:budget\s+(?:is\s+|of\s+)?around|around|about|roughly|approximately|near|~)"
    r"\s*\$?([0-9]+(?:\.[0-9]+)?)",
    re.I,
)
BUDGET_MAX_RE = re.compile(
    r"(?:under|below|up\s+to|<=|less\s+than|no\s+more\s+than|at\s+most|cheaper\s+than"
    r"|max(?:imum)?(?:\s+of)?|within|keep\s+it\s+(?:under|below))\s*\$?([0-9]+(?:\.[0-9]+)?)",
    re.I,
)
BUDGET_OR_LESS_RE = re.compile(r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:or\s+less|or\s+under)", re.I)
# Keyword vocabularies per simulator attribute, used both to type a disclosed
# value and to estimate how much a question about that attribute would narrow
# the candidate tier.
SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck")
USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")
# Single-valued slots: a newly disclosed value of one of these types replaces
# the stored value rather than accumulating beside it.
SINGLE_VALUED_SLOTS = frozenset({"material", "color", "budget"})
MAX_FALLBACK_NGRAM = 8


def is_override(message: str) -> bool:
    return bool(OVERRIDE_RE.search(message))


def is_exploratory(message: str) -> bool:
    return bool(EXPLORATORY_RE.search(message))


def has_preference(message: str) -> bool:
    """False for replies that disclose nothing: no-preference and rejections."""
    return not (NO_PREFERENCE_RE.search(message) or NEGATIVE_RE.search(message))


def base_intent(message: str) -> str:
    # The initial category precedes the first sentence boundary.  Keeping only
    # that clause prevents an Intent Override from retaining the stale value.
    match = BASE_INTENT_RE.search(message)
    return message[:match.start()] if match else message


def constraint_type(value: str) -> str:
    """Slot type for a disclosed value, mirroring the simulator's attributes."""
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    value_terms = set(terms(lowered))
    if value_terms & MATERIAL_TERMS:
        return "material"
    if value_terms & COLOR_TERMS or "color" in lowered:
        return "color"
    if any(word in lowered for word in SIZE_WORDS):
        return "size"
    if any(word in lowered for word in STYLE_WORDS):
        return "style"
    if any(word in lowered for word in USE_CASE_WORDS):
        return "use_case"
    return "feature"


def contradicts(slot: str, stored: str, disclosed: str) -> bool:
    """True when a newly disclosed value cannot hold beside a stored one.

    Same-type values are usually complementary (``cotton`` and a full
    fabric-composition string describe one product), so only disjoint
    vocabularies count as a contradiction.
    """
    if slot == "budget":
        return normalized_value(stored) != normalized_value(disclosed)
    vocabulary = MATERIAL_TERMS if slot == "material" else COLOR_TERMS
    stored_terms = vocabulary & set(terms(stored))
    disclosed_terms = vocabulary & set(terms(disclosed))
    if not stored_terms or not disclosed_terms:
        return False
    return stored_terms.isdisjoint(disclosed_terms)


def budget_score(query_text: str, price: float | None) -> float:
    if price is None:
        return 0.0
    # A cap is a harder statement than a target, so it is read first when a
    # paraphrase happens to contain both readings.
    maximum = BUDGET_MAX_RE.search(query_text) or BUDGET_OR_LESS_RE.search(query_text)
    around = BUDGET_AROUND_RE.search(query_text)
    if maximum and (not around or maximum.start() <= around.start()):
        return 1.0 if price <= float(maximum.group(1)) else -1.0
    if around:
        target = float(around.group(1))
        scale = max(10.0, target * 0.25)
        return max(0.0, 1.0 - abs(price - target) / scale)
    return 0.0


class IntentRouter:
    """Message classification and index-gated constraint extraction."""

    def __init__(self, evidence: ExactEvidencePool, generic_token_df: int = 12000) -> None:
        self.evidence = evidence
        self.generic_token_df = generic_token_df
        self.token_df = evidence.index.token_df
        self._message_cache: dict[tuple[str, bool], list[str]] = {}
        self._category_cache: dict[str, tuple[str, frozenset[str]]] = {}
        self._budget_cache: dict[str, bool] = {}

    def marker_phrases(self, message: str) -> list[str] | None:
        """Values after a recognized introduction, or None if none is present."""
        lowered = str(message).lower()
        for marker in CONSTRAINT_MARKERS:
            index = lowered.find(marker)
            if index < 0:
                continue
            remainder = lowered[index + len(marker):]
            # A disclosure ends at its own sentence; anything after belongs to
            # a different statement.
            break_match = SENTENCE_BREAK_RE.search(remainder)
            if break_match:
                remainder = remainder[:break_match.start()]
            remainder = remainder.strip(" .;,-")
            if not remainder:
                return []
            if ";" in remainder:
                # The public list delimiter.
                phrases = [part.strip(" .;,-") for part in remainder.split(";")]
                return [phrase for phrase in phrases if phrase]
            if self.evidence.postings(remainder):
                # One verbatim catalog value that happens to contain a
                # delimiter character, such as "Pvc,Resin".
                return [remainder]
            # Other delimiters are ambiguous: catalog values contain commas
            # and the word "and".  Accept such a split only when every part is
            # itself an indexed value, so a paraphrased list is separated
            # while a long prose value stays whole.
            parts = [
                part.strip(" .;,-") for part in CONSTRAINT_DELIMITERS.split(remainder)
            ]
            parts = [part for part in parts if part]
            if len(parts) > 1 and all(self.evidence.postings(part) for part in parts):
                return parts
            return [remainder]
        return None

    def ngram_constraints(self, text: str) -> list[str]:
        """Longest-first, non-overlapping token n-grams present in the index.

        This is the paraphrase fallback: it recovers catalog-derived values
        that arrive without a recognized introduction, and it can only ever
        return strings that the exact-evidence index already contains, so
        unrecognized framing cannot enter the exact lane.
        """
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        if not tokens or not self.evidence.enabled:
            return []
        used = [False] * len(tokens)
        found: list[tuple[int, str]] = []
        for size in range(min(MAX_FALLBACK_NGRAM, len(tokens)), 0, -1):
            for start in range(len(tokens) - size + 1):
                if any(used[start:start + size]):
                    continue
                gram = tokens[start:start + size]
                phrase = " ".join(gram)
                if not terms(phrase):
                    # Pure stopwords or single letters: "i'm" must never match
                    # the indexed size value "m".
                    continue
                if size == 1 and self.token_df.get(gram[0], 0) > self.generic_token_df:
                    # A lone catalog-wide token is framing noise, not evidence.
                    continue
                if not self.evidence.postings(phrase):
                    continue
                for index in range(start, start + size):
                    used[index] = True
                found.append((start, phrase))
        return [phrase for _, phrase in sorted(found)]

    def message_constraints(self, message: str, is_opening: bool = False) -> list[str]:
        cache_key = (message, is_opening)
        cached = self._message_cache.get(cache_key)
        if cached is not None:
            return cached
        if is_opening:
            # The opening clause names the category, which the category routes
            # and coarse-category features already handle.  Only what follows
            # it is a disclosure.
            message = message[len(base_intent(message)):]
        phrases = self.marker_phrases(message)
        if phrases is None:
            # No recognized introduction: recover disclosed values from the
            # index.  An introduced value is trusted as written, because the
            # simulator truncates long catalog strings and mining fragments
            # out of a truncated value only adds weak evidence.
            values = self.ngram_constraints(message)
        else:
            values = list(phrases)
        values = list(dict.fromkeys(values))
        self._message_cache[cache_key] = values
        return values

    def requested_category(self, base_message: str) -> tuple[str, frozenset[str]]:
        """Category phrase disclosed on turn 1, as normalized string and terms."""
        cached = self._category_cache.get(base_message)
        if cached is not None:
            return cached
        lowered = str(base_message).lower()
        marker = "looking for"
        index = lowered.find(marker)
        phrase = ""
        if index >= 0:
            remainder = lowered[index + len(marker):]
            for stop in (",", ".", ";", " but "):
                position = remainder.find(stop)
                if position >= 0:
                    remainder = remainder[:position]
            phrase = remainder.strip()
        cached = (normalized_value(phrase), frozenset(terms(phrase)))
        self._category_cache[base_message] = cached
        return cached

    def budget_disclosed(self, query_text: str) -> bool:
        cached = self._budget_cache.get(query_text)
        if cached is None:
            cached = bool(
                BUDGET_AROUND_RE.search(query_text)
                or BUDGET_MAX_RE.search(query_text)
                or BUDGET_OR_LESS_RE.search(query_text)
            )
            self._budget_cache[query_text] = cached
        return cached
