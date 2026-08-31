"""Question policy: ask about the attribute that should narrow the tier most.

For every attribute the harness allows, the policy partitions the current top
dominance tier by that attribute's values and computes the expected tier size
that remains once the customer names one value. The attribute with the largest
expected reduction wins. Asking stops entirely once the top tier already fits
in one page of results.
"""

from __future__ import annotations

import math

from src.catalog_index import CatalogIndex
from src.intent_router import SIZE_WORDS, STYLE_WORDS, USE_CASE_WORDS
from src.text_utils import COLOR_TERMS, MATERIAL_TERMS


ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)
KEYWORD_VOCABULARIES: dict[str, frozenset[str]] = {
    "material": frozenset(MATERIAL_TERMS),
    "color": frozenset(COLOR_TERMS),
    "size": frozenset(SIZE_WORDS),
    "style": frozenset(STYLE_WORDS),
    "use_case": frozenset(USE_CASE_WORDS),
}
# Estimating over the whole tier would cost more than it can inform; the head
# of the tier is a representative sample because ties are broken identically.
POLICY_SAMPLE = 60
TIER_SATISFIED = 10


class QuestionPolicy:
    """Expected-reduction estimator over the current candidate tier."""

    def __init__(self, index: CatalogIndex, enabled: bool = True) -> None:
        self.index = index
        self.enabled = enabled

    def _values(self, parent_asin: str, attribute: str) -> frozenset[str]:
        field_terms, combined_terms, _ = self.index.profile(parent_asin)
        vocabulary = KEYWORD_VOCABULARIES.get(attribute)
        if vocabulary is not None:
            return frozenset(combined_terms & vocabulary)
        if attribute == "brand":
            return field_terms[4]
        if attribute == "category":
            return self.index.coarse_category_views.get(
                parent_asin, ("", frozenset())
            )[1]
        if attribute == "budget":
            price = self.index.price_of(parent_asin)
            if price is None:
                return frozenset()
            # Log-spaced buckets: customers state budgets by order of size.
            return frozenset({f"price_{int(math.log1p(max(0.0, price)) * 2)}"})
        if attribute == "feature":
            return frozenset(field_terms[2] | field_terms[3])
        return frozenset()

    def _expected_remaining(self, tier: list[str], attribute: str) -> float:
        """Expected tier size after the customer names one value."""
        buckets: dict[frozenset[str], int] = {}
        for parent_asin in tier:
            if attribute == "other":
                # An open question accepts a value of any attribute, so its
                # partition is the join of every attribute's partition.
                key = frozenset(
                    (name, value)
                    for name in ALLOWED_ATTRIBUTES
                    if name != "other"
                    for value in self._values(parent_asin, name)
                )
            else:
                key = self._values(parent_asin, attribute)
            buckets[key] = buckets.get(key, 0) + 1
        total = max(1, len(tier))
        return sum(count * count for count in buckets.values()) / total

    def choose(
        self, tier: list[str], asked: set[str] | None = None
    ) -> tuple[str | None, list[dict]]:
        """Return the chosen attribute (None to stop asking) and the estimates."""
        if not self.enabled:
            return "other", []
        if len(tier) <= TIER_SATISFIED:
            # Every remaining candidate already fits on one page, so another
            # question cannot improve the answer.
            return None, []
        sample = tier[:POLICY_SAMPLE]
        size = float(len(sample))
        estimates: list[dict] = []
        for attribute in ALLOWED_ATTRIBUTES:
            if asked and attribute in asked and attribute != "other":
                continue
            remaining = self._expected_remaining(sample, attribute)
            estimates.append({
                "attribute": attribute,
                "expected_remaining": round(remaining, 4),
                "expected_reduction": round(size - remaining, 4),
            })
        # Ties go to the open question: any constraint type can answer it, so
        # at equal expected reduction it is likelier to get an answer at all.
        estimates.sort(key=lambda item: (
            -item["expected_reduction"],
            item["attribute"] != "other",
            item["attribute"],
        ))
        return (estimates[0]["attribute"] if estimates else "other"), estimates
