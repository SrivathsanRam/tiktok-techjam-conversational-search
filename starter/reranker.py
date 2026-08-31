"""Tiered reranker: dominance tiers ordered internally by a linear model."""

from __future__ import annotations

import json
import math
from pathlib import Path

from starter.catalog_index import CatalogIndex
from starter.exact_evidence import ExactEvidencePool
from starter.intent_router import IntentRouter, budget_score
from starter.text_utils import COLOR_TERMS, MATERIAL_TERMS, terms


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
    "exact_evidence_coverage",
    "exact_evidence_rarity",
    "satisfied_constraints",
    "coarse_category_equality",
    "coarse_category_overlap",
    "max_matched_rarity",
    "unmatched_rarity",
    "price_presence_when_budget",
    "preference_tag_overlap",
)


def load_reranker_weights(
    weights_path: Path | None = None,
) -> tuple[float, ...] | None:
    path = weights_path or Path(__file__).with_name("reranker_weights.json")
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("reranker_weights.json must contain a weights object")
    # Features absent from the fitted file carry zero weight, so an older
    # 16-feature file scores identically on the extended feature vector.
    return tuple(float(weights.get(name, 0.0)) for name in RERANK_FEATURE_NAMES)


def fallback_structured_score(features: tuple[float, ...]) -> float:
    (
        coverage, title_coverage, category_coverage, attribute_coverage,
        description_coverage, constraint_coverage, exact_fraction,
        material_match, color_match, budget_match,
    ) = features[1:11]
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


class TieredReranker:
    """Scores candidates and orders them strictly within dominance tiers."""

    def __init__(
        self,
        index: CatalogIndex,
        evidence: ExactEvidencePool,
        router: IntentRouter,
        weights: tuple[float, ...] | None,
    ) -> None:
        self.index = index
        self.evidence = evidence
        self.router = router
        self.weights = weights

    def _compatibility_features(
        self,
        parent_asin: str,
        query_terms: list[str],
        constraints: list[str],
        query_text: str,
    ) -> tuple[float, ...]:
        price = self.index.price_of(parent_asin)
        field_terms, combined_terms, combined_token_string = self.index.profile(parent_asin)
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
            value_terms = self.index.value_terms(constraint)
            if not value_terms:
                continue
            constraint_coverages.append(len(value_terms & combined_terms) / len(value_terms))
            normalized = " ".join(terms(constraint))
            if normalized and normalized in combined_token_string:
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
        budget_match = budget_score(query_text, price)

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

    def feature_vector(
        self,
        parent_asin: str,
        rank: int,
        state: dict[str, object],
        query_terms: list[str],
        constraints: list[str],
        query_text: str,
    ) -> tuple[float, ...]:
        compatibility = self._compatibility_features(
            parent_asin, query_terms, constraints, query_text
        )
        popularity = self.index.popularity.get(parent_asin, 0.0) / max(
            1.0, self.index.max_popularity
        )
        exploratory = bool(state["exploratory"])
        retrieval = 1.0 / math.log2(rank + 1.0)
        (
            exact_coverage,
            exact_rarity,
            satisfied_constraints,
            max_matched_rarity,
            unmatched_rarity,
        ) = self.evidence.features(parent_asin, constraints)

        requested_norm, requested_terms = self.router.requested_category(
            str(state.get("base_message") or "")
        )
        coarse_norm, coarse_terms = self.index.coarse_category_views.get(
            parent_asin, ("", frozenset())
        )
        coarse_equality = 1.0 if requested_norm and requested_norm == coarse_norm else 0.0
        coarse_overlap = (
            len(requested_terms & coarse_terms) / len(requested_terms)
            if requested_terms else 0.0
        )
        price = self.index.price_of(parent_asin)
        price_presence = (
            1.0 if price is not None and self.router.budget_disclosed(query_text) else 0.0
        )
        profile = state.get("user_profile")
        tags = profile.get("preference_tags") if isinstance(profile, dict) else None
        tag_overlap = 0.0
        if isinstance(tags, list) and tags:
            combined_terms = self.index.profile(parent_asin)[1]
            matched_tags = sum(
                1 for tag in tags
                if (tag_terms := self.index.value_terms(str(tag)))
                and tag_terms <= combined_terms
            )
            tag_overlap = matched_tags / len(tags)

        return (
            retrieval,
            *compatibility,
            popularity,
            popularity if exploratory else 0.0,
            compatibility[5] if not exploratory else 0.0,
            exact_coverage,
            exact_rarity,
            satisfied_constraints,
            coarse_equality,
            coarse_overlap,
            max_matched_rarity,
            unmatched_rarity,
            price_presence,
            tag_overlap,
        )

    def rerank(
        self,
        candidates: list[str],
        state: dict[str, object],
        query_terms: list[str],
        constraints: list[str],
        top_k: int,
        tier_counts: dict[str, int] | None = None,
    ) -> list[str]:
        messages = state["messages"]
        query_text = (
            " ".join(str(item) for item in messages) if isinstance(messages, list) else ""
        )
        structured_weight = 0.10 if state["exploratory"] else 0.15
        counts = tier_counts or {}
        scored: list[tuple[int, float, int, str]] = []
        for rank, parent_asin in enumerate(candidates, start=1):
            features = self.feature_vector(
                parent_asin, rank, state, query_terms, constraints, query_text
            )
            if self.weights is None:
                compatibility = fallback_structured_score(features)
                final_score = (
                    (1.0 - structured_weight) * features[0]
                    + structured_weight * compatibility
                )
            else:
                final_score = sum(
                    weight * value
                    for weight, value in zip(self.weights, features, strict=True)
                )
            scored.append((counts.get(parent_asin, 0), final_score, rank, parent_asin))
        # Dominance tiers: satisfying more exact constraints always outranks
        # satisfying fewer.  The learned score orders products within one tier.
        scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        return [parent_asin for _, _, _, parent_asin in scored[:top_k]]
