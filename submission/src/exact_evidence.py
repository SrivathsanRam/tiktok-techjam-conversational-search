"""Exact-evidence pool: dominance tiers over verbatim catalog values."""

from __future__ import annotations

import math
from collections import Counter

from src.catalog_index import CatalogIndex
from src.text_utils import normalized_value


CATALOG_SIZE = 50000.0


class ExactEvidencePool:
    """Looks constraints up by exact value and partitions products into tiers."""

    def __init__(
        self,
        index: CatalogIndex,
        candidate_limit: int = 1000,
        precision_limit: int = 60,
        single_max_df: int = 50000,
    ) -> None:
        self.index = index
        self.enabled = index.use_exact_evidence
        self.candidate_limit = candidate_limit
        self.precision_limit = precision_limit
        self.single_max_df = single_max_df
        self._postings_cache: dict[str, tuple[str, ...]] = {}
        self._membership_cache: dict[str, frozenset[str]] = {}

    def postings(self, phrase: str) -> tuple[str, ...]:
        normalized = normalized_value(phrase)
        if not normalized or not self.enabled:
            return ()
        cached = self._postings_cache.get(normalized)
        if cached is not None:
            return cached
        rows = self.index.connection.execute(
            "SELECT parent_asin FROM evidence_values WHERE normalized_value = ?",
            (normalized,),
        ).fetchall()
        result = tuple(str(row[0]) for row in rows)
        self._postings_cache[normalized] = result
        return result

    def pool(self, constraints: list[str]) -> tuple[list[str], dict[str, int]]:
        """Rank products by the amount and rarity of verbatim catalog evidence.

        Returns the full top dominance tier (every product satisfying the
        maximum number of exact constraints, up to the safety cap) plus a map
        from parent_asin to satisfied-constraint count for every product that
        matched at least one exact constraint.  An empty count map means
        dominance ordering is disabled for this turn.
        """
        posting_lists = [
            values for constraint in constraints
            if (values := self.postings(constraint))
        ]
        if not posting_lists:
            return [], {}
        matches: Counter[str] = Counter()
        rarity: dict[str, float] = {}
        for values in posting_lists:
            evidence_weight = math.log((CATALOG_SIZE + 1.0) / (len(values) + 1.0))
            for parent_asin in values:
                matches[parent_asin] += 1
                rarity[parent_asin] = rarity.get(parent_asin, 0.0) + evidence_weight
        strongest_match_count = max(matches.values(), default=0)
        if strongest_match_count == 1 and len(posting_lists[0]) > self.single_max_df:
            return [], {}
        strongest = [
            parent_asin
            for parent_asin, match_count in matches.items()
            if match_count == strongest_match_count
        ]
        popularity = self.index.popularity
        ordered = sorted(
            strongest,
            key=lambda asin: (
                -matches[asin],
                -rarity[asin],
                -popularity.get(asin, 0.0),
                asin,
            ),
        )
        if len(ordered) > self.candidate_limit:
            # A tier wider than the safety cap is not a discriminative
            # dominance signal (for example every cotton product), so keep the
            # legacy precision slice and let the learned reranker order freely.
            return ordered[:self.precision_limit], {}
        return ordered, dict(matches)

    def features(
        self, parent_asin: str, constraints: list[str]
    ) -> tuple[float, float, float, float, float]:
        """Coverage, mean rarity, raw count, max rarity, and unmatched rarity."""
        matched_values = 0
        matched_rarity = 0.0
        max_matched_rarity = 0.0
        unmatched_rarity = 0.0
        indexed_constraints = 0
        max_idf = math.log((CATALOG_SIZE + 1.0) / 2.0)
        for constraint in constraints:
            normalized = normalized_value(constraint)
            posting_list = self.postings(constraint)
            if not normalized or not posting_list:
                continue
            indexed_constraints += 1
            members = self._membership_cache.get(normalized)
            if members is None:
                members = frozenset(posting_list)
                self._membership_cache[normalized] = members
            rarity = math.log((CATALOG_SIZE + 1.0) / (len(posting_list) + 1.0)) / max_idf
            if parent_asin in members:
                matched_values += 1
                matched_rarity += rarity
                max_matched_rarity = max(max_matched_rarity, rarity)
            else:
                unmatched_rarity += rarity
        denominator = max(1, indexed_constraints)
        return (
            matched_values / denominator,
            matched_rarity / denominator,
            float(matched_values),
            max_matched_rarity,
            unmatched_rarity / denominator,
        )
