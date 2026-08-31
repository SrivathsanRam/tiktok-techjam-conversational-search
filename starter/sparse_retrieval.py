"""Sparse retrieval: per-constraint FTS5 routes fused by reciprocal rank."""

from __future__ import annotations

from starter.catalog_index import CatalogIndex
from starter.text_utils import COLOR_TERMS, MATERIAL_TERMS, TOKEN_RE, terms


RRF_RANK_CONSTANT = 20.0
ROUTE_LIMIT = 150


class SparseRetrieval:
    """Builds one FTS expression per disclosed requirement and fuses them.

    Generic tokens are recognized by catalog document frequency rather than a
    hardcoded list, so boilerplate such as "imported" or a bare percentage
    cannot drive a precision route.
    """

    def __init__(self, index: CatalogIndex, generic_token_df: int = 12000) -> None:
        self.index = index
        self.generic_token_df = generic_token_df

    def _routes(
        self,
        query_terms: list[str],
        constraints: tuple[str, ...],
        category_phrase: str,
        disjunctive_weight: float,
    ) -> list[tuple[str, float]]:
        token_df = self.index.token_df
        quoted = [f'"{term}"' for term in query_terms]
        category_terms = [
            term for term in terms(category_phrase) if token_df.get(term, 0) > 0
        ]
        materials = [term for term in query_terms if term in MATERIAL_TERMS]
        colors = [term for term in query_terms if term in COLOR_TERMS]
        routes: list[tuple[str, float]] = []
        if category_terms:
            category_quoted = [f'"{term}"' for term in category_terms]
            routes.append((" AND ".join(category_quoted), 2.0))
            if materials:
                routes.append((
                    " AND ".join(category_quoted + [f'"{term}"' for term in materials]),
                    2.5,
                ))
            if colors:
                routes.append((
                    " AND ".join(category_quoted + [f'"{term}"' for term in colors]),
                    2.0,
                ))
        constraint_phrases: list[str] = []
        for constraint in constraints:
            tokens = [token.lower() for token in TOKEN_RE.findall(constraint)]
            # A constraint made only of generic tokens matches most of the
            # catalog and adds noise.
            if not tokens or not any(
                token_df.get(token, 0) <= self.generic_token_df for token in tokens
            ):
                continue
            constraint_phrases.append('"' + " ".join(tokens) + '"')
        if constraint_phrases:
            routes.append((" OR ".join(dict.fromkeys(constraint_phrases)), 2.5))
        if not routes:
            # No category or usable constraint yet: the concatenated
            # conjunctive bag remains the precision fallback.
            routes.append((" AND ".join(quoted), 2.5))
        routes.append((
            " OR ".join(
                f'"{query_terms[index]} {query_terms[index + 1]}"'
                for index in range(len(query_terms) - 1)
            ),
            1.25,
        ))
        routes.append((" OR ".join(quoted), disjunctive_weight))
        return routes

    def search(
        self,
        query_terms: list[str],
        top_k: int,
        disjunctive_weight: float = 1.0,
        popularity_weight: float = 0.0,
        constraints: tuple[str, ...] = (),
        category_phrase: str = "",
    ) -> list[str]:
        if not query_terms:
            return []
        token_df = self.index.token_df
        # A token absent from the catalog cannot match anything, and inside a
        # conjunctive route it empties the whole route.  Unrecognized framing
        # words are therefore dropped before any expression is built.
        query_terms = [
            term for term in query_terms if token_df.get(term, 0) > 0
        ] or query_terms
        scores: dict[str, float] = {}
        best_route_rank: dict[str, int] = {}
        for expression, weight in self._routes(
            query_terms, constraints, category_phrase, disjunctive_weight
        ):
            if not expression:
                continue
            for rank, parent_asin in enumerate(
                self.index.ranked_asins(expression, ROUTE_LIMIT), start=1
            ):
                scores[parent_asin] = (
                    scores.get(parent_asin, 0.0) + weight / (RRF_RANK_CONSTANT + rank)
                )
                best_route_rank[parent_asin] = min(
                    best_route_rank.get(parent_asin, rank), rank
                )
        if popularity_weight > 0.0:
            popularity = self.index.popularity
            popularity_ranking = sorted(
                scores, key=lambda asin: (-popularity.get(asin, 0.0), asin)
            )
            for rank, parent_asin in enumerate(popularity_ranking, start=1):
                scores[parent_asin] += popularity_weight / (RRF_RANK_CONSTANT + rank)
        ordered = sorted(
            scores, key=lambda asin: (-scores[asin], best_route_rank[asin], asin)
        )
        return ordered[:top_k]
