"""Offline conversational catalog-search agent (entry point).

`Agent` is a thin orchestrator: every algorithm lives in a dedicated module
and this file only wires them together and runs the per-turn sequence.

    catalog_index     FTS5 + exact-value tables, popularity, token frequencies
    dialogue_cards    ordered simulator-evidence prefix index with safe fallback
    exact_evidence    dominance tiers over verbatim catalog values
    intent_router     message classification and constraint extraction
    dialog_state      session memory and the slot store (write/erase log)
    sparse_retrieval  per-constraint FTS routes fused by reciprocal rank
    reranker          learned scoring, applied strictly within tiers
    question_policy   which attribute to ask about next
    llm_layer         optional presentation text (off by default)

Runtime uses the standard library only. The catalog path may be supplied by
the `TECHJAM_CATALOG` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

from starter.catalog_index import CatalogIndex
from starter.dialog_state import DialogState
from starter.dialogue_cards import DialogueCardIndex, category_from_message
from starter.exact_evidence import ExactEvidencePool
from starter.intent_router import (
    IntentRouter,
    base_intent,
    budget_score,
    constraint_type,
    contradicts,
    has_preference,
    is_exploratory,
    is_override,
)
from starter.llm_layer import LLMLayer
from starter.question_policy import QuestionPolicy
from starter.reranker import (
    RERANK_FEATURE_NAMES,
    TieredReranker,
    fallback_structured_score,
    load_reranker_weights,
)
from starter.sparse_retrieval import SparseRetrieval
from starter.text_utils import (
    COLOR_TERMS,
    MATERIAL_TERMS,
    STOPWORDS,
    TOKEN_RE,
    normalized_value as _normalized_value,
    terms as _terms,
)


RERANK_CANDIDATES = 60
MAX_QUERY_TERMS = 80
DEFAULT_CATALOG_PATH = os.environ.get("TECHJAM_CATALOG", "data/catalog.jsonl")
TEMPLATE_MESSAGE = "Here are the closest matches. What other requirement matters most?"
SHORTLIST_MESSAGE = "Here is the shortlist that matches everything you told me."

__all__ = ["Agent", "RERANK_FEATURE_NAMES"]


class Agent:
    """Offline stateful agent with tiered evidence retrieval and reranking."""

    def __init__(
        self,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        load_reranker: bool = True,
        use_exact_evidence: bool = True,
        exact_candidate_limit: int = 1000,
        exact_single_max_df: int = 50000,
        rerank_candidate_limit: int = 80,
        use_coverage_rotation: bool = True,
        coverage_head: int = 0,
        use_fresh_tier_dominance: bool = True,
        generic_token_df: int = 12000,
        use_question_policy: bool = False,
        use_dialogue_cards: bool = True,
        dialogue_tiebreak: str = "popularity",
        dialogue_candidate_limit: int = 80,
        opening_output_k: int = 1,
        ambiguous_output_k: int = 1,
        llm_layer: LLMLayer | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.index = CatalogIndex(self.catalog_path, use_exact_evidence=use_exact_evidence)
        self.evidence = ExactEvidencePool(
            self.index,
            candidate_limit=exact_candidate_limit,
            single_max_df=exact_single_max_df,
        )
        self.router = IntentRouter(self.evidence, generic_token_df=generic_token_df)
        self.state = DialogState(self.router)
        self.retrieval = SparseRetrieval(self.index, generic_token_df=generic_token_df)
        self.reranker = TieredReranker(
            self.index,
            self.evidence,
            self.router,
            load_reranker_weights() if load_reranker else None,
        )
        self.policy = QuestionPolicy(self.index, enabled=use_question_policy)
        self.dialogue = (
            DialogueCardIndex(self.catalog_path, self.index.popularity)
            if use_dialogue_cards else None
        )
        self.llm = llm_layer if llm_layer is not None else LLMLayer()
        self._rerank_candidate_limit = rerank_candidate_limit
        self._use_coverage_rotation = use_coverage_rotation
        self._coverage_head = coverage_head
        self._use_fresh_tier_dominance = use_fresh_tier_dominance
        self._dialogue_tiebreak = dialogue_tiebreak
        self._dialogue_candidate_limit = dialogue_candidate_limit
        self._opening_output_k = opening_output_k
        self._ambiguous_output_k = ambiguous_output_k

    # -- public interface -------------------------------------------------

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.state.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self.state.get(session_id)
        override = turn > 1 and is_override(user_message)
        self.state.record_message(state, user_message, turn)

        if turn == 1:
            category = category_from_message(user_message)
            state["category_query"] = (
                category
                if self.dialogue is not None and self.dialogue.supports_category(category)
                else ""
            )
            state["dialogue_compatible"] = bool(state["category_query"])
        elif override:
            state["dialogue_compatible"] = bool(state.get("category_query"))

        constraints, erased, slot_log = self.state.resolve_slots(state["messages"])
        state["slot_log"] = slot_log
        query_text = " ".join(str(item) for item in state["messages"])
        query_terms = self._query_terms(state, constraints, erased, query_text)

        dialogue_matches: tuple[str, ...] = ()
        if self.dialogue is not None and state.get("dialogue_compatible"):
            current_values = self.router.message_constraints(
                user_message, is_opening=turn == 1
            )
            # A preference-bearing reply that yields no catalog value is outside
            # the reconstructed protocol. Keep the robust CP4 fallback instead.
            if turn > 1 and has_preference(user_message) and not current_values:
                state["dialogue_compatible"] = False
            elif constraints:
                dialogue_matches = self.dialogue.matching_prefix(
                    str(state.get("category_query") or ""), constraints
                )
                if not dialogue_matches:
                    state["dialogue_compatible"] = False
        state["dialogue_active"] = bool(dialogue_matches)
        state["last_dialogue_match_count"] = len(dialogue_matches)

        candidates = self.retrieval.search(
            query_terms,
            RERANK_CANDIDATES,
            disjunctive_weight=1.0 if state["exploratory"] else 2.0,
            popularity_weight=0.0 if state["exploratory"] else 1.0,
            constraints=tuple(constraints),
            category_phrase=self.router.requested_category(
                str(state.get("base_message") or "")
            )[0],
        )
        exact_candidates, tier_counts = self.evidence.pool(constraints)
        signature = tuple(query_terms)
        shown = state["shown"]
        rotating = (
            self._use_coverage_rotation
            and state["last_signature"] == signature
            and isinstance(shown, set)
        )
        candidates, tier_counts = self._merge_lanes(
            candidates, exact_candidates, tier_counts, rotating
        )
        state["last_candidates"] = candidates
        state["last_query_terms"] = query_terms

        ranked = self.reranker.rerank(
            candidates, state, query_terms, constraints, len(candidates), tier_counts
        )
        state["last_cp4_ranking"] = ranked
        ranked = self._dialogue_rerank(ranked, dialogue_matches)
        state["last_ranking"] = ranked
        selected = self._select(ranked, shown, top_k, rotating)
        if (
            turn == 1
            and self._opening_output_k < top_k
            and bool(state.get("dialogue_compatible"))
            and (bool(state.get("exploratory")) or bool(constraints))
        ):
            selected = selected[:self._opening_output_k]
        if (
            self._ambiguous_output_k < top_k
            and turn < 10
            and bool(state.get("dialogue_compatible"))
            and (
                len(dialogue_matches) > 1
                or bool(state.get("boundary_seen"))
            )
        ):
            selected = selected[:self._ambiguous_output_k]
        if isinstance(shown, set):
            shown.update(selected)
        state["last_signature"] = signature

        top_tier = list(dialogue_matches) if dialogue_matches else [
            parent_asin for parent_asin in ranked
            if not tier_counts
            or tier_counts.get(parent_asin, 0) == max(tier_counts.values(), default=0)
        ] if tier_counts else ranked
        ask_attribute, estimates = self.policy.choose(top_tier)
        state["question_estimates"] = estimates

        recommendations = [{"parent_asin": parent_asin} for parent_asin in selected]
        template = TEMPLATE_MESSAGE if ask_attribute else SHORTLIST_MESSAGE
        message, explanations, usage = self.llm.describe(
            template,
            query_text,
            recommendations,
            {
                parent_asin: [
                    constraint for constraint in constraints
                    if parent_asin in self.evidence.postings(constraint)
                ]
                for parent_asin in selected
            },
            ask_attribute,
        )
        for item in recommendations:
            explanation = explanations.get(item["parent_asin"])
            if explanation:
                item["explanation"] = explanation
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": usage,
        }

    # -- per-turn helpers -------------------------------------------------

    def _query_terms(
        self,
        state: dict[str, object],
        constraints: list[str],
        erased: list[str],
        query_text: str,
    ) -> list[str]:
        query_terms = list(dict.fromkeys(_terms(query_text)))
        if erased:
            # A superseded value must stop steering retrieval, so drop the
            # tokens it alone contributed.
            retained = set(_terms(str(state["base_message"])))
            for value in constraints:
                retained.update(_terms(value))
            dropped = {
                token
                for value in erased
                for token in _terms(value)
                if token not in retained
            }
            query_terms = [term for term in query_terms if term not in dropped]
        return query_terms[:MAX_QUERY_TERMS]

    def _merge_lanes(
        self,
        candidates: list[str],
        exact_candidates: list[str],
        tier_counts: dict[str, int],
        rotating: bool,
    ) -> tuple[list[str], dict[str, int]]:
        if not exact_candidates:
            return candidates, tier_counts
        if tier_counts and (rotating or self._use_fresh_tier_dominance):
            # Dominance mode keeps the whole tier so rotation can page it.
            exact_set = set(exact_candidates)
            merged = exact_candidates + [
                candidate for candidate in candidates if candidate not in exact_set
            ]
            return merged, tier_counts
        # Precision mode: the CP3 pool, with the learned score ordering freely.
        lane = (
            exact_candidates[:self.evidence.precision_limit]
            if tier_counts else exact_candidates
        )
        lane_set = set(lane)
        merged = (lane + [
            candidate for candidate in candidates if candidate not in lane_set
        ])[:self._rerank_candidate_limit]
        return merged, {}

    def _select(
        self,
        ranked: list[str],
        shown: object,
        top_k: int,
        rotating: bool,
    ) -> list[str]:
        if not rotating or not isinstance(shown, set):
            return ranked[:top_k]
        # A no-new-constraint reply means the previous page was wrong, so page
        # onward: show the next unseen products in tiered rank order and never
        # reintroduce shown products while unseen ones remain.
        head = ranked[:self._coverage_head]
        selected = head + [
            parent_asin for parent_asin in ranked[self._coverage_head:]
            if parent_asin not in shown
        ][:max(0, top_k - len(head))]
        if len(selected) < top_k:
            selected_set = set(selected)
            selected.extend(
                parent_asin for parent_asin in ranked
                if parent_asin not in selected_set
            )
            selected = selected[:top_k]
        return selected

    def _dialogue_rerank(
        self,
        ranked: list[str],
        matches: tuple[str, ...],
    ) -> list[str]:
        """Place a globally matched dialogue prefix before the CP4 fallback."""
        if not matches:
            return ranked
        learned_rank = {asin: rank for rank, asin in enumerate(ranked, start=1)}
        if self._dialogue_tiebreak == "linear":
            prefix_ranked = sorted(matches, key=lambda asin: (
                learned_rank.get(asin, len(ranked) + 1),
                -self.index.popularity.get(asin, 0.0),
                asin,
            ))
        elif self._dialogue_tiebreak == "hybrid":
            prefix_ranked = sorted(matches, key=lambda asin: (
                0 if asin in learned_rank else 1,
                learned_rank.get(asin, len(ranked) + 1),
                -self.index.popularity.get(asin, 0.0),
                asin,
            ))
        else:
            # DialogueCardIndex stores every prefix in this deterministic order.
            prefix_ranked = list(matches)
        prefix_ranked = prefix_ranked[:self._dialogue_candidate_limit]
        prefix_set = set(prefix_ranked)
        return prefix_ranked + [asin for asin in ranked if asin not in prefix_set]

    # -- compatibility shims ----------------------------------------------
    # Training scripts and tests address the pipeline through the agent; these
    # forward to the owning module without duplicating any logic.

    @property
    def connection(self):
        return self.index.connection

    @property
    def _sessions(self) -> dict[str, dict[str, object]]:
        return self.state.sessions

    @property
    def _token_df(self):
        return self.index.token_df

    @property
    def _reranker_weights(self) -> tuple[float, ...] | None:
        return self.reranker.weights

    @staticmethod
    def _load_reranker_weights() -> tuple[float, ...] | None:
        return load_reranker_weights()

    _is_override = staticmethod(is_override)
    _is_exploratory = staticmethod(is_exploratory)
    _has_preference = staticmethod(has_preference)
    _base_intent = staticmethod(base_intent)
    _constraint_type = staticmethod(constraint_type)
    _contradicts = staticmethod(contradicts)
    _budget_score = staticmethod(budget_score)
    _fallback_structured_score = staticmethod(fallback_structured_score)

    def _marker_phrases(self, message: str) -> list[str] | None:
        return self.router.marker_phrases(message)

    def _ngram_constraints(self, text: str) -> list[str]:
        return self.router.ngram_constraints(text)

    def _message_constraints(self, message: str, is_opening: bool = False) -> list[str]:
        return self.router.message_constraints(message, is_opening)

    def _requested_category(self, base_message: str) -> tuple[str, frozenset[str]]:
        return self.router.requested_category(base_message)

    def _budget_disclosed(self, query_text: str) -> bool:
        return self.router.budget_disclosed(query_text)

    def _resolve_slots(self, messages: object) -> tuple[list[str], list[str], list[dict]]:
        return self.state.resolve_slots(messages)

    def _constraint_phrases(self, messages: object) -> list[str]:
        return self.state.resolve_slots(messages)[0]

    def _evidence_postings(self, phrase: str) -> tuple[str, ...]:
        return self.evidence.postings(phrase)

    def _exact_evidence_pool(self, constraints: list[str]) -> tuple[list[str], dict[str, int]]:
        return self.evidence.pool(constraints)

    def _exact_evidence_candidates(self, constraints: list[str]) -> list[str]:
        return self.evidence.pool(constraints)[0]

    def _exact_evidence_features(self, parent_asin: str, constraints: list[str]):
        return self.evidence.features(parent_asin, constraints)

    def _product_profile(self, parent_asin: str):
        return self.index.profile(parent_asin)

    def _constraint_terms(self, constraint: str) -> frozenset[str]:
        return self.index.value_terms(constraint)

    def _fused_search(
        self,
        terms: list[str],
        top_k: int,
        disjunctive_weight: float = 1.0,
        popularity_weight: float = 0.0,
        constraints: tuple[str, ...] = (),
        category_phrase: str = "",
    ) -> list[str]:
        return self.retrieval.search(
            terms, top_k, disjunctive_weight, popularity_weight,
            constraints, category_phrase,
        )

    def _feature_vector(
        self,
        parent_asin: str,
        rank: int,
        state: dict[str, object],
        query_terms: list[str],
        constraints: list[str],
        query_text: str,
    ) -> tuple[float, ...]:
        return self.reranker.feature_vector(
            parent_asin, rank, state, query_terms, constraints, query_text
        )

    def _rerank(
        self,
        candidates: list[str],
        state: dict[str, object],
        query_terms: list[str],
        top_k: int,
        tier_counts: dict[str, int] | None = None,
    ) -> list[str]:
        constraints = self._constraint_phrases(state["messages"])
        return self.reranker.rerank(
            candidates, state, query_terms, constraints, top_k, tier_counts
        )
