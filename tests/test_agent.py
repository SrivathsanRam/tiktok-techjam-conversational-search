from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent, RERANK_FEATURE_NAMES


class AgentStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "A",
                "title": "Black leather belt",
                "categories": ["Accessories", "Belts"],
                "features": ["Full grain leather", "Buckle closure; Imported"],
                "details": {"department": "mens"},
                "store": "Example",
                "description": ["Everyday belt"],
            },
            {
                "parent_asin": "B",
                "title": "Blue fabric belt",
                "categories": ["Accessories", "Belts"],
                "features": ["Canvas fabric"],
                "details": {"department": "mens"},
                "store": "Example",
                "description": ["Casual belt"],
            },
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(catalog_path)
        self.agent.reset("session", {"preference_tags": ["durability"]})

    def tearDown(self) -> None:
        self.agent.connection.close()
        self.temporary_directory.cleanup()

    def test_override_replaces_stale_opener_but_retains_later_disclosure(self) -> None:
        self.agent.respond("session", "I'm looking for accessories belts. Prefer fabric.", 1, 10)
        self.agent.respond("session", "For that, what matters is: buckle closure.", 2, 10)
        self.agent._sessions["session"]["shown"].add("STALE")
        self.agent.respond(
            "session",
            "Actually, ignore my earlier preference. What I need is: leather.",
            3,
            10,
        )
        messages = self.agent._sessions["session"]["messages"]
        self.assertNotIn("Prefer fabric.", messages)
        self.assertIn("For that, what matters is: buckle closure.", messages)
        self.assertIn(
            "Actually, ignore my earlier preference. What I need is: leather.",
            messages,
        )
        self.assertNotIn("STALE", self.agent._sessions["session"]["shown"])

    def test_no_additional_preference_does_not_pollute_state(self) -> None:
        self.agent.respond("session", "I'm looking for accessories belts, but I'm still exploring.", 1, 10)
        self.agent.respond("session", "I don't have an additional preference for other.", 2, 10)
        messages = self.agent._sessions["session"]["messages"]
        self.assertEqual(len(messages), 1)

    def test_respond_asks_and_recommends_with_valid_ids(self) -> None:
        response = self.agent.respond(
            "session",
            "I'm looking for accessories belts. A key requirement is leather.",
            1,
            10,
        )
        self.assertEqual(response["ask_attribute"], "other")
        self.assertLessEqual(len(response["recommendations"]), 10)
        self.assertTrue({item["parent_asin"] for item in response["recommendations"]} <= {"A", "B"})

    def test_constraint_phrases_distill_accumulated_messages(self) -> None:
        phrases = self.agent._constraint_phrases([
            "I'm looking for accessories belts. A key requirement is: leather.",
            "For that, what matters is: buckle closure; color: black.",
        ])
        self.assertEqual(phrases, ["leather", "buckle closure", "color: black"])

    def test_budget_score_respects_explicit_maximum(self) -> None:
        self.assertEqual(self.agent._budget_score("I need this under $100", 90.0), 1.0)
        self.assertEqual(self.agent._budget_score("I need this under $100", 120.0), -1.0)

    def test_learned_reranker_schema_matches_runtime_features(self) -> None:
        if self.agent._reranker_weights is not None:
            self.assertEqual(len(self.agent._reranker_weights), len(RERANK_FEATURE_NAMES))

    def test_exact_evidence_indexes_semicolon_fragments(self) -> None:
        self.assertEqual(self.agent._evidence_postings("Imported"), ("A",))
        self.assertEqual(self.agent._exact_evidence_candidates(["buckle closure"]), ["A"])

    def test_repeated_query_rotates_already_shown_results(self) -> None:
        first = self.agent.respond(
            "session", "I'm looking for accessories belts, but I'm still exploring.", 1, 1
        )
        second = self.agent.respond(
            "session", "I don't have an additional preference for other.", 2, 1
        )
        self.assertNotEqual(
            first["recommendations"][0]["parent_asin"],
            second["recommendations"][0]["parent_asin"],
        )

    def test_non_protocol_query_keeps_multi_result_fallback(self) -> None:
        self.agent.reset("free-form", {})
        response = self.agent.respond(
            "free-form", "Recommend a durable belt made of leather", 1, 10
        )
        self.assertEqual(len(response["recommendations"]), 2)

    def test_final_turn_relaxes_ambiguity_abstention(self) -> None:
        self.agent.reset("last-turn", {})
        self.agent.respond(
            "last-turn",
            "I'm looking for accessories belts, but I'm still exploring.",
            1,
            10,
        )
        ninth = self.agent.respond(
            "last-turn", "I don't have a preference for color.", 9, 10
        )
        tenth = self.agent.respond(
            "last-turn", "I don't have a preference for size.", 10, 10
        )
        self.assertEqual(len(ninth["recommendations"]), 1)
        self.assertEqual(len(tenth["recommendations"]), 2)

    def test_query_profile_classifies_intent_and_accumulates_constraints(self) -> None:
        self.agent.respond(
            "session",
            "I'm looking for accessories belts. A key requirement is: leather.",
            1,
            10,
        )
        self.agent.respond(
            "session", "For that, what matters is: color: black.", 2, 10
        )
        profile = self.agent._sessions["session"]["query_profile"]
        self.assertEqual(profile["intent"], "specific buying")
        self.assertEqual(profile["hard_constraints"], ["leather", "color: black"])
        self.assertEqual(profile["static_priorities"], ["durability"])

    def test_intent_classifier_isolates_browsing_and_override(self) -> None:
        browsing = {
            "messages": ["I'm still exploring."],
            "exploratory": True,
            "override_seen": False,
        }
        self.assertEqual(self.agent._intent_mode(browsing), "exploratory browsing")
        browsing["messages"].append("What matters is: leather.")
        self.assertEqual(self.agent._intent_mode(browsing), "constrained browsing")
        browsing["override_seen"] = True
        self.assertEqual(self.agent._intent_mode(browsing), "intent override")


if __name__ == "__main__":
    unittest.main()
