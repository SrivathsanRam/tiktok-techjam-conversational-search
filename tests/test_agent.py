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
                "features": ["Full grain leather", "Buckle closure"],
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


if __name__ == "__main__":
    unittest.main()
