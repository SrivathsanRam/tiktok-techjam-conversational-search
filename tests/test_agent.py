from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent, RERANK_FEATURE_NAMES, _normalized_value


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

    def test_boundary_reply_triggers_rotation(self) -> None:
        first = self.agent.respond(
            "session", "I'm looking for accessories belts, but I'm still exploring.", 1, 1
        )
        second = self.agent.respond(
            "session",
            "I don't have a preference for other; please use your judgment.",
            2,
            1,
        )
        self.assertEqual(len(self.agent._sessions["session"]["messages"]), 1)
        self.assertNotEqual(
            first["recommendations"][0]["parent_asin"],
            second["recommendations"][0]["parent_asin"],
        )

    def test_invalid_question_reply_triggers_rotation(self) -> None:
        first = self.agent.respond(
            "session", "I'm looking for accessories belts, but I'm still exploring.", 1, 1
        )
        second = self.agent.respond(
            "session",
            "Those options are not quite right yet. Ask me about one specific attribute.",
            2,
            1,
        )
        self.assertEqual(len(self.agent._sessions["session"]["messages"]), 1)
        self.assertNotEqual(
            first["recommendations"][0]["parent_asin"],
            second["recommendations"][0]["parent_asin"],
        )


class ParaphraseRobustnessTest(unittest.TestCase):
    """One test per wording dependency listed in CP3_STATE.md section 7."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "A",
                "title": "Black leather belt",
                "categories": ["Accessories", "Belts"],
                "features": ["Full grain leather", "100% Leather", "Buckle closure; Imported"],
                "details": {"department": "mens"},
                "store": "Example",
                "description": ["Everyday belt"],
                "price": 40.0,
            },
            {
                "parent_asin": "B",
                "title": "Blue cotton belt",
                "categories": ["Accessories", "Belts"],
                "features": ["Woven cotton", "100% Cotton", "Buckle closure"],
                "details": {"department": "mens"},
                "store": "Example",
                "description": ["Casual belt"],
                "price": 20.0,
            },
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(catalog_path)
        self.agent.reset("session", {})

    def tearDown(self) -> None:
        self.agent.connection.close()
        self.temporary_directory.cleanup()

    def test_unmarked_message_recovers_values_from_the_index(self) -> None:
        # No template marker: n-grams are matched against the exact index and
        # returned in their normalized token form.
        self.assertEqual(
            self.agent._message_constraints("I really need 100% leather here"),
            ["100 leather"],
        )

    def test_unmatched_message_yields_no_constraints(self) -> None:
        self.assertEqual(
            self.agent._message_constraints("something entirely unrelated please"),
            [],
        )

    def test_override_without_the_official_marker_still_parses_constraints(self) -> None:
        self.agent.respond("session", "I'm looking for belts. Prefer cotton.", 1, 10)
        self.agent.respond(
            "session", "Scratch that, I changed my mind. It must have 100% leather.", 2, 10
        )
        constraints = self.agent._constraint_phrases(
            self.agent._sessions["session"]["messages"]
        )
        self.assertIn(
            "100 leather", [_normalized_value(value) for value in constraints]
        )

    def test_paraphrased_override_rebuilds_state(self) -> None:
        self.assertTrue(self.agent._is_override("On second thought, disregard that."))
        self.assertTrue(self.agent._is_override("Never mind my earlier note."))
        self.assertFalse(self.agent._is_override("For that, what matters is: leather."))

    def test_paraphrased_browsing_is_exploratory(self) -> None:
        for message in (
            "Browsing belts for now, nothing decided.",
            "Checking out belts — just looking at the moment.",
            "Belts please, though I'm undecided so far.",
        ):
            self.assertTrue(self.agent._is_exploratory(message), message)

    def test_paraphrased_no_preference_and_rejection_disclose_nothing(self) -> None:
        for message in (
            "No strong preference on color, honestly.",
            "color doesn't matter to me.",
            "I'm easy about color — anything is fine.",
            "Whatever you think is best for color.",
            "Up to you regarding color.",
            "None of those work for me.",
            "These aren't right — try again.",
            "That's off the mark.",
        ):
            self.assertFalse(self.agent._has_preference(message), message)

    def test_disclosure_paraphrases_are_still_preference_bearing(self) -> None:
        for message in (
            "I care about 100% leather.",
            "It must have buckle closure and imported.",
        ):
            self.assertTrue(self.agent._has_preference(message), message)

    def test_semantic_override_erases_contradicting_slot(self) -> None:
        _, erased, log = self.agent._resolve_slots([
            "I'm looking for belts. What I need is: cotton.",
            "I care about leather.",
        ])
        self.assertEqual(erased, ["cotton"])
        self.assertTrue(
            any(entry["action"] == "erase" and entry["value"] == "cotton" for entry in log)
        )

    def test_complementary_same_type_values_are_retained(self) -> None:
        active, erased, _ = self.agent._resolve_slots([
            "I'm looking for belts. What I need is: cotton.",
            "For that, what matters is: 100% cotton.",
        ])
        self.assertEqual(erased, [])
        self.assertEqual(active, ["cotton", "100% cotton"])

    def test_alternate_delimiters_split_only_when_every_part_is_indexed(self) -> None:
        self.assertEqual(
            self.agent._marker_phrases("what i need is: buckle closure and imported"),
            ["buckle closure", "imported"],
        )
        whole = "a long prose value that is not indexed, with a comma"
        self.assertEqual(
            self.agent._marker_phrases(f"what i need is: {whole}"), [whole]
        )

    def test_base_intent_accepts_other_sentence_boundaries(self) -> None:
        self.assertEqual(self.agent._base_intent("I want belts! Prefer cotton"), "I want belts")
        self.assertEqual(self.agent._base_intent("I want belts; prefer cotton"), "I want belts")
        self.assertEqual(
            self.agent._base_intent("I want belts, but I'm still exploring"), "I want belts"
        )

    def test_budget_paraphrases_are_scored(self) -> None:
        for text in ("no more than $100", "at most $100", "$100 or less", "within $100"):
            self.assertEqual(self.agent._budget_score(text, 90.0), 1.0, text)
            self.assertEqual(self.agent._budget_score(text, 120.0), -1.0, text)
        self.assertGreater(self.agent._budget_score("roughly $100", 100.0), 0.9)
        self.assertGreater(self.agent._budget_score("about $100", 100.0), 0.9)

    def test_unrecognized_framing_never_enters_the_exact_lane(self) -> None:
        # Framing words are not catalog values, so no posting list exists.
        self.assertEqual(
            self.agent._message_constraints("could you kindly show me something nice"),
            [],
        )


class RetrievalRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "LEATHER",
                "title": "Leather belt",
                "categories": ["Accessories", "Belts"],
                "features": ["Imported", "Full grain leather"],
                "details": {"department": "mens"},
                "store": "Example",
                "description": ["Everyday belt"],
            },
            {
                "parent_asin": "COTTON",
                "title": "Cotton belt",
                "categories": ["Accessories", "Belts"],
                "features": ["Imported", "Woven cotton"],
                "details": {"department": "mens"},
                "store": "Example",
                "description": ["Casual belt"],
            },
            {
                "parent_asin": "HAT",
                "title": "Imported wool hat",
                "categories": ["Accessories", "Hats"],
                "features": ["Imported"],
                "details": {"department": "mens"},
                "store": "Example",
                "description": ["Winter hat"],
            },
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        # "imported" appears in every product, so any threshold below the
        # catalog size marks it generic.
        self.agent = Agent(catalog_path, generic_token_df=2)
        self.agent.reset("session", {})

    def tearDown(self) -> None:
        self.agent.connection.close()
        self.temporary_directory.cleanup()

    def test_document_frequency_index_counts_products_not_occurrences(self) -> None:
        self.assertEqual(self.agent._token_df["imported"], 3)
        self.assertEqual(self.agent._token_df["leather"], 1)

    def test_generic_only_constraint_does_not_add_a_route(self) -> None:
        without = self.agent._fused_search(
            ["belts"], 10, constraints=(), category_phrase="accessories belts"
        )
        with_generic = self.agent._fused_search(
            ["belts"], 10, constraints=("Imported",), category_phrase="accessories belts"
        )
        self.assertEqual(without, with_generic)

    def test_category_material_route_prefers_matching_material(self) -> None:
        ranked = self.agent._fused_search(
            ["belts", "leather"],
            10,
            constraints=("leather",),
            category_phrase="accessories belts",
        )
        self.assertEqual(ranked[0], "LEATHER")


class DominanceTierRotationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": f"P{index}",
                "title": f"Imported leather belt {index}",
                "categories": ["Accessories", "Belts"],
                "features": ["Imported"],
                "details": {"department": "mens"},
                "store": "Example",
                "description": ["Everyday belt"],
                "rating_number": 100 - index,
            }
            for index in range(4)
        ]
        products.append(
            {
                "parent_asin": "Z0",
                "title": "Domestic canvas belt",
                "categories": ["Accessories", "Belts"],
                "features": ["Canvas fabric"],
                "details": {"department": "mens"},
                "store": "Example",
                "description": ["Casual belt"],
                "rating_number": 500,
            }
        )
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(catalog_path)
        self.agent.reset("session", {})

    def tearDown(self) -> None:
        self.agent.connection.close()
        self.temporary_directory.cleanup()

    def test_rotation_pages_unseen_tier_members_before_reintroducing(self) -> None:
        first = self.agent.respond(
            "session", "I'm looking for belts. A key requirement is: Imported.", 1, 2
        )
        shown_first = {item["parent_asin"] for item in first["recommendations"]}
        second = self.agent.respond(
            "session", "I don't have an additional preference for other.", 2, 2
        )
        shown_second = {item["parent_asin"] for item in second["recommendations"]}
        self.assertFalse(shown_first & shown_second)
        self.assertTrue(shown_second <= {"P0", "P1", "P2", "P3"})

    def test_rotation_ranks_satisfied_tier_above_unsatisfied(self) -> None:
        self.agent.respond(
            "session", "I'm looking for belts. A key requirement is: Imported.", 1, 1
        )
        second = self.agent.respond(
            "session", "I don't have an additional preference for other.", 2, 3
        )
        ranked = [item["parent_asin"] for item in second["recommendations"]]
        # Every unseen Imported tier member outranks the popular non-member.
        self.assertNotIn("Z0", ranked)


if __name__ == "__main__":
    unittest.main()
