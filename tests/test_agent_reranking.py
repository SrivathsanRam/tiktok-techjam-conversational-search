from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent


def _catalog_row(
    parent_asin: str,
    title: str,
    features: list[str],
    details: dict[str, str],
    price: float,
    average_rating: float = 4.2,
    rating_number: int = 100,
) -> dict:
    return {
        "parent_asin": parent_asin,
        "title": title,
        "features": features,
        "details": details,
        "description": features,
        "categories": ["Clothing", "Shoes & Jewelry"],
        "store": "Example",
        "average_rating": average_rating,
        "rating_number": rating_number,
        "price": price,
    }


class AgentRerankingTest(unittest.TestCase):
    def _agent_for(self, rows: list[dict]) -> Agent:
        self.directory = tempfile.TemporaryDirectory()
        catalog_path = Path(self.directory.name) / "catalog.jsonl"
        catalog_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return Agent(catalog_path)

    def tearDown(self) -> None:
        if hasattr(self, "directory"):
            self.directory.cleanup()

    def test_clarified_material_and_color_promote_best_candidate(self) -> None:
        agent = self._agent_for([
            _catalog_row(
                "TARGET",
                "Black winter boot",
                ["leather upper", "warm lined boot"],
                {"color": "black", "material": "leather"},
                89.0,
            ),
            _catalog_row(
                "DISTRACTOR",
                "Black winter boot",
                ["synthetic upper", "warm lined boot"],
                {"color": "black", "material": "polyester"},
                79.0,
                average_rating=4.9,
                rating_number=2000,
            ),
        ])
        agent.reset("s1", {"summary": "Prior purchases emphasize material and fit.", "preference_tags": ["material"]})
        agent.respond("s1", "I'm looking for winter boots.", 1, 2)
        result = agent.respond("s1", "For that, what matters is: leather; color: black.", 2, 2)

        self.assertEqual(result["recommendations"][0]["parent_asin"], "TARGET")

    def test_budget_constraint_promotes_closest_matching_price(self) -> None:
        agent = self._agent_for([
            _catalog_row("EXPENSIVE", "Blue cotton running shirt", ["cotton", "breathable"], {}, 72.0),
            _catalog_row("TARGET", "Blue cotton running shirt", ["cotton", "breathable"], {}, 35.0),
        ])
        agent.reset("s2", {"summary": "Prior purchases emphasize comfort.", "preference_tags": ["comfort"]})
        result = agent.respond("s2", "I need a blue cotton running shirt under $40.", 1, 2)

        self.assertEqual(result["recommendations"][0]["parent_asin"], "TARGET")

    def test_override_downweights_stale_earlier_intent(self) -> None:
        agent = self._agent_for([
            _catalog_row("STALE", "Red cotton dress", ["cotton", "summer"], {"color": "red"}, 45.0),
            _catalog_row("TARGET", "Black leather boot", ["leather", "winter"], {"color": "black"}, 90.0),
        ])
        agent.reset("s3", {"summary": "Prior purchases emphasize style.", "preference_tags": ["style"]})
        first = agent.respond("s3", "I'm looking for a red cotton dress.", 1, 2)
        result = agent.respond(
            "s3",
            "Actually, ignore my earlier preference. What I need is: black leather boot.",
            2,
            2,
        )

        self.assertEqual(first["recommendations"][0]["parent_asin"], "STALE")
        self.assertEqual(result["recommendations"][0]["parent_asin"], "TARGET")

    def test_synonym_expansion_matches_sneakers_to_shoes(self) -> None:
        agent = self._agent_for([
            _catalog_row(
                "TARGET",
                "Comfortable running shoe",
                ["breathable trainer", "cushioned"],
                {"department": "womens"},
                64.0,
            ),
            _catalog_row(
                "DISTRACTOR",
                "Formal leather shoe",
                ["office loafer", "polished"],
                {"department": "womens"},
                70.0,
                average_rating=4.9,
                rating_number=3000,
            ),
        ])
        agent.reset("s4", {"summary": "Prior purchases emphasize comfort.", "preference_tags": ["comfort"]})
        result = agent.respond("s4", "I need comfy sneakers for running.", 1, 2)

        self.assertEqual(result["recommendations"][0]["parent_asin"], "TARGET")


if __name__ == "__main__":
    unittest.main()
