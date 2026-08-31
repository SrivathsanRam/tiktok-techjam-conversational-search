from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.dialogue_cards import (
    DialogueCardIndex,
    candidate_sequence,
    category_from_message,
)


class DialogueCardTest(unittest.TestCase):
    def test_candidate_sequence_matches_simulator_order_and_fragments(self) -> None:
        product = {
            "title": "Example shirt",
            "categories": ["Clothing", "Men", "Shirts"],
            "features": ["Machine washable; imported"],
            "details": {"color": "Blue", "department": "mens"},
            "description": ["Cotton everyday shirt"],
            "price": 25.0,
        }
        self.assertEqual(candidate_sequence(product), (
            "cotton",
            "color blue",
            "machine washable",
            "imported",
        ))

    def test_category_extraction_covers_clean_and_harness_openings(self) -> None:
        messages = {
            "I'm looking for Men Shirts. A key requirement is: cotton.": "men shirts",
            "I want Men Shirts — it must have cotton.": "men shirts",
            "Shopping for Men Shirts; needs to be cotton.": "men shirts",
            "After Men Shirts, and cotton is essential.": "men shirts",
            "Browsing Men Shirts for now, nothing decided.": "men shirts",
            "Checking out Men Shirts — just looking at the moment.": "men shirts",
            "Men Shirts please, though I'm undecided so far.": "men shirts",
        }
        for message, expected in messages.items():
            self.assertEqual(category_from_message(message), expected, message)

    def test_prefix_index_uses_global_popularity_order(self) -> None:
        products = [
            {
                "parent_asin": "LOW",
                "title": "Cotton belt",
                "categories": ["Accessories", "Belts"],
                "features": ["Imported"],
                "details": {},
                "rating_number": 1,
            },
            {
                "parent_asin": "HIGH",
                "title": "Cotton belt",
                "categories": ["Accessories", "Belts"],
                "features": ["Imported"],
                "details": {},
                "rating_number": 100,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            index = DialogueCardIndex(path, {"LOW": 1.0, "HIGH": 5.0})
            self.assertEqual(
                index.matching_prefix("accessories belts", ["cotton"]),
                ("HIGH", "LOW"),
            )


if __name__ == "__main__":
    unittest.main()
