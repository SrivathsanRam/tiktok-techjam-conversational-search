from __future__ import annotations

import unittest

from starter.cp5_dialogue import (
    candidate_sequence,
    category_from_message,
    message_is_protocol_compatible,
)


class DialogueCardTest(unittest.TestCase):
    def test_candidate_sequence_matches_material_color_then_catalog_order(self) -> None:
        product = {
            "title": "Example shirt",
            "categories": ["Clothing", "Men", "Shirts"],
            "features": ["Machine washable; imported"],
            "details": {"color": "Blue", "department": "mens"},
            "description": ["Cotton everyday shirt"],
        }
        self.assertEqual(
            candidate_sequence(product),
            (
                "cotton",
                "color blue",
                "machine washable",
                "imported",
            ),
        )

    def test_category_is_extracted_from_both_initial_templates(self) -> None:
        self.assertEqual(
            category_from_message("I'm looking for Men Shirts. A key requirement is: cotton."),
            "men shirts",
        )
        self.assertEqual(
            category_from_message("I'm looking for Men Shirts, but I'm still exploring."),
            "men shirts",
        )

    def test_protocol_parser_accepts_curly_apostrophe_and_extra_whitespace(self) -> None:
        message = "  I’m   looking for Men Shirts, but I’m still exploring.  "
        self.assertTrue(message_is_protocol_compatible(message))
        self.assertEqual(category_from_message(message), "men shirts")


if __name__ == "__main__":
    unittest.main()
