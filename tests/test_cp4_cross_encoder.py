from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from starter.cp4_cross_encoder import LocalCrossEncoder, WordPieceTokenizer


class WordPieceTokenizerTest(unittest.TestCase):
    def test_common_tokens_and_punctuation_match_vocab(self) -> None:
        tokenizer = WordPieceTokenizer()
        self.assertEqual(
            tokenizer.tokenize("Leather boots!"),
            [
                tokenizer.vocab["leather"],
                tokenizer.vocab["boots"],
                tokenizer.vocab["!"],
            ],
        )

    def test_repeated_text_uses_bounded_token_cache(self) -> None:
        tokenizer = WordPieceTokenizer()
        tokenizer.tokenize("waterproof walking shoes")
        before = tokenizer._tokenize_cached.cache_info().hits
        tokenizer.tokenize("waterproof walking shoes")
        self.assertEqual(tokenizer._tokenize_cached.cache_info().hits, before + 1)

    def test_missing_model_uses_deterministic_fallback(self) -> None:
        with patch("starter.cp4_cross_encoder.MODEL_PATH", Path("missing-cp4.onnx")):
            self.assertIsNone(LocalCrossEncoder.try_load())


if __name__ == "__main__":
    unittest.main()
