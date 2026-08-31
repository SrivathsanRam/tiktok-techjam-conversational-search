from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path


MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "models" / "cp4-tinybert-reranker"
MODEL_PATH = MODEL_DIRECTORY / "model_quint8_avx2.onnx"
TOKENIZER_PATH = MODEL_DIRECTORY / "tokenizer.json"


def _is_control(character: str) -> bool:
    return character not in "\t\n\r" and unicodedata.category(character).startswith("C")


def _is_punctuation(character: str) -> bool:
    codepoint = ord(character)
    if 33 <= codepoint <= 47 or 58 <= codepoint <= 64:
        return True
    if 91 <= codepoint <= 96 or 123 <= codepoint <= 126:
        return True
    return unicodedata.category(character).startswith("P")


class WordPieceTokenizer:
    """Minimal uncased BERT tokenizer backed by the exported tokenizer JSON."""

    def __init__(self, tokenizer_path: Path = TOKENIZER_PATH) -> None:
        payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        self.vocab = {
            str(token): int(index) for token, index in payload["model"]["vocab"].items()
        }
        self.unknown_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.pad_id = self.vocab["[PAD]"]

    @staticmethod
    def _basic_tokens(text: str) -> list[str]:
        cleaned = "".join(
            " " if character.isspace() else character
            for character in text
            if not _is_control(character)
        ).lower()
        cleaned = "".join(
            character
            for character in unicodedata.normalize("NFD", cleaned)
            if unicodedata.category(character) != "Mn"
        )
        tokens: list[str] = []
        current: list[str] = []
        for character in cleaned:
            if character.isspace():
                if current:
                    tokens.append("".join(current))
                    current.clear()
            elif _is_punctuation(character):
                if current:
                    tokens.append("".join(current))
                    current.clear()
                tokens.append(character)
            else:
                current.append(character)
        if current:
            tokens.append("".join(current))
        return tokens

    @lru_cache(maxsize=16_384)
    def _tokenize_cached(self, text: str) -> tuple[int, ...]:
        output: list[int] = []
        for token in self._basic_tokens(text):
            if len(token) > 100:
                output.append(self.unknown_id)
                continue
            start = 0
            pieces: list[int] = []
            while start < len(token):
                end = len(token)
                piece_id: int | None = None
                while start < end:
                    piece = token[start:end]
                    if start:
                        piece = "##" + piece
                    if piece in self.vocab:
                        piece_id = self.vocab[piece]
                        break
                    end -= 1
                if piece_id is None:
                    pieces = [self.unknown_id]
                    break
                pieces.append(piece_id)
                start = end
            output.extend(pieces)
        return tuple(output)

    def tokenize(self, text: str) -> list[int]:
        # Candidate documents recur across turns and sessions.  Caching their
        # immutable WordPiece form is a cheap, bounded catalog-side preprocess.
        return list(self._tokenize_cached(text))

    def encode_pair(
        self, query: str, document: str, max_length: int
    ) -> tuple[list[int], list[int], list[int]]:
        query_ids = self.tokenize(query)
        document_ids = self.tokenize(document)
        while len(query_ids) + len(document_ids) > max_length - 3:
            if len(document_ids) > len(query_ids):
                document_ids.pop()
            else:
                query_ids.pop()
        input_ids = [self.cls_id, *query_ids, self.sep_id, *document_ids, self.sep_id]
        first_segment = len(query_ids) + 2
        token_types = [0] * first_segment + [1] * (len(document_ids) + 1)
        return input_ids, [1] * len(input_ids), token_types


class LocalCrossEncoder:
    def __init__(self, max_length: int = 160) -> None:
        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.max_length = max_length
        self.tokenizer = WordPieceTokenizer()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def try_load(cls) -> LocalCrossEncoder | None:
        if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
            return None
        try:
            return cls()
        except (ImportError, OSError, RuntimeError, ValueError, KeyError):
            return None

    def predict(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        encoded = [
            self.tokenizer.encode_pair(query, document, self.max_length)
            for document in documents
        ]
        sequence_length = max(len(item[0]) for item in encoded)
        inputs: list[list[int]] = []
        masks: list[list[int]] = []
        types: list[list[int]] = []
        for ids, mask, token_types in encoded:
            padding = sequence_length - len(ids)
            inputs.append(ids + [self.tokenizer.pad_id] * padding)
            masks.append(mask + [0] * padding)
            types.append(token_types + [0] * padding)
        logits = self.session.run(
            ["logits"],
            {
                "input_ids": self.np.asarray(inputs, dtype=self.np.int64),
                "attention_mask": self.np.asarray(masks, dtype=self.np.int64),
                "token_type_ids": self.np.asarray(types, dtype=self.np.int64),
            },
        )[0]
        return [float(value) for value in logits.reshape(-1)]
