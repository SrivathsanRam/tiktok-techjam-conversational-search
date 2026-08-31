from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from starter.cp4_cross_encoder import WordPieceTokenizer


def reciprocal_rank(scores: list[float]) -> float:
    positive = scores[0]
    return 1.0 / (1 + sum(score > positive for score in scores[1:]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CP4 tokenizer and runtime parity")
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--pytorch-model", type=Path, required=True)
    parser.add_argument("--float-onnx", type=Path, required=True)
    parser.add_argument("--quantized-onnx", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = [
        json.loads(line) for line in args.groups.open(encoding="utf-8") if line.strip()
    ]
    if args.limit is not None:
        groups = groups[: args.limit]
    hf_tokenizer = AutoTokenizer.from_pretrained(args.pytorch_model)
    runtime_tokenizer = WordPieceTokenizer(args.tokenizer_json)
    model = AutoModelForSequenceClassification.from_pretrained(args.pytorch_model).eval()
    float_session = ort.InferenceSession(
        str(args.float_onnx), providers=["CPUExecutionProvider"]
    )
    quantized_session = ort.InferenceSession(
        str(args.quantized_onnx), providers=["CPUExecutionProvider"]
    )

    pair_count = 0
    token_matches = 0
    reciprocal_ranks: dict[str, list[float]] = {
        "pytorch_fp32": [], "onnx_fp32": [], "onnx_quint8": [], "runtime_quint8": []
    }
    top_indices: dict[str, list[int]] = {name: [] for name in reciprocal_ranks}
    for group in groups:
        query = str(group["query"])
        documents = [str(group["positive"]), *map(str, group["negatives"])]
        for document in documents:
            expected = hf_tokenizer(
                query,
                document,
                truncation=True,
                max_length=args.max_length,
            )
            actual = runtime_tokenizer.encode_pair(query, document, args.max_length)
            pair_count += 1
            token_matches += int(
                expected["input_ids"] == actual[0]
                and expected["attention_mask"] == actual[1]
                and expected["token_type_ids"] == actual[2]
            )

        encoded = hf_tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
        )
        ort_inputs = {
            name: np.asarray(encoded[name], dtype=np.int64)
            for name in ("input_ids", "attention_mask", "token_type_ids")
        }
        with torch.inference_mode():
            torch_inputs = {
                name: torch.from_numpy(value) for name, value in ort_inputs.items()
            }
            pytorch_scores = model(**torch_inputs).logits.reshape(-1).tolist()
        float_scores = float_session.run(["logits"], ort_inputs)[0].reshape(-1).tolist()
        quantized_scores = (
            quantized_session.run(["logits"], ort_inputs)[0].reshape(-1).tolist()
        )

        runtime_encoded = [
            runtime_tokenizer.encode_pair(query, document, args.max_length)
            for document in documents
        ]
        width = max(len(item[0]) for item in runtime_encoded)
        runtime_inputs = {
            "input_ids": np.asarray([
                ids + [runtime_tokenizer.pad_id] * (width - len(ids))
                for ids, _, _ in runtime_encoded
            ], dtype=np.int64),
            "attention_mask": np.asarray([
                mask + [0] * (width - len(mask))
                for _, mask, _ in runtime_encoded
            ], dtype=np.int64),
            "token_type_ids": np.asarray([
                types + [0] * (width - len(types))
                for _, _, types in runtime_encoded
            ], dtype=np.int64),
        }
        runtime_scores = (
            quantized_session.run(["logits"], runtime_inputs)[0].reshape(-1).tolist()
        )
        score_sets = {
            "pytorch_fp32": pytorch_scores,
            "onnx_fp32": float_scores,
            "onnx_quint8": quantized_scores,
            "runtime_quint8": runtime_scores,
        }
        for name, scores in score_sets.items():
            reciprocal_ranks[name].append(reciprocal_rank(scores))
            top_indices[name].append(max(range(len(scores)), key=scores.__getitem__))

    reference = top_indices["pytorch_fp32"]
    report = {
        "groups": len(groups),
        "pairs": pair_count,
        "tokenizer_exact_pair_matches": token_matches,
        "tokenizer_pair_parity": token_matches / max(1, pair_count),
        "group_mrr": {
            name: sum(values) / max(1, len(values))
            for name, values in reciprocal_ranks.items()
        },
        "top1_agreement_with_pytorch": {
            name: sum(int(left == right) for left, right in zip(reference, values))
            / max(1, len(reference))
            for name, values in top_indices.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
