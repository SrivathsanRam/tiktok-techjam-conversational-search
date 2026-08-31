from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LogitsWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and quantize the CP4 reranker")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    wrapper = LogitsWrapper(model)
    example = tokenizer(
        ["intent: specific buying. request: leather boots."],
        ["title: black leather boots. features: waterproof."],
        return_tensors="pt",
    )
    float_path = args.output / "model.onnx"
    quantized_path = args.output / "model_quint8_avx2.onnx"
    torch.onnx.export(
        wrapper,
        (example["input_ids"], example["attention_mask"], example["token_type_ids"]),
        float_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(float_path, quantized_path, weight_type=QuantType.QUInt8)
    for name in (
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.txt", "training_report.json",
    ):
        source = args.model / name
        if source.exists():
            shutil.copy2(source, args.output / name)

    session = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
    encoded = tokenizer(
        ["intent: specific buying. request: leather boots."],
        ["title: black leather boots. features: waterproof."],
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="np",
    )
    output = session.run(
        ["logits"],
        {name: np.asarray(encoded[name], dtype=np.int64) for name in (
            "input_ids", "attention_mask", "token_type_ids"
        )},
    )[0]
    print(json.dumps({
        "float_bytes": float_path.stat().st_size,
        "quantized_bytes": quantized_path.stat().st_size,
        "smoke_logit": float(output.reshape(-1)[0]),
    }, indent=2))


if __name__ == "__main__":
    main()
