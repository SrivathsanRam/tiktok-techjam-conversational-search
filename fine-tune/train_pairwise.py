from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PairDataset(Dataset):
    def __init__(self, groups: list[dict]) -> None:
        self.pairs = [
            (group["query"], group["positive"], negative)
            for group in groups
            for negative in group["negatives"]
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str, str]:
        return self.pairs[index]


def score_pairs(model, tokenizer, pairs: list[tuple[str, str]], max_length: int) -> torch.Tensor:
    encoded = tokenizer(
        [query for query, _ in pairs],
        [document for _, document in pairs],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    return model(**encoded).logits.reshape(-1)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def evaluate(model, tokenizer, groups: list[dict], max_length: int) -> dict:
    model.eval()
    reciprocal_ranks: list[float] = []
    pairwise_correct = 0
    pairwise_total = 0
    for group in groups:
        documents = [group["positive"], *group["negatives"]]
        scores = score_pairs(
            model, tokenizer, [(group["query"], document) for document in documents], max_length
        )
        positive = float(scores[0])
        rank = 1 + sum(float(score) > positive for score in scores[1:])
        reciprocal_ranks.append(1.0 / rank)
        pairwise_correct += sum(positive > float(score) for score in scores[1:])
        pairwise_total += len(scores) - 1
    return {
        "group_mrr": sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)),
        "pairwise_accuracy": pairwise_correct / max(1, pairwise_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TinyBERT with pairwise ranking loss")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default="cross-encoder/ms-marco-TinyBERT-L2-v2",
    )
    parser.add_argument(
        "--base-revision",
        default="81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    train_groups = load_jsonl(args.train)
    validation_groups = load_jsonl(args.validation)
    dataset = PairDataset(train_groups)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    base_path = Path(args.base_model)
    revision = None if base_path.exists() else args.base_revision
    base_reference = str(base_path) if base_path.exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(base_reference, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_reference, revision=revision
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = max(1, args.epochs * len(loader))
    warmup_steps = max(1, round(total_steps * 0.1))

    baseline_metrics = evaluate(model, tokenizer, validation_groups, args.max_length)
    best_mrr = baseline_metrics["group_mrr"]
    history: list[dict] = [{"epoch": 0, **baseline_metrics}]
    print(json.dumps(history[0]), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            queries, positives, negatives = batch
            positive_scores = score_pairs(
                model, tokenizer, list(zip(queries, positives)), args.max_length
            )
            negative_scores = score_pairs(
                model, tokenizer, list(zip(queries, negatives)), args.max_length
            )
            loss = F.softplus(negative_scores - positive_scores).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            absolute_step = (epoch - 1) * len(loader) + step
            if absolute_step <= warmup_steps:
                scale = absolute_step / warmup_steps
            else:
                scale = max(
                    0.0,
                    (total_steps - absolute_step) / max(1, total_steps - warmup_steps),
                )
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach())

        metrics = evaluate(model, tokenizer, validation_groups, args.max_length)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(loader)),
            **metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["group_mrr"] > best_mrr:
            best_mrr = metrics["group_mrr"]
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)

    report = {
        "base_model": str(args.base_model),
        "base_revision": revision,
        "device": str(device),
        "train_groups": len(train_groups),
        "train_pairs": len(dataset),
        "validation_groups": len(validation_groups),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "history": history,
        "best_group_mrr": best_mrr,
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
