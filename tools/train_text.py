from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


class GoEmotionsDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        self.rows: list[tuple[str, list[int]]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) < 2:
                    continue
                ids = [int(value) for value in row[1].split(",") if value.strip()]
                self.rows.append((row[0], ids))
        self.tokenizer = tokenizer
        self.max_length = max_length
        if not self.rows:
            raise ValueError(f"No examples found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        text, label_ids = self.rows[index]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length)
        target = torch.zeros(28, dtype=torch.float32)
        for label_id in label_ids:
            if 0 <= label_id < 28:
                target[label_id] = 1.0
        return {**encoding, "labels": target}


def make_collator(tokenizer, use_cuda: bool):
    def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
        labels = torch.stack([row.pop("labels") for row in rows])
        batch = tokenizer.pad(
            rows,
            padding=True,
            pad_to_multiple_of=8 if use_cuda else None,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch

    return collate


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def collect_predictions(model, loader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            all_logits.append(outputs.logits.float().cpu().numpy())
            all_targets.append(batch["labels"].cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def multilabel_metrics(logits: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    predictions = (1 / (1 + np.exp(-logits)) >= thresholds.reshape(1, -1)).astype(np.uint8)
    return {
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
    }


def tune_thresholds(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    probabilities = 1 / (1 + np.exp(-logits))
    grid = np.arange(0.10, 0.71, 0.05)
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    for label in range(targets.shape[1]):
        best_f1 = -1.0
        for threshold in grid:
            score = f1_score(
                targets[:, label], probabilities[:, label] >= threshold, zero_division=0
            )
            if score > best_f1:
                best_f1 = score
                thresholds[label] = threshold
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a local DistilRoBERTa on GoEmotions.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/text/goemotions"))
    parser.add_argument("--base-model", type=Path, default=Path("models/text_base"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/text_model"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable in this PyTorch environment.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    use_amp = device.type == "cuda" and not args.no_amp
    if not (args.base_model / "config.json").exists():
        raise SystemExit("Local base weights are missing. Run `python tools/download_assets.py` first.")
    for name in ("train.tsv", "dev.tsv", "test.tsv", "emotions.txt"):
        if not (args.data_dir / name).exists():
            raise SystemExit(f"Missing dataset file: {args.data_dir / name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    labels = [line.strip() for line in (args.data_dir / "emotions.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 28:
        raise SystemExit(f"Expected GoEmotions' 28 labels; found {len(labels)} in emotions.txt")
    label2id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=False,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label={index: label for index, label in enumerate(labels)},
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_set = GoEmotionsDataset(args.data_dir / "train.tsv", tokenizer, args.max_length)
    dev_set = GoEmotionsDataset(args.data_dir / "dev.tsv", tokenizer, args.max_length)
    test_set = GoEmotionsDataset(args.data_dir / "test.tsv", tokenizer, args.max_length)
    collate = make_collator(tokenizer, device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * 0.06)),
        num_training_steps=total_updates,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_dev_loss = math.inf
    best_state = args.output_dir / "best_state.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            group_start = (step // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(train_loader) - group_start)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = output.loss / group_size
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * group_size
            end_group = (step + 1) % args.gradient_accumulation == 0 or step + 1 == len(train_loader)
            if end_group:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        dev_logits, dev_targets = collect_predictions(model, dev_loader, device, use_amp)
        dev_probs = 1 / (1 + np.exp(-dev_logits))
        dev_loss = float(
            torch.nn.functional.binary_cross_entropy(
                torch.from_numpy(dev_probs), torch.from_numpy(dev_targets)
            ).item()
        )
        dev_metrics = multilabel_metrics(dev_logits, dev_targets, np.full(len(labels), 0.5))
        print(
            f"epoch {epoch}/{args.epochs} train_loss={total_loss / len(train_loader):.4f} "
            f"dev_loss={dev_loss:.4f} dev_micro_f1={dev_metrics['micro_f1']:.4f} "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
        )
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), best_state)

    model.load_state_dict(torch.load(best_state, map_location=device, weights_only=True))
    dev_logits, dev_targets = collect_predictions(model, dev_loader, device, use_amp)
    thresholds = tune_thresholds(dev_logits, dev_targets)
    test_logits, test_targets = collect_predictions(model, test_loader, device, use_amp)
    metrics = multilabel_metrics(test_logits, test_targets, thresholds)
    model.config.id2label = {index: label for index, label in enumerate(labels)}
    model.config.label2id = label2id
    model.save_pretrained(args.output_dir / "model", safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir / "model")
    (args.output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (args.output_dir / "thresholds.json").write_text(
        json.dumps({label: float(thresholds[index]) for index, label in enumerate(labels)}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Held-out test metrics: {json.dumps(metrics)}")
    print(f"Saved offline text model to {args.output_dir / 'model'}")


if __name__ == "__main__":
    main()
