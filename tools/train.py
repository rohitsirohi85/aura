from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fer_app.model import ACTION_UNITS, EXPRESSION_CLASSES, FacialAffectNet  # noqa: E402


EXPRESSION_COLUMNS = [f"expr_{name}" for name in EXPRESSION_CLASSES]
VA_COLUMNS = ["valence", "arousal"]
AU_COLUMNS = [f"au{unit}" for unit in ACTION_UNITS]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, split: str) -> None:
        frame = pd.read_csv(manifest)
        required = {"image", "split"}
        if not required.issubset(frame.columns):
            raise ValueError("Manifest needs 'image' and 'split' columns.")
        frame = frame[frame["split"].astype(str).str.lower() == split].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No '{split}' rows found in {manifest}.")
        self.frame = frame
        self.root = manifest.resolve().parent
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                *(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(8),
                        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                    ]
                    if split == "train"
                    else []
                ),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def _values(self, row: pd.Series, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
        values = np.full(len(columns), np.nan, dtype=np.float32)
        for index, column in enumerate(columns):
            if column in row.index:
                try:
                    values[index] = float(row[column])
                except (TypeError, ValueError):
                    pass
        mask = np.isfinite(values)
        values[~mask] = 0.0
        return values, mask.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["image"]))
        if not image_path.is_absolute():
            image_path = self.root / image_path
        with Image.open(image_path) as image_file:
            image = self.transform(image_file.convert("RGB"))
        expression, expression_mask = self._values(row, EXPRESSION_COLUMNS)
        valence_arousal, va_mask = self._values(row, VA_COLUMNS)
        action_units, au_mask = self._values(row, AU_COLUMNS)
        return {
            "image": image,
            "expression": torch.from_numpy(expression),
            "expression_mask": torch.tensor(float(expression_mask.all() and expression.sum() > 0)),
            "valence_arousal": torch.from_numpy(valence_arousal),
            "va_mask": torch.from_numpy(va_mask),
            "action_units": torch.from_numpy(action_units),
            "au_mask": torch.from_numpy(au_mask),
        }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    mask = mask.to(dtype=values.dtype)
    denominator = mask.sum()
    if denominator.item() == 0:
        return None
    return (values * mask).sum() / denominator


def losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms: dict[str, torch.Tensor] = {}
    expression_mask = batch["expression_mask"]
    if expression_mask.sum().item() > 0:
        soft_ce = -(batch["expression"] * F.log_softmax(outputs["expression"], dim=1)).sum(dim=1)
        terms["expression"] = masked_mean(soft_ce, expression_mask)  # type: ignore[assignment]

    va_mask = batch["va_mask"]
    if va_mask.sum().item() > 0:
        va_error = F.smooth_l1_loss(outputs["valence_arousal"], batch["valence_arousal"], reduction="none")
        terms["valence_arousal"] = masked_mean(va_error, va_mask)  # type: ignore[assignment]

    au_mask = batch["au_mask"]
    if au_mask.sum().item() > 0:
        au_error = F.binary_cross_entropy_with_logits(
            outputs["action_units"], batch["action_units"], reduction="none"
        )
        terms["action_units"] = masked_mean(au_error, au_mask)  # type: ignore[assignment]

    if not terms:
        raise ValueError("This batch has no observed labels for any task.")
    weights = {"expression": 1.0, "valence_arousal": 0.5, "action_units": 0.5}
    total = sum(weights[name] * value for name, value in terms.items())
    return total, terms


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def class_balanced_sampler(dataset: ManifestDataset) -> WeightedRandomSampler | None:
    if not set(EXPRESSION_COLUMNS).issubset(dataset.frame.columns):
        return None
    targets = dataset.frame[EXPRESSION_COLUMNS].fillna(0).to_numpy(dtype=np.float64)
    if np.any(targets.sum(axis=1) <= 0):
        return None
    class_mass = targets.sum(axis=0)
    inverse_mass = np.divide(1.0, class_mass, out=np.zeros_like(class_mass), where=class_mass > 0)
    sample_weights = targets @ inverse_mass
    if not np.isfinite(sample_weights).all() or not (sample_weights > 0).any():
        return None
    sample_weights[sample_weights <= 0] = sample_weights[sample_weights > 0].min()
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def find_trained_heads(manifest: Path) -> tuple[list[str], list[str], list[int]]:
    frame = pd.read_csv(manifest)
    training = frame[frame["split"].astype(str).str.lower() == "train"]
    heads: list[str] = []
    trained_va_axes: list[str] = []
    trained_action_units: list[int] = []
    if set(EXPRESSION_COLUMNS).issubset(training.columns):
        expression = training[EXPRESSION_COLUMNS].to_numpy(dtype=np.float32)
        usable = np.isfinite(expression).all(axis=1) & (np.nan_to_num(expression).sum(axis=1) > 0)
        if usable.any():
            heads.append("expression")
    trained_va_axes = [
        axis for axis in VA_COLUMNS if axis in training.columns and training[axis].notna().any()
    ]
    if trained_va_axes:
        heads.append("valence_arousal")
    trained_action_units = [
        unit
        for unit in ACTION_UNITS
        if f"au{unit}" in training.columns and training[f"au{unit}"].notna().any()
    ]
    if trained_action_units:
        heads.append("action_units")
    if not heads:
        raise ValueError("Training split contains no labeled targets supported by the model.")
    return heads, trained_va_axes, trained_action_units


def run_epoch(
    model: FacialAffectNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    collect_metrics: bool | None = None,
    scaler=None,
    gradient_accumulation: int = 1,
    use_amp: bool = False,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    if collect_metrics is None:
        collect_metrics = not training
    model.train(training)
    total_loss = 0.0
    batches = 0
    expression_confusion = np.zeros((len(EXPRESSION_CLASSES), len(EXPRESSION_CLASSES)), dtype=np.int64)
    va_stats = np.zeros((2, 6), dtype=np.float64)  # n, sum-pred, sum-target, sum-pred2, sum-target2, sum-cross
    au_counts = np.zeros((len(AU_COLUMNS), 3), dtype=np.int64)  # true-positive, false-positive, false-negative
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        if training and step % gradient_accumulation == 0:
            optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with torch.set_grad_enabled(training), autocast:
            outputs = model(batch["image"])
            loss, _ = losses(outputs, batch)
            if training:
                group_start = (step // gradient_accumulation) * gradient_accumulation
                group_size = min(gradient_accumulation, len(loader) - group_start)
                scaler.scale(loss / group_size).backward()
        if training:
            end_group = (step + 1) % gradient_accumulation == 0 or step + 1 == len(loader)
            if end_group:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu())
        batches += 1
        if collect_metrics:
            selected = batch["expression_mask"].bool()
            if selected.any():
                y_true = batch["expression"][selected].argmax(dim=1).cpu().numpy()
                y_pred = outputs["expression"][selected].argmax(dim=1).cpu().numpy()
                np.add.at(expression_confusion, (y_true, y_pred), 1)

            va_target = batch["valence_arousal"].detach().cpu().numpy()
            va_prediction = outputs["valence_arousal"].detach().cpu().numpy()
            va_mask = batch["va_mask"].detach().cpu().numpy().astype(bool)
            for axis in range(2):
                valid = va_mask[:, axis]
                if valid.any():
                    pred = va_prediction[valid, axis]
                    target = va_target[valid, axis]
                    va_stats[axis] += (
                        len(pred), pred.sum(), target.sum(), np.square(pred).sum(),
                        np.square(target).sum(), (pred * target).sum(),
                    )

            au_target = batch["action_units"].detach().cpu().numpy() >= 0.5
            au_prediction = torch.sigmoid(outputs["action_units"]).detach().cpu().numpy() >= 0.5
            au_mask = batch["au_mask"].detach().cpu().numpy().astype(bool)
            for unit in range(len(ACTION_UNITS)):
                valid = au_mask[:, unit]
                if valid.any():
                    pred = au_prediction[valid, unit]
                    target = au_target[valid, unit]
                    au_counts[unit, 0] += np.logical_and(pred, target).sum()
                    au_counts[unit, 1] += np.logical_and(pred, ~target).sum()
                    au_counts[unit, 2] += np.logical_and(~pred, target).sum()

    if batches == 0:
        raise ValueError("No batches were produced.")
    metrics: dict[str, float] = {}
    if collect_metrics and expression_confusion.sum() > 0:
        true_positive = np.diag(expression_confusion).astype(np.float64)
        precision_denominator = expression_confusion.sum(axis=0)
        recall_denominator = expression_confusion.sum(axis=1)
        f1_denominator = precision_denominator + recall_denominator
        per_class_f1 = np.divide(
            2 * true_positive,
            f1_denominator,
            out=np.zeros_like(true_positive),
            where=f1_denominator > 0,
        )
        metrics["expression_accuracy"] = float(true_positive.sum() / expression_confusion.sum())
        metrics["expression_macro_f1"] = float(per_class_f1.mean())
        present = recall_denominator > 0
        recalls = np.divide(
            true_positive,
            recall_denominator,
            out=np.zeros_like(true_positive),
            where=present,
        )
        metrics["expression_balanced_accuracy"] = float(recalls[present].mean())
        for index, label in enumerate(EXPRESSION_CLASSES):
            if present[index]:
                metrics[f"recall_{label}"] = float(recalls[index])
    for axis, name in enumerate(VA_COLUMNS):
        count, sum_pred, sum_target, sum_pred2, sum_target2, sum_cross = va_stats[axis]
        if count > 1:
            covariance = sum_cross - sum_pred * sum_target / count
            variance_pred = sum_pred2 - sum_pred * sum_pred / count
            variance_target = sum_target2 - sum_target * sum_target / count
            mean_difference = (sum_pred - sum_target) / count
            denominator = variance_pred + variance_target + count * mean_difference**2
            if denominator > 0:
                metrics[f"{name}_ccc"] = float(2 * covariance / denominator)
    unit_denominators = 2 * au_counts[:, 0] + au_counts[:, 1] + au_counts[:, 2]
    observed_units = unit_denominators > 0
    if observed_units.any():
        unit_f1 = np.divide(
            2 * au_counts[:, 0],
            unit_denominators,
            out=np.zeros(len(ACTION_UNITS), dtype=np.float64),
            where=observed_units,
        )
        metrics["action_units_macro_f1"] = float(unit_f1[observed_units].mean())
    return total_loss / batches, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the shared facial-affect model.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", default=Path("artifacts/best.pt"), type=Path)
    parser.add_argument(
        "--base-weights",
        default=Path("models/vision_base/convnext_tiny-983f1562.pth"),
        type=Path,
        help="Locally downloaded TorchVision ImageNet checkpoint.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Continue fine-tuning from a checkpoint produced by this training script.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("epochs, batch-size, and gradient-accumulation must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable in this PyTorch environment.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    use_amp = device.type == "cuda" and not args.no_amp
    trained_heads, trained_va_axes, trained_action_units = find_trained_heads(args.manifest)
    train_data = ManifestDataset(args.manifest, "train")
    validation_data = ManifestDataset(args.manifest, "val")
    sampler = class_balanced_sampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    previous_checkpoint = None
    if args.init_checkpoint:
        if not args.init_checkpoint.exists():
            raise SystemExit(f"Initial checkpoint not found: {args.init_checkpoint}")
        previous_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        model = FacialAffectNet(pretrained=False)
        model.load_state_dict(previous_checkpoint["state_dict"])
        trained_heads = sorted(set(trained_heads) | set(previous_checkpoint.get("trained_heads", [])))
        trained_va_axes = sorted(set(trained_va_axes) | set(previous_checkpoint.get("trained_va_axes", [])))
        trained_action_units = sorted(
            set(trained_action_units) | set(previous_checkpoint.get("trained_action_units", []))
        )
    else:
        if not args.base_weights.exists():
            raise SystemExit(
                "Local ConvNeXt weights missing. Run `python tools/download_assets.py --include-vision-weights`."
            )
        model = FacialAffectNet(pretrained_path=args.base_weights)
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 2e-5},
            {"params": model.shared.parameters(), "lr": 2e-4},
            {"params": model.expression_head.parameters(), "lr": 2e-4},
            {"params": model.au_head.parameters(), "lr": 2e-4},
            {"params": model.va_head.parameters(), "lr": 2e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            collect_metrics=False,
            scaler=scaler,
            gradient_accumulation=args.gradient_accumulation,
            use_amp=use_amp,
        )
        val_loss, metrics = run_epoch(model, validation_loader, device, use_amp=use_amp)
        scheduler.step()
        report = " | ".join(f"{name} {value:.3f}" for name, value in metrics.items())
        print(f"epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | {report}")
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "trained_heads": trained_heads,
                "trained_va_axes": trained_va_axes,
                "trained_action_units": trained_action_units,
                "expression_classes": list(EXPRESSION_CLASSES),
                "action_units": list(ACTION_UNITS),
                "input_size": 224,
                "normalization_mean": MEAN,
                "normalization_std": STD,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, args.output)
    print(f"Best validation loss: {best_loss:.4f}; checkpoint: {args.output}")
    test_data = ManifestDataset(args.manifest, "test")
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    best_checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"])
    test_loss, test_metrics = run_epoch(model, test_loader, device, use_amp=use_amp)
    report = " | ".join(f"{name} {value:.3f}" for name, value in test_metrics.items())
    print(f"Held-out test loss {test_loss:.4f} | {report}")
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "test_loss": test_loss,
                "metrics": test_metrics,
                "expression_classes": list(EXPRESSION_CLASSES),
                "expression_confusion_matrix": expression_test_confusion(model, test_loader, device, use_amp),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def expression_test_confusion(
    model: FacialAffectNet, loader: DataLoader, device: torch.device, use_amp: bool
) -> list[list[int]]:
    matrix = np.zeros((len(EXPRESSION_CLASSES), len(EXPRESSION_CLASSES)), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with autocast:
                outputs = model(batch["image"])
            valid = batch["expression_mask"].bool()
            if valid.any():
                true = batch["expression"][valid].argmax(dim=1).cpu().numpy()
                pred = outputs["expression"][valid].argmax(dim=1).cpu().numpy()
                np.add.at(matrix, (true, pred), 1)
    return matrix.tolist()


if __name__ == "__main__":
    main()
