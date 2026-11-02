from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
)
SPLITS = {
    "training": "train",
    "publictest": "val",
    "privatetest": "test",
}


def normalized(row: dict[str, str]) -> dict[str, str]:
    return {str(key).strip().lower(): str(value).strip() for key, value in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge local FER2013 pixels with Microsoft's FER+ vote annotations."
    )
    parser.add_argument("--fer-csv", required=True, type=Path)
    parser.add_argument("--ferplus-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.fer_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        fer_rows = [normalized(row) for row in csv.DictReader(handle)]
    with args.ferplus_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        plus_rows = [normalized(row) for row in csv.DictReader(handle)]

    if not fer_rows or not plus_rows:
        raise SystemExit("Both input CSV files must contain data rows.")
    if len(fer_rows) != len(plus_rows):
        raise SystemExit(
            f"Row count mismatch: FER2013 has {len(fer_rows)} rows and FER+ has {len(plus_rows)}."
        )
    if not {"pixels", "usage"}.issubset(fer_rows[0]):
        raise SystemExit("FER2013 CSV must have 'pixels' and 'Usage' columns.")
    if not set(CLASSES).issubset(plus_rows[0]):
        raise SystemExit("FER+ CSV is missing one or more expected expression vote columns.")

    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    fieldnames = ["image", "split", *(f"expr_{name}" for name in CLASSES)]
    written = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (fer, plus) in enumerate(zip(fer_rows, plus_rows)):
            usage = fer.get("usage", plus.get("usage", "")).lower().replace(" ", "")
            split = SPLITS.get(usage)
            if split is None:
                skipped += 1
                continue

            try:
                pixels = np.fromstring(fer["pixels"], sep=" ", dtype=np.uint8)
                votes = np.asarray([float(plus[name]) for name in CLASSES], dtype=np.float32)
            except (KeyError, ValueError):
                skipped += 1
                continue
            if pixels.size != 48 * 48 or not np.isfinite(votes).all():
                skipped += 1
                continue

            known_votes = float(votes.sum())
            excluded_votes = sum(
                float(plus.get(name, "0") or 0) for name in ("contempt", "unknown", "nf")
            )
            if known_votes <= 0 or excluded_votes >= known_votes:
                skipped += 1
                continue

            relative_image = Path("images") / f"fer_{index:06d}.jpg"
            Image.fromarray(pixels.reshape(48, 48), mode="L").convert("RGB").save(
                output_dir / relative_image, quality=95
            )
            target = {f"expr_{name}": float(votes[i] / known_votes) for i, name in enumerate(CLASSES)}
            writer.writerow({"image": relative_image.as_posix(), "split": split, **target})
            written[split] += 1

    print(f"Wrote manifest: {manifest_path}")
    print(f"Kept rows: {written}; skipped rows: {skipped}")


if __name__ == "__main__":
    main()
