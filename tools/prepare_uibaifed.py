from __future__ import annotations

import argparse
import csv
import random
import re
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image


CLASSES = ("neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear")
ALIASES = {
    "neutral": "neutral",
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "disdain": "disgust",
    "fear": "fear",
    "joy": "happiness",
    "happy": "happiness",
    "happiness": "happiness",
    "sad": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
}


def label_for(path: PurePosixPath) -> str | None:
    for parent in reversed(path.parts[:-1]):
        key = re.sub(r"[^a-z]", "", parent.lower())
        if key in ALIASES:
            return ALIASES[key]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a grouped train/validation/test manifest from the UIBAIFED archive."
    )
    parser.add_argument("--zip", dest="archive", type=Path, default=Path("data/visual/uibaifed/UIBAIFED.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/uibaifed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.archive.exists():
        raise SystemExit(f"Dataset archive is missing: {args.archive}")

    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        records: list[tuple[str, str, str]] = []
        for member in archive.infolist():
            if member.is_dir():
                continue
            path = PurePosixPath(member.filename)
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            label = label_for(path)
            if label is None:
                continue
            fields = path.stem.split("_")
            group = "_".join(fields[1:4]) if len(fields) >= 4 else path.stem
            records.append((member.filename, label, group))

    if not records:
        raise SystemExit("No labeled images were found; inspect the archive's folder names.")
    groups = sorted({record[2] for record in records})
    if len(groups) < 3:
        raise SystemExit("At least three demographic prompt groups are needed for train/val/test splits.")
    random.Random(args.seed).shuffle(groups)
    train_end = max(1, int(len(groups) * 0.70))
    val_end = max(train_end + 1, int(len(groups) * 0.85))
    split_by_group = {
        group: "train" if index < train_end else "val" if index < val_end else "test"
        for index, group in enumerate(groups)
    }

    manifest = output_dir / "manifest.csv"
    columns = ["image", "split", *(f"expr_{label}" for label in CLASSES)]
    counts = {split: 0 for split in ("train", "val", "test")}
    with zipfile.ZipFile(args.archive) as image_archive:
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for index, (member_name, label, group) in enumerate(records):
                relative = Path("images") / f"uibaifed_{index:05d}.jpg"
                with image_archive.open(member_name) as member_stream, Image.open(member_stream) as image:
                    image.convert("RGB").save(output_dir / relative, format="JPEG", quality=92)
                split = split_by_group[group]
                row = {
                    "image": relative.as_posix(),
                    "split": split,
                    **{f"expr_{name}": int(name == label) for name in CLASSES},
                }
                writer.writerow(row)
                counts[split] += 1

    attribution = (
        "Dataset: UIBAIFED – Artificial Intelligence Facial Expression Dataset (Zenodo record 15181380).\n"
        "Source: https://doi.org/10.5281/zenodo.15181380\n"
        "Paper: https://ceur-ws.org/Vol-4000/paper15.pdf\n"
        "The paper is distributed under CC BY 4.0. Images are synthetic; report this and use a real-image\n"
        "held-out set for any claim about performance on webcam photos.\n"
    )
    (output_dir / "ATTRIBUTION.txt").write_text(attribution, encoding="utf-8")
    print(f"Wrote {manifest}; rows={counts}; groups={len(groups)}")


if __name__ == "__main__":
    main()
