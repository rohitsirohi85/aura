"""Run the project's face and text fine-tuning jobs in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(paths: list[Path], description: str) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing {description}: " + ", ".join(missing))


def run_stage(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    print("Command: " + " ".join(command), flush=True)
    log_dir = ROOT / "artifacts/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "face" if name.startswith("Face") else "text"
    log_path = log_dir / f"{timestamp}_{slug}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise SystemExit(f"{name} failed with exit code {return_code}. See {log_path}.")
    print(f"Training log: {log_path.relative_to(ROOT)}", flush=True)
    print(f"=== {name} complete ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight and train the face and text models sequentially."
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--skip-face", action="store_true", help="Use an existing face checkpoint.")
    parser.add_argument("--skip-text", action="store_true", help="Do not train the text branch.")
    parser.add_argument("--face-epochs", type=int, default=20)
    parser.add_argument("--face-batch-size", type=int, default=4)
    parser.add_argument("--face-gradient-accumulation", type=int, default=4)
    parser.add_argument("--face-output", type=Path, default=Path("artifacts/best.pt"))
    parser.add_argument("--text-epochs", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=4)
    parser.add_argument("--text-gradient-accumulation", type=int, default=8)
    parser.add_argument("--text-output-dir", type=Path, default=Path("artifacts/text_model"))
    args = parser.parse_args()

    if args.skip_face and args.skip_text:
        raise SystemExit("Nothing to run: both training stages were skipped.")
    positive_values = (
        args.face_epochs,
        args.face_batch_size,
        args.face_gradient_accumulation,
        args.text_epochs,
        args.text_batch_size,
        args.text_gradient_accumulation,
    )
    if any(value < 1 for value in positive_values):
        raise SystemExit("Epoch, batch-size, and gradient-accumulation values must be positive.")

    face_output = args.face_output if args.face_output.is_absolute() else ROOT / args.face_output
    text_output = args.text_output_dir if args.text_output_dir.is_absolute() else ROOT / args.text_output_dir
    if args.skip_face:
        require([face_output], "face checkpoint")
    else:
        require(
            [
                ROOT / "data/processed/uibaifed/manifest.csv",
                ROOT / "models/vision_base/convnext_tiny-983f1562.pth",
            ],
            "face training inputs",
        )
    if not args.skip_text:
        require(
            [
                ROOT / "data/text/goemotions/train.tsv",
                ROOT / "data/text/goemotions/dev.tsv",
                ROOT / "data/text/goemotions/test.tsv",
                ROOT / "data/text/goemotions/emotions.txt",
                ROOT / "models/text_base/config.json",
                ROOT / "models/text_base/model.safetensors",
            ],
            "text training inputs",
        )

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable in this PyTorch environment.")
    if args.device == "auto":
        print(
            "Selected device: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
            flush=True,
        )
    else:
        print(f"Selected device: {args.device}", flush=True)

    if not args.skip_face:
        run_stage(
            "Face model (UIBAIFED, expression head)",
            [
                sys.executable,
                "-u",
                "tools/train.py",
                "--manifest",
                "data/processed/uibaifed/manifest.csv",
                "--output",
                str(face_output),
                "--device",
                args.device,
                "--epochs",
                str(args.face_epochs),
                "--batch-size",
                str(args.face_batch_size),
                "--gradient-accumulation",
                str(args.face_gradient_accumulation),
            ],
        )
        require([face_output, face_output.with_suffix(".metrics.json")], "face training outputs")

    if not args.skip_text:
        run_stage(
            "Text model (GoEmotions)",
            [
                sys.executable,
                "-u",
                "tools/train_text.py",
                "--device",
                args.device,
                "--epochs",
                str(args.text_epochs),
                "--batch-size",
                str(args.text_batch_size),
                "--gradient-accumulation",
                str(args.text_gradient_accumulation),
                "--output-dir",
                str(text_output),
            ],
        )
        require(
            [
                text_output / "model/config.json",
                text_output / "test_metrics.json",
                text_output / "thresholds.json",
            ],
            "text training outputs",
        )

    print("\nAll requested training stages completed successfully.", flush=True)


if __name__ == "__main__":
    main()
