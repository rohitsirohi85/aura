from __future__ import annotations

import argparse
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOEMOTIONS = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions"
TEXT_MODEL = "https://huggingface.co/distilbert/distilroberta-base/resolve/main"
VISION_MODEL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
UIBAIFED = "https://zenodo.org/records/15181380/files/UIBAIFED.zip?download=1"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "local-affect-project/1.0"}
    head = urllib.request.Request(url, headers=headers, method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as response:
        total_size = int(response.headers.get("Content-Length", "0") or 0)
    if destination.exists() and total_size and destination.stat().st_size == total_size:
        print(f"Already present: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url} -> {destination}")
    if total_size >= 64 * 1024 * 1024:
        parts_dir = destination.with_suffix(destination.suffix + ".parts")
        parts_dir.mkdir(parents=True, exist_ok=True)
        chunk_size = 32 * 1024 * 1024
        chunk_count = math.ceil(total_size / chunk_size)

        def fetch_part(index: int) -> Path:
            start = index * chunk_size
            end = min(total_size - 1, start + chunk_size - 1)
            part = parts_dir / f"{index:04d}.part"
            expected = end - start + 1
            if part.exists() and part.stat().st_size == expected:
                return part
            request = urllib.request.Request(
                url,
                headers={**headers, "Range": f"bytes={start}-{end}"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"Server did not honor byte range {start}-{end} (status {response.status}).")
                expected_range = f"bytes {start}-{end}/"
                if not response.headers.get("Content-Range", "").startswith(expected_range):
                    raise RuntimeError(f"Unexpected byte range returned for {start}-{end}.")
                with part.open("wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
            if part.stat().st_size != expected:
                raise RuntimeError(f"Incomplete ranged download for bytes {start}-{end}.")
            return part

        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(fetch_part, index) for index in range(chunk_count)]
                for future in as_completed(futures):
                    future.result()
            with temporary.open("wb") as out:
                for index in range(chunk_count):
                    with (parts_dir / f"{index:04d}.part").open("rb") as part:
                        shutil.copyfileobj(part, out, length=1024 * 1024)
            if temporary.stat().st_size != total_size:
                raise RuntimeError("Combined ranged download has the wrong size.")
            temporary.replace(destination)
            for part in parts_dir.iterdir():
                part.unlink()
            parts_dir.rmdir()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public training data and pretrained weights for offline use."
    )
    parser.add_argument(
        "--include-vision-weights",
        action="store_true",
        help="Also fetch the 109 MB TorchVision ConvNeXt-Tiny ImageNet checkpoint.",
    )
    parser.add_argument(
        "--include-ferplus-labels",
        action="store_true",
        help="Fetch FER+ annotations only; FER2013 source images are not included.",
    )
    parser.add_argument(
        "--include-uibaifed",
        action="store_true",
        help="Fetch the 1.2 GB UIBAIFED synthetic facial-expression archive.",
    )
    args = parser.parse_args()

    data_dir = ROOT / "data" / "text" / "goemotions"
    for name in ("train.tsv", "dev.tsv", "test.tsv", "emotions.txt"):
        download(f"{GOEMOTIONS}/data/{name}", data_dir / name)
    download(f"{GOEMOTIONS}/README.md", data_dir / "GOEMOTIONS_README.md")
    download(
        "https://raw.githubusercontent.com/google-research/google-research/master/LICENSE",
        data_dir / "GOOGLE_RESEARCH_LICENSE.txt",
    )

    text_dir = ROOT / "models" / "text_base"
    for name in (
        "config.json",
        "model.safetensors",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        download(f"{TEXT_MODEL}/{name}", text_dir / name)
    download(f"{TEXT_MODEL}/README.md", text_dir / "MODEL_CARD.md")

    if args.include_vision_weights:
        download(VISION_MODEL, ROOT / "models" / "vision_base" / "convnext_tiny-983f1562.pth")
    if args.include_ferplus_labels:
        download(
            "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv",
            ROOT / "data" / "visual" / "ferplus" / "fer2013new.csv",
        )
    if args.include_uibaifed:
        download(UIBAIFED, ROOT / "data" / "visual" / "uibaifed" / "UIBAIFED.zip")

    print("Assets are local. Training and app inference do not call a hosted inference API.")


if __name__ == "__main__":
    main()
