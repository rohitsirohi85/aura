from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "tts" / "kokoro"
REPOSITORY = "hexgrad/Kokoro-82M"
REVISION = "9871713c6a29032f914cee65b617e2b6b3323770"
FILES = (
    "config.json",
    "kokoro-v1_0.pth",
    "voices/af_heart.pt",
    "voices/af_bella.pt",
    "voices/am_michael.pt",
)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        print(f"Fetching {filename} from {REPOSITORY}@{REVISION}")
        destination = hf_hub_download(
            repo_id=REPOSITORY,
            filename=filename,
            revision=REVISION,
            local_dir=MODEL_DIR,
        )
        print(f"Saved {Path(destination).relative_to(ROOT)}")
    print("Local Kokoro model and three speaker packs are ready.")


if __name__ == "__main__":
    main()
