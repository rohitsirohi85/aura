from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "stt" / "whisper-tiny.en"
REPOSITORY = "Systran/faster-whisper-tiny.en"
REVISION = "0d3d19a32d3338f10357c0889762bd8d64bbdeba"
FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")


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
    print("Local Whisper speech-to-text model is ready.")


if __name__ == "__main__":
    main()
