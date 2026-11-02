from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app


def main() -> None:
    app.TTS_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for voice, label in app.TTS_VOICES.items():
        output = app.TTS_SAMPLE_DIR / f"{voice}.wav"
        if output.is_file():
            print(f"Keeping existing preview: {output.relative_to(ROOT)}")
            continue
        print(f"Rendering {label} sample with local Kokoro weights")
        audio = app.synthesize_local_speech(app.TTS_SAMPLE_TEXT, voice)
        output.write_bytes(audio)
        print(f"Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
