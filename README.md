# AURA · Emotion Studio

A local demonstration app for independent facial-expression and text-emotion models. Face, text, speech recognition, and speech synthesis run on the machine using downloaded checkpoints. The Face studio offers a live camera preview, camera snapshot, or image upload; the Text studio accepts typed text or microphone recording transcribed by local Whisper. Auri can read four-line text and image summaries with three bundled local voices.

## Architecture

The two input types have separate models because image and text are different modalities and the available datasets do not contain paired face/text examples with aligned labels. They share one offline app, not a fused prediction head.

| Branch | Model | Outputs | Primary training data |
| --- | --- | --- | --- |
| Face | ImageNet-pretrained ConvNeXt-Tiny, fine-tuned with a shared feature layer and masked task heads | Seven shared expression categories; optional AU and valence/arousal outputs when those labels are present | UIBAIFED is ready as a synthetic starter; FER+ is the natural-image upgrade; Aff-Wild2 is the best integrated research set after access approval |
| Text | DistilRoBERTa-base fine-tuned for 28 independent labels | GoEmotions' 27 emotion labels plus Neutral; several labels can be active at once | GoEmotions train/dev/test splits |

Facial expression categories and emotion categories from text use different label taxonomies. The face model's action-unit and valence/arousal heads are separate tasks from its categorical expression head. The app displays only heads with training labels in the saved checkpoint.

The interface uses an editorial AURA theme with a full-body, expression-changing custom avatar. While Auri reads a summary, the avatar moves and animates along with the audio. Speech synthesis uses the downloaded Kokoro-82M model and local Heart, Bella, or Michael speaker packs; it does not call ElevenLabs or another speech API. Speech recognition records audio until you click **Stop & transcribe**, then uses the downloaded Whisper tiny.en model on this device. Pauses do not stop recording, and the browser's SpeechRecognition service is not used. Webcam streaming is provided by `streamlit-webrtc`; click **START** and allow camera access when prompted. While streaming, you can save and download the most recent eight seconds with the expression overlay. If you do not want to use a microphone, type in the text box instead.

## Datasets and their status

- **GoEmotions (downloaded):** Google's official 43,410/5,426/5,427 train/dev/test split files and 28-label map are in `data/text/goemotions/`. It is English Reddit text and uses multi-label annotations. Google notes demographic and content limitations; cite the dataset paper and review the included README/license before distributing text or a trained model.
- **DistilRoBERTa-base (downloaded):** local 82M-parameter base weights and tokenizer are in `models/text_base/`. The model card states Apache 2.0. It is initialized locally and fine-tuned by the project.
- **ConvNeXt-Tiny ImageNet weights (downloaded):** local checkpoint is in `models/vision_base/`.
- **Kokoro-82M voice model (download separately):** the speech model and Heart, Bella, and Michael speaker files are stored in `models/tts/kokoro/` and loaded from those paths. Kokoro is an 82M-parameter open-weight TTS model published under Apache-2.0; see the [official model card](https://huggingface.co/hexgrad/Kokoro-82M) and [voice list](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md). The English phoneme pipeline is installed locally by `requirements.txt`.
- **Whisper tiny.en speech model (download separately):** the CTranslate2 model files are stored in `models/stt/whisper-tiny.en/` and loaded locally with faster-whisper. The model is English-only, about 76 MB, and published under MIT; see the [official model card](https://huggingface.co/Systran/faster-whisper-tiny.en).
- **FER+ annotations (downloaded):** Microsoft's `fer2013new.csv` is in `data/visual/ferplus/`. FER+ has ten annotator votes per image. Microsoft does not host the FER2013 source photographs; this workspace does not contain them. Obtain `fer2013.csv` through its official Kaggle competition page and follow its access/use terms. FER+'s repository license does not grant rights to the photos.
- **UIBAIFED (downloaded and prepared):** synthetic starter data from the 2025 UIB Artificial Intelligence Facial Expression Dataset. Its 2,948 generated images are in `data/processed/uibaifed/`, with grouped train/validation/test splits of 2,038/460/450. It brings up the visual training pipeline; synthetic performance is not evidence of real-webcam accuracy. The preparation script groups prompt variants before splitting and records attribution.
- **Aff-Wild2 (not downloaded):** best fit for the full in-the-wild face model because it has expression, valence/arousal, and AU labels. The official data route requires academic registration and returns a dataset EULA. The project is ready for a normalized manifest, but the authorized source videos/annotations must come from your institution's approved access.
- **AffectNet (not downloaded):** image expression plus valence/arousal, but access requires a request and the academic-use license restricts use. Do not use unofficial mirrors.

GoEmotions and the model assets were fetched from their official public static-download locations. FER+ annotations were fetched from Microsoft's official repository. Dataset acquisition is a setup/download step; the app itself has no network path for inference.

## Setup on the training machine

Python 3.10 or 3.11 is recommended. Python 3.13 uses pinned official Kokoro and Misaki source revisions because their current PyPI wheels still exclude 3.13. Install a PyTorch build compatible with the GPU driver first, using the current [official PyTorch install selector](https://pytorch.org/get-started/locally/), then install the remaining requirements:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you are moving the project to another computer, fetch its local assets once:

```powershell
python tools/download_assets.py --include-vision-weights --include-ferplus-labels --include-uibaifed
python tools/download_tts_assets.py
python tools/download_stt_assets.py
python tools/generate_tts_samples.py
```

The UIBAIFED option downloads about 1.2 GB; the TTS command downloads about 330 MB of model weights plus three small speaker files, and the STT command downloads about 76 MB. The sample-generation command creates three short local WAV previews for the voice picker. The checked-in scripts use local paths for model loading; after this setup, training and app inference, including speech recognition and synthesis, can run offline.

### Train the text branch

```powershell
python tools/train_text.py --device cuda --batch-size 8 --gradient-accumulation 4 --epochs 4
```

The script fine-tunes the local DistilRoBERTa weights, evaluates against the fixed dev split, tunes a decision threshold per label on dev, reports micro/macro/weighted F1 on the untouched test split, and saves the model, tokenizer, label map, thresholds, and test metrics in `artifacts/text_model/`.

For a smaller GPU, reduce `--batch-size` to 4 and increase `--gradient-accumulation` to 8. Automatic mixed precision is enabled for CUDA by default. Set `--device cpu` only if necessary; transformer fine-tuning on CPU is slow.

### Run both training branches automatically

After preparing the datasets and installing dependencies, run the sequential launcher:

```powershell
python tools/train_all.py --device auto
```

It checks the local inputs, trains the face model first, then the text model, saves each stage's console output under `artifacts/logs/`, and stops if a stage fails. Defaults use smaller batches suitable for a 6 GB GPU. Use `--skip-face` or `--skip-text` to run only one branch; `--face-epochs`, `--text-epochs`, batch sizes, and gradient accumulation can be overridden from the command line.

### Train the face branch

When the UIBAIFED archive is present, build its grouped manifest and train:

```powershell
python tools/prepare_uibaifed.py
python tools/train.py --manifest data\processed\uibaifed\manifest.csv --output artifacts\best.pt --device cuda --batch-size 8 --gradient-accumulation 2 --epochs 20
```

This first model is trained on synthetic faces and predicts expression categories only. After obtaining FER2013 images through the official route, fine-tune that checkpoint on FER+:

```powershell
python tools/prepare_ferplus.py --fer-csv C:\data\fer2013.csv --ferplus-csv data\visual\ferplus\fer2013new.csv --output-dir data\processed\ferplus
python tools/train.py --manifest data\processed\ferplus\manifest.csv --init-checkpoint artifacts\best.pt --output artifacts\best.pt --device cuda --batch-size 8 --gradient-accumulation 2 --epochs 20
```

The FER+ starter uses seven categories shared with Aff-Wild2 and discards examples dominated by contempt, unknown, or non-face votes. For the full multi-task result, map the approved Aff-Wild2 labels into the manifest schema in `ARCHITECTURE.md` and train the AU/valence-arousal heads. Keep every source video/person in one split.

For another authorized face dataset, `tools/train.py` expects one manifest row per face crop with `image`, `split` (`train`, `val`, `test`), seven `expr_<class>` vote proportions, optional `valence`/`arousal`, and optional AU columns `au1`, `au2`, `au4`, `au6`, `au7`, `au10`, `au12`, `au15`, `au23`, `au24`, `au25`, `au26`. Leave missing annotations blank. Expression classes are `neutral`, `happiness`, `surprise`, `sadness`, `anger`, `disgust`, and `fear`. Relative image paths resolve from the manifest directory.

### Run the app

For the packaged demo, double-click `AURA_Studio\Launch AURA.bat`. It starts the bundled app and model files in your browser. If that folder reports missing Python packages, double-click `AURA_Studio\Setup AURA.bat` once, then launch again. To run the source project directly:

```powershell
streamlit run app.py
```

The app forces Transformers and Hugging Face offline mode and loads local face, text, Whisper, Kokoro, and speaker files. If a checkpoint is absent, the related studio explains which download or training step is needed. OpenCV 4's bundled frontal-face cascade provides the starter face cropper; compare it with a modern local detector before making accuracy claims. Voice dictation records locally in the browser and transcribes from the downloaded Whisper model on this machine. The transcript and spoken summaries are processed locally without a hosted speech API.

## GPU notes

The packaged app was smoke-checked on an NVIDIA GeForce RTX 3050 Laptop GPU with 6 GB VRAM. The app selects CUDA when PyTorch exposes it and falls back to CPU otherwise. For text-model training, 8 GB of NVIDIA VRAM is recommended; use the lower batch settings above on smaller GPUs.

## Report and limitations

- Text: micro-F1, macro-F1, weighted-F1, per-label precision/recall, and the dev-tuned thresholds. GoEmotions is English Reddit text, not a universal or clinical emotion ground truth.
- Face expression: macro-F1, balanced accuracy, per-class recall, and confusion matrix. Report UIBAIFED results as synthetic-domain results and evaluate separately on permitted natural images.
- AUs: macro-F1 and per-unit F1. Valence/arousal: CCC per axis. Keep the official test data untouched during model selection.
- Facial movements and text labels are contextual annotator judgments; neither determines a person's private emotional state. Do not use the outputs for consequential decisions.

See [ARCHITECTURE.md](ARCHITECTURE.md) and [DATA_SOURCES.md](DATA_SOURCES.md) for design, citations, and dataset access details.
