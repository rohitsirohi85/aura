# AURA Emotion Studio: Model Guide

This guide uses simple words to explain the models in this folder, the data used to train them, their test results, and how to try the app.

## What the app does

AURA has two separate emotion models. One looks for visible face-expression patterns in a photo or webcam frame. The other looks for emotion words and patterns in typed text. The scores from these two models are not mixed together.

The app also has two speech helpers. Whisper turns recorded speech into words. The text model then reads those words. Kokoro reads Auri's short summary out loud. These speech tools run from files on this computer. The app does not judge emotion from the sound or tone of a voice.

## The simple architecture

```text
PHOTO OR WEBCAM
  -> OpenCV finds a face
  -> ConvNeXt-Tiny reads the face crop
  -> 7 visible-expression scores

TYPED WORDS
  -> DistilRoBERTa
  -> 28 emotion scores

RECORDED SPEECH
  -> local Whisper turns speech into words
  -> the same DistilRoBERTa text model scores the words

MODEL RESULTS
  -> Auri makes a short summary
  -> local Kokoro voice can read it aloud
```

### Face model

The face model starts with ConvNeXt-Tiny. This is an image model first trained on the ImageNet picture collection. We added a small shared layer and a separate output for each of seven expression labels: neutral, happiness, surprise, sadness, anger, disgust, and fear. A face finder from OpenCV crops a face, then the model returns scores for the seven labels.

The saved face checkpoint trained only the expression output. Other outputs in the code, such as action units and valence/arousal, were not trained for this checkpoint and are not shown as results.

### Text model

The text model starts with DistilRoBERTa-base, a model that reads English words. We added an output for each of 28 GoEmotions labels: 27 emotions plus Neutral. Each label gets its own score, so more than one label can be active at the same time.

### Speech and Auri's voice

Whisper tiny.en turns an English recording into text on this computer. AURA sends only that text to the text model. Kokoro creates Auri's spoken summary using one of the local voice files. The microphone stays on until you press **Stop & transcribe**.

## Data and model sources

- [UIBAIFED dataset record](https://zenodo.org/records/15181380) and [paper](https://ceur-ws.org/Vol-4000/paper15.pdf)
- [GoEmotions data](https://github.com/google-research/google-research/tree/master/goemotions) and [research paper](https://aclanthology.org/2020.acl-main.372/)
- [DistilRoBERTa-base model card](https://huggingface.co/distilbert/distilroberta-base)
- [TorchVision ConvNeXt-Tiny weights](https://download.pytorch.org/models/convnext_tiny-983f1562.pth)

<!-- PDF_PAGE_BREAK -->

## Data used

| Model | Data used | What to know |
| --- | --- | --- |
| Face expression | UIBAIFED: 2,948 AI-made face images | The saved checkpoint uses 2,038 training, 460 validation, and 450 test images. These are synthetic images, not webcam photos. |
| Text emotion | GoEmotions: 43,410 train, 5,426 validation, and 5,427 test examples | English Reddit comments with 27 emotion labels plus Neutral. A comment may have several labels. |

The workspace also has FER+ vote labels, but not the original FER2013 photos. FER+ was not used to train the saved face checkpoint. Aff-Wild2 and AffectNet were not downloaded or used.

## How training worked

### Face

The face training script resized each face to 224 by 224 pixels. It started from ImageNet ConvNeXt-Tiny weights, then learned from the UIBAIFED training split. It used small image changes during training, such as flips, turns, and color changes. At each step, it compared its seven scores with the example label and adjusted the weights to reduce mistakes. The best version was chosen using the validation split. The separate test split was used for the final report.

Run these from the main project folder, one level above `AURA_Studio`:

```powershell
python tools/prepare_uibaifed.py
python tools/train.py --manifest data\processed\uibaifed\manifest.csv --output artifacts\best.pt --device auto --batch-size 8 --gradient-accumulation 2 --epochs 20
```

### Text

The text script started from the local DistilRoBERTa-base files. It fine-tuned the model on GoEmotions for four passes through the training data. It checked the validation data after each pass and kept the best version. It used the validation data to choose a score cutoff for each label. It then measured results on the untouched test split.

Run this from the main project folder:

```powershell
python tools/train_text.py --device auto --batch-size 8 --gradient-accumulation 4 --epochs 4
```

Training again is not needed to use the demo. The app folder already has the trained weights.

## Held-out test results

| Model and test set | Measure | Result | Plain meaning |
| --- | --- | --- | --- |
| Face, 450 synthetic images | Accuracy | 98.4% | Correct top label on this test set. |
| Face, 450 synthetic images | Macro F1 | 96.5% | Average score across the seven labels, giving each label equal weight. |
| Face, 450 synthetic images | Balanced accuracy | 96.8% | Average of the seven label recall scores. |
| Face, 450 synthetic images | Anger recall | 81.8% | The model found about 8 out of 10 anger examples in this test set. |
| Text, 5,427 comments | Micro F1 | 58.7% | Overall match of predicted and labeled emotion tags. |
| Text, 5,427 comments | Macro F1 | 46.2% | Average match across emotion labels, giving rare labels the same weight. |
| Text, 5,427 comments | Weighted F1 | 59.0% | Average match that gives common labels more weight. |

## Try the app

1. Double-click **Launch AURA.bat** in this folder. Keep its black console window open while you use the app.
2. In **Face studio**, choose **Upload image** and select a clear, front-facing photo. You can also use the webcam options.
3. In **Text studio**, try: `I am worried about the interview, but excited to start my new job.` Then click **Read the emotion signals**.
4. For speech, click **Start recording**, speak at your own pace, and click **Stop & transcribe** when you are finished. Check the words, then run the text analysis.
5. Click **Auri reads this** to hear the short summary in the selected local voice.

## Simple commands

Open PowerShell in this folder to run the app from a terminal:

```powershell
python -m streamlit run app.py
```

Check that the main Python files have no syntax errors:

```powershell
python -m py_compile app.py src\fer_app\model.py
```

Print the top three text scores from a short example:

```powershell
python -c "import app; rows, _ = app.analyze_text('I feel hopeful today.'); print([(r['label'], round(r['score'], 3)) for r in rows[:3]])"
```

## Important limits

- The face model describes visible patterns in an image. It cannot know a person's real feelings.
- The face checkpoint was trained and tested on synthetic images. Its high test score should not be treated as real-world accuracy.
- GoEmotions comes from English Reddit comments. Its labels may not fit every person, culture, or situation.
- Spoken input is judged from the words Whisper recognized. AURA does not read emotion from voice pitch or tone.
