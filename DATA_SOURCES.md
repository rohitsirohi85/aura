# Data and model sources

| Asset | Local path | Source | Use and limitation |
| --- | --- | --- | --- |
| GoEmotions curated splits and label map | `data/text/goemotions/` | [Google Research repository](https://github.com/google-research/google-research/tree/master/goemotions) | English Reddit comments, 27 emotions plus Neutral; cite Demszky et al. (ACL 2020). Includes upstream README and repository license copy. Review data provenance/terms and bias notes before redistribution. |
| DistilRoBERTa-base tokenizer and weights | `models/text_base/` | [Hugging Face model card](https://huggingface.co/distilbert/distilroberta-base) | Apache 2.0 model; 82M parameters. Fine-tuned locally for multi-label GoEmotions output. |
| ConvNeXt-Tiny ImageNet weights | `models/vision_base/` | [TorchVision model weights](https://download.pytorch.org/models/convnext_tiny-983f1562.pth) | ImageNet initialization only; project replaces the classifier with its own face task heads. |
| FER+ vote annotations | `data/visual/ferplus/fer2013new.csv` | [Microsoft FER+](https://github.com/microsoft/FERPlus) | Ten crowd votes per image. Original FER2013 images are not part of Microsoft's repository; acquire them separately through the official Kaggle page and respect those terms. |
| UIBAIFED images | `data/visual/uibaifed/UIBAIFED.zip` | [Zenodo record](https://zenodo.org/records/15181380), [paper](https://ceur-ws.org/Vol-4000/paper15.pdf) | 2,948 AI-generated images; CC BY 4.0 is stated for the paper, and the dataset listing reports CC BY. Synthetic-only starter data; not a webcam generalization set. Preserve attribution. |
| Aff-Wild2 | Not downloaded | [Official ABAW/iBUG access route](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/) | Best integrated in-the-wild face dataset for expressions, valence/arousal, and AUs. Requires academic registration/EULA, so this project does not fetch it without the user's institution obtaining access. |
| AffectNet | Not downloaded | [Official project](https://mohammadmahoor.com/pages/databases/affectnet/), [academic-use license](https://mohammadmahoor.com/wp-content/uploads/2024/06/AffectNet-Agreement-v2.1-10May2024.pdf) | Useful for still-image expressions and valence/arousal; access and use are restricted by request/license. |

The app does not call any of these sites. `tools/download_assets.py` is a one-time static file downloader for moving the project to a training machine.
