AURA · EMOTION STUDIO
=====================

For model details, training data, test results, and sample commands, see README.md or output/pdf/AURA_Model_Guide.pdf.

START
-----
Double-click "Launch AURA.bat". It checks Python, the app libraries, and the
OpenCV face detector, then starts the local Streamlit app in your browser at
http://localhost:8501. Keep the console window open while using the app; close
it to stop the server.

This folder includes the trained face and text checkpoints, the local Whisper
speech-to-text model, Kokoro local voices, and the microphone capture component.
Inference and speech processing run on this computer. Webcam and microphone use
require browser permission for localhost.

IF THE LAUNCHER REPORTS MISSING LIBRARIES
-----------------------------------------
Install Python 3.11 or 3.13, then open a terminal in this folder and run:
  python -m pip install -r requirements.txt
Then double-click "Launch AURA.bat" again. On Python 3.13, Git is needed to
install the pinned Kokoro and Misaki packages. An NVIDIA GPU is used when
PyTorch can access one; otherwise the app falls back to CPU.

FILES
-----
app.py                  Streamlit application
artifacts/              Trained face/text model checkpoints and metrics
models/                 Bundled local speech models and voices
components/              Local microphone recorder UI
src/                    Face model architecture
requirements.txt         Python runtime dependencies
tools/                  Optional speech model download helpers
