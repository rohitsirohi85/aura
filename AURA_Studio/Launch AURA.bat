@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "AURA_PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" call :check "%~dp0.venv\Scripts\python.exe"
if not defined AURA_PYTHON (
  for /f "delims=" %%P in ('where python.exe 2^>nul') do call :check "%%P"
)
if not defined AURA_PYTHON (
  echo.
  echo A compatible Python environment with AURA's dependencies was not found.
  echo Install Python 3.11 or 3.13 and run: python -m pip install -r requirements.txt
  echo Then double-click this launcher again.
  echo.
  pause
  exit /b 1
)
"%AURA_PYTHON%" -m streamlit run "%~dp0app.py" --server.address 127.0.0.1 --browser.gatherUsageStats false
set "AURA_EXIT=!errorlevel!"
if not "!AURA_EXIT!"=="0" (
  echo.
  echo AURA stopped with an error. Read the message above for details.
  pause
)
exit /b !AURA_EXIT!

:check
"%~1" -c "import importlib.util as u, cv2; names=['streamlit','streamlit_webrtc','av','numpy','torch','torchvision','PIL','transformers','faster_whisper','kokoro','spacy','misaki','en_core_web_sm']; missing=[n for n in names if u.find_spec(n) is None]; assert not missing, 'missing: '+','.join(missing); assert getattr(cv2,'CascadeClassifier',None) is not None or getattr(getattr(cv2,'objdetect',None),'CascadeClassifier',None) is not None, 'OpenCV CascadeClassifier missing'; assert getattr(cv2,'data',None) and cv2.data.haarcascades, 'OpenCV Haar cascade files missing'" >nul 2>&1
if not errorlevel 1 if not defined AURA_PYTHON set "AURA_PYTHON=%~1"
exit /b 0
