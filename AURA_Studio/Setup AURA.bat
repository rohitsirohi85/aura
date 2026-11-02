@echo off
setlocal
cd /d "%~dp0"
set "AURA_BASE_PYTHON="
for /f "delims=" %%P in ('where python.exe 2^>nul') do if not defined AURA_BASE_PYTHON call :pick "%%P"
if not defined AURA_BASE_PYTHON (
  echo Python was not found. Install Python 3.11 or 3.13, then run this setup again.
  pause
  exit /b 1
)
if not exist "%~dp0.venv\Scripts\python.exe" (
  "%AURA_BASE_PYTHON%" -m venv "%~dp0.venv"
  if errorlevel 1 goto failed
)
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto failed
echo.
echo Setup finished. Double-click "Launch AURA.bat" to start the app.
pause
exit /b 0
:pick
"%~1" -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1
if not errorlevel 1 set "AURA_BASE_PYTHON=%~1"
exit /b 0
:failed
echo.
echo Setup did not finish. Check the error above and confirm internet access.
pause
exit /b 1
