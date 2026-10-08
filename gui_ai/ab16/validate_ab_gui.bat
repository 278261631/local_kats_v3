@echo off
setlocal

REM Inspect the A/B classifier results on the test split (PySide6).
REM Run from this script directory (train_AB16pix\train_ab\).
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=validate_ab_gui.py"

%PY% "%SCRIPT%" %*
if errorlevel 1 pause

endlocal
