@echo off
setlocal

REM Run the A/B classifier on one FITS pair or a folder of them.
REM Run from this script directory (train_AB16pix\train_ab\).
REM   predict_ab.bat models_ab\best.pt ..\ai_split_16pix\test\target
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=predict_ab.py"

%PY% "%SCRIPT%" %*
if errorlevel 1 pause

endlocal
