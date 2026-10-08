@echo off
setlocal

REM Train the dual-channel A/B classifier (noise / pixelshift / target).
REM Run from this script directory (train_AB16pix\train_ab\).
REM Extra arguments are appended, e.g.:
REM   train_ab.bat --epochs 40 --use-snr
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=train_ab.py"

%PY% "%SCRIPT%" %*
if errorlevel 1 pause

endlocal
