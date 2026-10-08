@echo off
setlocal

REM Build leakage-free train/val/test split for the A/B classifier.
REM Run from this script directory (train_AB16pix\train_ab\).
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=make_split_dataset.py"

%PY% "%SCRIPT%" %*
if errorlevel 1 pause

endlocal
