@echo off
setlocal

REM Manual A/B candidate vetting: noise / pixel-shift / suspicious-target.
REM Run from this script directory (train_AB16pix\).
REM Extra arguments are appended, e.g.:
REM   label_ab_gui.bat --src ai_train_16pix --dst ai_labeled_16pix
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=label_ab_gui.py"

%PY% "%SCRIPT%" %*
if errorlevel 1 pause

endlocal
