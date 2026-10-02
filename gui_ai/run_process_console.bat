@echo off
cd /d "%~dp0"
echo === gui_ai batch PROCESS (console) ===
python cli_process.py %*
