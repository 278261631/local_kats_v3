@echo off
cd /d "%~dp0"
echo === gui_ai batch QUERY VSX/MPC (console) ===
python cli_query.py %*
