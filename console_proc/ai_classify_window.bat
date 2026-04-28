@echo off
REM Start PyMPC server from repository root

set SCRIPT_DIR=%~dp0
start "ai classify" cmd /c ""%SCRIPT_DIR%\start_console_ai_classify_filtered_A_-1_day.bat" %*"
