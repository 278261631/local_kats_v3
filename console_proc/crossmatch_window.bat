@echo off
REM Start PyMPC server from repository root

set SCRIPT_DIR=%~dp0
start "cross match" cmd /c ""%SCRIPT_DIR%\start_console_rerun_crossmatch_filtered_B_-1_day.bat" %*"
