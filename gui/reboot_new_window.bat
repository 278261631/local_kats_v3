@echo off
REM Start PyMPC server from repository root

set SCRIPT_DIR=%~dp0
start "reboot" cmd /c ""%SCRIPT_DIR%\clean_reboot.bat" %*"
