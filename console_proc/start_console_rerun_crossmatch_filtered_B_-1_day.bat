@echo off
cd /d "%~dp0"
set "SCRIPT_DIR=%~dp0"

for /f "tokens=*" %%i in ('powershell -Command "(Get-Date).AddDays(-1).ToString('yyyyMMdd')"') do (
    set YESTERDAY=%%i
)

python "%SCRIPT_DIR%crossmatch_rerun_runner.py" --config "%SCRIPT_DIR%config.json" --date %YESTERDAY% --profile B

