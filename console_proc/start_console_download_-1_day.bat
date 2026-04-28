@echo off
cd /d "%~dp0"

for /f "tokens=*" %%i in ('powershell -Command "(Get-Date).AddDays(-1).ToString('yyyyMMdd')"') do (
    set YESTERDAY=%%i
)

python cli.py --config "config.json" --date %YESTERDAY%

