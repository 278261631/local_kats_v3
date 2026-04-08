@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON_CMD=python"

%PYTHON_CMD% "%SCRIPT_DIR%diff_runner.py" --config "%SCRIPT_DIR%config.json" --date 20260404 --telescope GY1

if errorlevel 1 (
  echo.
  echo Diff任务执行失败，退出码: %errorlevel%
) else (
  echo.
  echo Diff任务执行完成。
)

pause
endlocal
