@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem 配置：在这里指定要清空的一组目录（每行一个）
call :clean "E:/fix_data/download/GY1"
call :clean "E:/fix_data/download/GY2"
call :clean "E:/fix_data/download/GY3"
call :clean "E:/fix_data/download/GY4"
call :clean "E:/fix_data/download/GY5"
call :clean "E:/fix_data/download/GY6"
rem call :clean "D:\another\folder"

echo.
echo 完成
exit /b 0

:clean
set "TARGET_DIR=%~1"
if not defined TARGET_DIR (
  echo [SKIP] 目标目录为空
  exit /b 0
)

if not exist "%TARGET_DIR%" (
  echo [SKIP] 目标目录不存在: "%TARGET_DIR%"
  exit /b 0
)

echo 将清空目录(仅内容，不删除目录本身):
echo   "%TARGET_DIR%"

del /f /q "%TARGET_DIR%\*" 1>nul 2>nul
for /d %%D in ("%TARGET_DIR%\*") do rd /s /q "%%D" 1>nul 2>nul
exit /b 0
