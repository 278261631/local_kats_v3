@echo off
setlocal EnableExtensions

rem Usage:
rem   clean_reboot.bat           -> 倒计时 60 秒后重启（Ctrl+C 可打断）
rem   clean_reboot.bat now       -> 立即重启
rem   clean_reboot.bat <seconds> -> 倒计时指定秒数后重启（Ctrl+C 可打断）

set "DELAY=300"

if /i "%~1"=="now" (
  set "DELAY=0"
) else if not "%~1"=="" (
  set "DELAY=%~1"
)

rem 简单数字校验：非数字则回退为 60
set "DELAY_NUM=%DELAY%"
for /f "delims=0123456789" %%A in ("%DELAY_NUM%") do set "DELAY_NUM="
if not defined DELAY_NUM set "DELAY=60"

echo 将在倒计时结束后重启系统（Ctrl+C 可打断）。
echo 倒计时秒数: %DELAY%
echo.

if "%DELAY%"=="0" goto :do_reboot

for /l %%S in (%DELAY%,-1,1) do (
  <nul set /p "=剩余 %%S 秒...`r"
  timeout /t 1 >nul
)
echo.

:do_reboot
shutdown /r /t 0 /c "clean_reboot.bat 请求重启"
exit /b %errorlevel%
