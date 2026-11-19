@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" %*
set "NOVA_EXIT=%errorlevel%"
if not "%NOVA_EXIT%"=="0" (
    echo Nova Stasis could not start.
    pause
)
exit /b %NOVA_EXIT%
