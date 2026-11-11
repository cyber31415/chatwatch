@echo off
rem chatwatch for Windows: double-click to start (the first time it sets WSL's Ubuntu up). Ctrl+C stops it.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0chatwatch.ps1" %*
echo.
pause
