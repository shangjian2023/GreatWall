@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_team_pair.ps1" %*
exit /b %ERRORLEVEL%
