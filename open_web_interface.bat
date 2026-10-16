@echo off
setlocal
cd /d "%~dp0"

REM Start the server in its own visible command window so that any error stays on screen.
start "SkinGuardAI server" cmd.exe /k call run_local_demo.bat

REM Give Python time to import the model API before opening the default browser.
timeout /t 5 /nobreak >nul
start "" "http://localhost:8501/"
