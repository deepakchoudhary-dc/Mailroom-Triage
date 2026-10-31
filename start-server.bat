@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo.
echo mailroom-triage - starting on http://127.0.0.1:8000
echo Open http://127.0.0.1:8000/docs in your browser.
echo Press Ctrl+C in this window to stop.
echo.

"%PY%" -m mailroom_triage.serve

echo.
echo Server stopped.
pause
