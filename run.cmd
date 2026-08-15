@echo off
setlocal

cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo uv not found. Installing uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo Setting up the virtual environment and installing dependencies...
uv sync
if errorlevel 1 (
    echo Dependency installation failed. See the output above.
    pause
    exit /b 1
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example - edit it to add your API keys, then re-run this file.
)

echo.
echo Starting Local-First Knowledge Base Agent at http://localhost:8943
echo Press Ctrl+C to stop.
echo.
uv run streamlit run app.py --server.port 8943

pause
