@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo [ERROR] .venv not found. Please install the environment first, see README.md
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0launch.pyw"
exit /b
