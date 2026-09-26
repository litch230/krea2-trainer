@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The environment is not installed yet.
    echo Run install.bat first.
    pause
    exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0krea2_trainer.py" --ui
if errorlevel 1 (
    echo.
    echo The trainer closed with an error. Review the message above.
    pause
)
