@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo Krea 2 Trainer - dependency installer
echo =====================================

if exist ".venv\Scripts\python.exe" goto install_packages

where py >nul 2>nul
if errorlevel 1 goto try_python

set "PY_LAUNCHER="
call :find_py_version 3.12
if defined PY_LAUNCHER goto create_venv_with_launcher
call :find_py_version 3.11
if defined PY_LAUNCHER goto create_venv_with_launcher
call :find_py_version 3.10
if defined PY_LAUNCHER goto create_venv_with_launcher
goto try_python

:create_venv_with_launcher
%PY_LAUNCHER% -m venv .venv
goto check_venv

:try_python
where python >nul 2>nul
if errorlevel 1 goto no_python
python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
if errorlevel 1 goto no_python
python -m venv .venv
goto check_venv

:find_py_version
set "PY_CHECK_FILE=%TEMP%\krea2-python-check-%RANDOM%-%RANDOM%.tmp"
del /q "%PY_CHECK_FILE%" >nul 2>nul
py -%1 -c "from pathlib import Path; Path(r'%PY_CHECK_FILE%').touch()" >nul 2>nul
if exist "%PY_CHECK_FILE%" set "PY_LAUNCHER=py -%1"
del /q "%PY_CHECK_FILE%" >nul 2>nul
exit /b 0

:check_venv
if not exist ".venv\Scripts\python.exe" goto venv_failed

:install_packages
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto install_failed

if defined KREA2_TORCH_INDEX_URL goto custom_torch
where nvidia-smi >nul 2>nul
if errorlevel 1 goto cpu_torch

echo.
echo NVIDIA GPU detected. Installing the CUDA build of PyTorch...
"%VENV_PYTHON%" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto install_failed
goto common_packages

:custom_torch
echo.
echo Installing PyTorch from %KREA2_TORCH_INDEX_URL%...
"%VENV_PYTHON%" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url "%KREA2_TORCH_INDEX_URL%"
if errorlevel 1 goto install_failed
goto common_packages

:cpu_torch
echo.
echo WARNING: No NVIDIA driver was detected.
echo Installing CPU PyTorch so the interface and validation tools can run.
echo Krea 2 training on Windows requires a CUDA-compatible NVIDIA GPU.
"%VENV_PYTHON%" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto install_failed

:common_packages
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto install_failed

"%VENV_PYTHON%" -c "import torch, tkinter, accelerate, transformers, diffusers, bitsandbytes; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
if errorlevel 1 goto install_failed

echo.
echo Installation completed. Run start_ui.bat to open the trainer.
pause
exit /b 0

:no_python
echo.
echo Python 3.10, 3.11, or 3.12 (64-bit) was not found.
echo Install Python from https://www.python.org/downloads/windows/
pause
exit /b 1

:venv_failed
echo.
echo Could not create the .venv environment.
pause
exit /b 1

:install_failed
echo.
echo Installation failed. Review the error above, then run install.bat again.
pause
exit /b 1
