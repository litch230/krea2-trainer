@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if /i "%~1"=="--no-pause" set "NO_PAUSE=1"

echo.
echo Krea 2 Trainer - dependency installer
echo =====================================

:detect_python
if exist ".venv\Scripts\python.exe" goto install_packages

where py >nul 2>nul
if errorlevel 1 goto try_python

set "PY_LAUNCHER="
call :find_py_version 3.13
if defined PY_LAUNCHER goto create_venv_with_launcher
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
if errorlevel 1 goto try_known_python_paths
python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul
if not errorlevel 1 (
    python -m venv .venv
    goto check_venv
)

:try_known_python_paths
if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    "%LocalAppData%\Programs\Python\Python313\python.exe" -m venv .venv
    goto check_venv
)
if exist "%ProgramFiles%\Python313\python.exe" (
    "%ProgramFiles%\Python313\python.exe" -m venv .venv
    goto check_venv
)
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    "%LocalAppData%\Programs\Python\Python312\python.exe" -m venv .venv
    goto check_venv
)
if exist "%ProgramFiles%\Python312\python.exe" (
    "%ProgramFiles%\Python312\python.exe" -m venv .venv
    goto check_venv
)
goto no_python

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
if not defined NO_PAUSE pause
exit /b 0

:no_python
echo.
echo Python 3.10, 3.11, 3.12, or 3.13 (64-bit) was not found.
if defined PYTHON_INSTALL_ATTEMPTED goto python_install_failed

set "INSTALL_PYTHON="
set /p "INSTALL_PYTHON=Install Python 3.12 now? [Y/N]: "
if /i not "%INSTALL_PYTHON%"=="Y" goto python_manual_install
set "PYTHON_INSTALL_ATTEMPTED=1"

where py >nul 2>nul
if errorlevel 1 goto install_python_with_winget
echo.
echo Installing Python 3.12...
py install 3.12
if not errorlevel 1 goto detect_python

:install_python_with_winget
where winget >nul 2>nul
if errorlevel 1 goto python_install_failed
echo.
echo Installing Python 3.12 with winget...
winget install --id Python.Python.3.12 --exact --source winget --scope user --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto python_install_failed
goto detect_python

:python_install_failed
echo.
echo Python could not be installed automatically.

:python_manual_install
echo Install Python from https://www.python.org/downloads/windows/
echo Then run install.bat again.
if not defined NO_PAUSE pause
exit /b 1

:venv_failed
echo.
echo Could not create the .venv environment.
if not defined NO_PAUSE pause
exit /b 1

:install_failed
echo.
echo Installation failed. Review the error above, then run install.bat again.
if not defined NO_PAUSE pause
exit /b 1
