@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo Krea 2 Trainer - updater
echo ========================

where git >nul 2>nul
if errorlevel 1 goto no_git

if not exist ".git" goto no_repository

git diff --quiet
if errorlevel 1 goto local_changes
git diff --cached --quiet
if errorlevel 1 goto local_changes

echo.
echo Checking for updates...
git pull --ff-only origin main
if errorlevel 1 goto update_failed

echo.
echo Updating dependencies...
call install.bat --no-pause
if errorlevel 1 goto dependency_failed

echo.
echo Update completed. Run start_ui.bat to open the trainer.
pause
exit /b 0

:no_git
echo.
echo Git was not found. Install Git for Windows, then run update.bat again.
echo https://git-scm.com/download/win
pause
exit /b 1

:no_repository
echo.
echo This folder was downloaded as a ZIP and cannot receive incremental updates.
echo Clone the repository with Git once, then use update.bat from that folder:
echo git clone https://github.com/litch230/krea2-trainer.git
pause
exit /b 1

:local_changes
echo.
echo The trainer files contain local changes, so the update was stopped.
echo Keep or discard those changes with Git before trying again.
pause
exit /b 1

:update_failed
echo.
echo The update failed. Review the Git error above and try again.
pause
exit /b 1

:dependency_failed
echo.
echo The code was updated, but dependency installation failed.
echo Review the error above, then run install.bat again.
pause
exit /b 1
