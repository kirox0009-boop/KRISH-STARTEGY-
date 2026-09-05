@echo off
REM KRISH - one-time setup on a Windows VPS.
REM Run this from the repo root:  scripts\windows\install.bat

setlocal enabledelayedexpansion
cd /d "%~dp0..\.."

echo ============================================================
echo  KRISH setup
echo ============================================================
echo.

REM ---- find a usable Python 3.11+ ---------------------------------------
set "PYEXE="
for %%V in (3.13 3.12 3.11) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>&1 && set "PYEXE=py -%%V"
    )
)
if not defined PYEXE (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1 ^
      && set "PYEXE=python"
)
if not defined PYEXE (
    echo [ERROR] Python 3.11 or newer was not found.
    echo.
    echo Install it from https://www.python.org/downloads/windows/
    echo IMPORTANT: tick "Add python.exe to PATH" in the installer.
    echo Then close this window, open a NEW terminal, and run this script again.
    exit /b 1
)
echo Using Python: %PYEXE%
%PYEXE% --version
echo.

REM ---- virtual environment ----------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PYEXE% -m venv .venv || (echo [ERROR] venv creation failed & exit /b 1)
) else (
    echo Virtual environment already exists, reusing it.
)

echo Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet

echo Installing KRISH and its dependencies (this takes 2-5 minutes) ...
".venv\Scripts\python.exe" -m pip install -e "backend" || (
    echo [ERROR] dependency installation failed. Scroll up for the reason.
    exit /b 1
)

REM ---- config -------------------------------------------------------------
if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
        echo Created .env from .env.example
    )
)

REM Windows single-box defaults: no Redis, no Postgres needed to start.
echo Verifying the install ...
".venv\Scripts\python.exe" -m krish.main roster || (
    echo [ERROR] KRISH did not start correctly.
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete.
echo.
echo  Next:  scripts\windows\start.bat
echo  Then open:  http://localhost:8000
echo ============================================================
endlocal
