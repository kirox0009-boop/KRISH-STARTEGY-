@echo off
REM KRISH - start the factory and the control room.
REM   scripts\windows\start.bat            full factory + control room
REM   scripts\windows\start.bat cycle      one research cycle, then exit
REM   scripts\windows\start.bat roster     list the agents

cd /d "%~dp0..\.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Not installed yet. Run scripts\windows\install.bat first.
    exit /b 1
)

REM Single Windows box: in-memory bus + SQLite. No Redis or Postgres required.
if not defined KRISH_BUS set "KRISH_BUS=memory"

if /i "%~1"=="cycle"  goto :cycle
if /i "%~1"=="roster" goto :roster

echo ============================================================
echo  KRISH is starting.  Control room: http://localhost:8000
echo  Press Ctrl+C to stop.
echo ============================================================
".venv\Scripts\python.exe" -m krish.main run
goto :eof

:cycle
set "ASSET=%~2"
set "TF=%~3"
if "%ASSET%"=="" set "ASSET=GOLD"
if "%TF%"=="" set "TF=H1"
echo Running one cycle on %ASSET% %TF% ...
".venv\Scripts\python.exe" -m krish.main cycle --asset %ASSET% --timeframe %TF% --count 3
goto :eof

:roster
".venv\Scripts\python.exe" -m krish.main roster
