@echo off
REM KRISH - run 24/7 on a Windows VPS, surviving reboots and logouts.
REM
REM Uses the built-in Task Scheduler, so nothing extra to download.
REM MUST be run as Administrator (right click -> Run as administrator).
REM
REM   scripts\windows\install-service.bat

cd /d "%~dp0..\.."
set "ROOT=%CD%"
set "TASKNAME=KRISH"

net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] This script needs Administrator rights.
    echo Right click it and choose "Run as administrator".
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Not installed yet. Run scripts\windows\install.bat first.
    exit /b 1
)

echo Registering scheduled task "%TASKNAME%" ...
schtasks /query /tn "%TASKNAME%" >nul 2>&1 && schtasks /delete /tn "%TASKNAME%" /f >nul

schtasks /create /tn "%TASKNAME%" /sc onstart /ru SYSTEM /rl HIGHEST /f ^
  /tr "cmd /c cd /d \"%ROOT%\" && \"%ROOT%\.venv\Scripts\python.exe\" -m krish.main run >> \"%ROOT%\var\logs\service.log\" 2>&1"
if errorlevel 1 (
    echo [ERROR] Could not create the task.
    exit /b 1
)

REM Restart automatically if the process dies.
powershell -NoProfile -Command ^
  "$s = New-Object -ComObject Schedule.Service; $s.Connect(); $f = $s.GetFolder('\'); $t = $f.GetTask('%TASKNAME%'); $d = $t.Definition; $d.Settings.RestartCount = 999; $d.Settings.RestartInterval = 'PT1M'; $d.Settings.ExecutionTimeLimit = 'PT0S'; $d.Settings.DisallowStartIfOnBatteries = $false; $d.Settings.StopIfGoingOnBatteries = $false; $f.RegisterTaskDefinition('%TASKNAME%', $d, 4, $null, $null, 5) | Out-Null" >nul 2>&1

if not exist "var\logs" mkdir "var\logs"

echo Starting it now ...
schtasks /run /tn "%TASKNAME%" >nul

echo.
echo ============================================================
echo  KRISH is now a Windows scheduled task.
echo.
echo    starts on boot, restarts itself if it crashes
echo    log:  %ROOT%\var\logs\service.log
echo.
echo  Control room:  http://localhost:8000
echo.
echo  Useful commands:
echo    schtasks /query  /tn KRISH       status
echo    schtasks /end    /tn KRISH       stop
echo    schtasks /run    /tn KRISH       start
echo    schtasks /delete /tn KRISH /f    remove
echo ============================================================
