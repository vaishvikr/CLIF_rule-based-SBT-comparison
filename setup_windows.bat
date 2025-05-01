@echo off
REM setup.bat with colored output using ANSI escape codes in CMD

REM Create an ESC variable (the ANSI escape character)
for /F "delims=" %%a in ('echo prompt $E^| cmd') do set "ESC=%%a"

REM Define color variables
set "YELLOW=%ESC%[33m"
set "CYAN=%ESC%[36m"
set "GREEN=%ESC%[32m"
set "RESET=%ESC%[0m"

echo %YELLOW%==================================================%RESET%
echo %CYAN%Creating virtual environment (.SBT)...%RESET%
python -m venv .SBT

echo %YELLOW%==================================================%RESET%
echo %CYAN%Activating virtual environment...%RESET%
call .\.SBT\Scripts\activate.bat

echo %YELLOW%==================================================%RESET%
echo %CYAN%Upgrading pip...%RESET%
python -m pip install --upgrade pip

echo %YELLOW%==================================================%RESET%
echo %CYAN%Installing required packages...%RESET%
pip install -r requirements.txt

echo %YELLOW%==================================================%RESET%
echo %CYAN%Installing Jupyter and IPykernel...%RESET%
pip install jupyter ipykernel

echo %YELLOW%==================================================%RESET%
echo %CYAN%Registering the virtual environment as a Jupyter kernel...%RESET%
python -m ipykernel install --user --name=.SBT --display-name="Python (SBT 2025)"

echo %YELLOW%==================================================%RESET%
echo %CYAN%Changing directory to code folder...%RESET%
cd code

echo %YELLOW%==================================================%RESET%
echo %CYAN% Execute 00_cohort_id.ipynb and stream its cell output only%RESET%
jupyter nbconvert --to script --stdout "01_cohort_id_script.ipynb" | python

setlocal enabledelayedexpansion

REM ── 1) Record start time in seconds since midnight ──────────────
for /f "tokens=1-3 delims=:.," %%h in ("%time%") do (
  set /A startSec=%%h*3600 + %%i*60 + %%j
)

REM ── 2) Launch each notebook in parallel, signaling when done ────
start "" /B cmd /C "echo ===== Execute 02_SBT_Standard ===== & jupyter nbconvert --to script 02_SBT_Standard.ipynb --stdout ^| python & WAITFOR /SI JOB1_DONE"
start "" /B cmd /C "echo ===== Execute 02_SBT_Both_stabilities ===== & jupyter nbconvert --to script 02_SBT_Both_stabilities.ipynb --stdout ^| python & WAITFOR /SI JOB2_DONE"
start "" /B cmd /C "echo ===== Execute 02_SBT_Hemodynamic_Stability ===== & jupyter nbconvert --to script 02_SBT_Hemodynamic_Stability.ipynb --stdout ^| python & WAITFOR /SI JOB3_DONE"
start "" /B cmd /C "echo ===== Execute 02_SBT_Respiratory_Stability ===== & jupyter nbconvert --to script 02_SBT_Respiratory_Stability.ipynb --stdout ^| python & WAITFOR /SI JOB4_DONE"

REM ── 3) Wait for all four signals ─────────────────────────────────
for %%S in (JOB1_DONE JOB2_DONE JOB3_DONE JOB4_DONE) do (
  waitfor %%S >nul
)

REM ── 4) Record end time and compute elapsed ───────────────────────
for /f "tokens=1-3 delims=:.," %%h in ("%time%") do (
  set /A endSec=%%h*3600 + %%i*60 + %%j
)
set /A elapsed=endSec - startSec
echo(
echo All notebooks done in %elapsed% seconds!
pause


echo %YELLOW%==================================================%RESET%
echo %CYAN%Execute 01_SAT_standard.ipynb and stream its cell output only%RESET%
jupyter nbconvert --to script --stdout "code\01_SAT_standard.ipynb" | python


echo %YELLOW%==================================================%RESET%
echo %GREEN%All tasks completed.%RESET%
pause
