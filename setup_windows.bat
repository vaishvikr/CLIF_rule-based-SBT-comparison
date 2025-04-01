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
echo %CYAN%Running Python script: code/01_cohort_id_script.py...%RESET%
python 01_cohort_id_script.py

echo %YELLOW%==================================================%RESET%
echo %CYAN%Running Python script: code/02_project_analysis_SBT_script.py...%RESET%
python 02_project_analysis_SBT_optimized_script.py

echo %YELLOW%==================================================%RESET%
echo %GREEN%All tasks completed.%RESET%
pause
