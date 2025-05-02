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
jupyter nbconvert --to script --stdout "00_cohort_id.ipynb" | python


echo %YELLOW%==================================================%RESET%
echo %CYAN%Execute 01_SAT_standard.ipynb and stream its cell output only%RESET%
jupyter nbconvert --to script --stdout "01_SAT_standard.ipynb" | python

echo %YELLOW%==================================================%RESET%
echo %CYAN%Execute 02_SBT_Standard.ipynb and stream its cell output only%RESET%
jupyter nbconvert --to script --stdout "02_SBT_Standard.ipynb" | python

echo %YELLOW%==================================================%RESET%
echo %CYAN%Execute 02_SBT_Both_stabilities.ipynb and stream its cell output only%RESET%
jupyter nbconvert --to script --stdout "02_SBT_Both_stabilities.ipynb" | python

echo %YELLOW%==================================================%RESET%
echo %CYAN%Execute 02_SBT_Hemodynamic_Stability.ipynb and stream its cell output only%RESET%
jupyter nbconvert --to script --stdout "02_SBT_Hemodynamic_Stability.ipynb" | python

echo %YELLOW%==================================================%RESET%
echo %CYAN%Execute 02_SBT_Respiratory_Stability.ipynb and stream its cell output only%RESET%
jupyter nbconvert --to script --stdout "02_SBT_Respiratory_Stability.ipynb" | python


echo %YELLOW%==================================================%RESET%
echo %GREEN%All tasks completed.%RESET%
pause
