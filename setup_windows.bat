@echo on
REM setup.bat

echo ===================================================
echo Creating virtual environment (.SBT)...
python -m venv .SBT

echo ===================================================
echo Activating virtual environment...
call .\.SBT\Scripts\activate.bat

echo ===================================================
echo Upgrading pip...
python -m pip install --upgrade pip

echo ===================================================
echo Installing required packages...
pip install -r requirements.txt

echo ===================================================
echo Installing Jupyter and IPykernel...
pip install jupyter ipykernel

echo ===================================================
echo Registering the virtual environment as a Jupyter kernel...
python -m ipykernel install --user --name=.SBT --display-name="Python (SBT 2025)"

echo ===================================================
echo Running Python script: code/01_cohort_id_script.py...
python code/01_cohort_id_script.py

echo ===================================================
echo Running Python script: code/02_project_analysis_SBT_script.py...
python code/02_project_analysis_SBT_script.py

echo ===================================================
echo All tasks completed.
pause
