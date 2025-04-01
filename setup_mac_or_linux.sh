#!/bin/bash

# setup.sh with colored output using ANSI escape codes

# Define color variables
YELLOW="\033[33m"
CYAN="\033[36m"
GREEN="\033[32m"
RESET="\033[0m"

echo -e "${YELLOW}==================================================${RESET}"
echo -e "${CYAN}Creating virtual environment (.SBT)...${RESET}"
python3 -m venv .SBT

echo -e "${YELLOW}==================================================${RESET}"
echo -e "${CYAN}Activating virtual environment...${RESET}"
source .SBT/bin/activate

echo -e "${YELLOW}==================================================${RESET}"
echo -e "${CYAN}Upgrading pip...${RESET}"
python -m pip install --upgrade pip

echo -e "${YELLOW}==================================================${RESET}"
echo -e "${CYAN}Installing required packages...${RESET}"
pip install -r requirements.txt

echo -e "${YELLOW}==================================================${RESET}"
echo -e "${CYAN}Installing Jupyter and IPykernel...${RESET}"
pip install jupyter ipykernel

echo -e "${YELLOW}==================================================${RESET}"
echo -e "${CYAN}Registering the virtual environment as a Jupyter kernel...${RESET}"
python -m ipykernel install --user --name=.SBT --display-name="Python (SBT 2025)"

echo -e "${YELLOW}==================================================${RESET}"
echo -e "${CYAN}Changing directory to code folder...${RESET}"
cd code || exit

echo -e "${YELLOW}==================================================${RESET}"
echo -e "${CYAN}Running Python script: code/01_cohort_id_script.py...${RESET}"
python 01_cohort_id_script.py

echo -e "${YELLOW}==================================================${RESET}"
echo -e "${CYAN}Running Python script: code/02_project_analysis_SBT_script.py...${RESET}"
python 02_project_analysis_SBT_optimized_script.py

echo -e "${YELLOW}==================================================${RESET}"
echo -e "${GREEN}All tasks completed.${RESET}"

# Optional pause-like behavior
read -p "Press [Enter] to continue..."