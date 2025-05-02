#!/usr/bin/env bash
# setup.sh — Bash version of setup.bat
# Stop on first error
set -e

# ── ANSI colour codes ──────────────────────────────────────────────────────────
YELLOW="\033[33m"
CYAN="\033[36m"
GREEN="\033[32m"
RESET="\033[0m"

separator() { echo -e "${YELLOW}==================================================${RESET}"; }

# ── 1. Create virtual environment ──────────────────────────────────────────────
separator
echo -e "${CYAN}Creating virtual environment (.SBT)…${RESET}"
python -m venv .SBT

# ── 2. Activate virtual environment ────────────────────────────────────────────
separator
echo -e "${CYAN}Activating virtual environment…${RESET}"
# shellcheck source=/dev/null
source .SBT/bin/activate   # (Unix path; note the forward slashes)

# ── 3. Upgrade pip ─────────────────────────────────────────────────────────────
separator
echo -e "${CYAN}Upgrading pip…${RESET}"
python -m pip install --upgrade pip

# ── 4. Install required packages ───────────────────────────────────────────────
separator
echo -e "${CYAN}Installing required packages…${RESET}"
pip install -r requirements.txt

# ── 5. Install Jupyter + IPython kernel ────────────────────────────────────────
separator
echo -e "${CYAN}Installing Jupyter and IPykernel…${RESET}"
pip install jupyter ipykernel

separator
echo -e "${CYAN}Registering the virtual environment as a Jupyter kernel…${RESET}"
python -m ipykernel install --user --name ".SBT" --display-name "Python (SBT 2025)"

# ── 6. Change to code directory ────────────────────────────────────────────────
separator
echo -e "${CYAN}Changing directory to code folder…${RESET}"
cd code || { echo "❌  'code' directory not found."; exit 1; }

# ── 7. Run notebooks, streaming cell output ────────────────────────────────────
for nb in \
    "00_cohort_id.ipynb" \
    "01_SAT_standard.ipynb" \
    "02_SBT_Standard.ipynb" \
    "02_SBT_Both_stabilities.ipynb" \
    "02_SBT_Hemodynamic_Stability.ipynb" \
    "02_SBT_Respiratory_Stability.ipynb"
do
    separator
    echo -e "${CYAN}Executing ${nb} and streaming its cell output…${RESET}"
    jupyter nbconvert --to script --stdout "${nb}" | python
done

# ── 8. Finish ──────────────────────────────────────────────────────────────────
separator
echo -e "${GREEN}All tasks completed.${RESET}"

# Keep the window open if run interactively (optional)
read -rp "Press [Enter] to exit…"
