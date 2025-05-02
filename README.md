# CLIF Rule based SBT comparison

## Objective

The aim of this project is to evaluate compliance in the delivery of SBT within healthcare settings. We have developed an algorithm that detects SBT events in Electronic Health Records (EHR) by identifying specific signatures. This algorithm allows for comparison of SAT occurrences with documented flowsheets to assess adherence to the SBT protocol.

## Required CLIF tables and fields

Please refer to the online [CLIF data dictionary](https://clif-consortium.github.io/website/data-dictionary.html), [ETL tools](https://github.com/clif-consortium/CLIF/tree/main/etl-to-clif-resources), and [specific table contacts](https://github.com/clif-consortium/CLIF?tab=readme-ov-file#relational-clif) for more information on constructing the required tables and fields. List all required tables for the project here, and provide a brief rationale for why they are required.


## Required CLIF Tables

The following tables and fields are required:

1. **`patient`**
   - Fields: `patient_id`, `race_category`, `ethnicity_category`, `sex_category`

2. **`hospitalization`**
   - Fields: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission`

3. **`medication_admin_continuous`**
   - Fields: `hospitalization_id`, `admin_dttm`, `med_category`, `med_dose`
   - Relevant `med_category` values:
     ```
     norepinephrine, epinephrine, phenylephrine, angiotensin, vasopressin, 
     dopamine, dobutamine, milrinone, isoproterenol, cisatracurium, vecuronium, 
     rocuronium, fentanyl, propofol, lorazepam, midazolam, hydromorphone, morphine
     ```

4. **`respiratory_support`**
   - Fields: `hospitalization_id`, `recorded_dttm`, `device_category`

5. **`patient_assessments`**
   - Fields: `hospitalization_id`, `recorded_dttm`, `assessment_category`, `numerical_value`, `categorical_value`
   - Relevant `assessment_category` values:
     ```
     sbt_delivery_pass_fail, sbt_screen_pass_fail,
     sat_delivery_pass_fail, sat_screen_pass_fail
     ```

6. **`vitals`**
   - Fields: `hospitalization_id`, `recorded_dttm`, `vitals_category`, `vitals_value`

7. **`crrt_therapy`**
   - Fields: `hospitalization_id`, `recorded_dttm`

---

## Cohort Identification

- **Study Period:** January 1, 2022 – December 31, 2024
- **Inclusion Criteria:**
  - At least one ICU admission with IMV during the study period
  - Age ≥ 18 years at the time of initial hospital admission

---

## Expected Output

Results will be written to the `output/final` directory and include:

- `table1` summary file
- Statistical metrics file

See [`output/README.md`](../output/README.md) for details.

---

## Running the Project

### 1. Configure the Project

Update the configuration file at `config/config.json`.

Refer to [`config/README.md`](config/README.md) for step-by-step instructions.

---

### 2. Set Up and Run the Environment

#### On Mac/Linux:

```bash
# Open a terminal and run:
bash setup_mac_or_linux.sh

# Open a command prompt and run:
setup_windows.bat
```

### 3. Troubleshooting

You can run notebooks one by one:
Refer to [`code/README.md`](code/README.md) for step-by-step instructions.

If you encounter an error:

- Open and run the following notebooks **in sequence, cell by cell**:
  1. `00_*.ipynb`
  2. `01_*.ipynb`
  3. `02_*.ipynb`

This will help identify exactly where the code is failing and allow for targeted debugging.
---
