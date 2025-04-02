# %%
import pandas as pd
import numpy as np
import pyCLIF as pc
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")


cohort = pd.read_csv('../output/intermediate/study_cohort.csv')

# %%
if pc.helper['site_name']=='RUSH':
    cohort.loc[cohort['sbt_timepoint'] == '3-5 minute evaluation', 'pressure_support_set'] = 6.1
    cohort.loc[cohort['sbt_timepoint'] == '3-5 minute evaluation', 'mode_category'] = 'Pressure Support/CPAP'
    print('its a rush thing')


# %% [markdown]
# # Eligibility Flag making

# %% [markdown]
# #### Device Fillforward After Waterfall

# %%
# vasoactive-> if the meds are missing from site then fill NaN
active_vasoactive_n_col = [
    "norepinephrine", "epinephrine", "phenylephrine", "angiotensin","vasopressin", "dopamine", "dobutamine", "milrinone", "isoproterenol"
]
for col in active_vasoactive_n_col:
    if col not in cohort.columns:
        cohort[col] = np.nan

# Ensure all time columns are in datetime format
cohort['event_time'] = pd.to_datetime(cohort['event_time'])
cohort['admission_dttm'] = pc.getdttm(cohort['admission_dttm'])
cohort['discharge_dttm'] = pc.getdttm(cohort['discharge_dttm'])

# Ensure the data is sorted by 'hosp_id_day_key' and 'event_time'
cohort = cohort.sort_values(by=['hospitalization_id', 'event_time']).reset_index(drop=True)


cohort['device_category'] = cohort['device_category'].str.lower()
cohort['mode_category'] = cohort['mode_category'].str.lower()

# Fill forward the meds by hospitalization columns by 'hosp_id'
cohort[['device_category', 'mode_category', 'mode_name',
        'location_category','hospital_id']] = cohort.groupby('hospitalization_id')[
    ['device_category', 'mode_category','mode_name',
     'location_category','hospital_id']
].ffill()

cohort[["norepinephrine", "epinephrine", "phenylephrine", "angiotensin",
    "vasopressin", "dopamine", "dobutamine", "milrinone", "isoproterenol"]] = cohort.groupby('hospitalization_id')[
    ["norepinephrine", "epinephrine", "phenylephrine", "angiotensin",
    "vasopressin", "dopamine", "dobutamine", "milrinone", "isoproterenol"]
].ffill()

cohort[["fio2_set","peep_set","spo2",'pressure_support_set']] = cohort.groupby('hospitalization_id')[
    ["fio2_set","peep_set","spo2",'pressure_support_set']
].ffill()

cohort[['norepinephrine', 'epinephrine', 'phenylephrine', 'dopamine', 'angiotensin', 'vasopressin']] = \
    cohort[['norepinephrine', 'epinephrine', 'phenylephrine', 'dopamine', 'angiotensin', 'vasopressin']].fillna(0)

cohort['NEE'] = cohort['norepinephrine'] + cohort['epinephrine'] + (cohort['phenylephrine']/10) + (cohort['vasopressin']*2.5) + (cohort['dopamine']/100) + (cohort['angiotensin']*10)

cohort["Hemodynamic_Stability_by_NEE"] = (
    ((cohort["NEE"] <= 0.2))
).astype(int)

# Define Respiratory Stability Flag
cohort["Respiratory_Stability"] = (
    (cohort["fio2_set"] <= 0.5) &
    (cohort["peep_set"] <= 8) &
    (cohort["spo2"] >= 88)
).astype(int)


# %% [markdown]
# ## SBT Eligibility Criteria

# %%
def process_cohort_conditions(cohort):
    # --- Preliminary processing ---
    # Ensure event_time is datetime and sort the dataframe
    cohort['event_time'] = pd.to_datetime(cohort['event_time'])
    cohort = cohort.sort_values(['hospitalization_id', 'event_time']).reset_index(drop=False)
    
    # IMV flag
    cohort['IMV_flag'] = (
        (cohort['device_category'] == 'imv') &
        (cohort['location_category'] == 'icu')
    )
    
    # --- Prepare new flag columns ---
    # For Condition 1, record the event_time when the threshold is reached.
    cohort['IMV_Controlled_met_time'] = pd.NaT
    # New flag for eligible day (1 if condition 1 is met that day, else 0)
    cohort['eligible_day'] = 0
    
    # For grouping by day, use the normalized event_time (midnight)
    cohort['current_day'] = cohort['event_time'].dt.normalize()
    
    # Build a dictionary of full hospitalization data to avoid repeated filtering.
    hosp_groups = {
        hosp_id: df.copy().sort_values('event_time')
        for hosp_id, df in cohort.groupby('hospitalization_id')
    }
    
    # --- Define thresholds and time windows ---
    cond1_threshold = pd.Timedelta(hours=6)  # Condition 1: 6 cumulative hours
   
    # For Condition 1: window is 10 PM (previous day) to 6 AM (current day)
    cond1_window_start_offset = pd.Timedelta(hours=22) - pd.Timedelta(days=1)  # previous day 10 PM
    cond1_window_end_offset = pd.Timedelta(hours=6)  # current day 6 AM
    
    # --- Process each hospitalization and day ---
    # Group by hospitalization and current day
    groups = cohort.groupby(['hospitalization_id', 'current_day'])
    
    for (hosp_id, curr_day), day_group in tqdm(groups, desc="Processing each Hosp & Day"):
        # --- Condition 1: IMV in controlled mode ---
        # Define window for condition 1 based on the current day
        cond1_start = curr_day + cond1_window_start_offset
        cond1_end = curr_day + cond1_window_end_offset
        
        # Use full hospitalization data so events before midnight can contribute.
        hosp_df = hosp_groups[hosp_id]
        cond1_df = hosp_df[(hosp_df['event_time'] >= cond1_start) & (hosp_df['event_time'] <= cond1_end)].copy()
        if cond1_df.empty:
            continue  # no events in this window
        
        if not cond1_df['IMV_flag'].any():
            continue
        
        # Identify contiguous segments where IMV_flag is True.
        cond1_df['seg'] = (cond1_df['IMV_flag'] != cond1_df['IMV_flag'].shift()).cumsum()
        valid_segs = cond1_df[cond1_df['IMV_flag']].groupby('seg')
        
        cond1_met = False  # flag indicating if condition 1 was met
        for seg_id, seg_df in valid_segs:
            seg_df = seg_df.sort_values('event_time')
            seg_df['duration'] = seg_df['event_time'].diff().fillna(pd.Timedelta(seconds=0))
            seg_df['cum_duration'] = seg_df['duration'].cumsum()
            if seg_df['cum_duration'].iloc[-1] >= cond1_threshold:
                # Find the first row where the cumulative duration reaches the threshold.
                flag_row = seg_df[seg_df['cum_duration'] >= cond1_threshold].iloc[0]
                flag_idx = flag_row.name  # this is the original index in hosp_df (and cohort)
                flag_time = flag_row['event_time']
                cohort.loc[flag_idx, 'IMV_Controlled_met_time'] = flag_time
                cond1_met = True
                break  # Only the first qualifying segment for this day is flagged.
        
        # --- Eligible Day Flag ---
        # If condition 1 is met for the day, mark all rows of this day as eligible_day = 1.
        if cond1_met:
            cohort.loc[day_group.index, 'eligible_day'] = 1
    
    return cohort

# Example usage:
final_df = process_cohort_conditions(cohort)




# %%
# Print statistics
print('By n = Days')
total_days = final_df['hosp_id_day_key'].nunique()
print('Total number of days for eval in cohort:', total_days)
eligible_days = final_df[final_df['eligible_day'] == 1]['hosp_id_day_key'].nunique()
percentage = (eligible_days / total_days) * 100 if total_days > 0 else 0
print(f"Eligible days: {eligible_days} / {total_days} ({percentage:.2f}%)")
print('Hospital days with atleast one IMV event: ',final_df[final_df['device_category'] == 'imv' ]['hosp_id_day_key'].nunique())
print('Hospital days with atleast one IMV & ICU event: ',final_df[(final_df['device_category'] == 'imv') &
        (final_df['location_category'] == 'icu')]['hosp_id_day_key'].nunique())

print('By n = Encounter')
h_total_days = final_df['hospitalization_id'].nunique()
print('Total number of days for eval in cohort:', h_total_days)
h_eligible_days = final_df[final_df['eligible_day'] == 1]['hospitalization_id'].nunique()
h_percentage = (h_eligible_days / h_total_days) * 100 if h_total_days > 0 else 0
print(f"Eligible days: {h_eligible_days} / {h_total_days} ({h_percentage:.2f}%)")
print('Hospital days with atleast one IMV event: ',final_df[final_df['device_category'] == 'imv' ]['hospitalization_id'].nunique())
print('Hospital days with atleast one IMV & ICU event: ',final_df[(final_df['device_category'] == 'imv') &
        (final_df['location_category'] == 'icu')]['hospitalization_id'].nunique())

# %% [markdown]
# ## FLIP Check

# %%
def process_diagnostic_flip_sbt_optimized_v2(cohort):
    # Ensure event_time is datetime.
    cohort['event_time'] = pd.to_datetime(cohort['event_time'])
    
    # Preinitialize diagnostic and flip evaluation columns.
    diag_cols = ['cond_device_imv', 'cond_location_icu', 'cond_mode_ps_cpap',
                 'cond_ps_set_le8', 'cond_peep_set_le8', 'cond_mode_tpiece',
                 'flip_skip_reason', 'first_flip_time']
    for col in diag_cols:
        cohort[col] = None
        
    # Initialize EHR delivery columns.
    for mins in [2, 30]:
        cohort[f"EHR_Delivery_{mins}mins"] = pd.NaT

    # --- Precompute diagnostic flags (vectorized) ---
    mask_eligible = cohort['eligible_day'] == 1
    
    # Normalize and compare strings.
    cond_imv = cohort['device_category'].fillna('').str.strip().str.lower() == 'imv'
    cond_icu = cohort['location_category'].fillna('').str.strip().str.lower() == 'icu'
    
    mode_cat_lower = cohort['mode_category'].fillna('').str.lower()
    cond_mode_ps = mode_cat_lower.str.contains('pressure support|cpap', regex=True)
    cond_ps_le8 = cohort['pressure_support_set'] <= 8
    cond_peep_le8 = cohort['peep_set'] <= 8
    conditionA = cond_mode_ps & cond_ps_le8 & cond_peep_le8
    mode_name_lower = cohort['mode_name'].fillna('').str.strip().str.lower()
    cond_mode_tpiece = mode_name_lower.str.match(r'^t[-]?piece$', na=False)
    composite = conditionA | cond_mode_tpiece
    passed = cond_imv & cond_icu & composite

    # Set diagnostic columns for eligible rows.
    cohort.loc[mask_eligible & (~cond_imv), 'cond_device_imv'] = \
        cohort.loc[mask_eligible & (~cond_imv), 'device_category']
    cohort.loc[mask_eligible & cond_imv & (~cond_icu), 'cond_location_icu'] = \
        cohort.loc[mask_eligible & cond_imv & (~cond_icu), 'location_category']
    
    mask_composite_fail = mask_eligible & cond_imv & cond_icu & (~composite)
    cohort.loc[mask_composite_fail & (~cond_mode_ps), 'cond_mode_ps_cpap'] = \
        cohort.loc[mask_composite_fail & (~cond_mode_ps), 'mode_category']
    mask_ps_fail = cohort['pressure_support_set'].isnull() | (cohort['pressure_support_set'] > 8)
    cohort.loc[mask_composite_fail & mask_ps_fail, 'cond_ps_set_le8'] = \
        cohort.loc[mask_composite_fail & mask_ps_fail, 'pressure_support_set']
    mask_peep_fail = cohort['peep_set'].isnull() | (cohort['peep_set'] > 8)
    cohort.loc[mask_composite_fail & mask_peep_fail, 'cond_peep_set_le8'] = \
        cohort.loc[mask_composite_fail & mask_peep_fail, 'peep_set']
    cohort.loc[mask_composite_fail & (~cond_mode_tpiece), 'cond_mode_tpiece'] = \
        cohort.loc[mask_composite_fail & (~cond_mode_tpiece), 'mode_name']
    
    # Mark candidate rows.
    cohort['flip_check_flag'] = False
    cohort.loc[mask_eligible, 'flip_check_flag'] = passed[mask_eligible]
    
    # Compute the minimum IMV_Controlled_met_time per eligible group.
    cohort.loc[mask_eligible, 'min_met_time'] = (
        cohort.loc[mask_eligible]
        .groupby(['hospitalization_id', 'current_day'])['IMV_Controlled_met_time']
        .transform('min')
    )
    
    # --- Process each eligible group using vectorized operations ---
    def process_group(group):
        # Work on a copy sorted by event_time.
        group = group.sort_values('event_time').copy()
        n = len(group)
        if n == 0:
            return group
        
        # Convert event_time to numpy array.
        times = group['event_time'].values.astype('datetime64[ns]')
        delta = np.timedelta64(2, 'm')
        # Use searchsorted to find the boundary index for each row's 2-minute window.
        boundaries = np.searchsorted(times, times + delta, side='right')
        cnt_total = boundaries - np.arange(n)
        group['cnt_total'] = cnt_total
        
        # Compute cumulative sum for flip_check_flag (converted to int).
        flip_int = group['flip_check_flag'].astype(int).values
        cumsum = np.cumsum(flip_int)
        cnt_pass = np.empty(n, dtype=int)
        for i in range(n):
            start = i
            end = boundaries[i] - 1
            cnt_pass[i] = cumsum[end] - (cumsum[start-1] if start > 0 else 0)
        group['cnt_pass'] = cnt_pass
        
        # Compute sustained condition for each row.
        group['sustained'] = (group['event_time'] > group['min_met_time']) & \
                             (group['cnt_total'] == group['cnt_pass']) & \
                             group['flip_check_flag']
        
        # Now iterate only over candidate rows (flip_check_flag True) in order.
        candidate_indices = group.index[group['flip_check_flag']].tolist()
        for idx in candidate_indices:
            # Set first_flip_time on the candidate row.
            group.at[idx, 'first_flip_time'] = group.at[idx, 'event_time']
            # If the candidate occurs before the min_met_time, mark the failure reason.
            if group.at[idx, 'event_time'] <= group.at[idx, 'min_met_time']:
                group.at[idx, 'flip_skip_reason'] = "Flip before IMV_Controlled_met_time"
                # Continue with next candidate in the same group.
                continue
            else:
                # Candidate event_time is after the min_met_time.
                if group.at[idx, 'sustained']:
                    # If sustained, mark success (EHR_Delivery_2mins = 1) on this candidate row.
                    group.at[idx, 'EHR_Delivery_2mins'] = 1
                    group.at[idx, 'flip_skip_reason'] = None
                    # Once a successful candidate is found, stop evaluating further candidates.
                    break
                else:
                    # Not sustained: mark failure reason on this candidate row.
                    group.at[idx, 'flip_skip_reason'] = "ehr_delivery_2min not possible"
                    # Continue to the next candidate.
                    continue
        return group

    # Apply the per-group processing only on eligible rows.
    eligible_df = cohort[mask_eligible].copy()
    processed = eligible_df.groupby(['hospitalization_id', 'current_day'], group_keys=False).apply(process_group)
    
    # Update only the eligible rows in the original DataFrame.
    cohort.update(processed)
    
    # Optionally, remove helper columns.
    helper_cols = ['cnt_total', 'cnt_pass', 'sustained', 'min_met_time']
    cohort.drop(columns=[col for col in helper_cols if col in cohort.columns], inplace=True)
    
    return cohort

# Example usage:
final_df = process_diagnostic_flip_sbt_optimized_v2(final_df)


# %%
final_df

# %%
final_df['sbt_bkp'] = final_df['sbt_delivery_pass_fail']
final_df['sbt_delivery_pass_fail'] = final_df['sbt_delivery_pass_fail'].map({0:1,1:1})
final_df['sbt_screen_pass_fail'] = final_df['sbt_screen_pass_fail'].map({0:1,1:1})

#fill forward reason of flip fail
final_df['flip_skip_reason'] = (
    final_df.groupby('hosp_id_day_key')['flip_skip_reason']
    .transform(lambda x: x.ffill().bfill())
)

# %% [markdown]
# ## Results Section

# %%
# Ensure the specified columns are treated as datetime before calculating percentages
datetime_columns = [
    'EHR_Delivery_2mins'
]

# for col in datetime_columns:
#     if col in final_df.columns:
#         final_df[col] = final_df[col].notna().astype(int)

# Group by hosp_id_day_key and aggregate using max
# grouped_df = final_df.groupby('hosp_id_day_key').agg({
#     'hospital_id' : lambda x: x.dropna().iloc[-1] if x.dropna().size > 0 else np.nan,
#     'eligible_day':'max',
#     'EHR_Delivery_2mins': 'max',
#     'sat_screen_pass_fail': 'max',
#     'sat_delivery_pass_fail': 'max',
#     'sbt_screen_pass_fail': 'max',
#     'sbt_delivery_pass_fail': 'max',
#     # 'sbt_bkp':'min',
#     'flip_skip_reason': lambda x: x.dropna().iloc[-1] if x.dropna().size > 0 else np.nan
# }).reset_index().fillna(0)

# mat_df = grouped_df[grouped_df['eligible_day']==1]


# Define a helper function to flag extubation
def check_extubation(series):
    found_imv = False
    # Iterate over non-missing values in order
    for device in series.dropna():
        if device == "imv":
            found_imv = True
        elif found_imv and device != "imv":
            return 1  # Transition occurred
    return 0  # No transition from IMV to a non-IMV device

# Group and aggregate the DataFrame including the extubation check
grouped_df = final_df.groupby('hosp_id_day_key').agg({
    'hospital_id': lambda x: x.dropna().iloc[-1] if x.dropna().size > 0 else np.nan,
    'eligible_day': 'max',
    'EHR_Delivery_2mins': 'max',
    'sat_screen_pass_fail': 'max',
    'sat_delivery_pass_fail': 'max',
    'sbt_screen_pass_fail': 'max',
    'sbt_delivery_pass_fail': 'max',
    # 'sbt_bkp': 'min',  # Uncomment if needed
    'flip_skip_reason': lambda x: x.dropna().iloc[-1] if x.dropna().size > 0 else np.nan,
    'device_category': check_extubation  # Apply the extubation logic
}).reset_index()

# Rename the aggregated device_category column to extubated and fill NaN values
grouped_df = grouped_df.rename(columns={'device_category': 'extubated'}).fillna(0)
mat_df = grouped_df[grouped_df['eligible_day']==1]

# %%
sbt_S = grouped_df[(grouped_df['sbt_screen_pass_fail']==1) & (grouped_df['eligible_day']==1)].hosp_id_day_key.unique()
sbt_D = grouped_df[(grouped_df['sbt_delivery_pass_fail']==1) & (grouped_df['eligible_day']==1)].hosp_id_day_key.unique()
ehr_2min = grouped_df[(grouped_df['EHR_Delivery_2mins']==1) & (grouped_df['eligible_day']==1)].hosp_id_day_key.unique()
print(f"Number of unique days passing SBT Screen: {len(sbt_S)}")
print(f"Number of unique days passing SBT Delivery: {len(sbt_D)}")
print(f"Number of unique days with EHR Delivery in 2 minutes: {len(ehr_2min)}")

# %%
hospital_ids = mat_df['hospital_id'].unique()

for hosp in hospital_ids:
    # Filter the DataFrame for the current hospital
    df_hosp = mat_df[mat_df['hospital_id'] == hosp]
    
    # Create the confusion matrix using pd.crosstab
    conf_matrix = pd.crosstab(df_hosp['EHR_Delivery_2mins'], df_hosp['sbt_delivery_pass_fail'])
    
    # Calculate percentages for each cell
    conf_matrix_percent = conf_matrix / conf_matrix.values.sum() * 100
    
    # Create annotation labels that combine count and percentage
    annot = conf_matrix.astype(str) + "\n" + conf_matrix_percent.round(1).astype(str) + "%"
    
    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=annot, fmt='', cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("SBT Delivery in Flowsheet")
    plt.ylabel("EHR Delivery in 2 minutes")
    plt.title(f"Confusion Matrix for Hospital {hosp}")
    # Save the plot as a PNG file
    plt.savefig(f"../output/final/confusion_matrix_hospital_{hosp}.png")
    plt.close()  # Close the plot to free memory
    
    # Extract ground truth and predictions for the current hospital
    y_true = df_hosp['EHR_Delivery_2mins']
    y_pred = df_hosp['sbt_delivery_pass_fail']
    
    # Compute the confusion matrix and extract TP, FP, FN, TN
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate individual metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    # Print metrics for current hospital (optional)
    print(f"Hospital ID: {hosp}")
    print(f"Accuracy    : {accuracy:.3f}")
    print(f"Precision   : {precision:.3f}")
    print(f"Recall      : {recall:.3f}")
    print(f"F1 Score    : {f1:.3f}")
    print(f"Specificity : {specificity:.3f}\n")
    
    # Create a dictionary with the computed metrics
    metrics_dict = {
        "True Positives (TP)": tp,
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "True Negatives (TN)": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity
    }
    
    # Build a DataFrame to store the metrics
    df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    
    # Save the metrics DataFrame as a CSV file
    df_metrics.to_csv(f"../output/final/df_metrics_hospital_{hosp}.csv", index=False)
    print(hosp,df_metrics)

# %%
hospital_ids = mat_df['hospital_id'].unique()

for hosp in hospital_ids:
    # -------------------------------
    # Filter the data for the current hospital
    # -------------------------------
    mat_hosp = mat_df[mat_df['hospital_id'] == hosp]
    
    # -------------------------------
    # Step 1: Extract filtered keys from mat_hosp
    # -------------------------------
    filtered_keys = mat_hosp.loc[
        (mat_hosp['EHR_Delivery_2mins'] == 0) & (mat_hosp['sbt_delivery_pass_fail'] == 1),
        'hosp_id_day_key'
    ].unique()
    
    # -------------------------------
    # Step 2: Filter final_hosp using these keys
    # -------------------------------
    final_filtered_df = final_df.loc[
        (final_df['sbt_delivery_pass_fail'] == 1) & 
        (final_df['hosp_id_day_key'].isin(filtered_keys))
    ]
    
    final_filtered_df = final_filtered_df.sort_values('event_time')
    final_filtered_df = final_filtered_df.drop_duplicates(subset='hosp_id_day_key', keep='first')
    
    print(f"Hospital: {hosp}, final_filtered_df shape: {final_filtered_df.shape}")
    
    # -------------------------------
    # Work on a copy for filtering steps
    # -------------------------------
    df = final_filtered_df.copy()
    results = []
    
    # ---------------------------------------
    # Step 1: Filter on 'flip_skip_reason'
    # ---------------------------------------
    step1 = df[~df['flip_skip_reason'].isna()]
    results.append({
        'Step': 'Step 1',
        'FilterColumn': 'flip_skip_reason',
        'UniqueKeys': step1['hosp_id_day_key'].nunique(),
        'RowCount': step1.shape[0],
        'ValueCounts': step1['flip_skip_reason'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step1['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 2: Filter on 'cond_device_imv'
    # ---------------------------------------
    step2 = df[~df['cond_device_imv'].isna()]
    results.append({
        'Step': 'Step 2',
        'FilterColumn': 'cond_device_imv',
        'UniqueKeys': step2['hosp_id_day_key'].nunique(),
        'RowCount': step2.shape[0],
        'ValueCounts': step2['cond_device_imv'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step2['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 3: Filter on 'cond_location_icu'
    # ---------------------------------------
    step3 = df[~df['cond_location_icu'].isna()]
    results.append({
        'Step': 'Step 3',
        'FilterColumn': 'cond_location_icu',
        'UniqueKeys': step3['hosp_id_day_key'].nunique(),
        'RowCount': step3.shape[0],
        'ValueCounts': step3['cond_location_icu'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step3['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 4: Filter on 'cond_peep_set_le8'
    # ---------------------------------------
    step4 = df[~df['cond_peep_set_le8'].isna()]
    results.append({
        'Step': 'Step 4',
        'FilterColumn': 'cond_peep_set_le8',
        'UniqueKeys': step4['hosp_id_day_key'].nunique(),
        'RowCount': step4.shape[0],
        'ValueCounts': step4['cond_peep_set_le8'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step4['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 5: Filter on 'cond_ps_set_le8'
    # ---------------------------------------
    step5 = df[~df['cond_ps_set_le8'].isna()]
    results.append({
        'Step': 'Step 5',
        'FilterColumn': 'cond_ps_set_le8',
        'UniqueKeys': step5['hosp_id_day_key'].nunique(),
        'RowCount': step5.shape[0],
        'ValueCounts': step5['cond_ps_set_le8'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step5['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 6: Filter on 'cond_mode_ps_cpap'
    # ---------------------------------------
    step6 = df[~df['cond_mode_ps_cpap'].isna()]
    results.append({
        'Step': 'Step 6',
        'FilterColumn': 'cond_mode_ps_cpap',
        'UniqueKeys': step6['hosp_id_day_key'].nunique(),
        'RowCount': step6.shape[0],
        'ValueCounts': step6['cond_mode_ps_cpap'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step6['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 7: Remaining (unmatched) rows
    # ---------------------------------------
    step7 = df.copy()
    results.append({
        'Step': 'Step 7 (Unmatched)',
        'FilterColumn': None,
        'UniqueKeys': step7['hosp_id_day_key'].nunique(),
        'RowCount': step7.shape[0],
        'ValueCounts': None
    })
    
    # ---------------------------------------
    # Create Detailed Step-by-Step Summary DataFrame
    # ---------------------------------------
    detailed_summary_df = pd.DataFrame(results)
    
    # Calculate total_failures as the sum of UniqueKeys across all steps for this hospital
    total_failures = detailed_summary_df['UniqueKeys'].sum()
    
   
    
    # Add "% Per 100" and "% of Total" columns
    detailed_summary_df['% by eligible_days'] = detailed_summary_df['UniqueKeys'].apply(
        lambda x: round((x / eligible_days) * 100, 2)
    )
    detailed_summary_df['% of Total'] = detailed_summary_df['UniqueKeys'].apply(
        lambda x: round((x / total_failures) * 100, 2) if total_failures != 0 else 0
    )
    
    # ---------------------------------------
    # Save the detailed summary DataFrame as a CSV file for the current hospital
    # ---------------------------------------
    output_filename = f"../output/final/dependent_summary_hospital_{hosp}.csv"
    detailed_summary_df.to_csv(output_filename, index=False)
    print(f"Saved detailed summary for hospital {hosp} to {output_filename}\n")
    print(hosp,detailed_summary_df)
    print()

    # ============================================================
    # B. Independent Filtering Summary (Apply each filter independently)
    # ============================================================
    ind_step1 = final_filtered_df[~final_filtered_df['flip_skip_reason'].isna()]
    ind_step2 = final_filtered_df[~final_filtered_df['cond_device_imv'].isna()]
    ind_step3 = final_filtered_df[~final_filtered_df['cond_location_icu'].isna()]
    ind_step4 = final_filtered_df[~final_filtered_df['cond_peep_set_le8'].isna()]
    ind_step5 = final_filtered_df[~final_filtered_df['cond_ps_set_le8'].isna()]
    ind_step6 = final_filtered_df[~final_filtered_df['cond_mode_ps_cpap'].isna()]
    
    # Determine the union of keys matched by any filter
    matched_keys = set().union(
        ind_step1['hosp_id_day_key'],
        ind_step2['hosp_id_day_key'],
        ind_step3['hosp_id_day_key'],
        ind_step4['hosp_id_day_key'],
        ind_step5['hosp_id_day_key'],
        ind_step6['hosp_id_day_key']
    )
    # Unmatched keys: those not included in any of the independent filters
    ind_step7 = final_filtered_df[~final_filtered_df['hosp_id_day_key'].isin(matched_keys)]
    
    # Compute unique key counts per filter
    failure_counts = {
        'flip_skip_reason': ind_step1['hosp_id_day_key'].nunique(),
        'cond_device_imv': ind_step2['hosp_id_day_key'].nunique(),
        'cond_location_icu': ind_step3['hosp_id_day_key'].nunique(),
        'cond_peep_set_le8': ind_step4['hosp_id_day_key'].nunique(),
        'cond_ps_set_le8': ind_step5['hosp_id_day_key'].nunique(),
        'cond_mode_ps_cpap': ind_step6['hosp_id_day_key'].nunique(),
        'unmatched': ind_step7['hosp_id_day_key'].nunique()
    }
    
    # Compute value counts for each filter column
    value_counts_map = {
        'flip_skip_reason': ind_step1['flip_skip_reason'].value_counts(dropna=False).to_dict(),
        'cond_device_imv': ind_step2['cond_device_imv'].value_counts(dropna=False).to_dict(),
        'cond_location_icu': ind_step3['cond_location_icu'].value_counts(dropna=False).to_dict(),
        'cond_peep_set_le8': ind_step4['cond_peep_set_le8'].value_counts(dropna=False).to_dict(),
        'cond_ps_set_le8': ind_step5['cond_ps_set_le8'].value_counts(dropna=False).to_dict(),
        'cond_mode_ps_cpap': ind_step6['cond_mode_ps_cpap'].value_counts(dropna=False).to_dict(),
        'unmatched': None
    }
    
    total_failures_ind = sum(failure_counts.values())
    summary_data = []
    for reason, count in failure_counts.items():
        summary_data.append({
            'Failure Reason': reason,
            'Count': count,
            '% by eligible_days': round((count / eligible_days) * 100, 2),
            '% of Total (out of total failed cases)': round((count / total_failures_ind) * 100, 2) if total_failures_ind else 0,
            'Value Counts': value_counts_map[reason]
        })
    
    independent_summary_df = pd.DataFrame(summary_data)
    independent_summary_df = independent_summary_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    
    ind_output_filename = f"../output/final/independent_summary_hospital_{hosp}.csv"
    independent_summary_df.to_csv(ind_output_filename, index=False)
    print(f"Saved independent summary for hospital {hosp} to {ind_output_filename}\n")
    print(hosp, independent_summary_df)
    print()

# %%
detailed_summary_df

# %%
independent_summary_df

# %% [markdown]
# #### Plots

# %%
hospital_ids = final_df['hospital_id'].dropna().unique()

# This list will hold the summary data for each hospital
hospital_summary_list = []

for hosp in hospital_ids:
    # Filter final_df for the current hospital
    final_hosp = final_df[final_df['hospital_id'] == hosp]
    
    # Extract event times for SBT delivery (pass) and EHR delivery (within 2 mins)
    sbt_d_time = final_hosp[
        (final_hosp['sbt_delivery_pass_fail'] == 1) & 
        (final_hosp['eligible_day'] == 1)
    ][['hosp_id_day_key', 'event_time']].drop_duplicates()
    
    ehr_d_time = final_hosp[
        (final_hosp['EHR_Delivery_2mins'] == 1) & 
        (final_hosp['eligible_day'] == 1)
    ][['hosp_id_day_key', 'event_time']].drop_duplicates()
    
    # Convert event_time to hour values
    sbt_hours = sbt_d_time['event_time'].dt.hour
    ehr_hours = ehr_d_time['event_time'].dt.hour

    # Create overlay histogram plot for the current hospital
    plt.figure(figsize=(10, 6))
    # Use bins from 0 to 24 (24 bins) to capture each hour of the day
    plt.hist(sbt_hours, bins=range(0, 25), alpha=0.5, label='SBT Delivery Time', edgecolor='black')
    plt.hist(ehr_hours, bins=range(0, 25), alpha=0.5, label='EHR Delivery Time', edgecolor='black')
    plt.xlabel('Hour of Day')
    plt.ylabel('Frequency')
    plt.title(f'Event Time Distribution (Hourly) - Hospital {hosp}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot for the current hospital
    plt.savefig(f"../output/final/event_time_distribution_hospital_{hosp}.png")
    plt.close()
    
    # Build a summary DataFrame for the current hospital:
    # Get counts per hour for each event type
    sbt_counts = sbt_hours.value_counts().sort_index()
    ehr_counts = ehr_hours.value_counts().sort_index()
    
    # Create a DataFrame with all hours 0-23, merging the counts (fill missing with 0)
    hours_df = pd.DataFrame({'hour': range(24)})
    hours_df['SBT_Delivery'] = hours_df['hour'].map(sbt_counts).fillna(0).astype(int)
    hours_df['EHR_Delivery'] = hours_df['hour'].map(ehr_counts).fillna(0).astype(int)
    hours_df['hospital_id'] = hosp
    
    hospital_summary_list.append(hours_df)

# Combine the summary data for all hospitals into one DataFrame
combined_summary_df = pd.concat(hospital_summary_list, ignore_index=True)
combined_summary_df.to_csv(f"../output/final/event_time_distribution_summary.csv", index=False)

print("Overlay plots created and summary CSV saved.")

# %%
# --- Calculate statistics from final_df ---

# By n = Days
total_days = final_df['hosp_id_day_key'].nunique()
eligible_days = final_df[final_df['eligible_day'] == 1]['hosp_id_day_key'].nunique()
percentage = (eligible_days / total_days) * 100 if total_days > 0 else 0
imv_days = final_df[final_df['device_category'] == 'imv']['hosp_id_day_key'].nunique()
imv_icu_days = final_df[(final_df['device_category'] == 'imv') & (final_df['location_category'] == 'icu')]['hosp_id_day_key'].nunique()

# By n = Encounter
h_total_days = final_df['hospitalization_id'].nunique()
h_eligible_days = final_df[final_df['eligible_day'] == 1]['hospitalization_id'].nunique()
h_percentage = (h_eligible_days / h_total_days) * 100 if h_total_days > 0 else 0
h_imv_days = final_df[final_df['device_category'] == 'imv']['hospitalization_id'].nunique()
h_imv_icu_days = final_df[(final_df['device_category'] == 'imv') & (final_df['location_category'] == 'icu')]['hospitalization_id'].nunique()

# --- Calculate statistics from mat_df ---

# Distribution of EHR_Delivery_2mins for extubated == 1 (in percentages)
ehr_delivery_counts = (
    mat_df[mat_df['extubated'] == 1]['EHR_Delivery_2mins']
    .value_counts(normalize=True) * 100
)

# Distribution of sbt_delivery_pass_fail for extubated == 1 (in percentages)
sbt_delivery_counts = (
    mat_df[mat_df['extubated'] == 1]['sbt_delivery_pass_fail']
    .value_counts(normalize=True) * 100
)

# --- Print the statistics ---

print('By n = Days')
print('Total number of days for eval in cohort:', total_days)
print(f"Eligible days: {eligible_days} / {total_days} ({percentage:.2f}%)")
print('Hospital days with at least one IMV event:', imv_days)
print('Hospital days with at least one IMV & ICU event:', imv_icu_days)

print('\nBy n = Encounter')
print('Total number of days for eval in cohort:', h_total_days)
print(f"Eligible days: {h_eligible_days} / {h_total_days} ({h_percentage:.2f}%)")
print('Hospital days with at least one IMV event:', h_imv_days)
print('Hospital days with at least one IMV & ICU event:', h_imv_icu_days)

print('\nEHR_Delivery_2mins distribution (for extubated == 1):')
print(ehr_delivery_counts)

print('\nsbt_delivery_pass_fail distribution (for extubated == 1):')
print(sbt_delivery_counts)

# --- Create a summary DataFrame for the final_df stats ---
stats_data = {
    'Metric': [
        'total_days', 'eligible_days', 'eligible_percentage', 
        'imv_days', 'imv_icu_days', 
        'h_total_days', 'h_eligible_days', 'h_eligible_percentage', 
        'h_imv_days', 'h_imv_icu_days'
    ],
    'Value': [
        total_days, eligible_days, percentage,
        imv_days, imv_icu_days,
        h_total_days, h_eligible_days, h_percentage,
        h_imv_days, h_imv_icu_days
    ]
}

stats_df = pd.DataFrame(stats_data)

print('\nCombined statistics DataFrame:')
print(stats_df)
stats_df.to_csv('../output/final/stats_df.csv')

# %%



