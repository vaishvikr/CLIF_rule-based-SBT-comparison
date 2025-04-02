# %%
import pandas as pd
from tqdm import tqdm
import numpy as np
import duckdb
import pyCLIF as pc
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Base Population

# %%
adt = pc.load_data('clif_adt')
adt['hospitalization_id'] = adt['hospitalization_id'].astype(str)
adt['in_dttm'] = pc.getdttm(adt['in_dttm'])
adt = pc.standardize_datetime_tz(adt,['in_dttm','out_dttm'],pc.helper['your_site_timezone'],pc.helper['data_timezone'])
pc.deftime(adt['in_dttm'])
adt.head()

# %%
hosp = pc.load_data('clif_hospitalization')
hosp['hospitalization_id'] = hosp['hospitalization_id'].astype(str)
if 'hospitalization_joined_id' not in hosp.columns:
    hosp['hospitalization_joined_id'] = hosp['hospitalization_id']

hosp['hospitalization_joined_id'] = hosp['hospitalization_joined_id'].astype(str)
hosp['admission_dttm'] = pc.getdttm(hosp['admission_dttm'])
hosp['discharge_dttm'] = pc.getdttm(hosp['discharge_dttm'])
hosp = pc.standardize_datetime_tz(hosp,['admission_dttm','discharge_dttm'],pc.helper['your_site_timezone'],pc.helper['data_timezone'])
hosp.head()

# %%
adt['Hosp_key_bkp'] = adt['hospitalization_id']
hosp['Hosp_key_bkp'] = hosp['hospitalization_id']

# %%
eblock = pc.stitch_encounters(hosp,adt)

# Create mapping dictionary
hospitalization_to_block = {
    hospital_id: block 
    for block, hospital_list in zip(eblock["encounter_block"], eblock["list_hospitalization_id"])
    for hospital_id in hospital_list
}

# %%
agg_rules_hosp = {
    'patient_id': 'first',  # Assuming patient_id is consistent across duplicates
    'zipcode_five_digit': 'first',  # Retain first occurrence
    'admission_dttm': 'min',  # Earliest admission date
    'discharge_dttm': 'max',  # Latest discharge date
    'discharge_name': 'last',  # Prioritize first value (change as per logic)
    'age_at_admission': 'mean',  # Take average if different
    'discharge_category': 'last',  # Keep the first occurrence
    'hospitalization_joined_id': lambda x: ', '.join(x.unique()),  # Retain first occurrence
    'Hosp_key_bkp': lambda x: ', '.join(x.unique())  # Backup key, take first occurrence
}

hosp['hospitalization_id'] = hosp['hospitalization_id'].map(hospitalization_to_block)
hosp['hospitalization_id'] = hosp['hospitalization_id'].astype(str)
hosp = hosp.sort_values(by=['hospitalization_id', 'admission_dttm'])

hosp = hosp.groupby('hospitalization_id').agg(agg_rules_hosp).reset_index()

# %%
adt = adt[['hospitalization_id', 'in_dttm', 'location_category', 'hospital_id']]
adt['hospitalization_id'] = adt['hospitalization_id'].map(hospitalization_to_block).astype(str)
adt = adt.sort_values(by=['hospitalization_id', 'in_dttm'])

# %% [markdown]
# #### cohort filters

# %%
rst = pc.load_data('clif_respiratory_support')
rst['hospitalization_id'] = rst['hospitalization_id'].astype(str)
rst['hospitalization_id'] = rst['hospitalization_id'].map(hospitalization_to_block).fillna(-1).astype(int).astype(str)
rst = rst[~rst['hospitalization_id'].isin(rst[rst['tracheostomy']==1].hospitalization_id.unique())] #exclude trach pats

# %%
rst = pc.standardize_datetime_tz(rst,['recorded_dttm'],pc.helper['your_site_timezone'],pc.helper['data_timezone'])

# %%
rst.head()

# %%
pat = pc.load_data('clif_patient')
pat = pc.standardize_datetime_tz(pat,['birth_date','death_dttm'],pc.helper['your_site_timezone'],pc.helper['data_timezone'])

# %%
imv_hosp_ids = rst[rst['device_category'].str.lower()=='imv'].hospitalization_id.unique()
icu_hosp_ids = adt[adt['location_category'].str.lower()=='icu'].hospitalization_id.unique()

icu_hosp_ids = [x for x in icu_hosp_ids if x is not None]
imv_hosp_ids = [x for x in imv_hosp_ids if x is not None]

hosp = hosp[
    (hosp['admission_dttm'].dt.year >= 2022) &
    (hosp['admission_dttm'].dt.year <= 2024) &
    (hosp['hospitalization_id'].isin(np.intersect1d(imv_hosp_ids, icu_hosp_ids))) &
    (hosp['age_at_admission'] <=119)
].reset_index(drop=True)

required_id= hosp['hospitalization_id'].unique()
print(len(required_id),' : potential cohort count')

base = pd.merge(hosp,pat,on='patient_id',how='inner')\
[['patient_id', 'hospitalization_id','admission_dttm', 'discharge_dttm','age_at_admission', 'discharge_category','sex_category','race_category', 'ethnicity_category']]

base['admission_dttm'] = pc.getdttm(base['admission_dttm'])

base.columns

adt = adt[adt['hospitalization_id'].isin(required_id)].reset_index(drop=True)
rst = rst[rst['hospitalization_id'].isin(required_id)].reset_index(drop=True)

# %%
base.head()

# %%
if pc.helper['site_name']=='RUSH':
    rst_col = [ 'hospitalization_id', 'recorded_dttm', 'device_category', 'mode_category','fio2_set','peep_set','resp_rate_set','pressure_support_set','mode_name','tube_comp_%','sbt_timepoint']
else:
    rst_col = [ 'hospitalization_id', 'recorded_dttm', 'device_category', 'mode_category','fio2_set','peep_set','resp_rate_set','pressure_support_set','mode_name']
rst = rst[rst_col]
rst['device_category'] = rst['device_category'].str.lower()
rst['mode_category'] = rst['mode_category'].str.lower()
rst['recorded_dttm'] = pc.getdttm(rst['recorded_dttm'])

# %%
pc.deftime(rst['recorded_dttm'])

# %%
rst.head()

# %% [markdown]
# ### MAC

# %%
mac = pc.load_data('clif_medication_admin_continuous')
mac['hospitalization_id'] = mac['hospitalization_id'].astype(str)
mac['hospitalization_id'] = mac['hospitalization_id'].map(hospitalization_to_block).astype(str)
mac_col = ['hospitalization_id', 'admin_dttm','med_dose','med_category','med_dose_unit']
mac = mac[(mac['hospitalization_id'].isin(required_id)) & (mac['med_category'].isin( [
        "norepinephrine",
        "epinephrine",
        "phenylephrine",
        "angiotensin",
        "vasopressin",
        "dopamine",
        "dobutamine",
        "milrinone",
        "isoproterenol",
        "cisatracurium",
        "vecuronium",
        "rocuronium",'fentanyl', 'propofol', 'lorazepam', 'midazolam','hydromorphone','morphine'
    ]))][mac_col].reset_index(drop=True)

mac['admin_dttm'] = pc.getdttm(mac['admin_dttm'])
mac = pc.standardize_datetime_tz(mac,['admin_dttm'],pc.helper['your_site_timezone'],pc.helper['data_timezone'])

mac['med_dose_unit']=mac['med_dose_unit'].str.lower()
mac = mac[(mac['med_dose_unit'].str.contains(r'/', na=False)) & (mac['med_dose_unit']!='units/hr')].reset_index(drop=True)

# %%
mac.head()

# %% [markdown]
# ### Patient_assessment

# %%
cat_values_mapping_dict = {
    'negative': 0,
    'fail': 0,
    'pass': 1,
    'positive': 1,
    None: np.nan ,
    np.nan : np.nan,
    'yes':1,
    'no':0
}

pat_assess_cats_rquired = [ 'sbt_delivery_pass_fail',
                            'sbt_screen_pass_fail','sat_delivery_pass_fail',
                            'sat_screen_pass_fail']

# %%
pat_at = pc.load_data('clif_patient_assessments',-1)
pat_at_col = ['hospitalization_id', 'recorded_dttm','numerical_value', 'categorical_value','assessment_category']
pat_at['assessment_category'] = pat_at['assessment_category'].str.lower()
pat_at = pat_at[(pat_at['assessment_category'].isin(pat_assess_cats_rquired)) ][pat_at_col].reset_index(drop=True)
pat_at = pc.standardize_datetime_tz(pat_at,['recorded_dttm'],pc.helper['your_site_timezone'],pc.helper['data_timezone'])

# %%
pat_at['hospitalization_id'] = pat_at['hospitalization_id'].astype(str)

# %%
pat_at['hospitalization_id'] = pat_at['hospitalization_id'].map(hospitalization_to_block).fillna(-1).astype(int).astype(str)

# %%
pat_at = pat_at[(pat_at['hospitalization_id'].isin(required_id))][pat_at_col].reset_index(drop=True)
pat_at['recorded_dttm'] = pc.getdttm(pat_at['recorded_dttm'])
pat_at['categorical_value'] = pat_at['categorical_value'].str.lower().map(cat_values_mapping_dict)
pat_at['assessment_value'] = pat_at['numerical_value'].combine_first(pat_at['categorical_value'])
pat_at.drop(columns=['numerical_value','categorical_value'],inplace=True)

# %%
pat_at.assessment_category.unique()

# %%
pat_at['assessment_value'].value_counts()

# %%
pat_at.head()

# %% [markdown]
# ### vitals

# %%
vit = pc.load_data('clif_vitals',-1)
vit['hospitalization_id'] = vit['hospitalization_id'].astype(str)
vit['hospitalization_id'] = vit['hospitalization_id'].map(hospitalization_to_block).astype(str)
vit_col = ['hospitalization_id','recorded_dttm','vital_category','vital_value' ]
vit['vital_category'] = vit['vital_category'].str.lower()
vit = pc.standardize_datetime_tz(vit,['recorded_dttm'],pc.helper['your_site_timezone'],pc.helper['data_timezone'])
vit = vit[(vit['hospitalization_id'].isin(required_id)) & (vit['vital_category'].isin(['map','heart_rate','sbp','dbp','spo2','respiratory_rate','weight_kg','height_cm'])) ][vit_col].reset_index(drop=True)

vit['recorded_dttm_min'] = pc.getdttm(vit['recorded_dttm'])

# Sort by hospitalization_id and recorded_dttm
vit = vit.sort_values(by=["hospitalization_id", "recorded_dttm"])

# Group by hospitalization_id, vital_category, and recorded_dttm_min, then take the first occurrence of vital_value
vit = vit.groupby(["hospitalization_id", "vital_category", "recorded_dttm_min"], as_index=False).agg({
    "vital_value": "first"
})
# make sure float
vit['vital_value']=vit['vital_value'].astype(float)

#for meds
vit_weight = vit[vit['vital_category']=='weight_kg'].reset_index(drop=True)

# %%
vit.head()

# %%
# Count duplicates
duplicates = vit.duplicated(subset=["hospitalization_id", "vital_category", "recorded_dttm_min"], keep=False)

# Show any duplicates (should be empty if grouping worked correctly)
vit[duplicates]

# %% [markdown]
# ### new mac and weight df for med unit conversion

# %%
vit_weight.rename({'vital_category': 'med_category', 'recorded_dttm_min': 'admin_dttm'}, axis='columns', inplace=True)

new_mac = pd.concat([mac, vit_weight], ignore_index=True)

new_mac = new_mac.sort_values(by=['hospitalization_id', 'admin_dttm'])

new_mac['vital_value'] = new_mac.groupby('hospitalization_id')['vital_value'].ffill().bfill()

new_mac = new_mac[~(new_mac['med_category']=='weight_kg')].reset_index(drop=True)

print('mac rows:',mac.shape,'New mac rows:', new_mac.shape)
#del vit_weight

# %%
new_mac.head(5)

# %%
# The med_unit_info dictionary
med_unit_info = {
    'norepinephrine': {
        'required_unit': 'mcg/kg/min',
        'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr'],
    },
    'epinephrine': {
        'required_unit': 'mcg/kg/min',
        'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr'],
    },
    'phenylephrine': {
        'required_unit': 'mcg/kg/min',
        'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr'],
    },
    'angiotensin': {
        'required_unit': 'ng/kg/min',
        'acceptable_units': ['ng/kg/min', 'ng/kg/hr'],
    },
    'vasopressin': {
        'required_unit': 'units/min',
        'acceptable_units': ['units/min', 'units/hr', 'milliunits/min', 'milliunits/hr'],
    },
    'dopamine': {
        'required_unit': 'mcg/kg/min',
        'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr'],
    },
    'dobutamine': {
        'required_unit': 'mcg/kg/min',
        'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr'],
    },
    'milrinone': {
        'required_unit': 'mcg/kg/min',
        'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr'],
    },
    'isoproterenol': {
        'required_unit': 'mcg/kg/min',
        'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr'],
    },
}

def convert_med_dose(row):
    category = row['med_category']
    # If the category is not in our dictionary, skip conversion.
    if category not in med_unit_info:
        return row
    
    info = med_unit_info[category]
    required_unit = info['required_unit']
    acceptable_units = info['acceptable_units']
    
    current_unit = row['med_dose_unit']
    dose = row['med_dose']
    weight = row['vital_value']  # patient's weight in kg

    # If the current unit already matches the required unit, nothing to do.
    if current_unit == required_unit:
        return row

    # If the current unit is not in the acceptable list, skip conversion.
    if current_unit not in acceptable_units:
        return row

    # Start with a conversion factor of 1.
    conversion_factor = 1.0

    # --------------------------------------------------
    # 1. Weight conversion: if the current unit is per kg but the required is not,
    # then multiply by the patient’s weight.
    if 'kg' in current_unit and 'kg' not in required_unit:
        conversion_factor *= weight
    elif 'kg' not in current_unit and 'kg' in required_unit:
        conversion_factor /= weight

    # --------------------------------------------------
    # 2. Time conversion: convert from per hour to per minute or vice versa.
    if 'hr' in current_unit and 'min' in required_unit:
        conversion_factor /= 60.0
    elif 'min' in current_unit and 'hr' in required_unit:
        conversion_factor *= 60.0

    # --------------------------------------------------
    # 3. Medication unit conversion (e.g., mg to mcg, milliunits to units)
    # We assume the first part (before the first '/') is the measurement unit.
    current_med_unit = current_unit.split('/')[0]
    required_med_unit = required_unit.split('/')[0]

    med_conversion = {
        ('mg', 'mcg'): 1000,
        ('mcg', 'mg'): 0.001,
        ('milliunits', 'units'): 0.001,
        ('units', 'milliunits'): 1000,
    }

    if current_med_unit != required_med_unit:
        factor = med_conversion.get((current_med_unit, required_med_unit))
        if factor is not None:
            conversion_factor *= factor
        else:
            # If no conversion factor is defined, skip conversion.
            return row

    # --------------------------------------------------
    # Apply the conversion
    new_dose = dose * conversion_factor

    # Update the row with the converted dose and unit.
    row['med_dose'] = new_dose
    row['med_dose_unit'] = required_unit
    return row

# Apply the conversion function with tqdm for progress tracking
tqdm.pandas(desc="Converting medication doses")
new_mac = new_mac.progress_apply(convert_med_dose, axis=1)

# %%
# Create a summary table for each med_category
summary_table = new_mac.groupby(['med_category','med_dose_unit']).agg(
    total_N=('med_category', 'size'),
    min=('med_dose', 'min'),
    max=('med_dose', 'max'),
    first_quantile=('med_dose', lambda x: x.quantile(0.25)),
    second_quantile=('med_dose', lambda x: x.quantile(0.5)),
    third_quantile=('med_dose', lambda x: x.quantile(0.75)),
    missing_values=('med_dose', lambda x: x.isna().sum())
).reset_index()

## check the distrbituon of required continuous meds
summary_table

# %% [markdown]
# ## Wide Dataset

# %%
duckdb.register("base", base)
duckdb.register("pat_at", pat_at)
duckdb.register("rst", rst)
duckdb.register("mac", new_mac)
duckdb.register('adt',adt)
duckdb.register('vit',vit)

q="""
WITH
    uni_event_dttm as (
        select distinct
            hospitalization_id,
            event_time
        from
            (
                SELECT
                    hospitalization_id,
                    in_dttm AS event_time
                FROM
                    adt
                where
                    in_dttm is not null
                UNION
                SELECT
                    hospitalization_id,
                    recorded_dttm AS event_time
                FROM
                    rst
                where
                    recorded_dttm is not null
                UNION
                SELECT
                    hospitalization_id,
                    recorded_dttm AS event_time
                FROM
                    pat_at
                where
                    recorded_dttm is not null
                UNION
                SELECT
                    hospitalization_id,
                    admin_dttm AS event_time
                FROM
                    mac
                where
                    admin_dttm is not null
                UNION
                SELECT
                    hospitalization_id,
                    recorded_dttm_min AS event_time
                FROM
                    vit
                where
                    recorded_dttm_min is not null
            ) uni_time
    )
select distinct
    patient_id,
    a.hospitalization_id,
    admission_dttm,
    discharge_dttm,
    age_at_admission,
    discharge_category,
    sex_category,
    race_category,
    ethnicity_category,
    event_time
from
    base a
    left join uni_event_dttm b on a.hospitalization_id = b.hospitalization_id
"""
wide_cohort_df = duckdb.sql(q).df()
pc.deftime(wide_cohort_df['event_time'])

# %% [markdown]
# #### pivots for assessment and mac table 

# %%
query = """
WITH pas_data AS (
    SELECT  distinct assessment_value ,	assessment_category	,
    hospitalization_id || '_' || strftime(recorded_dttm, '%Y%m%d%H%M') AS combo_id
    FROM pat_at where recorded_dttm is not null 
) 
PIVOT pas_data
ON assessment_category
USING first(assessment_value)
GROUP BY combo_id
"""
p_pas = duckdb.sql(query).df()

query = """
WITH mac_data AS (
    SELECT  distinct med_dose ,	med_category	,
    hospitalization_id || '_' || strftime(admin_dttm, '%Y%m%d%H%M') AS combo_id
    FROM mac where admin_dttm is not null 
) 
PIVOT mac_data
ON med_category
USING min(med_dose)
GROUP BY combo_id
"""
p_mac = duckdb.sql(query).df()



# %%

query = """
WITH vital_data AS (
    SELECT  distinct vital_category,	vital_value	,
    hospitalization_id || '_' || strftime(recorded_dttm_min, '%Y%m%d%H%M') AS combo_id
    FROM vit where recorded_dttm_min is not null 
)
PIVOT vital_data
ON vital_category
USING first(vital_value)
GROUP BY combo_id
"""
p_vitals = duckdb.sql(query).df()

# %% [markdown]
# #### id-ing all unique timestamps

# %%
duckdb.register("expanded_df", wide_cohort_df)
duckdb.register("p_pas", p_pas)
duckdb.register("p_mac", p_mac)

q="""
  WITH
    u_rst as (
        select
            *,
            hospitalization_id || '_' || strftime (recorded_dttm, '%Y%m%d%H%M') AS combo_id
        from
            rst
    ),
    u_adt as (
        select
            *,
            hospitalization_id || '_' || strftime (in_dttm, '%Y%m%d%H%M') AS combo_id
        from
            adt
    ),
    u_expanded_df as (
        select
            *,
            hospitalization_id || '_' || strftime (event_time, '%Y%m%d%H%M') AS combo_id
        from
            expanded_df
    )
select
    *
from
    u_expanded_df a
    left join u_adt d on a.combo_id = d.combo_id
    left join u_rst e on a.combo_id = e.combo_id
    left join p_mac g on a.combo_id = g.combo_id
    left join p_pas h on a.combo_id = h.combo_id
    left join p_vitals i on a.combo_id=i.combo_id 

                    
"""

all_join_df = duckdb.sql(q).df().drop_duplicates()

# %%
if all_join_df.shape[0] != wide_cohort_df.shape[0]:
    print('Data has duplicates or same timestamp issue, contact project owner')
else:
    del rst,mac,pat_at

# %% [markdown]
# #### removing wide-supporting columns and adding forward fills

# %%
all_join_df.columns

# %%
# all_join_df.drop(columns= ['hospitalization_id_2','hospitalization_id_3','combo_id', 'combo_id_2' ,'combo_id_3','combo_id_4','combo_id_5','recorded_dttm','combo_id_6','in_dttm'], axis = 1,inplace=True)

all_join_df['event_time'] = pd.to_datetime(all_join_df['event_time'])
all_join_df['date'] = all_join_df['event_time'].dt.date

all_join_df = all_join_df.sort_values(['hospitalization_id', 'event_time']).reset_index(drop=True)

# Assign day numbers to each 'hospitalization_id'
all_join_df['day_number'] = all_join_df.groupby('hospitalization_id')['date'].rank(method='dense').astype(int)

# Create the combo_key by combining 'hospitalization_id' and 'day_number'
all_join_df['hosp_id_day_key'] = all_join_df['hospitalization_id'].astype(str) + '_day_' + all_join_df['day_number'].astype(str)

# %%
columns_to_check = ['sbt_delivery_pass_fail','sbt_screen_pass_fail','sat_delivery_pass_fail','sat_screen_pass_fail']
for col in columns_to_check:
    if col not in all_join_df.columns:
        all_join_df[col] = np.nan

# %%
if pc.helper['site_name']=='RUSH':
    all_join_df[['patient_id',
 'hospitalization_id',
 'admission_dttm',
 'discharge_dttm',
 'age_at_admission',
 'discharge_category',
 'sex_category',
 'race_category',
 'ethnicity_category',
 'event_time',
 'location_category',
 'hospital_id',
 'recorded_dttm',
 'device_category',
 'mode_category',
 'fio2_set',
 'peep_set',
 'resp_rate_set',
 'pressure_support_set',
 'mode_name',
 'tube_comp_%',
 'sbt_timepoint',
 'cisatracurium',
 'dobutamine',
 'dopamine',
 'epinephrine',
 'fentanyl',
 'hydromorphone',
 'midazolam',
 'milrinone',
 'morphine',
 'norepinephrine',
 'phenylephrine',
 'propofol',
 'vasopressin',
 'sat_delivery_pass_fail',
 'sat_screen_pass_fail',
 'sbt_delivery_pass_fail',
 'sbt_screen_pass_fail',
 'dbp',
 'heart_rate',
 'height_cm',
 'map',
 'respiratory_rate',
 'sbp',
 'spo2',
 'weight_kg',
 'date',
 'day_number',
 'hosp_id_day_key']].to_csv('../output/intermediate/study_cohort.csv', index=False)
else:
    all_join_df[['patient_id',
 'hospitalization_id',
 'admission_dttm',
 'discharge_dttm',
 'age_at_admission',
 'discharge_category',
 'sex_category',
 'race_category',
 'ethnicity_category',
 'event_time',
 'location_category',
 'hospital_id',
 'recorded_dttm',
 'device_category',
 'mode_category',
 'fio2_set',
 'peep_set',
 'resp_rate_set',
 'pressure_support_set',
 'mode_name',
 'cisatracurium',
 'dobutamine',
 'dopamine',
 'epinephrine',
 'fentanyl',
 'hydromorphone',
 'midazolam',
 'milrinone',
 'morphine',
 'norepinephrine',
 'phenylephrine',
 'propofol',
 'vasopressin',
 'sat_delivery_pass_fail',
 'sat_screen_pass_fail',
 'sbt_delivery_pass_fail',
 'sbt_screen_pass_fail',
 'dbp',
 'heart_rate',
 'height_cm',
 'map',
 'respiratory_rate',
 'sbp',
 'spo2',
 'weight_kg',
 'date',
 'day_number',
 'hosp_id_day_key']].to_csv('../output/intermediate/study_cohort.csv', index=False)


