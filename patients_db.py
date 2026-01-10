import pandas as pd
import numpy as np

def generate_patient_records(n_patients=5):
    """
    Generates robust, realistic synthetic patient data for HCT survival analysis.
    Ensures no NaNs and correct data types for 60+ columns.
    """
    
    # 1. Base Profiles (The "Identity" of the patient)
    profiles = [
        {"first_name": "James", "last_name": "Wilson", "age": 58, "picture": "https://randomuser.me/api/portraits/men/32.jpg"},
        {"first_name": "Linda", "last_name": "Martinez", "age": 42, "picture": "https://randomuser.me/api/portraits/women/44.jpg"},
        {"first_name": "Robert", "last_name": "Chen", "age": 65, "picture": "https://randomuser.me/api/portraits/men/11.jpg"},
        {"first_name": "Sarah", "last_name": "Johnson", "age": 29, "picture": "https://randomuser.me/api/portraits/women/65.jpg"},
        {"first_name": "Michael", "last_name": "Brown", "age": 52, "picture": "https://randomuser.me/api/portraits/men/85.jpg"}
    ]
    
    data = []
    
    # 2. Define Clinical Constants
    # Categorical Options
    cat_opts = {
        'dri_score': ['Low', 'Intermediate', 'High', 'Very High'],
        'tbi_status': ['No TBI', 'TBI + Cy +- Other', 'TBI +- Other, <cGy'],
        'graft_type': ['Peripheral blood', 'Bone marrow', 'Cord blood'],
        'prim_disease_hct': ['AML', 'ALL', 'MDS', 'NHL', 'CML'],
        'cmv_status': ['+/-', '-/+', '-/-', '+/+'],
        'cyto_score': ['Poor', 'Intermediate', 'Favorable'],
        'sex_match': ['M-M', 'F-F', 'M-F', 'F-M'],
        'race_group': ['White', 'Asian', 'Black', 'Other'],
        'donor_related': ['Related', 'Unrelated'],
        'prod_type': ['Hoffman', 'Other', 'Unknown'],
        'conditioning_intensity': ['MAC', 'RIC', 'NMA']
    }

    # Binary Columns (Yes/No)
    binary_cols = [
        'psych_disturb', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue',
        'pulm_severe', 'rituximab', 'obesity', 'in_vivo_tcd', 'hepatic_severe',
        'prior_tumor', 'peptic_ulcer', 'rheum_issue', 'hepatic_mild', 'cardiac',
        'pulm_moderate', 'mrd_hct'
    ]

    # HLA Match Columns (Usually 0-10 scores or binary)
    hla_cols = [
        'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6', 
        'hla_high_res_10', 'hla_match_dqb1_high', 'hla_nmdp_6', 'hla_match_c_low',
        'hla_match_drb1_low', 'hla_match_dqb1_low', 'hla_match_a_high', 
        'hla_match_b_low', 'hla_match_a_low', 'hla_match_b_high', 
        'hla_low_res_8', 'hla_match_drb1_high', 'hla_low_res_10'
    ]

    # 3. Generate Data Row by Row
    for i, p in enumerate(profiles):
        row = p.copy()
        row['ID'] = 1000 + i
        
        # --- NUMERICAL CLINICAL SCORES ---
        # We use explicit floats to avoid type issues
        row['karnofsky_score'] = float(np.random.choice([80, 90, 100]))
        row['comorbidity_score'] = float(np.random.randint(0, 6))
        row['donor_age'] = float(np.random.randint(20, 65))
        row['age_at_hct'] = float(p['age'])
        row['year_hct'] = float(np.random.randint(2015, 2024))
        
        # Fill HLA columns with safe numbers (e.g., 0, 1, or 2 for mismatch count)
        for hla in hla_cols:
            row[hla] = float(np.random.choice([0, 1, 2]))

        # --- BINARY DATA ---
        for col in binary_cols:
            row[col] = np.random.choice(['Yes', 'No'], p=[0.15, 0.85])

        # --- CATEGORICAL DATA ---
        for col, choices in cat_opts.items():
            row[col] = np.random.choice(choices)

        # --- RANDOM FILLER FOR OTHER POTENTIAL COLUMNS ---
        # To ensure we don't have gaps if the model expects 'melphalan_dose' etc.
        extra_cols = ['melphalan_dose', 'tce_match', 'tce_div_match', 'gvhd_proph']
        for col in extra_cols:
            row[col] = 0.0 # Safe numeric default

        data.append(row)

    return pd.DataFrame(data)