import pandas as pd
import json
import os

def load_variable_labels(json_path='scratch/variable_labels.json'):
    """Loads labels from a pre-extracted JSON file."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}

def rename_df_to_labels(df, labels_dict):
    """Renames a dataframe columns to their human-readable labels."""
    # Flatten the labels dict if it's nested by dataset
    flat_labels = {}
    if any(isinstance(v, dict) for v in labels_dict.values()):
        for dataset, mapping in labels_dict.items():
            flat_labels.update(mapping)
    else:
        flat_labels = labels_dict
        
    # Clean labels for column names
    clean_mapping = {}
    for k, v in flat_labels.items():
        if k in df.columns:
            clean_v = v.replace(' ', '_').replace('.', '').replace(':', '').replace('-', '_')
            # Keep it reasonable length
            clean_v = clean_v[:50].strip('_')
            clean_mapping[k] = clean_v
            
    return df.rename(columns=clean_mapping)

# Common mappings for quick reference
COMMON_MAP = {
    'RO3': 'Sex',
    'RO5': 'Age',
    'RO6': 'Marital_Status',
    'DB5': 'HH_Debt_Amount',
    'DB6': 'Debt_from_Shopkeeper',
    'DB6A': 'Shopkeeper_Debt_Amount',
    'DB1C': 'Debt_from_Money_Lender',
    'DB2C': 'Largest_Loan_Purpose',
    'NWKNONAG': 'Num_NonAg_Workers',
    'NWKAGLAB': 'Num_AgLab_Workers',
    'NWKSALARY': 'Num_Salary_Workers',
    'NWKBUSINESS': 'Num_Business_Workers',
    'NWKFARM': 'Num_Farm_Workers',
}
