import pandas as pd
import numpy as np
import os
import json

def get_variable_labels(file_path):
    """Extracts variable labels from a Stata file."""
    try:
        reader = pd.io.stata.StataReader(file_path)
        return reader.variable_labels()
    except Exception as e:
        print(f"Warning: Could not extract labels from {file_path}: {e}")
        return {}

def preprocess_ihds(include_women=False, rename_to_labels=False):
    """
    Preprocesses IHDS-II data by merging Individual, Household, and optionally Eligible Women datasets.
    """
    print("Starting IHDS-II Preprocessing...")
    
    # Define paths
    ds0001_path = 'DS0001/36151-0001-Data.dta'
    ds0002_path = 'DS0002/36151-0002-Data.dta'
    ds0003_path = 'DS0003/36151-0003-Data.dta'
    
    # 1. Load Individual Data (Base)
    print(f"Loading Individual data (DS0001)...")
    df_ind = pd.read_stata(ds0001_path, convert_categoricals=False)
    ind_labels = get_variable_labels(ds0001_path)
    print(f"Loaded Individual data: {df_ind.shape}")
    
    # 2. Load Household Data
    print(f"Loading Household data (DS0002)...")
    df_hh = pd.read_stata(ds0002_path, convert_categoricals=False)
    hh_labels = get_variable_labels(ds0002_path)
    # Remove duplicate ID columns from HH before merge to avoid many suffixes
    hh_cols_to_drop = ['SURVEY', 'IDPSU', 'IDHH', 'WT', 'FWT', 'DIST01', 'DISTRICT', 'URBAN2011', 'URBAN4_2011', 'METRO', 'METRO6']
    df_hh = df_hh.drop(columns=[c for c in hh_cols_to_drop if c in df_hh.columns])
    print(f"Loaded Household data: {df_hh.shape}")
    
    # 3. Merge Keys
    # Note: PERSONID is only in Individual and Women datasets
    hh_merge_keys = ['STATEID', 'DISTID', 'PSUID', 'HHID', 'HHSPLITID']
    
    # 4. Merge Individual + Household
    print(f"Merging Individual and Household data...")
    df_merged = pd.merge(df_ind, df_hh, on=hh_merge_keys, how='left')
    print(f"Merged Base shape: {df_merged.shape}")
    
    # 5. Optional: Merge Eligible Women (DS0003)
    if include_women and os.path.exists(ds0003_path):
        print(f"Loading Eligible Women data (DS0003)...")
        df_women = pd.read_stata(ds0003_path, convert_categoricals=False)
        women_labels = get_variable_labels(ds0003_path)
        # Person-level merge keys
        women_merge_keys = hh_merge_keys + ['PERSONID']
        
        # Drop redundant columns from women dataset
        women_cols_to_drop = ['SURVEY', 'IDPSU', 'IDHH', 'IDPERSON', 'WT', 'FWT', 'DIST01', 'DISTRICT']
        df_women = df_women.drop(columns=[c for c in women_cols_to_drop if c in df_women.columns])
        
        print(f"Merging Women data...")
        df_merged = pd.merge(df_merged, df_women, on=women_merge_keys, how='left', suffixes=('', '_women'))
        print(f"Final Merged shape: {df_merged.shape}")

    # 6. Handle Missing Values
    print("Recoding missing values...")
    # IHDS-II standard missing codes
    # Caution: This replaces all occurrences of -9, -8, -7. Ensure these aren't valid for any needed var.
    for val in [-9, -8, -7]:
        df_merged.replace(val, np.nan, inplace=True)
    
    # Dataset-specific fix from supplemental syntax
    if 'MB21B' in df_merged.columns:
        df_merged.loc[df_merged['MB21B'] == 8, 'MB21B'] = np.nan

    # 7. Create Unique Identifiers
    print("Creating unique IDs...")
    df_merged['hh_unique_id'] = df_merged[hh_merge_keys].astype(str).agg('-'.join, axis=1)
    df_merged['person_unique_id'] = df_merged[hh_merge_keys + ['PERSONID']].astype(str).agg('-'.join, axis=1)
    
    # 8. Rename to Labels (Optional)
    if rename_to_labels:
        print("Renaming columns to human-readable labels...")
        all_labels = {**ind_labels, **hh_labels}
        if include_women:
            all_labels.update(women_labels)
        
        # Clean labels to be valid column names (no spaces, special chars)
        clean_labels = {k: v.replace(' ', '_').replace('.', '').replace(':', '').replace('-', '_')[:50] 
                        for k, v in all_labels.items()}
        df_merged.rename(columns=clean_labels, inplace=True)
    
    # 9. Export
    output_path = 'ihds_preprocessed.parquet' # Parquet is much faster/smaller for this size
    try:
        import pyarrow
        print(f"Saving to Parquet: {output_path}...")
        df_merged.to_parquet(output_path, index=False)
    except ImportError:
        output_path = 'ihds_preprocessed.csv'
        print(f"Pyarrow not found. Saving to CSV: {output_path}...")
        df_merged.to_csv(output_path, index=False)
    
    print("✨ Preprocessing Successfully Completed!")
    return df_merged

if __name__ == "__main__":
    # Change include_women=True if you need the fertility/women-specific variables
    df = preprocess_ihds(include_women=False)

