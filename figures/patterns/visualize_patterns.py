import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ihds_utils import load_variable_labels, COMMON_MAP

def create_visualizations(data_path='ihds_preprocessed.parquet', output_dir='figures/patterns'):
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use consistent styling
    plt.style.use('ggplot')
    
    # 1. Age and Sex Distribution (Population Pyramid)
    print("Generating Population Pyramid...")
    if 'RO5' in df.columns and 'RO3' in df.columns:
        # Filter valid ages (0-100) and sex (1, 2)
        pyramid_df = df[(df['RO5'] >= 0) & (df['RO5'] <= 100) & (df['RO3'].isin([1, 2]))]
        
        ages = np.arange(0, 101, 5)
        males = []
        females = []
        
        for age in ages:
            m = len(pyramid_df[(pyramid_df['RO5'] >= age) & (pyramid_df['RO5'] < age + 5) & (pyramid_df['RO3'] == 1)])
            f = len(pyramid_df[(pyramid_df['RO5'] >= age) & (pyramid_df['RO5'] < age + 5) & (pyramid_df['RO3'] == 2)])
            males.append(-m) # Negative for left side
            females.append(f)
            
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(ages, males, height=4, color='skyblue', label='Male')
        ax.barh(ages, females, height=4, color='lightpink', label='Female')
        
        ax.set_xlabel('Population Count')
        ax.set_ylabel('Age Group')
        ax.set_title('IHDS-II Population Pyramid')
        ax.legend()
        
        # Fix x-axis labels to show positive values on both sides
        ticks = ax.get_xticks()
        ax.set_xticklabels([str(abs(int(tick))) for tick in ticks])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/population_pyramid.png')
        plt.close()

    # 2. Income Distribution by Urban/Rural
    print("Generating Income Distribution...")
    income_col = 'INCOME_x' if 'INCOME_x' in df.columns else 'INCOME'
    urban_col = 'URBAN2011'

    if income_col in df.columns and urban_col in df.columns:
        # One row per household (income is repeated per person otherwise)
        hh_df = df.drop_duplicates(subset=['STATEID', 'DISTID', 'PSUID', 'HHID', 'HHSPLITID'])
        n_total = len(hh_df)

        # Drop non-positive (net income can be ≤ 0) and implausibly low (< ₹1000/yr)
        # reports, which would otherwise create a misleading left tail under log10.
        income_floor = 1000
        hh_df = hh_df[hh_df[income_col] >= income_floor]
        n_kept = len(hh_df)

        plt.figure(figsize=(10, 6))
        for val, label in [(0, 'Rural'), (1, 'Urban')]:
            subset = hh_df[hh_df[urban_col] == val]
            if not subset.empty:
                plt.hist(np.log10(subset[income_col]), bins=50, alpha=0.5, label=label, density=True)

        plt.xlabel('Annual Household Income (Log10 scale, ₹)')
        plt.ylabel('Density')
        plt.title(
            f'Income Distribution: Urban vs Rural\n'
            f'(n={n_kept:,} of {n_total:,} HHs; {n_total - n_kept:,} dropped with income < ₹{income_floor:,})'
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/income_distribution.png')
        plt.close()

    # 3. Education Levels
    # Use ED6 (highest standard completed, 0–16), NOT ED4 which is a literacy
    # yes/no flag and collapses everyone into two bars.
    print("Generating Education Profile...")
    if 'ED6' in df.columns:
        adults = df[df['RO5'] >= 18]
        ed_counts = adults['ED6'].dropna().astype(int).value_counts().sort_index()
        ed_counts = ed_counts.reindex(range(0, 17), fill_value=0)

        plt.figure(figsize=(12, 6))
        ed_counts.plot(kind='bar', color='teal')
        plt.xlabel('Highest Standard Completed (ED6)')
        plt.ylabel('Number of Individuals (Age 18+)')
        plt.title('Education Profile of Adults in IHDS-II')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/education_levels.png')
        plt.close()

    # 4. Household Size Distribution
    print("Generating Household Size Distribution...")
    npersons_col = 'NPERSONS_x' if 'NPERSONS_x' in df.columns else 'NPERSONS'
    if npersons_col in df.columns:
        hh_df = df.drop_duplicates(subset=['STATEID', 'DISTID', 'PSUID', 'HHID', 'HHSPLITID'])
        
        plt.figure(figsize=(10, 6))
        hh_df[npersons_col].value_counts().sort_index().plot(kind='bar', color='orange')
        plt.xlabel('Number of Persons in Household')
        plt.ylabel('Number of Households')
        plt.title('Household Size Distribution')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/household_size.png')
        plt.close()

    print(f"✅ All visualizations saved to {output_dir}")

if __name__ == "__main__":
    create_visualizations()
