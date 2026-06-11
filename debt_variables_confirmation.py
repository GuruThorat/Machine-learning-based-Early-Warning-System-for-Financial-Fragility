import pandas as pd
df = pd.read_stata('DS0002/36151-0002-Data.dta', convert_categoricals=False)

# Inspect the debt block
debt_cols = [c for c in df.columns if c.startswith('DB')]
print("Debt columns:", debt_cols)
print("\nDescribe:")
print(df[debt_cols].describe().T[['count', 'mean', 'min', 'max']])

# Also check loan source question
print("\nDB2A-G value counts (first non-null sample):")
for c in ['DB2A', 'DB2B', 'DB2C', 'DB2D', 'DB2E', 'DB2F', 'DB2G']:
    if c in df.columns:
        print(c, df[c].value_counts(dropna=False).head().to_dict())