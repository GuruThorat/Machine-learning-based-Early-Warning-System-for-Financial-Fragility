"""
IHDS-II Financial Fragility Index (FFI) - household-level construction.
Reads DS0002 directly (household file), computes six FFI components,
z-standardizes, aggregates with equal weights, assigns fragility states.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HH_FILE = Path('DS0002/36151-0002-Data.dta')
OUT_FILE = Path('ihds2_ffi.parquet')

# ---------------------------------------------------------------
# 1. Load household file (only required columns to avoid fragmentation)
# ---------------------------------------------------------------
print("Loading household file...")
REQUIRED_COLS = [
    'STATEID', 'DISTID', 'PSUID', 'HHID', 'HHSPLITID', # keys
    'INCOME', 'COTOTAL', 'ASSETS', 'NPERSONS', 'URBAN2011', # financials
    'NWKNONAG', 'NWKAGLAB', 'NWKSALARY', 'NWKBUSINESS', 'NWKFARM', # employment
    'DB5', 'DB6', 'DB6A', 'DB1C', 'DB2C' # debt
]
df = pd.read_stata(HH_FILE, columns=REQUIRED_COLS, convert_categoricals=False)
print(f"  shape: {df.shape}")  # expect ~ (42152, 20)


# ---------------------------------------------------------------
# 2. Targeted missing-code handling
# ---------------------------------------------------------------
# IHDS-II uses -9, -8, -7 for various types of missingness/non-response
for code in (-9, -8, -7):
    df = df.replace(code, np.nan)


# ---------------------------------------------------------------
# 3. Build household unique ID
# ---------------------------------------------------------------
hh_keys = ['STATEID', 'DISTID', 'PSUID', 'HHID', 'HHSPLITID']
df['hh_id'] = df[hh_keys].astype(str).agg('-'.join, axis=1)

# ---------------------------------------------------------------
# 4. Winsorize monetary variables at 1%/99% to kill tail pollution
# ---------------------------------------------------------------
def winsorize(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

for c in ['INCOME', 'COTOTAL']:
    df[c] = winsorize(df[c])

# ---------------------------------------------------------------
# 5. FFI components  (each oriented so HIGHER = MORE FRAGILE)
# ---------------------------------------------------------------
eps = 1.0  # guard against divide-by-zero

# Total outstanding debt = Household debt (DB5) + Shopkeeper debt (DB6A)
df['debt_total'] = df['DB5'].fillna(0) + df['DB6A'].fillna(0)

comp = pd.DataFrame(index=df.index)
comp['c1_debt_burden']  =  df['debt_total'] / (df['INCOME'].abs() + eps)
comp['c2_cons_stress']  =  df['COTOTAL']    / (df['INCOME'].abs() + eps)
comp['c3_asset_deficit'] = -np.log1p(df['ASSETS'].clip(lower=0))

# Employment concentration (Herfindahl on the five worker-count buckets)
work_cols = ['NWKNONAG', 'NWKAGLAB', 'NWKSALARY', 'NWKBUSINESS', 'NWKFARM']
w = df[work_cols].fillna(0).clip(lower=0)
w_total = w.sum(axis=1).replace(0, np.nan)
shares = w.div(w_total, axis=0)
comp['c4_emp_concentration'] = (shares ** 2).sum(axis=1)   # 1 = single-source

# Dependency pressure
earners    = w_total.fillna(0)
non_earners = (df['NPERSONS'] - earners).clip(lower=0)
comp['c5_dependency'] = non_earners / (earners + 1.0)

# Distress-borrowing flag: 1 if purpose is Consumption(6) or Medical(11), 
# or if borrowing from Shopkeeper(DB6) or Money Lender(DB1C)
comp['c6_distress_borrow'] = (
    df['DB2C'].isin([6, 11]) | 
    (df['DB6'] == 1) | 
    (df['DB1C'] == 1)
).astype(float)


# ---------------------------------------------------------------
# 6. Z-score standardization and FFI aggregation (equal weights)
# ---------------------------------------------------------------
z = (comp - comp.mean()) / comp.std(ddof=0)
weights = np.ones(z.shape[1]) / z.shape[1]
df['FFI'] = z.values @ weights

# Join components back to main df for export
df = pd.concat([df, comp], axis=1)

# ---------------------------------------------------------------
# 7. Quartile-based fragility states
# ---------------------------------------------------------------
q = df['FFI'].quantile([0.25, 0.50, 0.75]).values

df['FRAG_STATE'] = np.select(
    [df['FFI'] <= q[0], df['FFI'] <= q[1], df['FFI'] <= q[2]],
    ['Stable', 'Stretched', 'Fragile'],
    default='Distressed',
)
df['FRAG_BINARY'] = df['FRAG_STATE'].isin(['Fragile', 'Distressed']).astype(int)

# ---------------------------------------------------------------
# 8. Quick summary
# ---------------------------------------------------------------
print("\nFFI components (summary):")
print(comp.describe().T[['count', 'mean', 'std', 'min', 'max']])

print("\nFFI distribution:")
print(df['FFI'].describe())

print("\nFragility states:")
print(df['FRAG_STATE'].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------
keep = ['hh_id'] + hh_keys + [
    'INCOME', 'COTOTAL', 'ASSETS', 'NPERSONS', 'URBAN2011',
    'debt_total',
] + list(comp.columns) + ['FFI', 'FRAG_STATE', 'FRAG_BINARY']
df[keep].to_parquet(OUT_FILE, index=False)
print(f"\nSaved: {OUT_FILE}")