"""
IHDS-II baseline classifier for FRAG_BINARY (Fragile/Distressed vs rest).
Features are disjoint from FFI inputs to avoid label leakage.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

HH_FILE  = Path('DS0002/36151-0002-Data.dta')
FFI_FILE = Path('ihds2_ffi.parquet')

# ---------------------------------------------------------------
# 1. Load FFI labels + non-leaky predictors from DS0002
# ---------------------------------------------------------------
print("Loading FFI labels...")
ffi = pd.read_parquet(FFI_FILE)
print(f"  FFI shape: {ffi.shape}")

# Predictor columns chosen to AVOID overlap with FFI inputs
# FFI inputs were: INCOME, COTOTAL, ASSETS, DB*, NWK*, NPERSONS
PREDICTOR_COLS = [
    # keys (drop after merge)
    'STATEID', 'DISTID', 'PSUID', 'HHID', 'HHSPLITID',
    # household head
    'MHEADAGE', 'FHEADAGE',
    # education of head (years)
    'HHEDUC', 'HHEDUCM', 'HHEDUCF',
    # social/identity
    'ID11', 'ID13', 'GROUPS',
    # geography
    'URBAN2011', 'URBAN4_2011', 'METRO', 'METRO6',
    # housing quality
    'HQ1', 'HQWALL', 'HQROOF', 'HQFLOOR',
    # water / sanitation / fuel
    'WATER', 'SATOILET', 'SAKITCHEN',
    'FU1', 'FULPG',
    # migration history of HH
    'MG1',
]

print("Loading predictors from DS0002...")
X_raw = pd.read_stata(HH_FILE, columns=PREDICTOR_COLS, convert_categoricals=False)
print(f"  Predictors shape: {X_raw.shape}")

# IHDS missing codes
for code in (-9, -8, -7):
    X_raw = X_raw.replace(code, np.nan)

# Build same hh_id used in FFI
hh_keys = ['STATEID', 'DISTID', 'PSUID', 'HHID', 'HHSPLITID']
X_raw['hh_id'] = X_raw[hh_keys].astype(str).agg('-'.join, axis=1)

# ---------------------------------------------------------------
# 2. Merge labels with predictors
# ---------------------------------------------------------------
data = X_raw.merge(
    ffi[['hh_id', 'FRAG_BINARY', 'FRAG_STATE', 'FFI']],
    on='hh_id', how='inner'
)
print(f"  Merged: {data.shape}")
print(f"  Class balance:\n{data['FRAG_BINARY'].value_counts(normalize=True).round(3)}")

# ---------------------------------------------------------------
# 3. Feature matrix
# ---------------------------------------------------------------
y = data['FRAG_BINARY'].values
X = data.drop(columns=['FRAG_BINARY', 'FRAG_STATE', 'FFI', 'hh_id'] + hh_keys)

# Treat categorical-coded columns as categories (one-hot)
CATEGORICAL = ['ID11', 'ID13', 'GROUPS', 'URBAN2011', 'URBAN4_2011',
               'METRO', 'METRO6', 'HQWALL', 'HQROOF', 'HQFLOOR',
               'WATER', 'SATOILET', 'SAKITCHEN', 'FU1', 'FULPG', 'MG1']
CATEGORICAL = [c for c in CATEGORICAL if c in X.columns]
X = pd.get_dummies(X, columns=CATEGORICAL, dummy_na=True, drop_first=True)
print(f"  Feature matrix after one-hot: {X.shape}")

# ---------------------------------------------------------------
# 4. Train/test split
# ---------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

# ---------------------------------------------------------------
# 5. Logistic Regression (with median imputation + scaling)
# ---------------------------------------------------------------
print("\n=== Logistic Regression ===")
logit = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  StandardScaler(with_mean=False)),  # sparse-friendly
    ('clf', LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0)),
])
logit.fit(X_tr, y_tr)
p_logit = logit.predict_proba(X_te)[:, 1]
yhat_logit = (p_logit >= 0.5).astype(int)
print(classification_report(y_te, yhat_logit, digits=3))
print(f"ROC-AUC: {roc_auc_score(y_te, p_logit):.4f}")
print("Confusion matrix:\n", confusion_matrix(y_te, yhat_logit))

# ---------------------------------------------------------------
# 6. Random Forest
# ---------------------------------------------------------------
print("\n=== Random Forest ===")
rf = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(
        n_estimators=300, max_depth=None,
        min_samples_leaf=20, n_jobs=-1, random_state=42)),
])
rf.fit(X_tr, y_tr)
p_rf = rf.predict_proba(X_te)[:, 1]
yhat_rf = (p_rf >= 0.5).astype(int)
print(classification_report(y_te, yhat_rf, digits=3))
print(f"ROC-AUC: {roc_auc_score(y_te, p_rf):.4f}")
print("Confusion matrix:\n", confusion_matrix(y_te, yhat_rf))

# ---------------------------------------------------------------
# 7. Feature importance (RF) — top 20
# ---------------------------------------------------------------
print("\n=== Top 20 features by RF importance ===")
importances = pd.Series(
    rf.named_steps['clf'].feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print(importances.head(20).to_string())

# ---------------------------------------------------------------
# 8. Save predictions and importances for the deck
# ---------------------------------------------------------------
importances.to_csv('rf_feature_importance.csv', header=['importance'])
pd.DataFrame({'y_true': y_te, 'p_logit': p_logit, 'p_rf': p_rf}).to_csv(
    'test_predictions.csv', index=False)
print("\nSaved: rf_feature_importance.csv, test_predictions.csv")