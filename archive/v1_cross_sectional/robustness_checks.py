"""
FFI robustness / sensitivity checks for the report.

Three checks:
  A) PCA-weighted FFI vs equal-weighted: how stable is the binary label?
  B) Alternative quartile cutoff (top tertile vs top half): re-train LightGBM
     on the alternative label, report held-out test AUC.
  C) Geographic out-of-sample: GroupKFold by STATEID, mean / std CV AUC for
     LightGBM (does the classifier generalise across states?).

Writes robustness.json and prints a summary table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    roc_auc_score,
)
from scipy.stats import spearmanr, skew
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

import lightgbm as lgb

HH_FILE = Path("DS0002/36151-0002-Data.dta")
FFI_FILE = Path("ihds2_ffi.parquet")
OUT = Path("robustness.json")
RNG = 42

# ----------------------------------------------------------------------
# 1. Load FFI parquet (already has all six components + FFI + states)
# ----------------------------------------------------------------------
ffi = pd.read_parquet(FFI_FILE)
print(f"FFI rows: {len(ffi)}")

COMP_COLS = [
    "c1_debt_burden", "c2_cons_stress", "c3_asset_deficit",
    "c4_emp_concentration", "c5_dependency", "c6_distress_borrow",
]
comp = ffi[COMP_COLS]
z = (comp - comp.mean()) / comp.std(ddof=0)
# Restrict PCA + comparison to rows with finite values in all components.
finite_mask = np.isfinite(z.values).all(axis=1) & np.isfinite(ffi["FFI"].values)
z_f = z.values[finite_mask]
ffi_eq_f = ffi["FFI"].values[finite_mask]
y_eq = ffi["FRAG_BINARY"].astype(int).values[finite_mask]
print(f"  Rows with all components finite: {finite_mask.sum():,} / {len(ffi):,}")

# ----------------------------------------------------------------------
# A) PCA-weighted FFI vs equal-weighted FFI
# ----------------------------------------------------------------------
pca = PCA(n_components=1).fit(z_f)
ffi_pca_raw = pca.transform(z_f).ravel()
# Align PC1 so it is positively correlated with the equal-weight FFI
sign = 1.0 if np.corrcoef(ffi_pca_raw, ffi_eq_f)[0, 1] >= 0 else -1.0
ffi_pca_raw *= sign
ffi_pca = (ffi_pca_raw - ffi_pca_raw.mean()) / ffi_pca_raw.std(ddof=0)

# Binary labels under each scheme (top 50% under each respective score)
y_pca = (ffi_pca > np.median(ffi_pca)).astype(int)

kappa_pca = cohen_kappa_score(y_eq, y_pca)
agree_pca = (y_eq == y_pca).mean()
pearson = float(np.corrcoef(ffi_eq_f, ffi_pca)[0, 1])
spearman = float(spearmanr(ffi_eq_f, ffi_pca).statistic)
skew_eq  = float(skew(ffi_eq_f))
skew_pca = float(skew(ffi_pca))

print(f"\n[A] PCA weights — explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}")
print(f"    PCA loadings (sign-aligned):")
loadings = pd.Series(pca.components_[0] * sign, index=COMP_COLS)
print(loadings.round(3).to_string())
print(f"    Pearson(equal, PCA)  = {pearson:.4f}")
print(f"    Spearman(equal, PCA) = {spearman:.4f}")
print(f"    Skewness: equal-FFI = {skew_eq:.2f}, PCA-FFI = {skew_pca:.2f}")
print(f"    Binary-label agreement = {agree_pca:.4f}  (kappa = {kappa_pca:.4f})")

# ----------------------------------------------------------------------
# B) Alternative cutoff: top tertile vs top half (re-train LightGBM)
# ----------------------------------------------------------------------
# Load predictors (same as training_full.py)
PREDICTOR_COLS = [
    "STATEID", "DISTID", "PSUID", "HHID", "HHSPLITID",
    "MHEADAGE", "FHEADAGE",
    "HHEDUC", "HHEDUCM", "HHEDUCF",
    "ID11", "ID13", "GROUPS",
    "URBAN2011", "URBAN4_2011", "METRO", "METRO6",
    "HQ1", "HQWALL", "HQROOF", "HQFLOOR",
    "WATER", "SATOILET", "SAKITCHEN",
    "FU1", "FULPG",
    "MG1",
]
print("\nLoading predictors...")
X_raw = pd.read_stata(HH_FILE, columns=PREDICTOR_COLS, convert_categoricals=False)
for code in (-9, -8, -7):
    X_raw = X_raw.replace(code, np.nan)
HH_KEYS = ["STATEID", "DISTID", "PSUID", "HHID", "HHSPLITID"]
X_raw["hh_id"] = X_raw[HH_KEYS].astype(str).agg("-".join, axis=1)

# Add alternative labels onto the FFI table
ffi_aug = ffi.copy()
ffi_aug["y_top_tertile"] = (ffi_aug["FFI"] >= ffi_aug["FFI"].quantile(2/3)).astype(int)

data = X_raw.merge(
    ffi_aug[["hh_id", "FRAG_BINARY", "y_top_tertile", "STATEID"]].rename(
        columns={"STATEID": "STATEID_lbl"}
    ),
    on="hh_id", how="inner",
)

CATEGORICAL = [
    "ID11", "ID13", "GROUPS",
    "URBAN2011", "URBAN4_2011", "METRO", "METRO6",
    "HQWALL", "HQROOF", "HQFLOOR",
    "WATER", "SATOILET", "SAKITCHEN", "FU1", "FULPG", "MG1",
]
X_df = data.drop(columns=["FRAG_BINARY", "y_top_tertile", "hh_id"] + HH_KEYS + ["STATEID_lbl"])
X_df = pd.get_dummies(X_df, columns=[c for c in CATEGORICAL if c in X_df.columns],
                      dummy_na=True, drop_first=True).astype(np.float32)

# Use STATEID from predictor frame for geographic split
state_arr = data["STATEID_lbl"].astype(int).values

def fit_lgbm_eval(y, X, name):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RNG, stratify=y
    )
    Xtr2, Xval, ytr2, yval = train_test_split(
        Xtr, ytr, test_size=0.15, random_state=RNG, stratify=ytr
    )
    m = lgb.LGBMClassifier(
        objective="binary", n_estimators=1500, learning_rate=0.05,
        num_leaves=31, min_child_samples=50,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=RNG, n_jobs=-1, verbose=-1,
    )
    m.fit(Xtr2, ytr2, eval_set=[(Xval, yval)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    ap = average_precision_score(yte, p)
    print(f"  {name}: AUC={auc:.4f}  AP={ap:.4f}  prevalence={y.mean():.3f}")
    return {"auc": float(auc), "ap": float(ap), "prevalence": float(y.mean())}

print("\n[B] Alternative binarization cutoff (LightGBM):")
res_b = {
    "top_half_equal_weights":     fit_lgbm_eval(data["FRAG_BINARY"].values,    X_df, "top-half (baseline)"),
    "top_tertile_equal_weights":  fit_lgbm_eval(data["y_top_tertile"].values, X_df, "top-tertile"),
}

# ----------------------------------------------------------------------
# C) Geographic out-of-sample: GroupKFold by STATEID
# ----------------------------------------------------------------------
print("\n[C] Geographic out-of-sample (GroupKFold by STATEID, 5 folds):")
y = data["FRAG_BINARY"].values
gkf = GroupKFold(n_splits=5)
aucs, aps, sizes = [], [], []
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_df, y, groups=state_arr), 1):
    Xtr, Xva = X_df.iloc[tr_idx], X_df.iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    held_states = sorted(set(state_arr[va_idx].tolist()))
    m = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=50,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=RNG, n_jobs=-1, verbose=-1,
    )
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(yva, p)
    ap = average_precision_score(yva, p)
    aucs.append(auc); aps.append(ap); sizes.append(len(yva))
    print(f"  fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  n_val={len(yva):,}  states_held_out={held_states}")
print(f"  mean AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"  mean AP  = {np.mean(aps):.4f} ± {np.std(aps):.4f}")

# ----------------------------------------------------------------------
# Save summary
# ----------------------------------------------------------------------
summary = {
    "A_pca_vs_equal_weights": {
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pca_loadings_sign_aligned": loadings.round(4).to_dict(),
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "skew_equal_weight_ffi": skew_eq,
        "skew_pca_ffi": skew_pca,
        "binary_label_agreement": float(agree_pca),
        "cohens_kappa": float(kappa_pca),
        "n_rows_used": int(finite_mask.sum()),
    },
    "B_alt_cutoff_lgbm": res_b,
    "C_geographic_oos_lgbm": {
        "fold_aucs": [float(a) for a in aucs],
        "fold_aps":  [float(a) for a in aps],
        "fold_sizes": [int(s) for s in sizes],
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "ap_mean": float(np.mean(aps)),
        "ap_std": float(np.std(aps)),
    },
}
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nWrote {OUT}")
