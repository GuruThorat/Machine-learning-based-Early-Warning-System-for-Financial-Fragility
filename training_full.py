"""
IHDS-II full training pipeline for FRAG_BINARY.

Adds, on top of training_baseline.py:
  * 5-fold stratified CV on the training split (LR, RF, LightGBM)
  * Light hyperparameter tuning (LR C, LightGBM num_leaves/min_child_samples)
  * LightGBM as the strong non-linear baseline (with early stopping)
  * Probability calibration (isotonic via CalibratedClassifierCV)
  * SHAP-based feature importance for the LightGBM model
  * Permutation importance for the random forest (as cross-check)
  * Persists metrics_full.json and regenerates figures 06-09 + 11 (calibration) + 12 (SHAP)

Predictors remain disjoint from FFI inputs (no leakage).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import shap

HH_FILE = Path("DS0002/36151-0002-Data.dta")
FFI_FILE = Path("ihds2_ffi.parquet")
FIG_DIR = Path("figures")
OUT_METRICS = Path("metrics_full.json")
OUT_IMP_RF = Path("rf_feature_importance.csv")
OUT_IMP_PERM = Path("rf_permutation_importance.csv")
OUT_IMP_SHAP = Path("lgbm_shap_importance.csv")
OUT_TEST = Path("test_predictions.csv")

RNG = 42

# ------------------------------------------------------------------
# 1. Load FFI labels and disjoint predictors
# ------------------------------------------------------------------
print("Loading FFI labels...")
ffi = pd.read_parquet(FFI_FILE)
print(f"  FFI shape: {ffi.shape}")

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

print("Loading predictors from DS0002...")
X_raw = pd.read_stata(HH_FILE, columns=PREDICTOR_COLS, convert_categoricals=False)
print(f"  Predictors shape: {X_raw.shape}")

for code in (-9, -8, -7):
    X_raw = X_raw.replace(code, np.nan)

HH_KEYS = ["STATEID", "DISTID", "PSUID", "HHID", "HHSPLITID"]
X_raw["hh_id"] = X_raw[HH_KEYS].astype(str).agg("-".join, axis=1)

data = X_raw.merge(
    ffi[["hh_id", "FRAG_BINARY", "FRAG_STATE", "FFI"]],
    on="hh_id", how="inner",
)
print(f"  Merged: {data.shape}")
class_balance = data["FRAG_BINARY"].value_counts(normalize=True).round(4)
print(f"  Class balance:\n{class_balance}")

y = data["FRAG_BINARY"].astype(int).values
X_df = data.drop(columns=["FRAG_BINARY", "FRAG_STATE", "FFI", "hh_id"] + HH_KEYS)

CATEGORICAL = [
    "ID11", "ID13", "GROUPS",
    "URBAN2011", "URBAN4_2011", "METRO", "METRO6",
    "HQWALL", "HQROOF", "HQFLOOR",
    "WATER", "SATOILET", "SAKITCHEN", "FU1", "FULPG", "MG1",
]
CATEGORICAL = [c for c in CATEGORICAL if c in X_df.columns]
X_df = pd.get_dummies(X_df, columns=CATEGORICAL, dummy_na=True, drop_first=True)
X_df = X_df.astype(np.float32)
print(f"  Feature matrix after one-hot: {X_df.shape}")

# ------------------------------------------------------------------
# 2. Held-out 80/20 stratified split
# ------------------------------------------------------------------
X_tr_df, X_te_df, y_tr, y_te = train_test_split(
    X_df, y, test_size=0.2, random_state=RNG, stratify=y
)
print(f"  Train: {X_tr_df.shape}, Test: {X_te_df.shape}")

# ------------------------------------------------------------------
# 3. Model factories (untuned base pipelines; tuning loops below)
# ------------------------------------------------------------------
def make_logit(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(C=C, max_iter=2000, n_jobs=-1, solver="lbfgs")),
    ])


def make_rf() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None,
            min_samples_leaf=20, n_jobs=-1, random_state=RNG,
        )),
    ])


def make_lgbm(num_leaves: int = 63, min_child_samples: int = 50,
              learning_rate: float = 0.05, n_estimators: int = 1500) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RNG,
        n_jobs=-1,
        verbose=-1,
    )


# ------------------------------------------------------------------
# 4. 5-fold stratified CV on training (AUC + AP)
# ------------------------------------------------------------------
def cv_score(model_factory, X, y, name: str, supports_es: bool = False) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    aucs, aps, briers = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = model_factory()
        if supports_es:
            # LightGBM with early stopping on the validation fold
            model.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        aucs.append(roc_auc_score(yva, p))
        aps.append(average_precision_score(yva, p))
        briers.append(brier_score_loss(yva, p))
        print(f"  [{name}] fold {fold}: AUC={aucs[-1]:.4f} AP={aps[-1]:.4f} Brier={briers[-1]:.4f}")
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "ap_mean":  float(np.mean(aps)),  "ap_std":  float(np.std(aps)),
        "brier_mean": float(np.mean(briers)), "brier_std": float(np.std(briers)),
        "folds_auc": [float(a) for a in aucs],
    }


print("\n=== 5-fold CV: Logistic Regression (C grid) ===")
lr_cv = {}
for C in (0.1, 1.0, 5.0):
    print(f"-- C={C} --")
    lr_cv[C] = cv_score(lambda C=C: make_logit(C=C), X_tr_df, y_tr, name=f"LR(C={C})")
best_C = max(lr_cv, key=lambda c: lr_cv[c]["auc_mean"])
print(f"  best LR C = {best_C} (CV AUC = {lr_cv[best_C]['auc_mean']:.4f})")

print("\n=== 5-fold CV: Random Forest ===")
rf_cv = cv_score(make_rf, X_tr_df, y_tr, name="RF")

print("\n=== 5-fold CV: LightGBM (small grid) ===")
lgbm_cv = {}
grid = [(31, 50), (63, 50), (63, 100), (127, 100)]
for nl, mcs in grid:
    key = f"nl={nl}_mcs={mcs}"
    print(f"-- {key} --")
    lgbm_cv[key] = cv_score(
        lambda nl=nl, mcs=mcs: make_lgbm(num_leaves=nl, min_child_samples=mcs),
        X_tr_df, y_tr, name=f"LGBM({key})", supports_es=True,
    )
best_lgbm_key = max(lgbm_cv, key=lambda k: lgbm_cv[k]["auc_mean"])
nl_best, mcs_best = grid[list(lgbm_cv).index(best_lgbm_key)]
print(f"  best LGBM = {best_lgbm_key} (CV AUC = {lgbm_cv[best_lgbm_key]['auc_mean']:.4f})")

# ------------------------------------------------------------------
# 5. Refit best configs on full training; evaluate on held-out test
# ------------------------------------------------------------------
print("\n=== Final fits on full training set ===")
logit = make_logit(C=best_C).fit(X_tr_df, y_tr)
rf = make_rf().fit(X_tr_df, y_tr)

# LightGBM gets a small internal val for ES on the refit
X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_tr_df, y_tr, test_size=0.15, random_state=RNG, stratify=y_tr
)
lgbm_uncal = make_lgbm(num_leaves=nl_best, min_child_samples=mcs_best)
lgbm_uncal.fit(
    X_tr2, y_tr2,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
best_iter = lgbm_uncal.best_iteration_ or lgbm_uncal.n_estimators
print(f"  LGBM best_iteration = {best_iter}")

# Refit on full training with the picked iteration count
lgbm = make_lgbm(num_leaves=nl_best, min_child_samples=mcs_best, n_estimators=best_iter)
lgbm.fit(X_tr_df, y_tr)

# Isotonic calibration for the best (uncalibrated) LightGBM
print("  Isotonic-calibrating LightGBM with 5-fold CV...")
lgbm_cal = CalibratedClassifierCV(
    lgb.LGBMClassifier(
        objective="binary", n_estimators=best_iter, learning_rate=0.05,
        num_leaves=nl_best, min_child_samples=mcs_best,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=RNG, n_jobs=-1, verbose=-1,
    ),
    cv=5, method="isotonic",
)
lgbm_cal.fit(X_tr_df, y_tr)


def evaluate(model, name: str) -> dict:
    p = model.predict_proba(X_te_df)[:, 1]
    yhat = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_te, yhat, average="binary", zero_division=0)
    auc = roc_auc_score(y_te, p)
    ap = average_precision_score(y_te, p)
    brier = brier_score_loss(y_te, p)
    cm = confusion_matrix(y_te, yhat).tolist()
    print(f"\n=== {name} (test) ===")
    print(classification_report(y_te, yhat, digits=3))
    print(f"ROC-AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}")
    print(f"Confusion matrix: {cm}")
    return {
        "precision": float(pr), "recall": float(rc), "f1": float(f1),
        "roc_auc": float(auc), "ap": float(ap), "brier": float(brier),
        "confusion_matrix": cm, "proba": p,
    }


res_logit = evaluate(logit, "Logistic Regression")
res_rf = evaluate(rf, "Random Forest")
res_lgbm = evaluate(lgbm, "LightGBM (uncalibrated)")
res_lgbm_cal = evaluate(lgbm_cal, "LightGBM (isotonic-calibrated)")

# ------------------------------------------------------------------
# 6. Figures
# ------------------------------------------------------------------
FIG_DIR.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10})

# 6a. ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, res in [("Logistic Regression", res_logit),
                   ("Random Forest", res_rf),
                   ("LightGBM", res_lgbm_cal)]:
    fpr, tpr, _ = roc_curve(y_te, res["proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Receiver Operating Characteristic — test set")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_DIR / "06_roc_curves.png"); plt.close(fig)

# 6b. PR curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, res in [("Logistic Regression", res_logit),
                   ("Random Forest", res_rf),
                   ("LightGBM", res_lgbm_cal)]:
    pr, rc, _ = precision_recall_curve(y_te, res["proba"])
    ax.plot(rc, pr, label=f"{name} (AP = {res['ap']:.3f})")
base = y_te.mean()
ax.axhline(base, color="k", ls="--", lw=0.8, label=f"baseline = {base:.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision–Recall — test set")
ax.legend(loc="lower left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_DIR / "07_pr_curves.png"); plt.close(fig)

# 6c. Confusion matrices at 0.5
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, res) in zip(axes, [("Logistic Regression", res_logit),
                                   ("Random Forest", res_rf),
                                   ("LightGBM (calibrated)", res_lgbm_cal)]):
    cm = np.array(res["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not fragile", "Fragile"])
    ax.set_yticklabels(["Not fragile", "Fragile"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(name)
fig.suptitle("Confusion matrices at threshold = 0.5", y=1.02)
fig.tight_layout(); fig.savefig(FIG_DIR / "08_confusion_matrices.png", bbox_inches="tight")
plt.close(fig)

# 6d. RF impurity-based importance (kept for reference)
rf_imp = pd.Series(
    rf.named_steps["clf"].feature_importances_, index=X_df.columns
).sort_values(ascending=False)
rf_imp.to_csv(OUT_IMP_RF, header=["importance"])

top20_rf = rf_imp.head(20)[::-1]
fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(top20_rf.index, top20_rf.values, color="steelblue")
ax.set_xlabel("Mean decrease in impurity")
ax.set_title("Random Forest impurity-based importance (top 20)")
fig.tight_layout(); fig.savefig(FIG_DIR / "09_rf_feature_importance_top20.png")
plt.close(fig)

# 6e. Calibration plot (raw vs isotonic)
fig, ax = plt.subplots(figsize=(6, 5))
for name, res in [("LightGBM raw", res_lgbm), ("LightGBM isotonic", res_lgbm_cal)]:
    frac_pos, mean_pred = calibration_curve(y_te, res["proba"], n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", label=f"{name} (Brier = {res['brier']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfectly calibrated")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Empirical fraction positive")
ax.set_title("Reliability diagram — LightGBM")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_DIR / "11_calibration.png"); plt.close(fig)

# 6f. SHAP for LightGBM (on a sample of test rows for speed)
print("\n=== SHAP (LightGBM) ===")
shap_sample = X_te_df.sample(n=min(3000, len(X_te_df)), random_state=RNG)
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(shap_sample)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # binary: take positive class
mean_abs = np.abs(shap_values).mean(axis=0)
shap_imp = pd.Series(mean_abs, index=X_df.columns).sort_values(ascending=False)
shap_imp.to_csv(OUT_IMP_SHAP, header=["mean_abs_shap"])

top20_shap = shap_imp.head(20)[::-1]
fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(top20_shap.index, top20_shap.values, color="darkorange")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("LightGBM SHAP importance (top 20)")
fig.tight_layout(); fig.savefig(FIG_DIR / "12_shap_top20.png"); plt.close(fig)

# Beeswarm summary plot (more informative than the bar chart)
plt.figure(figsize=(8, 8))
shap.summary_plot(shap_values, shap_sample, show=False, max_display=20)
plt.tight_layout(); plt.savefig(FIG_DIR / "13_shap_beeswarm.png"); plt.close()

# 6g. Permutation importance for the RF (cross-check; on a sample for speed)
print("\n=== Permutation importance (RF, sample of 4000 test rows) ===")
perm_sample_idx = np.random.RandomState(RNG).choice(
    len(X_te_df), size=min(4000, len(X_te_df)), replace=False
)
perm = permutation_importance(
    rf, X_te_df.iloc[perm_sample_idx], y_te[perm_sample_idx],
    n_repeats=5, random_state=RNG, n_jobs=-1, scoring="roc_auc",
)
perm_imp = pd.Series(perm.importances_mean, index=X_df.columns).sort_values(ascending=False)
perm_imp.to_csv(OUT_IMP_PERM, header=["perm_importance"])
print(perm_imp.head(15).to_string())

# ------------------------------------------------------------------
# 7. Persist everything
# ------------------------------------------------------------------
def strip_proba(d):
    return {k: v for k, v in d.items() if k != "proba"}

metrics = {
    "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    "class_balance": {str(k): float(v) for k, v in class_balance.items()},
    "n_features": int(X_df.shape[1]),
    "cv": {
        "logistic_regression": {str(k): v for k, v in lr_cv.items()},
        "logistic_regression_best_C": float(best_C),
        "random_forest": rf_cv,
        "lightgbm_grid": lgbm_cv,
        "lightgbm_best": best_lgbm_key,
        "lightgbm_best_iteration": int(best_iter),
    },
    "test": {
        "logistic_regression": strip_proba(res_logit),
        "random_forest": strip_proba(res_rf),
        "lightgbm_uncalibrated": strip_proba(res_lgbm),
        "lightgbm_calibrated": strip_proba(res_lgbm_cal),
    },
    "top_features": {
        "rf_impurity_top10": rf_imp.head(10).round(4).to_dict(),
        "rf_permutation_top10": perm_imp.head(10).round(4).to_dict(),
        "lgbm_shap_top10": shap_imp.head(10).round(4).to_dict(),
    },
}

with open(OUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nWrote {OUT_METRICS}")

pd.DataFrame({
    "y_true":  y_te,
    "p_logit": res_logit["proba"],
    "p_rf":    res_rf["proba"],
    "p_lgbm":  res_lgbm["proba"],
    "p_lgbm_cal": res_lgbm_cal["proba"],
}).to_csv(OUT_TEST, index=False)
print(f"Wrote {OUT_TEST}")
print("Done.")
