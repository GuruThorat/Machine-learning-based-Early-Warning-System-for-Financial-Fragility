"""
make_visuals.py
---------------
Generates presentation-grade figures from the IHDS-II Financial Fragility
pipeline outputs:
  - ihds2_ffi.parquet              (from build_ffi.py)
  - test_predictions.csv           (from training_baseline.py)
  - rf_feature_importance.csv      (from training_baseline.py)

All figures are written to ./figures/ as 150-dpi PNGs, ready for PPT.
Run from the same folder as the three input files.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, average_precision_score
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
FFI_FILE  = Path("ihds2_ffi.parquet")
PRED_FILE = Path("test_predictions.csv")
IMP_FILE  = Path("rf_feature_importance.csv")
OUT_DIR   = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# Visual style (matplotlib only, no seaborn)
# ---------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi":         100,
    "savefig.dpi":        150,
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
})

PALETTE = {
    "stable":     "#2E7D32",
    "stretched":  "#F9A825",
    "fragile":    "#EF6C00",
    "distressed": "#C62828",
    "logit":      "#1f77b4",
    "rf":         "#d62728",
    "mute":       "#888888",
}

# IHDS / 2011 Census state codes -> short names
STATE_NAMES = {
    1: "J&K", 2: "Himachal Pradesh", 3: "Punjab", 4: "Chandigarh",
    5: "Uttarakhand", 6: "Haryana", 7: "Delhi", 8: "Rajasthan",
    9: "Uttar Pradesh", 10: "Bihar", 11: "Sikkim", 12: "Arunachal Pradesh",
    13: "Nagaland", 14: "Manipur", 15: "Mizoram", 16: "Tripura",
    17: "Meghalaya", 18: "Assam", 19: "West Bengal", 20: "Jharkhand",
    21: "Odisha", 22: "Chhattisgarh", 23: "Madhya Pradesh", 24: "Gujarat",
    25: "Daman & Diu", 26: "Dadra & Nagar Haveli", 27: "Maharashtra",
    28: "Andhra Pradesh", 29: "Karnataka", 30: "Goa", 31: "Lakshadweep",
    32: "Kerala", 33: "Tamil Nadu", 34: "Puducherry", 35: "Andaman & Nicobar",
}


def save(fig, name):
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------
# Load
# ---------------------------------------------------------------
print("Loading inputs...")
ffi  = pd.read_parquet(FFI_FILE)
pred = pd.read_csv(PRED_FILE)
imp  = pd.read_csv(IMP_FILE, index_col=0)["importance"]
print(f"  FFI: {ffi.shape}   Pred: {pred.shape}   Imp: {len(imp)} features")


# ---------------------------------------------------------------
# 1. FFI distribution + quartile cuts
# ---------------------------------------------------------------
print("\n[1/10] FFI distribution")
fig, ax = plt.subplots(figsize=(8, 4.5))
clip_lo, clip_hi = ffi["FFI"].quantile([0.001, 0.999])
ax.hist(ffi["FFI"].clip(clip_lo, clip_hi), bins=80,
        color=PALETTE["mute"], edgecolor="white")
q = ffi["FFI"].quantile([0.25, 0.50, 0.75]).values
labels = ["Q1: Stable→Stretched", "Q2: Stretched→Fragile", "Q3: Fragile→Distressed"]
colors = [PALETTE["stretched"], PALETTE["fragile"], PALETTE["distressed"]]
for v, lab, c in zip(q, labels, colors):
    ax.axvline(v, color=c, linestyle="--", linewidth=1.5,
               label=f"{lab} = {v:.2f}")
ax.set_xlabel("FFI score (clipped to 0.1–99.9 pct for display)")
ax.set_ylabel("Households")
ax.set_title("Financial Fragility Index — distribution and quartile cut-points")
ax.legend(loc="upper right", frameon=False, fontsize=9)
save(fig, "01_ffi_distribution.png")


# ---------------------------------------------------------------
# 2. Component distributions (2x3)
# ---------------------------------------------------------------
print("[2/10] FFI components")
comp_cols = [
    "c1_debt_burden", "c2_cons_stress", "c3_asset_deficit",
    "c4_emp_concentration", "c5_dependency", "c6_distress_borrow",
]
titles = [
    "Debt burden\n(debt / income)",
    "Consumption stress\n(consumption / income)",
    "Asset deficit\n(−log(1 + assets))",
    "Employment concentration\n(Herfindahl on worker types)",
    "Dependency pressure\n(non-earners / (earners+1))",
    "Distress borrowing\n(0 / 1 flag)",
]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, c, t in zip(axes.flatten(), comp_cols, titles):
    s = ffi[c].dropna()
    if c in ("c1_debt_burden", "c2_cons_stress"):
        s_clip = s[s <= s.quantile(0.99)]
        ax.hist(s_clip, bins=60, color=PALETTE["fragile"], edgecolor="white")
        ax.set_yscale("log")
        ax.set_xlabel("value (clipped at 99th pct)")
    elif c == "c6_distress_borrow":
        ax.hist(s, bins=[-0.25, 0.25, 0.75, 1.25],
                color=PALETTE["distressed"], edgecolor="white", rwidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xlabel("flag")
    else:
        ax.hist(s, bins=60, color=PALETTE["stretched"], edgecolor="white")
        ax.set_xlabel("value")
    ax.set_title(t)
    ax.set_ylabel("households")
fig.suptitle("FFI component distributions", y=1.00, fontsize=14)
save(fig, "02_ffi_components_dist.png")


# ---------------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------------
print("[3/10] Component correlation")
corr = ffi[comp_cols].corr()
short = ["debt_burden", "cons_stress", "asset_def",
         "emp_conc", "depend", "distr_borrow"]
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(short))); ax.set_yticks(range(len(short)))
ax.set_xticklabels(short, rotation=45, ha="right")
ax.set_yticklabels(short)
for i in range(len(short)):
    for j in range(len(short)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.5 else "black", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="Pearson r")
ax.set_title("Pairwise correlations among FFI components")
save(fig, "03_ffi_components_correlation.png")


# ---------------------------------------------------------------
# 4. Fragility state by urban / rural
# ---------------------------------------------------------------
print("[4/10] Fragility by urban/rural")
if "URBAN2011" in ffi.columns:
    sub = ffi.dropna(subset=["URBAN2011"]).copy()
    sub["URBAN"] = sub["URBAN2011"].map({0: "Rural", 1: "Urban"})
    tab = (sub.groupby(["URBAN", "FRAG_STATE"]).size()
              .unstack(fill_value=0))
    tab = tab.reindex(columns=["Stable", "Stretched", "Fragile", "Distressed"])
    tab_pct = tab.div(tab.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(tab_pct))
    cols = [PALETTE[k] for k in ["stable", "stretched", "fragile", "distressed"]]
    for state, color in zip(tab_pct.columns, cols):
        ax.bar(tab_pct.index, tab_pct[state], bottom=bottom,
               label=state, color=color)
        bottom += tab_pct[state].values
    ax.set_ylabel("% of households")
    ax.set_title("Fragility composition: Urban vs Rural")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    save(fig, "04_fragility_by_urban_rural.png")


# ---------------------------------------------------------------
# 5. State-level fragile share
# ---------------------------------------------------------------
print("[5/10] Fragility by state")
if "STATEID" in ffi.columns:
    by_state = ffi.groupby("STATEID").agg(
        n=("FRAG_BINARY", "size"),
        frag_share=("FRAG_BINARY", "mean"),
        mean_ffi=("FFI", "mean"),
    ).reset_index()
    by_state = by_state[by_state["n"] >= 100].sort_values("frag_share")
    by_state["label"] = by_state["STATEID"].map(STATE_NAMES).fillna(
        by_state["STATEID"].astype(str))

    fig, ax = plt.subplots(figsize=(8, 0.30 * len(by_state) + 1))
    bars = ax.barh(by_state["label"], by_state["frag_share"] * 100,
                   color=PALETTE["fragile"])
    ax.axvline(50, color="black", linestyle=":", linewidth=1, alpha=0.6,
               label="national mean (50%)")
    ax.set_xlabel("% of households in Fragile or Distressed state")
    ax.set_title("Share of fragile households by state (n ≥ 100)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "05_fragility_by_state.png")


# ---------------------------------------------------------------
# 6. ROC curves
# ---------------------------------------------------------------
print("[6/10] ROC curves")
y   = pred["y_true"].values
p_l = pred["p_logit"].values
p_r = pred["p_rf"].values
fpr_l, tpr_l, _ = roc_curve(y, p_l)
fpr_r, tpr_r, _ = roc_curve(y, p_r)
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(fpr_l, tpr_l, color=PALETTE["logit"],
        label=f"Logistic Regression (AUC = {auc(fpr_l, tpr_l):.3f})")
ax.plot(fpr_r, tpr_r, color=PALETTE["rf"],
        label=f"Random Forest         (AUC = {auc(fpr_r, tpr_r):.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC — predicting Fragile/Distressed households")
ax.legend(loc="lower right", frameon=False)
ax.set_aspect("equal")
save(fig, "06_roc_curves.png")


# ---------------------------------------------------------------
# 7. Precision-Recall curves
# ---------------------------------------------------------------
print("[7/10] PR curves")
prec_l, rec_l, _ = precision_recall_curve(y, p_l)
prec_r, rec_r, _ = precision_recall_curve(y, p_r)
ap_l = average_precision_score(y, p_l)
ap_r = average_precision_score(y, p_r)
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(rec_l, prec_l, color=PALETTE["logit"],
        label=f"Logistic (AP = {ap_l:.3f})")
ax.plot(rec_r, prec_r, color=PALETTE["rf"],
        label=f"Random Forest (AP = {ap_r:.3f})")
ax.axhline(y.mean(), color="gray", linestyle=":",
           label=f"baseline = {y.mean():.2f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall curves")
ax.legend(loc="lower left", frameon=False)
save(fig, "07_pr_curves.png")


# ---------------------------------------------------------------
# 8. Confusion matrices side-by-side
# ---------------------------------------------------------------
print("[8/10] Confusion matrices")

def plot_cm(ax, cm, title):
    cm_pct = cm / cm.sum() * 100
    im = ax.imshow(cm_pct, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_pct[i, j] > 25 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    ax.grid(False)

yhat_l = (p_l >= 0.5).astype(int)
yhat_r = (p_r >= 0.5).astype(int)
cm_l = confusion_matrix(y, yhat_l)
cm_r = confusion_matrix(y, yhat_r)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
plot_cm(axes[0], cm_l, "Logistic Regression")
plot_cm(axes[1], cm_r, "Random Forest")
fig.suptitle("Confusion matrices @ threshold 0.5", y=1.02, fontsize=13)
save(fig, "08_confusion_matrices.png")


# ---------------------------------------------------------------
# 9. Top-20 RF feature importance
# ---------------------------------------------------------------
print("[9/10] RF feature importance (top 20)")
top = imp.sort_values(ascending=False).head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top.index, top.values, color=PALETTE["rf"])
ax.set_xlabel("RF feature importance (mean Gini decrease)")
ax.set_title("Top-20 predictors of household fragility")
save(fig, "09_rf_feature_importance_top20.png")


# ---------------------------------------------------------------
# 10. Predicted-probability distribution by true class
# ---------------------------------------------------------------
print("[10/10] Score distributions")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, p, name in zip(axes, [p_l, p_r], ["Logistic Regression", "Random Forest"]):
    ax.hist(p[y == 0], bins=40, alpha=0.65,
            label="True = 0 (not fragile)", color=PALETTE["stable"])
    ax.hist(p[y == 1], bins=40, alpha=0.65,
            label="True = 1 (fragile)", color=PALETTE["distressed"])
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Predicted P(fragile)")
    ax.set_ylabel("Households")
    ax.set_title(name)
    ax.legend(frameon=False, fontsize=9)
fig.suptitle("Predicted probability distribution by true class", y=1.02)
save(fig, "10_score_distributions.png")

print("\nAll 10 figures written to ./figures/")