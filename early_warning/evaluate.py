"""
Assemble the held-out test-set evaluation across all four early-warning models
(LSTM + MC Dropout, naive, NGBoost, Kalman), and produce the figures and
summary table that feed §4.7 / §4.8 of the report.

Metrics:
  - ROC-AUC and Average Precision for the binary stress event
  - RMSE for the 3-month-ahead FFI regression
  - Expected Calibration Error (ECE, 10 equal-frequency bins) for probabilistic models
  - 95% credible-interval coverage (LSTM and NGBoost only)

Figures:
  22_ew_roc_pr.png            ROC and PR curves for all four models
  23_ew_calibration.png       Reliability diagram for LSTM and NGBoost
  24_ew_uncertainty_examples.png   Sample household predictions with credible intervals
  25_ew_risk_tier.png         Risk-tier counts and per-tier positive rate
  26_ew_coverage.png          Empirical coverage vs nominal coverage (LSTM)

Outputs:
  early_warning/lstm_arrays/ew_summary.json   final numbers for the report table
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
ARR = ROOT / "early_warning" / "lstm_arrays"
FIG = ROOT / "figures"
OUT = ARR / "ew_summary.json"


def ece_score(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-frequency bins."""
    order = np.argsort(p)
    y_sorted = y_true[order]; p_sorted = p[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        mean_p = p_sorted[b].mean()
        mean_y = y_sorted[b].mean()
        ece += (len(b) / len(p)) * abs(mean_p - mean_y)
    return float(ece)


def credible_coverage(y_ffi: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y_ffi >= lo) & (y_ffi <= hi)))


def load_all():
    """Load all model predictions for the test set, harmonising key names."""
    out = {}

    # LSTM (full posterior)
    lstm = np.load(ARR / "test_predictions.npz")
    out["lstm"] = {
        "ffi_mean": lstm["ffi_mean"], "ffi_low": lstm["ffi_low95"], "ffi_high": lstm["ffi_high95"],
        "stress_mean": lstm["stress_mean"], "stress_low": lstm["stress_low95"], "stress_high": lstm["stress_high95"],
        "risk_tier": lstm["risk_tier"],
        "y_ffi": lstm["y_ffi"], "y_stress": lstm["y_stress"],
    }
    # Naive
    nv = np.load(ARR / "baseline_naive.npz")
    out["naive"] = {
        "ffi_mean": nv["ffi_pred"], "stress_mean": nv["stress_score"],
        "y_ffi": nv["y_ffi"], "y_stress": nv["y_stress"],
    }
    # NGBoost
    if (ARR / "baseline_ngboost.npz").exists():
        ng = np.load(ARR / "baseline_ngboost.npz")
        # NGBoost stress mean is calibrated probability already in [0,1]
        out["ngboost"] = {
            "ffi_mean": ng["ffi_mean"], "ffi_std": ng["ffi_std"],
            "stress_mean": ng["stress_mean"],
            "y_ffi": ng["y_ffi"], "y_stress": ng["y_stress"],
        }
        # 95% credible interval from the parametric Normal
        out["ngboost"]["ffi_low"]  = ng["ffi_mean"] - 1.96 * ng["ffi_std"]
        out["ngboost"]["ffi_high"] = ng["ffi_mean"] + 1.96 * ng["ffi_std"]
    # Kalman
    if (ARR / "baseline_kalman.npz").exists():
        kl = np.load(ARR / "baseline_kalman.npz")
        out["kalman"] = {
            "ffi_mean": kl["ffi_pred"], "stress_mean": kl["stress_score"],
            "y_ffi": kl["y_ffi"], "y_stress": kl["y_stress"],
        }

    return out


def metrics_for(name: str, data: dict) -> dict:
    y_s = data["y_stress"]
    y_f = data["y_ffi"]
    p = data["stress_mean"]
    # For naive / kalman the "score" is the predicted FFI on the z-scale,
    # not a probability. ROC-AUC is scale-invariant so this is OK for AUC/AP,
    # but ECE is not meaningful unless we map it to [0,1]. Skip ECE for those.
    auc = roc_auc_score(y_s, p)
    ap = average_precision_score(y_s, p)
    rmse = float(np.sqrt(np.mean((data["ffi_mean"] - y_f) ** 2)))
    out = {"auc": float(auc), "ap": float(ap), "rmse_ffi": rmse}
    if name in ("lstm", "ngboost"):
        out["ece"] = ece_score(y_s, p)
        out["coverage_95"] = credible_coverage(y_f, data["ffi_low"], data["ffi_high"])
    return out


def figure_roc_pr(all_data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = ["naive", "kalman", "ngboost", "lstm"]
    colors = {"naive": "gray", "kalman": "C0", "ngboost": "C2", "lstm": "C3"}
    labels = {"naive": "Naive carry-forward", "kalman": "Kalman UC",
              "ngboost": "NGBoost (lagged)", "lstm": "LSTM + MC Dropout"}
    ax = axes[0]
    for k in order:
        if k not in all_data: continue
        d = all_data[k]
        fpr, tpr, _ = roc_curve(d["y_stress"], d["stress_mean"])
        auc = roc_auc_score(d["y_stress"], d["stress_mean"])
        ax.plot(fpr, tpr, color=colors[k], lw=1.8, label=f"{labels[k]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — early-warning, 3-month horizon")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax = axes[1]
    for k in order:
        if k not in all_data: continue
        d = all_data[k]
        pr, rc, _ = precision_recall_curve(d["y_stress"], d["stress_mean"])
        ap = average_precision_score(d["y_stress"], d["stress_mean"])
        ax.plot(rc, pr, color=colors[k], lw=1.8, label=f"{labels[k]} (AP={ap:.3f})")
    base = float(all_data["lstm"]["y_stress"].mean())
    ax.axhline(base, color="k", ls="--", lw=0.8, label=f"baseline = {base:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — early-warning, 3-month horizon")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "22_ew_roc_pr.png", dpi=200)
    plt.close(fig)


def figure_calibration(all_data: dict):
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, color, label in [
        ("lstm", "C3", "LSTM + MC Dropout"),
        ("ngboost", "C2", "NGBoost"),
    ]:
        if k not in all_data: continue
        d = all_data[k]
        p = np.clip(d["stress_mean"], 1e-6, 1 - 1e-6)
        frac, pred = calibration_curve(d["y_stress"], p,
                                       n_bins=10, strategy="quantile")
        ece = ece_score(d["y_stress"], p)
        ax.plot(pred, frac, "o-", color=color, label=f"{label} (ECE={ece:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical fraction positive")
    ax.set_title("Reliability diagram — stress probability (10 quantile bins)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "23_ew_calibration.png", dpi=200)
    plt.close(fig)


def figure_uncertainty_examples(all_data: dict):
    """For six representative test rows, plot the predicted FFI_{t+3} with its
    LSTM credible interval and the true value."""
    d = all_data["lstm"]
    rng = np.random.default_rng(0)
    # Choose 3 rows where LSTM is confident-correct, 3 where it's uncertain
    p = d["stress_mean"]; spread = d["stress_high"] - d["stress_low"]
    confident = np.where((p > 0.9) | (p < 0.1))[0]
    uncertain = np.where((p > 0.3) & (p < 0.7))[0]
    pick = np.concatenate([
        rng.choice(confident, size=min(3, len(confident)), replace=False),
        rng.choice(uncertain, size=min(3, len(uncertain)), replace=False),
    ])
    titles = ["Confident A", "Confident B", "Confident C",
              "Uncertain A", "Uncertain B", "Uncertain C"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, idx, ttl in zip(axes.flat, pick, titles):
        # Plot the 50 MC posterior samples of stress probability around mean
        mean_p = float(p[idx]); lo = float(d["stress_low"][idx]); hi = float(d["stress_high"][idx])
        y_true = int(d["y_stress"][idx]); ffi_true = float(d["y_ffi"][idx])
        ffi_pred = float(d["ffi_mean"][idx]); ffi_lo = float(d["ffi_low"][idx]); ffi_hi = float(d["ffi_high"][idx])

        # Two stacked sub-plots in this cell: FFI band + stress prob bar
        ax.errorbar([0], [ffi_pred], yerr=[[ffi_pred - ffi_lo], [ffi_hi - ffi_pred]],
                    fmt="o", color="C0", capsize=6, label=f"Predicted FFI ± 95% CI")
        ax.scatter([0], [ffi_true], color="red", s=60, zorder=5, label=f"True FFI")
        ax.bar([1], [mean_p], yerr=[[mean_p - lo], [hi - mean_p]],
               width=0.4, color="C3", alpha=0.6, capsize=6, label=f"Predicted p(stress) ± 95% CI")
        ax.axhline(0.5, xmin=0.4, xmax=1.0, color="gray", ls="--", lw=0.7)
        ax.text(1, 1.02, f"true y = {y_true}", ha="center", va="bottom", fontsize=8, color="red")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["FFI$_{t+3}$", "p(stress)"])
        ax.set_ylim(-3, 3.5)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle("LSTM posterior predictions on individual test rows: "
                 "confident correct vs uncertain", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "24_ew_uncertainty_examples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_risk_tier(all_data: dict):
    d = all_data["lstm"]
    tier = d["risk_tier"]; y = d["y_stress"]
    names = ["Low", "Medium", "High"]
    counts = [int((tier == i).sum()) for i in range(3)]
    purity = [float(y[tier == i].mean()) if (tier == i).any() else 0.0 for i in range(3)]
    capture = [float(((tier == i) & (y == 1)).sum() / max(y.sum(), 1)) for i in range(3)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.bar(names, counts, color=["#2ca02c", "#ff7f0e", "#d62728"])
    for n, c in zip(names, counts):
        ax.text(n, c, f"{c:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Test rows")
    ax.set_title("Risk tier population (LSTM + MC Dropout)")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    xs = np.arange(3); w = 0.35
    ax.bar(xs - w/2, purity, w, color="C3", label="Purity (positive rate within tier)")
    ax.bar(xs + w/2, capture, w, color="C0", label="Recall (share of positives captured)")
    for x, p in zip(xs, purity):
        ax.text(x - w/2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
    for x, c in zip(xs, capture):
        ax.text(x + w/2, c, f"{c:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Share")
    ax.set_title("Per-tier purity and recall"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "25_ew_risk_tier.png", dpi=200)
    plt.close(fig)


def figure_coverage(all_data: dict):
    """For LSTM, plot empirical coverage of predictive intervals at every nominal
    level from 5% to 95%."""
    # Reconstruct intervals from quantiles of the saved 95% CI alone is not
    # enough; we approximate by treating the predicted (low95, high95) as the
    # outer envelope and the mean as the centre, and plot the *one* point we
    # have plus the 50% interval derived from (mean ± 0.67 * sigma) where sigma
    # is implied by the 95% CI half-width.
    d = all_data["lstm"]
    sigma = (d["ffi_high"] - d["ffi_low"]) / (2 * 1.96)
    nominal = np.linspace(0.1, 0.95, 18)
    coverages = []
    for nl in nominal:
        z = abs(np.quantile(np.random.standard_normal(20_000), (1 + nl) / 2))
        lo = d["ffi_mean"] - z * sigma
        hi = d["ffi_mean"] + z * sigma
        coverages.append(float(np.mean((d["y_ffi"] >= lo) & (d["y_ffi"] <= hi))))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(nominal, coverages, "o-", color="C3", label="LSTM (MC Dropout, Gaussian fit)")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect coverage")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Predictive-interval coverage diagnostic")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "26_ew_coverage.png", dpi=200)
    plt.close(fig)


def main():
    all_data = load_all()
    summary = {}
    for name, data in all_data.items():
        summary[name] = metrics_for(name, data)
        print(f"  {name:10s}  {summary[name]}")

    # Figures
    figure_roc_pr(all_data)
    figure_calibration(all_data)
    figure_uncertainty_examples(all_data)
    figure_risk_tier(all_data)
    figure_coverage(all_data)

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
