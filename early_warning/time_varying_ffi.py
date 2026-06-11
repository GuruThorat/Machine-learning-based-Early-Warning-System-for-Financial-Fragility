"""
Compute the time-varying Financial Fragility Index FFI_t on the simulated panel,
per the project plan:

    FFI_t = w1 * DebtRatio_t  +  w2 * ExpenseVolatility_t  -  w3 * SavingsBuffer_t

with:
    DebtRatio_t          = debt_t / max(monthly_income_t, eps)
    ExpenseVolatility_t  = rolling 6-month std(expense) / rolling 6-month mean(expense)
    SavingsBuffer_t      = savings_t / max(monthly_expense_t, eps)        (months of cover)

We standardize each component (z-score across the entire panel pool) and use
equal weights w1 = w2 = w3 = 1/3, mirroring the cross-sectional FFI from §2 of
the report. Higher FFI_t = more fragile.

A "stress event" within a 3-month look-ahead window is defined as

    y_t = I[ max(FFI_{t+1}, FFI_{t+2}, FFI_{t+3}) > tau ]

with the threshold tau set to the cross-sectional 75th percentile of FFI_t
(matching the "Fragile or Distressed" quartile split used in the cross-sectional
work). This means the binary problem is to predict whether a household will
enter the top quartile of fragility within the next quarter, given its
preceding history.

Outputs:
  early_warning/panel_with_ffi.parquet          (hh × month × features + ffi_t + components)
  early_warning/panel_supervised.parquet        (one row per (hh, t) with all features and y_t)
  early_warning/ffi_calibration.json            (weights, threshold tau, summary stats)
  figures/18_ffi_t_distribution.png             (marginal of FFI_t and tau)
  figures/19_ffi_t_components_corr.png          (component correlation heatmap)
  figures/20_ffi_t_sample_paths.png             (FFI_t paths for 6 illustrative households)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_PARQUET = ROOT / "early_warning" / "simulated_panel.parquet"
OUT_PANEL = ROOT / "early_warning" / "panel_with_ffi.parquet"
OUT_SUPERVISED = ROOT / "early_warning" / "panel_supervised.parquet"
OUT_CAL = ROOT / "early_warning" / "ffi_calibration.json"
FIG_DIR = ROOT / "figures"

ROLLING_WINDOW = 6        # months for expense-volatility window
HORIZON = 3               # months ahead for stress-event lookahead
W1, W2, W3 = 1/3, 1/3, 1/3
EPS = 1.0                 # divide-by-zero guard, in rupees


def load_panel() -> pd.DataFrame:
    print(f"Loading {IN_PARQUET}...")
    df = pd.read_parquet(IN_PARQUET)
    print(f"  rows: {len(df):,}  households: {df['hh_id'].nunique():,}  months: {df['month'].nunique()}")
    return df.sort_values(["hh_id", "month"]).reset_index(drop=True)


def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add DebtRatio_t, ExpenseVolatility_t, SavingsBuffer_t columns (raw, not z-scored)."""
    print("Computing per-month FFI components (raw)...")
    df = df.copy()
    df["debt_ratio_raw"] = df["debt"] / (df["income"].abs() + EPS)
    df["savings_buffer_raw"] = df["savings"] / (df["expense"].abs() + EPS)

    # Rolling 6-month expense volatility (coefficient of variation), per household
    g = df.groupby("hh_id", sort=False)["expense"]
    roll_mean = g.transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=2).mean())
    roll_std  = g.transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=2).std(ddof=0))
    df["expense_vol_raw"] = roll_std / (roll_mean.abs() + EPS)

    # First month with no rolling window: forward-fill from t=2 onwards within hh
    # (rolling at t=1 is undefined; we mask later when building supervised set)
    return df


def standardize_components(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Z-score the three components across the entire pool (households × months),
    using only months where all three are finite. Return df with z-cols and the
    standardization statistics for inversion / reporting."""
    print("Standardizing components...")
    comps = ["debt_ratio_raw", "expense_vol_raw", "savings_buffer_raw"]
    # Winsorize at the 99th percentile to tame heavy tails before z-scoring
    stats = {}
    for c in comps:
        finite = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        p1 = float(finite.quantile(0.01))
        p99 = float(finite.quantile(0.99))
        df[c + "_w"] = df[c].clip(lower=p1, upper=p99)
        finite_w = df[c + "_w"].replace([np.inf, -np.inf], np.nan).dropna()
        mu = float(finite_w.mean())
        sd = float(finite_w.std(ddof=0))
        stats[c] = {"p1": p1, "p99": p99, "mean": mu, "std": sd}
        df[c.replace("_raw", "_z")] = (df[c + "_w"] - mu) / (sd if sd > 0 else 1.0)
        df = df.drop(columns=[c + "_w"])
    return df, stats


def aggregate_ffi(df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted aggregation per the plan formula."""
    print("Aggregating FFI_t = w1*DebtRatio + w2*ExpenseVol - w3*SavingsBuffer...")
    df["ffi_t"] = (
        W1 * df["debt_ratio_z"]
      + W2 * df["expense_vol_z"]
      - W3 * df["savings_buffer_z"]
    )
    return df


def build_supervised(df: pd.DataFrame, horizon: int = HORIZON,
                     warmup: int = ROLLING_WINDOW) -> tuple[pd.DataFrame, float]:
    """For each (hh, t), compute the binary stress event:

        y_t = 1 if max(ffi_{t+1}, ..., ffi_{t+horizon}) > tau, else 0

    tau is set to the 75th percentile of ffi_t over all rows where the rolling
    window is fully populated. Returns the supervised long-format frame and tau.
    """
    print(f"Building supervised target (horizon={horizon}, warmup={warmup})...")
    df = df.copy()

    # Threshold tau (top-quartile of FFI_t after warmup, matching cross-sectional cut)
    valid_for_tau = df[df["month"] > warmup]["ffi_t"]
    tau = float(valid_for_tau.quantile(0.75))
    print(f"  tau = 75th percentile of FFI_t (after warmup) = {tau:.4f}")

    # Build max future FFI within (t+1, t+horizon) per household
    g = df.groupby("hh_id", sort=False)["ffi_t"]
    fwd_max = g.transform(lambda s: s.shift(-1).rolling(horizon, min_periods=1).max())
    df["fwd_max_ffi"] = fwd_max
    df["y_stress"] = (df["fwd_max_ffi"] > tau).astype("Int8")

    # Drop rows where look-ahead is incomplete (last `horizon` months per hh)
    # and where rolling-vol warmup is not yet done.
    df["valid"] = (
        df["month"].between(warmup + 1, df["month"].max() - horizon).astype(int)
    )
    sup = df[df["valid"] == 1].copy()
    print(f"  supervised rows: {len(sup):,}  positive rate: {sup['y_stress'].mean():.3f}")
    return sup, tau


def make_figures(df: pd.DataFrame, sup: pd.DataFrame, tau: float, stats: dict):
    print("Making figures 18–20...")
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10})
    rng = np.random.default_rng(0)

    # --- Fig 18: Distribution of FFI_t and the tau line ---
    fig, ax = plt.subplots(figsize=(7, 4))
    sample = df.loc[df["month"] > ROLLING_WINDOW, "ffi_t"].sample(
        n=min(200_000, len(df)), random_state=0
    )
    ax.hist(sample, bins=120, color="C0", alpha=0.85)
    ax.axvline(tau, color="C3", ls="--", lw=2,
               label=f"$\\tau$ = 75th pct = {tau:.2f}")
    ax.set_xlabel("FFI$_t$ (time-varying FFI, equal-weight z-score)")
    ax.set_ylabel("Number of (household, month) cells")
    ax.set_title("Distribution of the time-varying FFI$_t$ across simulated panel")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "18_ffi_t_distribution.png"); plt.close(fig)

    # --- Fig 19: Component correlation heatmap (after standardisation) ---
    corr = sup[["debt_ratio_z", "expense_vol_z", "savings_buffer_z", "ffi_t"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(corr.columns)
    for (i, j), v in np.ndenumerate(corr.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Pearson correlation of FFI$_t$ components and aggregate")
    fig.tight_layout(); fig.savefig(FIG_DIR / "19_ffi_t_components_corr.png"); plt.close(fig)

    # --- Fig 20: Six sample household FFI_t paths (3 always-stable, 3 crossing tau) ---
    grouped = df.groupby("hh_id")
    max_ffi = grouped["ffi_t"].max()
    min_ffi = grouped["ffi_t"].min()

    always_stable = max_ffi[(max_ffi < tau - 0.1)].index.tolist()[:3]
    transitions = grouped["ffi_t"].apply(
        lambda s: ((s.shift(1) <= tau) & (s > tau)).sum()
    )
    crossers = transitions[transitions >= 1].index.tolist()[:3]

    pick = list(always_stable) + list(crossers)
    titles = ["Always stable A", "Always stable B", "Always stable C",
              "Crosses τ — case A", "Crosses τ — case B", "Crosses τ — case C"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for ax, hh, ttl in zip(axes.flat, pick, titles):
        series = df.loc[df["hh_id"] == hh, ["month", "ffi_t"]].sort_values("month")
        ax.plot(series["month"], series["ffi_t"], color="C0", lw=1.5)
        ax.axhline(tau, color="C3", ls="--", lw=1, label=f"τ = {tau:.2f}")
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("FFI$_t$"); ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[-1, 1].set_xlabel("Month (1 = Jan 2011)")
    fig.suptitle("Sample FFI$_t$ trajectories — three stable, three crossing the threshold", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "20_ffi_t_sample_paths.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_panel()
    df = compute_components(df)
    df, stats = standardize_components(df)
    df = aggregate_ffi(df)
    sup, tau = build_supervised(df, horizon=HORIZON, warmup=ROLLING_WINDOW)

    # Persist the (hh, t) panel with FFI for the LSTM input pipeline
    keep_panel = [
        "hh_id", "month", "income", "expense", "savings", "debt",
        "shock_medical", "shock_disaster", "shock_crop", "cpi",
        "debt_ratio_raw", "expense_vol_raw", "savings_buffer_raw",
        "debt_ratio_z", "expense_vol_z", "savings_buffer_z",
        "ffi_t",
    ]
    df[keep_panel].to_parquet(OUT_PANEL, index=False)
    print(f"Wrote {OUT_PANEL}  ({len(df):,} rows)")

    keep_sup = keep_panel + ["fwd_max_ffi", "y_stress"]
    sup[keep_sup].to_parquet(OUT_SUPERVISED, index=False)
    print(f"Wrote {OUT_SUPERVISED}  ({len(sup):,} rows)")

    with open(OUT_CAL, "w") as f:
        json.dump({
            "weights": {"w1_debt_ratio": W1, "w2_expense_vol": W2, "w3_savings_buffer": W3},
            "rolling_window_months": ROLLING_WINDOW,
            "horizon_months": HORIZON,
            "tau_75th_pct": tau,
            "component_stats": stats,
            "positive_rate": float(sup["y_stress"].mean()),
            "n_supervised_rows": int(len(sup)),
        }, f, indent=2)
    print(f"Wrote {OUT_CAL}")

    make_figures(df, sup, tau, stats)
    print("\nDone.")


if __name__ == "__main__":
    main()
