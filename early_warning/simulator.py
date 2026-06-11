"""
Macro-anchored monthly household-trajectory simulator for the temporal early-warning
extension.

For each IHDS-II household h with annual snapshot (INCOME, COTOTAL, ASSETS, DB5+DB6A,
earner mix), the simulator produces a 24-month trajectory of (income, expense, savings,
debt) and a binary indicator of shocks each month. Trajectories are anchored to the
household's annual values and modulated by:

  (i)   the real India CPI-Combined monthly series (2011-01 .. 2012-12),
  (ii)  a household-specific income-volatility coefficient derived from earner mix,
  (iii) a household-specific consumption-smoothing coefficient,
  (iv)  Poisson-distributed shocks (medical, disaster, crop-failure) at rates
        calibrated to the IHDS-II 5-year-recall MI-module prevalence.

The mathematical model is:

  CPI scaling     m_t        = CPI_t / CPI_anchor

  Income          y_h(t)     = (Y_h / 12) * m_t * exp( eps_y^h(t) - sigma_y^h^2 / 2 )
                               * (1 - delta_crop * I[shock_crop(t)])

  Expense         x_h(t)     = (C_h / 12) * m_t * seasonal(t)
                               * exp( eps_x^h(t) - sigma_x^h^2 / 2 )
                               + (medical_spike + disaster_spike)

  Cash-flow gap   g_h(t)     = y_h(t) - x_h(t) - i * D_h(t-1)        (i = 2%/month service)
  Savings         S_h(t)     = max(0, S_h(t-1) + g_h(t))             if g_h(t) > 0,
                               max(0, S_h(t-1) + g_h(t))             else
  Debt            D_h(t)     = D_h(t-1) + max(0, -(S_h(t-1) + g_h(t))) (debt absorbs negative cash-flow
                               beyond exhausted savings)
                               - min(repay_h(t), D_h(t-1))           if g_h(t) > 0 and S_h(t-1) > 0

The shock variables eps_y, eps_x are i.i.d. N(0, sigma^2) and the shock indicators are
independent Bernoulli per month at calibrated rates.

Outputs:
  early_warning/simulated_panel.parquet         (~24 * 42152 rows; long format)
  early_warning/simulator_calibration.json      (empirical shock rates and CV used)
  figures/15_simulator_trajectories.png         (6 example household trajectories)
  figures/16_simulator_validation_marginals.png (simulated annual sums vs IHDS anchors)
  figures/17_simulator_shock_calibration.png    (empirical vs realised shock incidence)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HH_FILE = ROOT / "DS0002" / "36151-0002-Data.dta"
FFI_FILE = ROOT / "ihds2_ffi.parquet"
CPI_FILE = ROOT / "data" / "cpi_india.csv"
MANIFEST = ROOT / "early_warning" / "sim_variable_manifest.json"
OUT_PARQUET = ROOT / "early_warning" / "simulated_panel.parquet"
OUT_CAL_JSON = ROOT / "early_warning" / "simulator_calibration.json"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

RNG_SEED = 42
N_MONTHS = 24
# Survey window centered at 2012-01 (mid-IHDS-II); month 1 = 2011-01, month 24 = 2012-12
CPI_ANCHOR_MONTH = "2012-01-01"
CPI_START_MONTH = "2011-01-01"

# Service rate on outstanding debt (monthly). Calibrated to ~25% APR on informal credit.
DEBT_MONTHLY_INTEREST = 0.02

# Initial liquid-savings buffer as a fraction of assets stock
INITIAL_SAVINGS_FRACTION = 0.05


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print("Loading manifest, household file, FFI labels, CPI...")
    with open(MANIFEST) as f:
        manifest = json.load(f)

    needed_cols = (
        ["STATEID", "DISTID", "PSUID", "HHID", "HHSPLITID"]
        + list(manifest["anchor_vars"].values())
        + manifest["earner_count_vars"]
        + manifest["composition_vars"]
        + [v["code"] for v in manifest["shock_vars"].values()]
        + manifest["debt_vars"]
    )
    needed_cols = list(dict.fromkeys(needed_cols))  # de-dup, preserve order

    hh = pd.read_stata(HH_FILE, columns=needed_cols, convert_categoricals=False)
    print(f"  household file: {hh.shape}")

    # IHDS missing codes -> NaN
    for code in (-9, -8, -7):
        hh = hh.replace(code, np.nan)

    # Re-build hh_id consistent with the cross-sectional pipeline
    hh_keys = ["STATEID", "DISTID", "PSUID", "HHID", "HHSPLITID"]
    hh["hh_id"] = hh[hh_keys].astype(str).agg("-".join, axis=1)

    cpi = pd.read_csv(CPI_FILE, parse_dates=["month"])
    return hh, cpi, manifest


def select_24mo_cpi_window(cpi: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Slice the bundled CPI series to the 24 months centered on IHDS-II."""
    cpi = cpi.set_index("month")
    start = pd.Timestamp(CPI_START_MONTH)
    window = cpi.loc[start:start + pd.DateOffset(months=N_MONTHS - 1)]
    assert len(window) == N_MONTHS, f"need {N_MONTHS} months, got {len(window)}"
    cpi_anchor = float(cpi.loc[CPI_ANCHOR_MONTH, "cpi"])
    cpi_series = window["cpi"].to_numpy()
    return cpi_series, cpi_anchor


def calibrate_shock_rates(hh: pd.DataFrame, manifest: dict) -> dict:
    """Convert 5-year-recall MI* prevalence to monthly Poisson rates.

    MI1/MI2/MI5 in IHDS-II are coded as 1 if the household reports having
    suffered the shock in the past 5 years, 0 otherwise. Under a constant-hazard
    assumption, monthly rate = prevalence / 60.
    """
    rates = {}
    for name, spec in manifest["shock_vars"].items():
        col = spec["code"]
        prev = hh[col].mean(skipna=True)
        rate = float(prev) / 60.0
        rates[name] = {
            "code": col,
            "five_year_prevalence": float(prev),
            "monthly_poisson_rate": rate,
            "shock_kind": spec["shock_kind"],
        }
        print(f"  shock {name:14s} ({col}): 5y prev = {prev:.3f}, monthly rate = {rate:.5f}")
    return rates


def income_volatility(earner_counts: np.ndarray) -> np.ndarray:
    """Return per-household monthly log-income volatility sigma_y.

    earner_counts shape: (N, 5) for [NONAG, AGLAB, SALARY, BUSINESS, FARM]
    Heuristic: more earner diversification -> lower volatility.
    """
    total = earner_counts.sum(axis=1)
    # Herfindahl across earner buckets (1 = single source, 0 = diversified)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total[:, None] > 0, earner_counts / np.where(total[:, None] > 0, total[:, None], 1), 0)
    herf = (share ** 2).sum(axis=1)
    # Sole-earner farm / business / aglab households are highest volatility
    farm_or_aglab = earner_counts[:, [1, 4]].sum(axis=1) > 0   # AGLAB or FARM
    salary_primary = earner_counts[:, 2] >= total / 2          # majority-salary
    sigma = np.where(salary_primary, 0.10, 0.20)
    sigma = sigma + 0.10 * (farm_or_aglab & (total <= 1))      # +0.10 if sole farm/aglab
    sigma = sigma * (1 + 0.5 * (herf - 1.0 / 5.0))             # scale by concentration
    return np.clip(sigma, 0.05, 0.45)


def seasonal_factor(months_idx: np.ndarray) -> np.ndarray:
    """Seasonal expense multiplier indexed by month-of-year (1..12).

    Two festive bumps (Oct-Nov: +15%) and a small lean stretch (Mar-Apr: -5%).
    """
    m = months_idx
    s = np.ones_like(m, dtype=np.float32)
    s = np.where((m == 10) | (m == 11), 1.15, s)
    s = np.where((m == 3) | (m == 4), 0.95, s)
    return s.astype(np.float32)


def simulate(hh: pd.DataFrame, cpi: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(RNG_SEED)
    cpi_series, cpi_anchor = select_24mo_cpi_window(cpi)
    m_t = cpi_series / cpi_anchor                              # (T,)
    print(f"  CPI window length: {len(m_t)}, anchor = {cpi_anchor}")

    # Calibrate Poisson rates from MI*
    rates = calibrate_shock_rates(hh, manifest)

    # Drop rows missing core anchors; impute zeros for the rest
    core = ["INCOME", "COTOTAL", "ASSETS", "DB5", "DB6A"]
    hh = hh.dropna(subset=["INCOME"]).reset_index(drop=True)
    for c in core:
        hh[c] = hh[c].fillna(0.0).astype(np.float64)

    N = len(hh)
    T = N_MONTHS
    print(f"  households simulated: {N:,}, T = {T} months")

    # Per-household constants
    Y = hh["INCOME"].to_numpy()                                # annual income
    C = hh["COTOTAL"].to_numpy().clip(min=1.0)                 # annual expense
    A = hh["ASSETS"].to_numpy().clip(min=0.0)                  # asset stock
    D0 = (hh["DB5"].fillna(0) + hh["DB6A"].fillna(0)).to_numpy()
    earner_counts = hh[manifest["earner_count_vars"]].fillna(0).to_numpy().astype(np.float32)
    sigma_y = income_volatility(earner_counts).astype(np.float32)
    sigma_x = np.full(N, 0.10, dtype=np.float32)               # consumption smoothing

    # Months-of-year for the 24-month window (Jan-2011 = month 1 ... Dec-2012 = month 12 wrap)
    start = pd.Timestamp(CPI_START_MONTH)
    moy = np.array([
        (start + pd.DateOffset(months=t)).month for t in range(T)
    ], dtype=np.int32)
    seasonal = seasonal_factor(moy)                            # (T,)

    # Base monthly anchors
    y_base = (Y[:, None] / 12.0) * m_t[None, :]                # (N, T)
    x_base = (C[:, None] / 12.0) * m_t[None, :] * seasonal[None, :]

    # Income / expense noise
    eps_y = rng.normal(0.0, sigma_y[:, None], size=(N, T)).astype(np.float32)
    eps_x = rng.normal(0.0, sigma_x[:, None], size=(N, T)).astype(np.float32)
    y = y_base * np.exp(eps_y - 0.5 * sigma_y[:, None] ** 2)
    x = x_base * np.exp(eps_x - 0.5 * sigma_x[:, None] ** 2)

    # ---- Shock injection ----
    # Medical: expense spike = lognormal(mu=log(0.5*income), sigma=0.8)
    p_med = rates["medical"]["monthly_poisson_rate"]
    p_dis = rates["disaster"]["monthly_poisson_rate"]
    p_crop = rates["crop_failure"]["monthly_poisson_rate"]

    # Restrict crop shock to households with farm earners (NWKFARM > 0)
    farm_mask = earner_counts[:, 4] > 0                        # (N,)

    sh_med = rng.binomial(1, p_med, size=(N, T)).astype(np.float32)
    sh_dis = rng.binomial(1, p_dis, size=(N, T)).astype(np.float32)
    sh_crop = rng.binomial(1, p_crop, size=(N, T)).astype(np.float32) * farm_mask[:, None]

    # Magnitudes (rupees), drawn from lognormals anchored on monthly income.
    # Clip Y to a small positive floor (~₹1000/yr) so log() is well-defined for
    # households with reported zero or negative annual income (rare, but present).
    Y_pos = np.clip(Y, 1000.0, None)
    med_amount  = np.exp(rng.normal(np.log(0.5 * (Y_pos[:, None] / 12.0)), 0.8, size=(N, T)))
    dis_amount  = np.exp(rng.normal(np.log(0.3 * (Y_pos[:, None] / 12.0)), 0.6, size=(N, T)))
    crop_drop   = rng.uniform(0.30, 0.70, size=(N, T)).astype(np.float32)  # multiplicative income drop

    # Apply shocks
    x = x + sh_med * med_amount + sh_dis * dis_amount
    y = y * (1.0 - sh_crop * crop_drop)

    # ---- Sequential debt / savings dynamics ----
    S = np.zeros((N, T), dtype=np.float32)
    D = np.zeros((N, T), dtype=np.float32)
    g = np.zeros((N, T), dtype=np.float32)
    S_prev = (INITIAL_SAVINGS_FRACTION * A).astype(np.float32)
    D_prev = D0.astype(np.float32)
    for t in range(T):
        service = DEBT_MONTHLY_INTEREST * D_prev
        gap_t = y[:, t] - x[:, t] - service
        g[:, t] = gap_t

        # Negative gap: pull from savings first; remainder pushed onto debt
        neg = -np.minimum(gap_t, 0.0)
        pull = np.minimum(neg, S_prev)
        roll_to_debt = neg - pull

        # Positive gap: top up savings; then repay debt up to 30% of monthly positive gap
        pos = np.maximum(gap_t, 0.0)
        repay = np.minimum(0.3 * pos, D_prev)
        save_add = pos - repay

        S_cur = np.clip(S_prev - pull + save_add, 0.0, None)
        D_cur = np.clip(D_prev + roll_to_debt - repay, 0.0, None)

        S[:, t] = S_cur
        D[:, t] = D_cur
        S_prev, D_prev = S_cur, D_cur

    # ---- Assemble long-format panel ----
    print("  assembling long-format DataFrame...")
    hh_id_arr = np.repeat(hh["hh_id"].to_numpy(), T)
    month_idx = np.tile(np.arange(1, T + 1), N)
    cpi_arr = np.tile(cpi_series, N)
    panel = pd.DataFrame({
        "hh_id":   hh_id_arr,
        "month":   month_idx,
        "income":  y.reshape(-1),
        "expense": x.reshape(-1),
        "gap":     g.reshape(-1),
        "savings": S.reshape(-1),
        "debt":    D.reshape(-1),
        "shock_medical":  sh_med.reshape(-1).astype(np.int8),
        "shock_disaster": sh_dis.reshape(-1).astype(np.int8),
        "shock_crop":     sh_crop.reshape(-1).astype(np.int8),
        "cpi":     cpi_arr,
    })

    print(f"  panel rows: {len(panel):,}")
    panel.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET}")

    calibration = {
        "n_households": int(N),
        "n_months": int(T),
        "cpi_window_start": CPI_START_MONTH,
        "cpi_anchor_month": CPI_ANCHOR_MONTH,
        "cpi_anchor_value": cpi_anchor,
        "debt_monthly_interest": DEBT_MONTHLY_INTEREST,
        "initial_savings_fraction": INITIAL_SAVINGS_FRACTION,
        "shock_rates": rates,
        "income_volatility_summary": {
            "mean": float(sigma_y.mean()),
            "p25":  float(np.percentile(sigma_y, 25)),
            "p50":  float(np.percentile(sigma_y, 50)),
            "p75":  float(np.percentile(sigma_y, 75)),
        },
    }
    with open(OUT_CAL_JSON, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"Wrote {OUT_CAL_JSON}")

    return panel, {"y": y, "x": x, "S": S, "D": D, "g": g,
                   "sh_med": sh_med, "sh_dis": sh_dis, "sh_crop": sh_crop,
                   "Y_anchor": Y, "C_anchor": C, "rates": rates, "hh": hh}


def make_validation_figures(arrays: dict, cpi_series: np.ndarray):
    print("Making validation figures...")
    y, x, S, D = arrays["y"], arrays["x"], arrays["S"], arrays["D"]
    Y_anchor, C_anchor = arrays["Y_anchor"], arrays["C_anchor"]
    sh_med, sh_dis, sh_crop = arrays["sh_med"], arrays["sh_dis"], arrays["sh_crop"]
    rates = arrays["rates"]
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10})

    rng = np.random.default_rng(0)
    months = np.arange(1, y.shape[1] + 1)

    # ---- Fig 15: 6 example trajectories (3 stable, 3 stressed) ----
    # Pick stable = first 3 households with no shocks AND positive ending savings
    no_shock = (sh_med.sum(axis=1) == 0) & (sh_dis.sum(axis=1) == 0) & (sh_crop.sum(axis=1) == 0)
    stable_idx = np.where(no_shock & (S[:, -1] > 0))[0][:3]
    # Stressed = households whose debt grows by more than 50% over the window
    growth = (D[:, -1] - D[:, 0])
    stressed_idx = np.argsort(-growth)[:3]
    pick = np.concatenate([stable_idx, stressed_idx])
    labels = ["Stable A", "Stable B", "Stable C", "Stressed A", "Stressed B", "Stressed C"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, idx, lbl in zip(axes.flat, pick, labels):
        ax.plot(months, y[idx] / 1000, label="Income", color="C0")
        ax.plot(months, x[idx] / 1000, label="Expense", color="C3")
        ax.plot(months, S[idx] / 1000, label="Savings", color="C2", ls="--")
        ax.plot(months, D[idx] / 1000, label="Debt", color="C1", ls=":")
        # Mark shocks
        sh_months = np.where(sh_med[idx] + sh_dis[idx] + sh_crop[idx] > 0)[0]
        for m in sh_months:
            ax.axvline(m + 1, color="purple", alpha=0.25, lw=2)
        ax.set_title(lbl, fontsize=10)
        ax.set_ylabel("₹ '000")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[-1, 1].set_xlabel("Month (1 = Jan 2011)")
    fig.suptitle("Simulated 24-month trajectories — three stable and three stressed households\n"
                 "Vertical purple lines mark shock months (medical / disaster / crop)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "15_simulator_trajectories.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 16: annual sums vs IHDS anchors (validation scatter) ----
    annual_income_sim = y.sum(axis=1)
    annual_expense_sim = x.sum(axis=1)
    sample = rng.choice(len(Y_anchor), size=min(2000, len(Y_anchor)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    ax.scatter(Y_anchor[sample] / 1e5, annual_income_sim[sample] / 1e5, s=4, alpha=0.4)
    lim = max(Y_anchor[sample].max(), annual_income_sim[sample].max()) / 1e5 * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("IHDS-II annual INCOME (₹ lakh)")
    ax.set_ylabel("Simulated annual sum (₹ lakh)")
    ax.set_title("Validation: simulated annual income matches IHDS anchor")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(C_anchor[sample] / 1e5, annual_expense_sim[sample] / 1e5, s=4, alpha=0.4, color="C3")
    lim2 = max(C_anchor[sample].max(), annual_expense_sim[sample].max()) / 1e5 * 1.05
    ax.plot([0, lim2], [0, lim2], "r--", lw=1)
    ax.set_xlabel("IHDS-II annual COTOTAL (₹ lakh)")
    ax.set_ylabel("Simulated annual sum (₹ lakh)")
    ax.set_title("Validation: simulated annual expense matches IHDS anchor")
    ax.set_xlim(0, lim2); ax.set_ylim(0, lim2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "16_simulator_validation_marginals.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 17: realised shock incidence vs target Poisson rate ----
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["medical", "disaster", "crop_failure"]
    targets = [rates[n]["monthly_poisson_rate"] for n in names]
    realised = [sh_med.mean(), sh_dis.mean(), sh_crop.mean()]
    width = 0.35
    xs = np.arange(len(names))
    ax.bar(xs - width / 2, targets, width, label="Target Poisson rate (prev/60)")
    ax.bar(xs + width / 2, realised, width, label="Realised monthly incidence")
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel("Monthly probability")
    ax.set_title("Shock-rate calibration: target vs realised")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "17_simulator_shock_calibration.png", bbox_inches="tight")
    plt.close(fig)

    print("  wrote figures/15, 16, 17")


def main():
    hh, cpi, manifest = load_inputs()
    panel, arrays = simulate(hh, cpi, manifest)
    cpi_series, _ = select_24mo_cpi_window(cpi)
    make_validation_figures(arrays, cpi_series)
    print("\nDone.")


if __name__ == "__main__":
    main()
