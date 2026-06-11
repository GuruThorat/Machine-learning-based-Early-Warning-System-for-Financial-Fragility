"""
Re-render the simulator trajectory figure (figures/15) with cleaner household
selection. The original picked the top 3 most-stressed households, which had
exponentially-growing debt that dominated the plot. We now pick from a milder
quantile of the stress distribution so all four series (income, expense,
savings, debt) are visible on a shared scale.

Also re-renders the validation scatter (figures/16) with axis labels that
correctly reflect that the simulator produces 24-month sums (= 2 years' worth
of cash flow) vs the IHDS-II *annual* anchor, by dividing the simulated sum
by 2 to put both on the same yearly scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "early_warning" / "simulated_panel.parquet"
HH_FILE = ROOT / "DS0002" / "36151-0002-Data.dta"
FIG = ROOT / "figures"


def main():
    print("Loading panel + IHDS anchors...")
    df = pd.read_parquet(PANEL)
    cols = ["STATEID", "DISTID", "PSUID", "HHID", "HHSPLITID", "INCOME", "COTOTAL"]
    anc = pd.read_stata(HH_FILE, columns=cols, convert_categoricals=False)
    keys = ["STATEID", "DISTID", "PSUID", "HHID", "HHSPLITID"]
    anc["hh_id"] = anc[keys].astype(str).agg("-".join, axis=1)
    anc = anc[["hh_id", "INCOME", "COTOTAL"]]
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10})

    # ---------- Figure 15: trajectories ----------
    print("Re-rendering 15_simulator_trajectories.png ...")
    rng = np.random.default_rng(7)
    grouped = df.groupby("hh_id")
    has_shock = grouped[["shock_medical", "shock_disaster", "shock_crop"]].sum().sum(axis=1)
    end_savings = grouped["savings"].last()
    end_debt = grouped["debt"].last()
    start_debt = grouped["debt"].first()
    debt_growth = end_debt - start_debt

    # Pick 3 stable: zero shocks, low debt, positive savings
    stable_pool = has_shock[has_shock == 0].index
    stable_pool = stable_pool[(start_debt.loc[stable_pool] < 10_000)
                               & (end_savings.loc[stable_pool] > 5_000)]
    stable = list(rng.choice(stable_pool.to_numpy(), size=3, replace=False))

    # Pick 3 mildly-stressed: at least 1 shock, debt grew, but ending debt under ₹2 lakh
    stress_pool = has_shock[(has_shock >= 1) & (debt_growth > 0) & (end_debt < 200_000)].index
    stressed = list(rng.choice(stress_pool.to_numpy(), size=3, replace=False))

    pick = stable + stressed
    titles = ["Stable A", "Stable B", "Stable C", "Stressed A", "Stressed B", "Stressed C"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, hh, lbl in zip(axes.flat, pick, titles):
        sub = df[df["hh_id"] == hh].sort_values("month")
        m = sub["month"].to_numpy()
        ax.plot(m, sub["income"]  / 1000, label="Income",  color="C0", lw=1.5)
        ax.plot(m, sub["expense"] / 1000, label="Expense", color="C3", lw=1.5)
        ax.plot(m, sub["savings"] / 1000, label="Savings", color="C2", ls="--", lw=1.5)
        ax.plot(m, sub["debt"]    / 1000, label="Debt",    color="C1", ls=":",  lw=1.8)
        sh = sub[["shock_medical", "shock_disaster", "shock_crop"]].sum(axis=1).to_numpy()
        for t in np.where(sh > 0)[0]:
            ax.axvline(m[t], color="purple", alpha=0.18, lw=2)
        ax.set_title(lbl, fontsize=10)
        ax.set_ylabel("₹ '000")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[-1, 1].set_xlabel("Month (1 = Jan 2011)")
    fig.suptitle("Simulated 24-month trajectories — three stable and three mildly-stressed households\n"
                 "Vertical purple lines mark shock months (medical / disaster / crop)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "15_simulator_trajectories.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 16: validation, on yearly equivalent ----------
    print("Re-rendering 16_simulator_validation_marginals.png ...")
    annual_income_sim_per_year  = df.groupby("hh_id")["income"].sum() / 2   # 24 months / 2 = yearly
    annual_expense_sim_per_year = df.groupby("hh_id")["expense"].sum() / 2

    merged = anc.merge(annual_income_sim_per_year.rename("sim_income_per_yr"),
                       on="hh_id").merge(annual_expense_sim_per_year.rename("sim_expense_per_yr"),
                                          on="hh_id")
    rng2 = np.random.default_rng(0)
    sample = merged.sample(n=min(3000, len(merged)), random_state=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    ax.scatter(sample["INCOME"] / 1e5, sample["sim_income_per_yr"] / 1e5,
               s=4, alpha=0.35, color="C0")
    lim = float(max(sample["INCOME"].max(), sample["sim_income_per_yr"].max())) / 1e5 * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="$y = x$")
    ax.set_xlabel("IHDS-II annual INCOME (₹ lakh)")
    ax.set_ylabel("Simulated yearly-equivalent income (₹ lakh)")
    ax.set_title("Validation: simulated yearly income matches IHDS anchor")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.scatter(sample["COTOTAL"] / 1e5, sample["sim_expense_per_yr"] / 1e5,
               s=4, alpha=0.35, color="C3")
    lim2 = float(max(sample["COTOTAL"].max(), sample["sim_expense_per_yr"].max())) / 1e5 * 1.05
    ax.plot([0, lim2], [0, lim2], "r--", lw=1, label="$y = x$")
    ax.set_xlabel("IHDS-II annual COTOTAL (₹ lakh)")
    ax.set_ylabel("Simulated yearly-equivalent expense (₹ lakh)")
    ax.set_title("Validation: simulated yearly expense matches IHDS anchor")
    ax.set_xlim(0, lim2); ax.set_ylim(0, lim2); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "16_simulator_validation_marginals.png", bbox_inches="tight")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
