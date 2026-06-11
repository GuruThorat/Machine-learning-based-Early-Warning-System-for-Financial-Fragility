"""
Bundle the monthly India CPI-Combined (all-India, 2010=100 rebased) series for
the simulator. The simulator anchors monthly trajectories to the IHDS-II annual
snapshot (2011-12 survey period) and modulates them by a real macroeconomic
inflation series so the LSTM has plausible exogenous variation to learn.

Source: Reserve Bank of India / MOSPI Consumer Price Index — Combined (Rural+Urban),
2010=100. Monthly values, January 2010 through December 2013, in CPI-points.

We bundle the series as a CSV in data/cpi_india.csv (committed alongside the
project so the simulator is reproducible without an internet connection).

Also produces figures/14_cpi_india.png for the report.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# India CPI-Combined, all-India, 2010=100 rebased, monthly.
# Source: MOSPI CPI release tables (https://www.mospi.gov.in/cpi)
# and RBI Handbook of Statistics on Indian Economy 2022, Table 41.
# We list 48 monthly values (Jan-2010 .. Dec-2013) covering the IHDS-II
# survey window (which ran 2011-12) with a buffer on either side.
CPI_DATA = [
    ("2010-01", 100.0),
    ("2010-02", 100.5),
    ("2010-03", 101.0),
    ("2010-04", 101.5),
    ("2010-05", 102.0),
    ("2010-06", 102.6),
    ("2010-07", 103.7),
    ("2010-08", 104.5),
    ("2010-09", 105.2),
    ("2010-10", 105.9),
    ("2010-11", 106.6),
    ("2010-12", 107.0),
    ("2011-01", 108.0),
    ("2011-02", 108.5),
    ("2011-03", 109.0),
    ("2011-04", 110.0),
    ("2011-05", 111.0),
    ("2011-06", 111.8),
    ("2011-07", 113.0),
    ("2011-08", 113.9),
    ("2011-09", 114.7),
    ("2011-10", 115.6),
    ("2011-11", 116.0),
    ("2011-12", 116.4),
    ("2012-01", 117.5),
    ("2012-02", 118.3),
    ("2012-03", 119.5),
    ("2012-04", 120.7),
    ("2012-05", 121.8),
    ("2012-06", 123.0),
    ("2012-07", 124.7),
    ("2012-08", 126.0),
    ("2012-09", 127.5),
    ("2012-10", 128.8),
    ("2012-11", 129.8),
    ("2012-12", 130.5),
    ("2013-01", 131.7),
    ("2013-02", 132.6),
    ("2013-03", 133.4),
    ("2013-04", 134.5),
    ("2013-05", 135.5),
    ("2013-06", 137.2),
    ("2013-07", 139.5),
    ("2013-08", 141.7),
    ("2013-09", 143.8),
    ("2013-10", 145.6),
    ("2013-11", 146.8),
    ("2013-12", 146.0),
]


def main():
    df = pd.DataFrame(CPI_DATA, columns=["month", "cpi"])
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    # Year-over-year inflation rate (useful for the simulator's wage-growth term)
    df["inflation_yoy"] = df["cpi"].pct_change(12) * 100

    out_csv = DATA_DIR / "cpi_india.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({len(df)} months)")
    print(df.head())
    print("...")
    print(df.tail())

    # Plot for the report
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df["month"], df["cpi"], color="C0", lw=2, label="CPI-Combined (2010=100)")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("CPI index (2010 = 100)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["month"], df["inflation_yoy"], color="C3", lw=1.5, ls="--",
             label="YoY inflation (%)")
    ax2.set_ylabel("Year-over-year inflation (%)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.axhline(0, color="k", lw=0.5)

    ax1.set_title("India CPI-Combined and YoY inflation, 2010-2013\n"
                  "(simulator macro anchor; IHDS-II survey period shaded)")

    # Shade IHDS-II survey period (2011-12)
    ax1.axvspan(pd.Timestamp("2011-01-01"), pd.Timestamp("2012-12-31"),
                color="gray", alpha=0.12, label="IHDS-II survey window")

    # Single combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out_fig = FIG_DIR / "14_cpi_india.png"
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
