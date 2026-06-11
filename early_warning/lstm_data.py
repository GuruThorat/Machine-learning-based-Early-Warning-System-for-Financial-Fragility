"""
Sequence construction and train/val/test split for the LSTM early-warning model.

For each household h, the time-varying panel `panel_with_ffi.parquet` carries
24 months. For each prediction point t in [HISTORY_LEN, T - HORIZON], we build:

    X_seq[h, t]   = shape (HISTORY_LEN, n_time_features)   monthly time-series
                    spanning months (t - HISTORY_LEN + 1) .. t.
    X_static[h]   = shape (n_static_features,)             cross-sectional features.
    y_ffi[h, t]   = ffi_t at month t + HORIZON (point regression target).
    y_stress[h,t] = supervised binary stress event from `panel_supervised.parquet`.

Households are split 70/15/15 into train / val / test by hh_id (no household
appears in more than one split, preventing temporal leakage between train and
test sets).

The time features are z-scored using statistics computed on the training split
only. Static features are one-hot encoded using the cross-sectional pipeline's
column list. Outputs are pickled tensors ready for the LSTM training loop.

Outputs:
  early_warning/lstm_arrays/{train,val,test}_X_seq.npy
  early_warning/lstm_arrays/{train,val,test}_X_static.npy
  early_warning/lstm_arrays/{train,val,test}_y_ffi.npy
  early_warning/lstm_arrays/{train,val,test}_y_stress.npy
  early_warning/lstm_arrays/{train,val,test}_hh_id.npy
  early_warning/lstm_arrays/feature_spec.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "early_warning" / "panel_with_ffi.parquet"
SUP = ROOT / "early_warning" / "panel_supervised.parquet"
HH_FILE = ROOT / "DS0002" / "36151-0002-Data.dta"
OUT_DIR = ROOT / "early_warning" / "lstm_arrays"
OUT_DIR.mkdir(exist_ok=True)

HISTORY_LEN = 12
HORIZON = 3
TOTAL_MONTHS = 24
RNG_SEED = 42

TIME_FEATURES = [
    "log_income", "log_expense", "log_savings", "log_debt",
    "debt_ratio_z", "expense_vol_z", "savings_buffer_z",
    "ffi_t", "cpi_n",
    "shock_medical", "shock_disaster", "shock_crop",
]

STATIC_COLS = [
    "MHEADAGE", "FHEADAGE",
    "HHEDUC", "HHEDUCM", "HHEDUCF",
    "ID11", "ID13", "GROUPS",
    "URBAN2011", "URBAN4_2011", "METRO", "METRO6",
    "HQ1", "HQWALL", "HQROOF", "HQFLOOR",
    "WATER", "SATOILET", "SAKITCHEN",
    "FU1", "FULPG",
    "MG1",
]
CATEGORICAL = [
    "ID11", "ID13", "GROUPS",
    "URBAN2011", "URBAN4_2011", "METRO", "METRO6",
    "HQWALL", "HQROOF", "HQFLOOR",
    "WATER", "SATOILET", "SAKITCHEN", "FU1", "FULPG", "MG1",
]


def load_static() -> pd.DataFrame:
    print("Loading static features from DS0002...")
    cols = ["STATEID", "DISTID", "PSUID", "HHID", "HHSPLITID"] + STATIC_COLS
    df = pd.read_stata(HH_FILE, columns=cols, convert_categoricals=False)
    for code in (-9, -8, -7):
        df = df.replace(code, np.nan)
    keys = ["STATEID", "DISTID", "PSUID", "HHID", "HHSPLITID"]
    df["hh_id"] = df[keys].astype(str).agg("-".join, axis=1)
    return df.drop(columns=keys)


def prepare_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    print("Preparing time-feature columns...")
    panel = panel.copy()
    panel["log_income"]  = np.log1p(panel["income"].clip(lower=0))
    panel["log_expense"] = np.log1p(panel["expense"].clip(lower=0))
    panel["log_savings"] = np.log1p(panel["savings"].clip(lower=0))
    panel["log_debt"]    = np.log1p(panel["debt"].clip(lower=0))
    panel["cpi_n"]       = (panel["cpi"] - panel["cpi"].mean()) / panel["cpi"].std()
    # The first month per household has NaN expense_vol_z / savings_buffer_z /
    # debt_ratio_z / ffi_t because the 6-month rolling-window stat is undefined
    # at t=1. Z-scored features default cleanly to 0 (the pool mean).
    for col in ["expense_vol_z", "savings_buffer_z", "debt_ratio_z", "ffi_t"]:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0.0)
    return panel


def fit_scalers(train_df: pd.DataFrame) -> dict:
    """Compute mean/std per time-feature on the training rows only."""
    stats = {}
    for c in TIME_FEATURES:
        if c.startswith("shock_"):
            stats[c] = {"mean": 0.0, "std": 1.0}
            continue
        mu = float(train_df[c].mean()); sd = float(train_df[c].std())
        if sd < 1e-8:
            sd = 1.0
        stats[c] = {"mean": mu, "std": sd}
    return stats


def apply_scalers(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    for c, s in stats.items():
        df[c] = (df[c] - s["mean"]) / s["std"]
    return df


def build_sequences(panel: pd.DataFrame, sup: pd.DataFrame,
                    static_oh: pd.DataFrame, hh_ids: np.ndarray):
    """Build (X_seq, X_static, y_ffi, y_stress, hh_id_per_row) for households in `hh_ids`."""
    print(f"  building sequences for {len(hh_ids):,} households...")
    panel_sub = panel[panel["hh_id"].isin(set(hh_ids))]
    panel_idx = panel_sub.set_index(["hh_id", "month"]).sort_index()
    panel_arr = panel_idx[TIME_FEATURES].to_numpy(dtype=np.float32)
    # Index lookup
    hh_to_offset = {}
    for hh, sub in panel_idx.groupby(level=0, sort=False):
        hh_to_offset[hh] = sub.index.get_level_values("month").to_numpy()

    sup_sub = sup[sup["hh_id"].isin(set(hh_ids))]
    sup_sub = sup_sub.merge(
        panel_idx[["ffi_t"]].reset_index().rename(columns={"month": "month_t"}),
        left_on=["hh_id"], right_on=["hh_id"], how="left", suffixes=("", "_lookup"),
    )
    # Skip rows where the requested history window is incomplete
    sup_sub = sup_sub[(sup_sub["month"] >= HISTORY_LEN)
                    & (sup_sub["month"] <= TOTAL_MONTHS - HORIZON)]
    sup_sub = sup_sub[["hh_id", "month", "y_stress"]].drop_duplicates().reset_index(drop=True)

    # Build arrays
    N = len(sup_sub)
    X_seq = np.zeros((N, HISTORY_LEN, len(TIME_FEATURES)), dtype=np.float32)
    y_ffi = np.zeros(N, dtype=np.float32)
    y_stress = np.zeros(N, dtype=np.int8)
    hh_arr = np.empty(N, dtype=object)

    # panel reshape: (n_hh, T, n_feat)
    grouped = panel_sub.sort_values(["hh_id", "month"]).set_index(["hh_id", "month"])
    hh_list = sorted(set(panel_sub["hh_id"]))
    hh_to_idx = {hh: i for i, hh in enumerate(hh_list)}
    grid = (
        grouped.reindex(
            pd.MultiIndex.from_product([hh_list, range(1, TOTAL_MONTHS + 1)],
                                       names=["hh_id", "month"])
        )[TIME_FEATURES]
        .to_numpy(dtype=np.float32)
        .reshape(len(hh_list), TOTAL_MONTHS, len(TIME_FEATURES))
    )
    ffi_grid = (
        grouped.reindex(
            pd.MultiIndex.from_product([hh_list, range(1, TOTAL_MONTHS + 1)],
                                       names=["hh_id", "month"])
        )["ffi_t"]
        .to_numpy(dtype=np.float32)
        .reshape(len(hh_list), TOTAL_MONTHS)
    )

    for i, row in sup_sub.iterrows():
        hh = row["hh_id"]; t = int(row["month"])
        idx = hh_to_idx[hh]
        # History window: months [t - HISTORY_LEN + 1, t] (inclusive)
        start = t - HISTORY_LEN
        X_seq[i] = grid[idx, start:t, :]      # rows for months start+1 .. t in 1-indexed
        # ffi target at month t + HORIZON
        y_ffi[i] = ffi_grid[idx, t + HORIZON - 1]
        y_stress[i] = int(row["y_stress"])
        hh_arr[i] = hh

    # Static features (broadcast across rows)
    static_lookup = static_oh.set_index("hh_id")
    static_mat = static_lookup.loc[hh_arr].to_numpy(dtype=np.float32)
    return X_seq, static_mat, y_ffi, y_stress, hh_arr


def main():
    print("Loading panels...")
    panel = pd.read_parquet(PANEL)
    sup = pd.read_parquet(SUP)
    panel = prepare_time_features(panel)
    print(f"  panel rows: {len(panel):,}  supervised rows: {len(sup):,}")

    static = load_static()
    # One-hot encode categorical static features
    static_oh = pd.get_dummies(
        static, columns=[c for c in CATEGORICAL if c in static.columns],
        dummy_na=True, drop_first=True,
    )
    # Impute remaining numeric NaNs with column median
    for c in static_oh.columns:
        if c == "hh_id":
            continue
        if static_oh[c].isna().any():
            static_oh[c] = static_oh[c].fillna(static_oh[c].median())
    static_oh = static_oh.astype({c: np.float32 for c in static_oh.columns if c != "hh_id"})
    print(f"  static after one-hot: {static_oh.shape}")

    # Train/val/test split by household
    rng = np.random.default_rng(RNG_SEED)
    all_hh = panel["hh_id"].unique()
    rng.shuffle(all_hh)
    n = len(all_hh)
    n_tr = int(0.70 * n); n_va = int(0.15 * n)
    hh_train = all_hh[:n_tr]; hh_val = all_hh[n_tr:n_tr + n_va]; hh_test = all_hh[n_tr + n_va:]
    print(f"  split: train={len(hh_train):,} val={len(hh_val):,} test={len(hh_test):,}")

    # Fit scalers on training panel only
    print("Fitting scalers on training rows...")
    train_panel = panel[panel["hh_id"].isin(set(hh_train))]
    stats = fit_scalers(train_panel)
    panel_s = apply_scalers(panel, stats)

    # Build sequences for each split
    def _clean(X_seq, X_st, y_ffi, y_s, hh, name):
        # Targets must be finite; X is fillna'd to zero in-place (z-scored features
        # default cleanly to the pool mean of 0).
        bad = np.isnan(y_ffi) | np.isinf(y_ffi)
        n_bad = int(bad.sum())
        if n_bad:
            print(f"  {name}: dropping {n_bad:,} rows with NaN target "
                  f"({n_bad / len(y_ffi):.2%} of {len(y_ffi):,})")
        keep = ~bad
        X_seq = np.nan_to_num(X_seq[keep], nan=0.0, posinf=0.0, neginf=0.0)
        X_st  = np.nan_to_num(X_st[keep],  nan=0.0, posinf=0.0, neginf=0.0)
        return X_seq, X_st, y_ffi[keep], y_s[keep], hh[keep]

    print("Building train sequences...")
    Xtr, Str, ytr_ffi, ytr_stress, hh_tr = build_sequences(panel_s, sup, static_oh, hh_train)
    Xtr, Str, ytr_ffi, ytr_stress, hh_tr = _clean(Xtr, Str, ytr_ffi, ytr_stress, hh_tr, "train")
    print(f"  train: X_seq {Xtr.shape}  static {Str.shape}  pos rate {ytr_stress.mean():.3f}")
    print("Building val sequences...")
    Xva, Sva, yva_ffi, yva_stress, hh_va = build_sequences(panel_s, sup, static_oh, hh_val)
    Xva, Sva, yva_ffi, yva_stress, hh_va = _clean(Xva, Sva, yva_ffi, yva_stress, hh_va, "val")
    print(f"  val:   X_seq {Xva.shape}  static {Sva.shape}  pos rate {yva_stress.mean():.3f}")
    print("Building test sequences...")
    Xte, Ste, yte_ffi, yte_stress, hh_te = build_sequences(panel_s, sup, static_oh, hh_test)
    Xte, Ste, yte_ffi, yte_stress, hh_te = _clean(Xte, Ste, yte_ffi, yte_stress, hh_te, "test")
    print(f"  test:  X_seq {Xte.shape}  static {Ste.shape}  pos rate {yte_stress.mean():.3f}")

    np.save(OUT_DIR / "train_X_seq.npy", Xtr); np.save(OUT_DIR / "train_X_static.npy", Str)
    np.save(OUT_DIR / "train_y_ffi.npy", ytr_ffi); np.save(OUT_DIR / "train_y_stress.npy", ytr_stress)
    np.save(OUT_DIR / "train_hh_id.npy", hh_tr)
    np.save(OUT_DIR / "val_X_seq.npy", Xva); np.save(OUT_DIR / "val_X_static.npy", Sva)
    np.save(OUT_DIR / "val_y_ffi.npy", yva_ffi); np.save(OUT_DIR / "val_y_stress.npy", yva_stress)
    np.save(OUT_DIR / "val_hh_id.npy", hh_va)
    np.save(OUT_DIR / "test_X_seq.npy", Xte); np.save(OUT_DIR / "test_X_static.npy", Ste)
    np.save(OUT_DIR / "test_y_ffi.npy", yte_ffi); np.save(OUT_DIR / "test_y_stress.npy", yte_stress)
    np.save(OUT_DIR / "test_hh_id.npy", hh_te)

    spec = {
        "history_len": HISTORY_LEN, "horizon": HORIZON,
        "time_features": TIME_FEATURES,
        "n_time_features": len(TIME_FEATURES),
        "static_features": [c for c in static_oh.columns if c != "hh_id"],
        "n_static_features": int(Str.shape[1]),
        "scalers": stats,
        "split": {
            "n_train_hh": int(len(hh_train)),
            "n_val_hh": int(len(hh_val)),
            "n_test_hh": int(len(hh_test)),
            "n_train_rows": int(len(ytr_stress)),
            "n_val_rows":   int(len(yva_stress)),
            "n_test_rows":  int(len(yte_stress)),
        },
    }
    with open(OUT_DIR / "feature_spec.json", "w") as f:
        json.dump(spec, f, indent=2, default=str)
    print(f"Wrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
