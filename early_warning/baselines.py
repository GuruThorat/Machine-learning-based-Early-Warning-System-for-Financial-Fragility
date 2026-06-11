"""
Three baselines the LSTM must beat:

  (1) Naive carry-forward: y_hat_{t+3} = y_t. The brick wall every time-series
      model has to clear.

  (2) NGBoost (Duan et al. 2020): natural-gradient boosting that natively
      predicts a parametric distribution (Normal mean+std for regression,
      Bernoulli for classification) rather than a point estimate. We fit on
      lagged-window features (the same 12-month window used by the LSTM,
      flattened to a vector + static features).

  (3) Kalman filter (Unobserved Components state-space model): treats
      ffi_t as a local level + AR(1) process. Forecasts 3 steps ahead per
      household. Provides a principled mean+variance forecast.

For each baseline we produce predictions for the held-out test households.
The regression target is FFI_{t+3} (in z-score units). The binary target is
the same y_stress (max ffi in next 3 months > tau).

Outputs:
  early_warning/lstm_arrays/baseline_naive.npz
  early_warning/lstm_arrays/baseline_ngboost.npz
  early_warning/lstm_arrays/baseline_kalman.npz
  early_warning/lstm_arrays/baselines_summary.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Bernoulli, Normal
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
ARR = ROOT / "early_warning" / "lstm_arrays"
PANEL = ROOT / "early_warning" / "panel_with_ffi.parquet"
CAL = ROOT / "early_warning" / "ffi_calibration.json"
OUT_SUMMARY = ARR / "baselines_summary.json"

RNG = 42


def load_split_arrays():
    def _load(split):
        return {
            "X_seq":   np.load(ARR / f"{split}_X_seq.npy"),
            "X_st":    np.load(ARR / f"{split}_X_static.npy"),
            "y_ffi":   np.load(ARR / f"{split}_y_ffi.npy"),
            "y_str":   np.load(ARR / f"{split}_y_stress.npy"),
            "hh_id":   np.load(ARR / f"{split}_hh_id.npy", allow_pickle=True),
        }
    return _load("train"), _load("val"), _load("test")


def baseline_naive(test: dict, spec: dict, tau: float):
    """y_hat_{t+3} = ffi_t  (carry forward the latest observed FFI)."""
    print("\n--- Baseline 1: naive carry-forward ---")
    feat_idx = spec["time_features"].index("ffi_t")
    # ffi_t at the LAST history month (column-major position L-1 of the sequence)
    last_ffi = test["X_seq"][:, -1, feat_idx]
    # The time features were z-scored across the training pool. ffi_t was
    # already in z-units before scaling, so "carry-forward" predictions are
    # given in the same z-scaled units as y_ffi. To compare on the original
    # FFI scale we need to invert the per-feature scaler.
    s = spec["scalers"]["ffi_t"]
    ffi_pred = last_ffi * s["std"] + s["mean"]
    # Stress probability: 1 if ffi_pred > tau, else 0; for AUC we use the raw
    # forecast as a score (higher -> more likely stress).
    stress_score = ffi_pred
    auc = roc_auc_score(test["y_str"], stress_score)
    ap  = average_precision_score(test["y_str"], stress_score)
    rmse = float(np.sqrt(np.mean((ffi_pred - test["y_ffi"]) ** 2)))
    print(f"  AUC {auc:.4f}  AP {ap:.4f}  RMSE {rmse:.4f}")
    np.savez(ARR / "baseline_naive.npz",
             ffi_pred=ffi_pred, stress_score=stress_score,
             y_ffi=test["y_ffi"], y_stress=test["y_str"])
    return {"auc": float(auc), "ap": float(ap), "rmse_ffi": rmse}


def _flatten_sequence_features(seq: np.ndarray, static: np.ndarray) -> np.ndarray:
    """For NGBoost: turn (N, L, F_t) + (N, F_s) into a single (N, L*F_t + F_s) matrix."""
    n, L, F = seq.shape
    return np.concatenate([seq.reshape(n, L * F), static], axis=1).astype(np.float32)


def baseline_ngboost(train, val, test, spec):
    """NGBoost: probabilistic gradient boosting that learns p(y | x) = N(mu, sigma).

    For the binary head we use NGBClassifier with Bernoulli."""
    print("\n--- Baseline 2: NGBoost on lagged-window features ---")
    Xtr = _flatten_sequence_features(train["X_seq"], train["X_st"])
    Xva = _flatten_sequence_features(val["X_seq"],   val["X_st"])
    Xte = _flatten_sequence_features(test["X_seq"],  test["X_st"])
    print(f"  flattened: train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    # Subsample training to keep wall-clock manageable; NGBoost is single-threaded
    # and 295k x 204 features is ~2h at default settings. 80k rows is plenty.
    rng = np.random.default_rng(RNG)
    sub = rng.choice(len(Xtr), size=min(80_000, len(Xtr)), replace=False)
    Xtr_s = Xtr[sub]; ytr_ffi_s = train["y_ffi"][sub]; ytr_str_s = train["y_str"][sub]

    base = DecisionTreeRegressor(max_depth=5, min_samples_leaf=50)
    t0 = time.time()
    reg = NGBRegressor(Dist=Normal, Base=base, n_estimators=300,
                       learning_rate=0.04, verbose=False, random_state=RNG)
    reg.fit(Xtr_s, ytr_ffi_s, X_val=Xva, Y_val=val["y_ffi"], early_stopping_rounds=20)
    print(f"  NGBRegressor fit done in {time.time()-t0:.1f}s")

    t0 = time.time()
    clf = NGBClassifier(Dist=Bernoulli, Base=base, n_estimators=300,
                        learning_rate=0.04, verbose=False, random_state=RNG)
    clf.fit(Xtr_s, ytr_str_s.astype(int), X_val=Xva, Y_val=val["y_str"].astype(int),
            early_stopping_rounds=20)
    print(f"  NGBClassifier fit done in {time.time()-t0:.1f}s")

    # Predict on test
    ffi_dist = reg.pred_dist(Xte)
    ffi_mean = ffi_dist.mean(); ffi_std = ffi_dist.std()
    rmse = float(np.sqrt(np.mean((ffi_mean - test["y_ffi"]) ** 2)))

    # NGBClassifier supports the sklearn predict_proba(X) -> (n, n_classes).
    p_mean = clf.predict_proba(Xte)[:, 1]

    auc = roc_auc_score(test["y_str"], p_mean)
    ap  = average_precision_score(test["y_str"], p_mean)
    print(f"  AUC {auc:.4f}  AP {ap:.4f}  RMSE {rmse:.4f}")

    np.savez(ARR / "baseline_ngboost.npz",
             ffi_mean=ffi_mean.astype(np.float32),
             ffi_std=ffi_std.astype(np.float32),
             stress_mean=p_mean.astype(np.float32),
             y_ffi=test["y_ffi"], y_stress=test["y_str"])
    return {"auc": float(auc), "ap": float(ap), "rmse_ffi": rmse,
            "mean_pred_std_ffi": float(np.mean(ffi_std))}


def baseline_kalman(panel: pd.DataFrame, test: dict, spec: dict, tau_scaled: float):
    """Per-household Kalman filter / Unobserved Components on ffi_t.

    OPTIMISED: fit one state-space model per household on its FULL history
    (months 1..21, the latest training point shared across all test rows for
    that hh), then use the fitted filter to extract dynamic 3-step-ahead
    forecasts from each prediction-time prefix t in [12..21]. That cuts the
    per-row fitting cost from 10 fits/hh to 1 fit/hh, taking total wall-clock
    from ~60 minutes to ~10 minutes.
    """
    print("\n--- Baseline 3: Kalman / UnobservedComponents per household (1 fit / hh) ---")
    from collections import defaultdict
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    s = spec["scalers"]["ffi_t"]
    history_len = spec["history_len"]
    horizon = spec["horizon"]
    test_hh_unique = np.unique(test["hh_id"])

    # Time-varying FFI series per household, on the original z-score scale
    p = panel.set_index(["hh_id", "month"]).sort_index()
    ffi_series = {hh: p.loc[hh, "ffi_t"].values for hh in test_hh_unique}

    # Build hh -> list of (row_idx, t) where t is the month of the last history obs
    hh_to_rows = defaultdict(list)
    for i, hh in enumerate(test["hh_id"]):
        hh_to_rows[hh].append(i)
    for hh in hh_to_rows:
        hh_to_rows[hh] = [(idx, history_len + off) for off, idx in enumerate(hh_to_rows[hh])]

    # We fit on the longest history available (max t per hh in the test set),
    # which equals TOTAL_MONTHS - HORIZON = 21 for almost all hh by construction.
    N = len(test["hh_id"])
    ffi_pred = np.zeros(N, dtype=np.float32)
    stress_score = np.zeros(N, dtype=np.float32)
    fail_count = 0
    t0 = time.time()
    n_hh = len(hh_to_rows)

    for i_hh, (hh, row_list) in enumerate(hh_to_rows.items()):
        series = ffi_series[hh]
        max_t = max(t for _, t in row_list)        # longest history we need
        fit_history = series[:max_t]
        try:
            mod = UnobservedComponents(fit_history, level="local level", autoregressive=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = mod.fit(disp=False, maxiter=30, method="lbfgs")
            # For each prediction-time prefix t, get dynamic 3-step-ahead forecasts
            for row_idx, t in row_list:
                # res.get_prediction(start=t, end=t+horizon-1, dynamic=True) gives
                # forecasts conditioned on data up to (start - 1).
                pred = res.get_prediction(start=t, end=t + horizon - 1, dynamic=True)
                mean = pred.predicted_mean
                # Per-prediction-time 3-step horizon prediction
                ffi_pred[row_idx] = mean[-1]
                stress_score[row_idx] = max(mean)
        except Exception:
            for row_idx, t in row_list:
                ffi_pred[row_idx] = series[t - 1] if t - 1 < len(series) else 0.0
                stress_score[row_idx] = ffi_pred[row_idx]
            fail_count += 1
        if (i_hh + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i_hh + 1) / elapsed
            eta = (n_hh - i_hh - 1) / rate
            print(f"  fitted {i_hh+1:,}/{n_hh:,} hh  ({rate:.1f} hh/s,  ETA {eta/60:.1f} min)")
    print(f"  Kalman finished in {time.time()-t0:.1f}s  (fallbacks: {fail_count})")

    # Both forecasts are on the original FFI scale; convert to z-scale to match y_ffi.
    # Fallback rows may contain NaN if the underlying series had NaN at the prefix
    # index; we substitute 0 (the z-scored pool mean) before computing metrics.
    ffi_pred_z = np.nan_to_num((ffi_pred - s["mean"]) / s["std"], nan=0.0, posinf=0.0, neginf=0.0)
    stress_score_z = np.nan_to_num((stress_score - s["mean"]) / s["std"], nan=0.0, posinf=0.0, neginf=0.0)

    rmse = float(np.sqrt(np.mean((ffi_pred_z - test["y_ffi"]) ** 2)))
    auc = roc_auc_score(test["y_str"], stress_score_z)
    ap  = average_precision_score(test["y_str"], stress_score_z)
    print(f"  AUC {auc:.4f}  AP {ap:.4f}  RMSE {rmse:.4f}")
    np.savez(ARR / "baseline_kalman.npz",
             ffi_pred=ffi_pred_z, stress_score=stress_score_z,
             y_ffi=test["y_ffi"], y_stress=test["y_str"])
    return {"auc": float(auc), "ap": float(ap), "rmse_ffi": rmse,
            "fallbacks": fail_count}


def main():
    with open(ARR / "feature_spec.json") as f:
        spec = json.load(f)
    with open(CAL) as f:
        ffi_cal = json.load(f)
    tau = float(ffi_cal["tau_75th_pct"])
    print(f"FFI threshold tau = {tau:.4f} (z-score scale)")

    train, val, test = load_split_arrays()
    print(f"Splits: train {len(train['y_ffi']):,}  val {len(val['y_ffi']):,}  test {len(test['y_ffi']):,}")

    results = {}

    # Re-use any already-saved baseline outputs to make this script restart-safe
    # (NGBoost is the bottleneck; we don't want to refit it on a re-run).
    def _reuse(name: str, runner):
        npz_path = ARR / f"baseline_{name}.npz"
        if npz_path.exists():
            print(f"\n--- {name}: re-using cached {npz_path.name} ---")
            d = np.load(npz_path)
            score_key = "stress_score" if name != "ngboost" else "stress_mean"
            stress = d[score_key]
            rmse = float(np.sqrt(np.mean((d["ffi_pred" if name != "ngboost" else "ffi_mean"]
                                          - d["y_ffi"]) ** 2)))
            auc = float(roc_auc_score(d["y_stress"], stress))
            ap  = float(average_precision_score(d["y_stress"], stress))
            print(f"  AUC {auc:.4f}  AP {ap:.4f}  RMSE {rmse:.4f}")
            return {"auc": auc, "ap": ap, "rmse_ffi": rmse, "reused": True}
        return runner()

    results["naive"]   = _reuse("naive",   lambda: baseline_naive(test, spec, tau))
    results["ngboost"] = _reuse("ngboost", lambda: baseline_ngboost(train, val, test, spec))
    panel = pd.read_parquet(PANEL)
    results["kalman"]  = _reuse("kalman",  lambda: baseline_kalman(panel, test, spec, tau))

    with open(OUT_SUMMARY, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_SUMMARY}")
    print("Summary:")
    for k, v in results.items():
        print(f"  {k:10s} AUC {v['auc']:.4f}  AP {v['ap']:.4f}  RMSE {v['rmse_ffi']:.4f}")


if __name__ == "__main__":
    main()
