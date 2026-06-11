"""Find which input tensor / column is contaminated with NaN or Inf."""
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent
ARR = ROOT / "early_warning" / "lstm_arrays"

with open(ARR / "feature_spec.json") as f:
    spec = json.load(f)
tf = spec["time_features"]

for split in ["train", "val", "test"]:
    X = np.load(ARR / f"{split}_X_seq.npy")
    S = np.load(ARR / f"{split}_X_static.npy")
    y_ffi = np.load(ARR / f"{split}_y_ffi.npy")
    y_s = np.load(ARR / f"{split}_y_stress.npy")
    print(f"\n=== {split} ===")
    print(f"X_seq:    nan={np.isnan(X).sum():,}  inf={np.isinf(X).sum():,}  shape={X.shape}")
    print(f"X_static: nan={np.isnan(S).sum():,}  inf={np.isinf(S).sum():,}  shape={S.shape}")
    print(f"y_ffi:    nan={np.isnan(y_ffi).sum():,}  inf={np.isinf(y_ffi).sum():,}  shape={y_ffi.shape}")
    if np.isnan(X).any() or np.isinf(X).any():
        print("  per-feature counts (time dim):")
        flat = X.reshape(-1, X.shape[-1])
        for i, name in enumerate(tf):
            n_nan = np.isnan(flat[:, i]).sum()
            n_inf = np.isinf(flat[:, i]).sum()
            mn = np.nanmin(flat[:, i]); mx = np.nanmax(flat[:, i])
            print(f"    [{i:2d}] {name:20s} nan={n_nan:>8,} inf={n_inf:>8,}  range=[{mn: .3f}, {mx: .3f}]")
