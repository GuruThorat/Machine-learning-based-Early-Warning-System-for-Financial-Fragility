"""
MC Dropout inference for the trained LSTM early-warning model.

Gal & Ghahramani (2016) show that a neural network trained with dropout, with
dropout left active at inference, is mathematically equivalent to a variational
approximation to a deep Gaussian process. K stochastic forward passes give K
samples from the approximate posterior over predictions; their mean is the
posterior point estimate and their dispersion is the (epistemic) uncertainty.

This script:
  1. loads the checkpoint and the test arrays
  2. with dropout layers in train() mode and grad disabled, runs K=50 forward passes
  3. assembles per-row predictive distributions for FFI_{t+3} (regression) and
     for the stress-event probability (classification)
  4. derives Low / Medium / High risk tiers from the posterior:
        High   = upper 95% credible bound of p(stress) > 0.5
        Medium = posterior mean of p(stress)         > 0.5
        Low    = otherwise
  5. saves outputs for downstream evaluation against the baselines

Outputs:
  early_warning/lstm_arrays/test_predictions.npz
    keys: ffi_mean, ffi_std, ffi_low95, ffi_high95,
          stress_mean, stress_std, stress_low95, stress_high95,
          risk_tier (int8: 0=Low, 1=Med, 2=High)
  early_warning/lstm_arrays/test_predictions_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lstm_train import LSTMEarlyWarning, pick_device

ROOT = Path(__file__).resolve().parent.parent
ARR = ROOT / "early_warning" / "lstm_arrays"
CKPT = ARR / "lstm_checkpoint.pt"
OUT_NPZ = ARR / "test_predictions.npz"
OUT_JSON = ARR / "test_predictions_summary.json"

K_MC = 50
BATCH = 512


def main():
    device = pick_device()
    print(f"Device: {device}")

    # Reproducibility for the MC samples (do not affect a re-run after retraining)
    torch.manual_seed(123)
    np.random.seed(123)

    with open(ARR / "feature_spec.json") as f:
        spec = json.load(f)
    n_time = spec["n_time_features"]; n_static = spec["n_static_features"]

    ckpt = torch.load(CKPT, map_location=device)
    model = LSTMEarlyWarning(n_time, n_static).to(device)
    model.load_state_dict(ckpt["state_dict"])

    # Critical: leave model in train() mode so dropout layers stay active at
    # inference. PyTorch's nn.LSTM dropout is *only* applied during train()
    # mode AND only between stacked layers (which is exactly what we have, 2-layer).
    model.train()

    X_seq = torch.from_numpy(np.load(ARR / "test_X_seq.npy"))
    X_st  = torch.from_numpy(np.load(ARR / "test_X_static.npy"))
    y_ffi = np.load(ARR / "test_y_ffi.npy")
    y_s   = np.load(ARR / "test_y_stress.npy")
    print(f"  test rows: {len(y_s):,}")

    loader = DataLoader(
        TensorDataset(X_seq, X_st), batch_size=BATCH, shuffle=False, num_workers=0,
    )

    # Pre-allocate per-sample collections to save memory
    N = len(y_s)
    ffi_samples = np.zeros((K_MC, N), dtype=np.float32)
    p_samples   = np.zeros((K_MC, N), dtype=np.float32)

    print(f"Running {K_MC} MC-Dropout forward passes (dropout active)...")
    with torch.no_grad():
        for k in range(K_MC):
            ptr = 0
            for X1, X2 in loader:
                X1 = X1.to(device); X2 = X2.to(device)
                out = model(X1, X2)
                b = out.shape[0]
                ffi_samples[k, ptr:ptr + b] = out[:, 0].cpu().numpy()
                p_samples[k,   ptr:ptr + b] = torch.sigmoid(out[:, 1]).cpu().numpy()
                ptr += b
            if (k + 1) % 10 == 0:
                print(f"  sample {k+1}/{K_MC} done")

    # Posterior summaries
    ffi_mean  = ffi_samples.mean(axis=0)
    ffi_std   = ffi_samples.std(axis=0)
    ffi_low95 = np.percentile(ffi_samples, 2.5,  axis=0)
    ffi_high95= np.percentile(ffi_samples, 97.5, axis=0)

    p_mean   = p_samples.mean(axis=0)
    p_std    = p_samples.std(axis=0)
    p_low95  = np.percentile(p_samples, 2.5,  axis=0)
    p_high95 = np.percentile(p_samples, 97.5, axis=0)

    # Risk-tier mapping that *actually uses* the posterior uncertainty.
    # The naive "High = upper > 0.5, Medium = mean > 0.5" rule collapses to two
    # tiers because mean <= upper always, so we cannot have Medium with the
    # upper bound below the cutoff. The Bayesian three-tier rule we want is:
    #
    #   High    = lower bound > 0.5     (confidently in stress)
    #   Medium  = mean        > 0.5  but lower bound <= 0.5   (likely-but-uncertain)
    #   Low     = mean        <= 0.5
    #
    # Then "Medium" precisely captures predictions whose posterior credible
    # interval straddles the decision boundary, which is the right operational
    # definition of "uncertain" for policy triage.
    tier = np.where(
        p_low95 > 0.5, 2,
        np.where(p_mean > 0.5, 1, 0),
    ).astype(np.int8)

    counts = {f"tier_{t}": int((tier == t).sum()) for t in [0, 1, 2]}
    pos = y_s == 1
    capture = {
        f"tier_{t}_positive_share": float(((tier == t) & pos).sum() / max(pos.sum(), 1))
        for t in [0, 1, 2]
    }
    purity = {
        f"tier_{t}_purity": (
            float(pos[(tier == t)].mean()) if (tier == t).any() else None
        )
        for t in [0, 1, 2]
    }

    print(f"  Tier counts: {counts}")
    print(f"  Positives captured by tier: {capture}")
    print(f"  Tier purity (= positive rate within tier): {purity}")

    np.savez(
        OUT_NPZ,
        ffi_mean=ffi_mean, ffi_std=ffi_std,
        ffi_low95=ffi_low95, ffi_high95=ffi_high95,
        stress_mean=p_mean, stress_std=p_std,
        stress_low95=p_low95, stress_high95=p_high95,
        risk_tier=tier,
        y_ffi=y_ffi, y_stress=y_s,
    )
    print(f"Wrote {OUT_NPZ}")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "K_MC": K_MC,
            "device": str(device),
            "tier_counts": counts,
            "tier_capture_of_positives": capture,
            "tier_purity": purity,
            "mean_predictive_std_ffi": float(ffi_std.mean()),
            "mean_predictive_std_p":   float(p_std.mean()),
        }, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
