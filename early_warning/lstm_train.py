"""
LSTM with MC Dropout for the joint regression / binary-classification of
financial fragility 3 months ahead.

Architecture:
    sequence input  (B, L=12, F_t=12)
       -> LSTM(F_t, hidden=64, num_layers=2, dropout=0.2)
       -> take last hidden state h_L  (B, 64)
       -> concat with static features  (B, 64 + F_s)
       -> MLP  Linear(124, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 2)
       -> outputs [ffi_pred, stress_logit]

Loss: 0.5 * MSE(ffi_pred, y_ffi)  +  BCEWithLogits(stress_logit, y_stress, pos_weight)

MC Dropout (Gal & Ghahramani 2016) makes the network a Bayesian approximation
by keeping dropout active at inference and treating K stochastic forward passes
as draws from the approximate posterior. We expose the dropout layers and the
model is set to .train() mode for inference sampling, with gradients disabled.

Outputs:
    early_warning/lstm_arrays/lstm_checkpoint.pt   (best-by-val-AUC weights)
    early_warning/lstm_arrays/lstm_train_log.json  (per-epoch losses and metrics)
    figures/21_lstm_training_curves.png            (train/val loss + val AUC vs epoch)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
ARR = ROOT / "early_warning" / "lstm_arrays"
FIG = ROOT / "figures"
CKPT = ARR / "lstm_checkpoint.pt"
LOG = ARR / "lstm_train_log.json"

EPOCHS = 30
BATCH = 512
LR = 1e-3
WD = 1e-5
PATIENCE = 5
SEED = 42

# --- Hyperparameters of the architecture ---
HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.2
LOSS_WEIGHT_REG = 0.5         # weight on MSE term


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LSTMEarlyWarning(nn.Module):
    def __init__(self, n_time_features: int, n_static_features: int,
                 hidden: int = HIDDEN, num_layers: int = LSTM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_time_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # MLP head on [last hidden state || static features]
        in_dim = hidden + n_static_features
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),                # [ffi_pred, stress_logit]
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_seq)
        h_last = h_n[-1]                          # (B, hidden), top-layer last hidden state
        z = torch.cat([h_last, x_static], dim=1)
        return self.head(z)                       # (B, 2)


def load_arrays():
    def _load(split):
        return (np.load(ARR / f"{split}_X_seq.npy"),
                np.load(ARR / f"{split}_X_static.npy"),
                np.load(ARR / f"{split}_y_ffi.npy"),
                np.load(ARR / f"{split}_y_stress.npy"))
    return _load("train"), _load("val"), _load("test")


def make_loader(X_seq, X_st, y_ffi, y_s, batch, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X_seq), torch.from_numpy(X_st),
        torch.from_numpy(y_ffi), torch.from_numpy(y_s.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, drop_last=False)


def evaluate(model, loader, device, pos_weight):
    model.eval()
    losses = []
    probs, ys, ffis_pred, ffis_true = [], [], [], []
    with torch.no_grad():
        for X_seq, X_st, y_ffi, y_s in loader:
            X_seq = X_seq.to(device); X_st = X_st.to(device)
            y_ffi = y_ffi.to(device); y_s = y_s.to(device)
            out = model(X_seq, X_st)
            ffi_pred, stress_logit = out[:, 0], out[:, 1]
            mse = F.mse_loss(ffi_pred, y_ffi)
            bce = F.binary_cross_entropy_with_logits(stress_logit, y_s, pos_weight=pos_weight)
            losses.append((LOSS_WEIGHT_REG * mse + bce).item())
            probs.append(torch.sigmoid(stress_logit).cpu().numpy())
            ys.append(y_s.cpu().numpy())
            ffis_pred.append(ffi_pred.cpu().numpy())
            ffis_true.append(y_ffi.cpu().numpy())
    probs = np.concatenate(probs); ys = np.concatenate(ys)
    ffis_pred = np.concatenate(ffis_pred); ffis_true = np.concatenate(ffis_true)
    return {
        "loss": float(np.mean(losses)),
        "auc":  float(roc_auc_score(ys, probs)),
        "ap":   float(average_precision_score(ys, probs)),
        "rmse_ffi": float(np.sqrt(np.mean((ffis_pred - ffis_true) ** 2))),
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = pick_device()
    print(f"Device: {device}")

    with open(ARR / "feature_spec.json") as f:
        spec = json.load(f)
    n_time = spec["n_time_features"]; n_static = spec["n_static_features"]
    print(f"  n_time_features={n_time}  n_static_features={n_static}")

    (Xtr, Str, ytr_ffi, ytr_s), (Xva, Sva, yva_ffi, yva_s), (Xte, Ste, yte_ffi, yte_s) = load_arrays()
    print(f"  train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    pos_weight = torch.tensor((1 - ytr_s.mean()) / max(ytr_s.mean(), 1e-6),
                              dtype=torch.float32, device=device)
    print(f"  pos_weight = {float(pos_weight):.3f}")

    tr_loader = make_loader(Xtr, Str, ytr_ffi, ytr_s, BATCH, shuffle=True)
    va_loader = make_loader(Xva, Sva, yva_ffi, yva_s, BATCH, shuffle=False)
    te_loader = make_loader(Xte, Ste, yte_ffi, yte_s, BATCH, shuffle=False)

    model = LSTMEarlyWarning(n_time, n_static).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")

    history = []
    best_val_auc = -np.inf
    bad_epochs = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_losses = []
        for X_seq, X_st, y_ffi, y_s in tr_loader:
            X_seq = X_seq.to(device); X_st = X_st.to(device)
            y_ffi = y_ffi.to(device); y_s = y_s.to(device)
            out = model(X_seq, X_st)
            ffi_pred, stress_logit = out[:, 0], out[:, 1]
            mse = F.mse_loss(ffi_pred, y_ffi)
            bce = F.binary_cross_entropy_with_logits(stress_logit, y_s, pos_weight=pos_weight)
            loss = LOSS_WEIGHT_REG * mse + bce
            opt.zero_grad(); loss.backward(); opt.step()
            ep_losses.append(loss.item())
        tr_loss = float(np.mean(ep_losses))
        val = evaluate(model, va_loader, device, pos_weight)
        history.append({
            "epoch": epoch, "train_loss": tr_loss,
            "val_loss": val["loss"], "val_auc": val["auc"],
            "val_ap": val["ap"], "val_rmse_ffi": val["rmse_ffi"],
        })
        print(f"  epoch {epoch:02d}  train {tr_loss:.4f}  val {val['loss']:.4f} "
              f"AUC {val['auc']:.4f}  AP {val['ap']:.4f}  RMSE {val['rmse_ffi']:.4f}")
        if val["auc"] > best_val_auc:
            best_val_auc = val["auc"]
            bad_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "n_time_features": n_time, "n_static_features": n_static,
                "epoch": epoch, "val_auc": val["auc"],
            }, CKPT)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  early stopping (no val-AUC improvement for {PATIENCE} epochs)")
                break
    secs = time.time() - t0

    # Load best, evaluate on test
    model.load_state_dict(torch.load(CKPT, map_location=device)["state_dict"])
    test_metrics = evaluate(model, te_loader, device, pos_weight)
    print(f"\nBest checkpoint val AUC = {best_val_auc:.4f}")
    print(f"Test metrics: AUC={test_metrics['auc']:.4f}  AP={test_metrics['ap']:.4f} "
          f"RMSE(FFI)={test_metrics['rmse_ffi']:.4f}  Loss={test_metrics['loss']:.4f}")

    with open(LOG, "w") as f:
        json.dump({
            "history": history,
            "best_val_auc": best_val_auc,
            "test": test_metrics,
            "wallclock_seconds": secs,
            "device": str(device),
            "hyperparams": {"hidden": HIDDEN, "lstm_layers": LSTM_LAYERS,
                            "dropout": DROPOUT, "loss_weight_reg": LOSS_WEIGHT_REG,
                            "lr": LR, "weight_decay": WD, "batch": BATCH,
                            "epochs_run": len(history)},
        }, f, indent=2)
    print(f"Wrote {LOG}")

    # Training curves figure
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history], label="Train loss")
    ax.plot(epochs, [h["val_loss"] for h in history], label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (0.5·MSE + BCE)")
    ax.set_title("LSTM training loss"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(epochs, [h["val_auc"] for h in history], color="C2", label="Val AUC")
    ax.plot(epochs, [h["val_ap"] for h in history], color="C3", ls="--", label="Val AP")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Metric")
    ax.set_title("Validation discrimination during training"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "21_lstm_training_curves.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {FIG / '21_lstm_training_curves.png'}")


if __name__ == "__main__":
    main()
