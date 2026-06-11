"""Sanity check: confirm all newly-installed packages import and MPS is available."""
import importlib, sys

packages = ["torch", "ngboost", "jupytext", "statsmodels", "shap", "lightgbm"]
print(f"Python {sys.version.split()[0]}")
for p in packages:
    try:
        m = importlib.import_module(p)
        print(f"  OK   {p:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  MISS {p:14s} {type(e).__name__}: {e}")

import torch
print()
print(f"torch MPS available: {torch.backends.mps.is_available()}")
print(f"torch CUDA available: {torch.cuda.is_available()}")
print(f"Default device pick: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
