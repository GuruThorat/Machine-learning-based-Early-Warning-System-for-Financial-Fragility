"""
Patch the report's §4 results table (and tier table) with the actual numbers
produced by evaluate.py. Reads early_warning/lstm_arrays/ew_summary.json and
early_warning/lstm_arrays/test_predictions_summary.json and substitutes the
[NAIVE_AUC], [LSTM_ECE], etc. placeholders in report/main.tex.

Run AFTER baselines.py and evaluate.py have completed successfully.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUM = ROOT / "early_warning" / "lstm_arrays" / "ew_summary.json"
TIER = ROOT / "early_warning" / "lstm_arrays" / "test_predictions_summary.json"
TEX = ROOT / "report" / "main.tex"


def fmt(x, decimals=4):
    if x is None:
        return "---"
    return f"{x:.{decimals}f}"


def main():
    with open(SUM) as f:
        s = json.load(f)
    with open(TIER) as f:
        t = json.load(f)
    print("Loaded:", list(s.keys()))

    subs = {
        "NAIVE_AUC":   fmt(s["naive"]["auc"], 4),
        "NAIVE_AP":    fmt(s["naive"]["ap"], 4),
        "NAIVE_RMSE":  fmt(s["naive"]["rmse_ffi"], 4),
        "KALMAN_AUC":  fmt(s.get("kalman", {}).get("auc"), 4),
        "KALMAN_AP":   fmt(s.get("kalman", {}).get("ap"), 4),
        "KALMAN_RMSE": fmt(s.get("kalman", {}).get("rmse_ffi"), 4),
        "NGB_AUC":     fmt(s.get("ngboost", {}).get("auc"), 4),
        "NGB_AP":      fmt(s.get("ngboost", {}).get("ap"), 4),
        "NGB_RMSE":    fmt(s.get("ngboost", {}).get("rmse_ffi"), 4),
        "NGB_ECE":     fmt(s.get("ngboost", {}).get("ece"), 4),
        "LSTM_ECE":    fmt(s.get("lstm", {}).get("ece"), 4),
        "LSTM_AUC":    fmt(s.get("lstm", {}).get("auc"), 4),
        "MED_PUR":     fmt((t["tier_purity"].get("tier_1_purity") or 0) * 100, 1),
        "HIGH_PUR":    fmt((t["tier_purity"].get("tier_2_purity") or 0) * 100, 1),
    }
    print("Substitutions:")
    for k, v in subs.items():
        print(f"  {k:14s} -> {v}")

    # The LaTeX source uses backslash-escaped underscores inside \texttt{...},
    # so the literal placeholder to match is e.g. \texttt{[NAIVE\_AUC]}.
    text = TEX.read_text()
    for k, v in subs.items():
        k_tex = k.replace("_", r"\_")  # match the escaped form in the .tex source
        text = text.replace(f"\\texttt{{[{k_tex}]}}", v)
    TEX.write_text(text)
    print(f"Patched {TEX}")


if __name__ == "__main__":
    main()
