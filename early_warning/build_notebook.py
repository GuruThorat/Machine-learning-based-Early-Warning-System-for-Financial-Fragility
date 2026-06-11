"""
Assemble the final ID5030_Group7.ipynb deliverable from the .py source files.

Uses jupytext-style logic: each .py file becomes one or two cells in the
notebook, prefaced with a markdown narrative block. The notebook is *not*
re-executed here -- the .py files were run individually and their outputs
(parquet, npz, png) are already on disk. The notebook walks through the same
code in a documented form so a reader can re-run and reproduce.

Output: ID5030_Group7.ipynb (committed alongside the report PDF).
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_OUT = ROOT / "ID5030_Group7.ipynb"

# Each entry: (markdown narrative, path-to-py-file, optional path-to-figure-to-display)
SECTIONS = [
    (
        "# ID5030 Project — Group 7\n\n"
        "## Predicting Household Financial Fragility in India\n"
        "*A composite index, cross-sectional ML, and a Bayesian LSTM early-warning extension.*\n\n"
        "**Author:** Gururaj Thorat (ED23B065), Department of Engineering Design, IIT Madras.\n\n"
        "This notebook is the full reproducible source for the project. Each section "
        "loads the cleaned data, builds the relevant model, and saves outputs to disk. "
        "The companion report is in `Group7_Project_Report.pdf`.",
        None, None,
    ),
    (
        "## 1. Preprocessing IHDS-II\n\n"
        "Loads the household file `DS0002/36151-0002-Data.dta`, recodes the IHDS missing "
        "codes (`-9`, `-8`, `-7`) to NaN, joins selected individual-level fields from "
        "`DS0001`, and writes `ihds_preprocessed.parquet`.",
        ROOT / "preprocess_ihds.py", None,
    ),
    (
        "## 2. Building the Cross-Sectional Financial Fragility Index (FFI)\n\n"
        "The FFI is the equal-weighted mean of six z-standardised components: debt burden, "
        "consumption stress, asset deficit, employment concentration, dependency pressure, "
        "and a distress-borrowing flag. Households are split into quartiles "
        "(Stable / Stretched / Fragile / Distressed). The binary high-fragility label is "
        "`Fragile OR Distressed`.",
        ROOT / "build_ffi.py",
        ROOT / "figures" / "01_ffi_distribution.png",
    ),
    (
        "## 3. Cross-Sectional Classification (LR + RF + LightGBM)\n\n"
        "Predicts the binary high-fragility label from a **disjoint** non-financial covariate "
        "set (no income, debt, consumption, asset, or worker-type column appears as a predictor) "
        "to ensure no label leakage. 5-fold stratified CV on the training fold drives "
        "hyperparameter selection; an isotonic-calibrated LightGBM is the headline.",
        ROOT / "training_full.py",
        ROOT / "figures" / "06_roc_curves.png",
    ),
    (
        "## 4. Robustness Checks for the Cross-Sectional FFI\n\n"
        "Three sensitivity analyses: PCA-derived weights vs equal weights; top-tertile vs "
        "top-half cutoff; GroupKFold-by-state out-of-sample generalisation.",
        ROOT / "robustness_checks.py", None,
    ),
    (
        "## 5. Bundling the India CPI Series\n\n"
        "We anchor the monthly simulator to a real India CPI-Combined series (rebased to "
        "2010 = 100). The 48-month series, January 2010 through December 2013, is bundled "
        "as a CSV inside the project for reproducibility.",
        ROOT / "early_warning" / "bundle_cpi.py",
        ROOT / "figures" / "14_cpi_india.png",
    ),
    (
        "## 6. Macro-Anchored Trajectory Simulator\n\n"
        "For each household, builds a 24-month trajectory of (income, expense, savings, debt) "
        "from its IHDS-II annual anchor, modulated by CPI and damped by Poisson-distributed "
        "shocks (medical, disaster, crop-failure) at empirically-calibrated rates from the "
        "IHDS-II MI module.",
        ROOT / "early_warning" / "simulator.py",
        ROOT / "figures" / "15_simulator_trajectories.png",
    ),
    (
        "## 7. Time-Varying FFI\n\n"
        "Implements the project plan's flow-based formula\n"
        "$\\text{FFI}_t = w_1 \\cdot \\text{DebtRatio}_t + w_2 \\cdot \\text{ExpenseVolatility}_t - w_3 \\cdot \\text{SavingsBuffer}_t$\n"
        "with equal weights, defines the supervised 3-month-ahead stress event\n"
        "$y_t = \\mathbb{I}\\{\\max(\\text{FFI}_{t+1}, \\text{FFI}_{t+2}, \\text{FFI}_{t+3}) > \\tau\\}$\n"
        "where $\\tau$ is the 75th percentile of FFI$_t$ (matching the cross-sectional Fragile cutoff).",
        ROOT / "early_warning" / "time_varying_ffi.py",
        ROOT / "figures" / "18_ffi_t_distribution.png",
    ),
    (
        "## 8. Building LSTM Training Tensors\n\n"
        "Splits households 70/15/15 train/val/test, then for each (household, prediction-time) "
        "pair extracts a 12-month input sequence and the 3-month-ahead targets.",
        ROOT / "early_warning" / "lstm_data.py", None,
    ),
    (
        "## 9. LSTM with MC Dropout — Training\n\n"
        "Two-layer LSTM (hidden = 64) with inter-layer dropout (p = 0.2), joint regression + "
        "classification heads, Adam optimiser, early stopping on validation ROC-AUC. Runs on "
        "the Apple-Silicon MPS device.",
        ROOT / "early_warning" / "lstm_train.py",
        ROOT / "figures" / "21_lstm_training_curves.png",
    ),
    (
        "## 10. MC Dropout Inference\n\n"
        "Runs 50 stochastic forward passes with dropout active at test time to draw samples "
        "from the approximate predictive posterior (Gal & Ghahramani 2016). Derives a Bayesian "
        "three-tier risk score (Low / Medium / High) from posterior quantiles.",
        ROOT / "early_warning" / "lstm_predict.py", None,
    ),
    (
        "## 11. Baselines (naive, NGBoost, Kalman)\n\n"
        "Three reference models the LSTM must beat: naive carry-forward; NGBoost (probabilistic "
        "gradient boosting with natural-gradient descent); and per-household Kalman state-space "
        "filter (local-level + AR(1) Unobserved Components).",
        ROOT / "early_warning" / "baselines.py", None,
    ),
    (
        "## 12. Evaluation, Figures, and Risk Tiers\n\n"
        "ROC and PR curves across all four models, reliability diagram for the probabilistic "
        "models, per-household uncertainty band illustrations, and the risk-tier triage table.",
        ROOT / "early_warning" / "evaluate.py",
        ROOT / "figures" / "22_ew_roc_pr.png",
    ),
    (
        "## 13. Patching the Report\n\n"
        "Convenience script that reads `ew_summary.json` + the MC Dropout tier summary and "
        "substitutes the corresponding numerical placeholders in `report/main.tex`.",
        ROOT / "early_warning" / "patch_report_table.py", None,
    ),
]


def main():
    nb = nbformat.v4.new_notebook()
    cells = []
    for narrative, py_path, fig_path in SECTIONS:
        cells.append(nbformat.v4.new_markdown_cell(narrative))
        if py_path is not None:
            code = py_path.read_text()
            # Light cleanup: drop the if __name__ == "__main__" guard so the
            # cell runs on import-style execution in the notebook.
            code = code.replace('if __name__ == "__main__":\n    main()',
                                "# (script entry point; uncomment to re-run)\n# main()")
            cells.append(nbformat.v4.new_code_cell(code))
        if fig_path is not None and fig_path.exists():
            md = (f"![{fig_path.name}]({fig_path.relative_to(ROOT)})\n\n"
                  f"_Figure produced by the cell above (saved to `{fig_path.relative_to(ROOT)}`)._")
            cells.append(nbformat.v4.new_markdown_cell(md))

    nb["cells"] = cells
    # Minimal metadata for the kernel
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    nbformat.write(nb, NB_OUT)
    print(f"Wrote {NB_OUT}")


if __name__ == "__main__":
    main()
