# Machine Learning Based Early-Warning System for Financial Fragility

Author: **Gururaj Dinesh Thorat** (ED23B065).

## What this project does

We build a 6-component **Financial Fragility Index (FFI)** for Indian households
using the India Human Development Survey, Wave 2 (IHDS-II, ICPSR study 36151),
and then extend it into a **temporal early-warning system** that flags households
likely to enter a stressed financial state in the next three months.

Two stages:

1. **Cross-sectional baseline.** Logistic Regression, Random Forest and LightGBM
   predict household FFI from socio-economic features. Best ROC-AUC ≈ 0.93,
   with SHAP-based attribution and robustness checks.
2. **Temporal early-warning extension.** A macro-anchored monthly trajectory
   simulator (Poisson shocks for medical, disaster, crop events, calibrated to
   IHDS-II MI-module prevalences; real India CPI 2010–2013 as macro anchor)
   produces a 1.01 M-row synthetic panel from the 42,152 IHDS households.
   A two-layer LSTM with **MC-Dropout** (Gal & Ghahramani 2016, K = 50) outputs
   a Bayesian posterior over FFI three months ahead, against three baselines
   (naive carry-forward, NGBoost, Kalman UnobservedComponents).

### Headline numbers (held-out test set, 63,240 household-months)

| Model | ROC-AUC | AP | RMSE (FFI z) | ECE |
|---|---|---|---|---|
| **LSTM + MC-Dropout** | **0.9821** | **0.9768** | **0.496** | 0.0328 |
| Naive carry-forward | 0.9457 | 0.9311 | 0.7340 | — |

Bayesian risk tiers (LSTM):
Low 43,002 (4.6 % positive rate), Medium 1,009 (40.5 %), High 19,229 (97.3 %; captures 89 % of all positives).

## Repository layout

```
.
├── ID5030_Group7.ipynb            Final notebook deliverable
├── Group7_Project_Report.pdf      Final report (LaTeX-built PDF)
├── plan..png                      Project plan / roadmap diagram
│
├── preprocess_ihds.py             Stage-1: merge IHDS DS0001+DS0002(+DS0003)
├── build_ffi.py                   Build the 6-component FFI
├── training_baseline.py           Logistic Regression + Random Forest baseline
├── training_full.py               Full training: LR / RF / LightGBM + SHAP
├── robustness_checks.py           Train/test split sensitivity, bootstrap CIs
├── ihds_utils.py                  Shared helpers
├── compress.py                    Parquet compression utilities
│
├── early_warning/                 Temporal extension pipeline
│   ├── bundle_cpi.py              India CPI 2010-2013 bundling
│   ├── simulator.py               Vectorised monthly-trajectory simulator
│   ├── time_varying_ffi.py        FFI computed per household-month
│   ├── lstm_data.py               Sequence + static feature builder
│   ├── lstm_train.py              Two-layer LSTM training
│   ├── lstm_predict.py            MC-Dropout predictive posterior (K=50)
│   ├── baselines.py               Naive / NGBoost / Kalman baselines
│   ├── evaluate.py                Joint regression + classification metrics
│   ├── patch_report_table.py      Inject numbers into report/main.tex
│   ├── build_notebook.py          Assemble final .ipynb from .py files
│   ├── finalize.py                Orchestrator (runs the full pipeline)
│   └── *.json                     Calibration / feature-spec artifacts
│
├── data/
│   ├── cpi_india.csv              Public RBI / MOSPI CPI series (committed)
│   └── README.md                  How to obtain IHDS-II (NOT committed)
│
├── figures/                       All report figures (PNG)
├── report/                        LaTeX source for the final report
│   ├── main.tex
│   ├── references.bib
│   └── asmeconf.{cls,bst}         Conference template
│
├── submission/Group7/             Sealed deliverables (PDFs, README.txt)
├── archive/v1_cross_sectional/    Frozen pre-extension snapshot
└── scratch/                       Exploratory inspection scripts
```

## Reproducing the results

The raw IHDS-II microdata is **not redistributed** here (see *Data access* below).
After you obtain it from ICPSR and place it under `DS0001/`…`DS0014/`, the pipeline is:

```bash
# Cross-sectional stage
python3 preprocess_ihds.py          # → ihds_preprocessed.parquet
python3 build_ffi.py                # → ihds2_ffi.parquet
python3 training_full.py            # → metrics, predictions, SHAP, figures
python3 robustness_checks.py        # → robustness.json

# Temporal early-warning stage
python3 early_warning/bundle_cpi.py
python3 early_warning/simulator.py          # → simulated_panel.parquet (~1.01 M rows)
python3 early_warning/time_varying_ffi.py   # → panel_with_ffi.parquet
python3 early_warning/lstm_data.py          # → panel_supervised.parquet + lstm_arrays/
python3 early_warning/lstm_train.py         # → lstm_checkpoint.pt
python3 early_warning/lstm_predict.py       # → test_predictions.npz
python3 early_warning/baselines.py
python3 early_warning/evaluate.py

# Build report + notebook
python3 early_warning/patch_report_table.py
( cd report && latexmk -pdf main.tex )
python3 early_warning/build_notebook.py
```

Or simply: `python3 early_warning/finalize.py` (runs the full early-warning chain).

## Data access

The India Human Development Survey, Wave 2 is distributed by **ICPSR** as
study **36151** under terms of use that prohibit redistribution.

To replicate this work you must:

1. Register as an authorised ICPSR user (free for academic / non-profit research).
2. Apply for and download study 36151 from
   <https://www.icpsr.umich.edu/web/ICPSR/studies/36151>.
3. Unpack into `DS0001/` through `DS0014/` at the repository root.

See [`data/README.md`](data/README.md) for the directory layout the scripts expect.

## Reports and deliverables

- `Group7_Project_Report.pdf` — final report (10 pages, ASME-conf template).
- `submission/Group7/Group7_Source_Code.ipynb` — submitted notebook with figures.
- `submission/Group7/Group7_Source_Code_with_Output_Results.pdf` — same, PDF.
- `submission/Group7/Group7_Supplementary_File.pdf` — supplementary material.

## Citation

If you reuse the IHDS-II data, cite:

> Desai, Sonalde, and Reeve Vanneman. **India Human Development Survey-II (IHDS-II), 2011–12.**
> Inter-university Consortium for Political and Social Research [distributor], 2018-08-08.
> <https://doi.org/10.3886/ICPSR36151.v6>

## License

Code in this repository is released under the MIT License (see `LICENSE` if present).
The IHDS-II data is **not** included and is **not** released under this license — it
remains governed by ICPSR's terms of use.
