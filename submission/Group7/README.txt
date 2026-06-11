================================================================================
  ID5030 Machine Learning — Final Project Submission
  Group 7
  Predicting Household Financial Fragility in India:
  A Composite Index and Machine Learning Approach on IHDS-II Data
================================================================================

Member
------
  Gururaj Thorat (ED23B065) — Department of Engineering Design, IIT Madras


Contents of this ZIP
--------------------

  Group7_PPT_Video.mp4
      10-minute project presentation video (title, motivation, methodology,
      results, conclusion, contributions).

  Group7_Project_Report.pdf
      Final project report written on the ASME LaTeX template.
      Covers: the cross-sectional Financial Fragility Index (FFI) constructed
      on IHDS-II data, three supervised classifiers (Logistic Regression,
      Random Forest, LightGBM), SHAP-based feature attribution, three FFI
      robustness checks, and a temporal early-warning extension built on a
      macro-anchored synthetic monthly panel using an LSTM with Monte-Carlo
      Dropout and three benchmark models (naive carry-forward, NGBoost,
      per-household Kalman filter).

  Group7_Source_Code.ipynb
      Single Jupyter notebook walking through the full pipeline in 13 sections:
      preprocessing, FFI construction, cross-sectional classification,
      robustness checks, CPI bundling, simulator, time-varying FFI, LSTM
      training, MC-Dropout inference, baselines, evaluation, and report
      patching.

  Group7_Source_Code_with_Output_Results.pdf
      The notebook above rendered to PDF with all figures and metric tables
      embedded as cell outputs.

  Group7_Supplementary_File.pdf
      Complete archive of every .py file produced during the project. Section 1
      is an annotated file index that groups every script by stage and gives
      a one-line role per file (pipeline scripts in the notebook + auxiliary
      and exploratory scripts that supported the work). Section 2 contains the
      verbatim source of every file with section headers.

  README.txt
      This file.


How to read the submission
--------------------------

  Start with: Group7_Project_Report.pdf (headline findings, math, figures).
  Then:       Group7_Source_Code_with_Output_Results.pdf  (code + live outputs).
  Reference:  Group7_Supplementary_File.pdf for the full code archive,
              including precursor / exploratory scripts not embedded in the
              notebook.
  Watch:      Group7_PPT_Video.mp4 for the spoken walkthrough.


Headline result
---------------

  Cross-sectional binary classifier (Fragile/Distressed vs not), held-out
  test set of N = 8,431 households, predictors disjoint from the FFI inputs:

        Logistic Regression      ROC-AUC 0.751
        Random Forest            ROC-AUC 0.754
        LightGBM (isotonic-cal.) ROC-AUC 0.760   <- headline cross-sectional model

  Temporal early-warning (3-month horizon, held-out test of 63,240 cells from
  6,324 unseen households, simulated macro-anchored monthly panel):

        Naive carry-forward      ROC-AUC 0.946
        Kalman UC                ROC-AUC 0.953
        NGBoost                  ROC-AUC 0.981   ECE 0.007   coverage_95 0.95
        LSTM + MC Dropout        ROC-AUC 0.982   ECE 0.033   coverage_95 0.28

  Risk-tier triage from the LSTM posterior (Low / Medium / High):
        High tier captures 89% of all stress events at 97% per-tier purity.


Data attribution
----------------

  All analysis uses the second wave of the India Human Development Survey
  (IHDS-II, 2011-12), distributed by ICPSR (Study 36151, DOI:10.3886/ICPSR36151.v6).
  The India CPI-Combined series bundled in data/cpi_india.csv is rebased to
  2010-01 = 100 from published RBI/MOSPI annual averages.


Tools / dependencies
--------------------

  Python 3.12, scikit-learn 1.8, LightGBM 4.6, SHAP 0.51, PyTorch 2.x with MPS
  (Apple-Silicon GPU), NGBoost 0.5, statsmodels 0.14, jupyter / nbformat /
  nbconvert / pandoc 3.9 (for PDF rendering). LaTeX: TeX Live 2026 with the
  ASME asmeconf class.


Last updated: 2026-05-24
================================================================================
