"""
Build Group7_Supplementary_File.pdf

Contents:
  1. Title page + introduction
  2. Annotated file index (pipeline + exploratory), one-line role per file
  3. Full source code dump of every .py file (LaTeX listings, syntax-highlighted)

Output: submission/Group7/Group7_Supplementary_File.pdf
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submission" / "Group7"
BUILD_DIR = ROOT / "submission" / "_build_supp"

# Two-tier classification: pipeline files (used in the notebook) and supplementary
# files (precursor/exploratory). Each entry: (relative path, one-line role).
PIPELINE_FILES = [
    ("preprocess_ihds.py",
     "Loads IHDS-II Stata files, recodes missing codes (-9/-8/-7), merges household with selected individual fields, saves ihds_preprocessed.parquet."),
    ("build_ffi.py",
     "Constructs the six-component Financial Fragility Index (debt/consumption/asset/employment/dependency/distress-borrowing); z-standardises, equally weights, quartile-bins into Stable/Stretched/Fragile/Distressed."),
    ("ihds_utils.py",
     "Helper functions: variable-label loader and column-renaming utilities for IHDS dataframes."),
    ("training_full.py",
     "5-fold CV-tuned cross-sectional classifiers (Logistic Regression, Random Forest, LightGBM with isotonic calibration); SHAP and permutation importances; writes metrics_full.json + figures 06-09 / 11-13."),
    ("robustness_checks.py",
     "FFI robustness sensitivity analysis: PCA-weighted alternative, top-tertile cutoff, GroupKFold-by-STATEID leave-states-out generalisation."),
    ("early_warning/bundle_cpi.py",
     "Bundles the India CPI-Combined series (2010-2013) and produces the macro-anchor figure used by the simulator."),
    ("early_warning/simulator.py",
     "Macro-anchored monthly trajectory simulator: lognormal income/expense walks, Poisson medical/disaster/crop shocks calibrated from the IHDS-II MI module, cash-flow gap accounting; emits 42,152 x 24 -month panel."),
    ("early_warning/time_varying_ffi.py",
     "Computes the time-varying FFI_t per the project plan formula and builds the supervised 3-month-ahead stress-event target."),
    ("early_warning/lstm_data.py",
     "Train/val/test split BY HOUSEHOLD, fits z-score scalers on training rows only, materialises (B, 12, 12) sequence tensors + 60-dim static features."),
    ("early_warning/lstm_train.py",
     "PyTorch 2-layer LSTM (hidden 64, dropout 0.2) with joint regression + binary heads, Adam optimiser, early-stopping on val ROC-AUC; runs on Apple-Silicon MPS."),
    ("early_warning/lstm_predict.py",
     "Monte-Carlo Dropout inference: K=50 stochastic forward passes; per-row posterior mean/quantiles; Bayesian Low/Medium/High risk tier assignment."),
    ("early_warning/baselines.py",
     "Three baselines: naive carry-forward, NGBoost (probabilistic gradient boosting with natural-gradient descent), per-household Kalman / UnobservedComponents state-space."),
    ("early_warning/evaluate.py",
     "Aggregates LSTM + 3 baselines; computes RMSE / AUC / AP / ECE / 95% credible-interval coverage; emits figures 22-26 and ew_summary.json."),
    ("early_warning/patch_report_table.py",
     "Substitutes computed metrics into the LaTeX placeholders \\texttt{[NAIVE\\_AUC]} etc. inside report/main.tex."),
    ("early_warning/finalize.py",
     "Orchestrator: chains evaluate -> patch -> latexmk -> notebook assembly."),
    ("early_warning/build_notebook.py",
     "Assembles the 13-section Jupyter notebook from the .py sources via nbformat, with narrative markdown between code cells."),
]

SUPPLEMENTARY_FILES = [
    ("training_baseline.py",
     "Initial single-split LR + RF cross-sectional baseline (superseded by training_full.py, kept as proof-of-work)."),
    ("debt_variables_confirmation.py",
     "One-off exploration confirming the debt-related IHDS variables (DB5, DB6A) used in FFI component c1."),
    ("compress.py",
     "Utility that produced the v1 report.zip distribution (no longer in the active pipeline)."),
    ("early_warning/replot_trajectories.py",
     "Re-renders the simulator trajectory and validation marginals figures with cleaner styling (used after the initial simulator run)."),
    ("scratch/inspect_ihds_for_sim.py",
     "Inventory script that enumerates IHDS-II shock-history (MI module), composition, and earner-count variables for the simulator manifest."),
    ("scratch/check_lstm_inputs.py",
     "Diagnostic script that reports NaN/Inf counts per tensor feature; surfaced the rolling-window cold-start NaN issue."),
    ("scratch/check_cols.py",
     "Column-name exploration of DS0001 and DS0002 IHDS files."),
    ("scratch/check_ds0001.py",
     "Exploration of individual-level fields in DS0001 prior to the household-level merge."),
    ("scratch/check_labels.py",
     "Extracts categorical value labels from Stata files for human-readable variable interpretation."),
    ("scratch/check_debt_labels.py",
     "Variable-label dump for the debt module (DB1-DB8) used while designing FFI component c1."),
    ("scratch/check_env.py",
     "Python-environment sanity check (numpy/pandas/sklearn versions)."),
    ("scratch/extract_all_labels.py",
     "Dumps every IHDS variable label to scratch/variable_labels.json for downstream consumption."),
]

INTRO = r"""
\section*{About this supplementary file}
This document is a complete archive of the Python source code produced for the Group 7 project on \emph{Predicting Household Financial Fragility in India} (ID5030, IIT Madras). It is organised in two parts:
\begin{itemize}
    \item \textbf{Section~\ref{sec:index}} is an annotated index that groups every script by stage and gives a one-line description of its role and outputs.
    \item \textbf{Section~\ref{sec:source}} contains the complete verbatim source of every \texttt{.py} file in the repository, with section headers per file.
\end{itemize}
The Jupyter notebook (\texttt{Group7\_Source\_Code.ipynb}) walks through the pipeline scripts in execution order; this supplementary file is the full, unedited code archive that includes the auxiliary and exploratory scripts not embedded in the notebook.
"""


def latex_escape(s: str) -> str:
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def file_section(rel_path: str) -> str:
    p = ROOT / rel_path
    if not p.exists():
        return ""
    code = p.read_text()
    return (f"\\filesubsection{{{latex_escape(rel_path)}}}\n"
            f"\\begin{{pythoncode}}\n{code}\n\\end{{pythoncode}}\n\n")


def index_table(rows) -> str:
    """Two-column table: file (typewriter) + one-line role."""
    out = ["\\begin{longtable}{@{}p{0.32\\textwidth}p{0.65\\textwidth}@{}}",
           "\\toprule",
           "\\textbf{File} & \\textbf{Role} \\\\",
           "\\midrule",
           "\\endhead"]
    for rel, role in rows:
        out.append(f"\\texttt{{{latex_escape(rel)}}} & {latex_escape(role)} \\\\")
        out.append("\\addlinespace[2pt]")
    out.append("\\bottomrule")
    out.append("\\end{longtable}\n")
    return "\n".join(out)


def main():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = BUILD_DIR / "supplementary.tex"

    # Use the listings package with a Python-friendly style.
    preamble = r"""
\documentclass[10pt,a4paper]{article}
\usepackage[a4paper,top=2cm,bottom=2.2cm,left=2cm,right=2cm]{geometry}
\usepackage{microtype}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}
\usepackage{xcolor}
\definecolor{kw}{rgb}{0.15,0.0,0.50}
\definecolor{str}{rgb}{0.40,0.30,0.00}
\definecolor{com}{rgb}{0.30,0.45,0.30}
\definecolor{bg}{rgb}{0.98,0.98,0.98}
\usepackage{listings}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{parskip}
\lstdefinestyle{py}{
    language=Python,
    basicstyle=\ttfamily\scriptsize,
    keywordstyle=\color{kw}\bfseries,
    stringstyle=\color{str},
    commentstyle=\color{com}\itshape,
    backgroundcolor=\color{bg},
    showstringspaces=false,
    breaklines=true,
    breakatwhitespace=false,
    columns=fullflexible,
    keepspaces=true,
    frame=single,
    framesep=4pt,
    framerule=0.4pt,
    numbers=left,
    numberstyle=\tiny\color{gray},
    numbersep=6pt,
    xleftmargin=8mm,
    aboveskip=4pt,
    belowskip=4pt,
    upquote=true,
    inputencoding=utf8,
    extendedchars=true,
    literate={—}{{---}}1 {–}{{--}}1 {≈}{{$\approx$}}1 {≤}{{$\leq$}}1 {≥}{{$\geq$}}1
             {α}{{$\alpha$}}1 {β}{{$\beta$}}1 {μ}{{$\mu$}}1 {σ}{{$\sigma$}}1
             {₂}{{$_2$}}1 {₃}{{$_3$}}1 {²}{{$^2$}}1 {¹}{{$^1$}}1
             {τ}{{$\tau$}}1 {θ}{{$\theta$}}1 {λ}{{$\lambda$}}1 {δ}{{$\delta$}}1
             {₁}{{$_1$}}1 {₄}{{$_4$}}1 {₅}{{$_5$}}1 {₆}{{$_6$}}1
             {₊}{{$_+$}}1 {₋}{{$_-$}}1 {₀}{{$_0$}}1 {₇}{{$_7$}}1 {₈}{{$_8$}}1 {₉}{{$_9$}}1
             {₍}{{$_($}}1 {₎}{{$_)$}}1
             {ε}{{$\varepsilon$}}1 {η}{{$\eta$}}1 {ρ}{{$\rho$}}1 {ξ}{{$\xi$}}1
             {Σ}{{$\Sigma$}}1 {Ψ}{{$\Psi$}}1 {ψ}{{$\psi$}}1
             {₂}{{$_2$}}1 {₃}{{$_3$}}1
             {π}{{$\pi$}}1 {ω}{{$\omega$}}1 {Δ}{{$\Delta$}}1 {∇}{{$\nabla$}}1
             {₊}{{$_+$}}1 {★}{{$\star$}}1 {≠}{{$\neq$}}1 {→}{{$\rightarrow$}}1
             {₁}{{$_1$}}1 {⊤}{{$\top$}}1 {∼}{{$\sim$}}1
}
\lstnewenvironment{pythoncode}{\lstset{style=py}}{}
\newcommand{\filesubsection}[1]{\subsection*{\texttt{#1}}\addcontentsline{toc}{subsection}{\texttt{#1}}}

\title{\textbf{Group 7 --- Supplementary Source Code}\\[2pt]
\large ID5030 Machine Learning, IIT Madras\\
\large Predicting Household Financial Fragility in India}
\author{Gururaj Thorat (ED23B065)}
\date{\today}

\begin{document}
\maketitle
"""

    body = INTRO + r"""
\tableofcontents
\newpage

\section{Annotated File Index}\label{sec:index}

\subsection{Pipeline scripts (executed by the notebook)}
""" + index_table(PIPELINE_FILES) + r"""

\subsection{Auxiliary and exploratory scripts}
""" + index_table(SUPPLEMENTARY_FILES) + r"""

\newpage
\section{Full Source Code}\label{sec:source}

\subsection*{Pipeline scripts}
"""

    for rel, _ in PIPELINE_FILES:
        body += file_section(rel)

    body += "\\newpage\n\\subsection*{Auxiliary and exploratory scripts}\n"
    for rel, _ in SUPPLEMENTARY_FILES:
        body += file_section(rel)

    body += "\n\\end{document}\n"

    tex_path.write_text(preamble + body)
    print(f"wrote {tex_path}  ({tex_path.stat().st_size / 1024:.1f} KB)")

    # Compile with pdflatex twice (TOC needs two passes)
    for i in range(2):
        res = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "supplementary.tex"],
            cwd=BUILD_DIR, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if res.returncode != 0 and i == 1:
            print(res.stdout[-3000:])
            raise SystemExit("pdflatex failed")

    out_pdf = BUILD_DIR / "supplementary.pdf"
    final = OUT_DIR / "Group7_Supplementary_File.pdf"
    shutil.copy(out_pdf, final)
    print(f"Wrote {final}  ({final.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
