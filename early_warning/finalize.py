"""
Orchestration script: assumes baselines.py and evaluate.py have run successfully.
Runs patch_report_table.py to fill in the report numbers, then compiles the
report via latexmk, copies the final PDF out, and assembles the .ipynb.

Safe to re-run; each step is idempotent.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
FINAL_PDF = ROOT / "Group7_Project_Report.pdf"


def run(cmd, cwd=None, check=True):
    # latexmk emits stray bytes that aren't UTF-8 from the underlying font tools,
    # so we decode permissively with errors="replace".
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd or ROOT})")
    res = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if res.stdout:
        print(res.stdout[-2000:])
    if check and res.returncode != 0:
        print("STDERR:", res.stderr[-2000:])
        raise SystemExit(f"Step failed: {' '.join(cmd)}")


def main():
    # Step 1: aggregate metrics + render the §4 results figures
    run(["python3", "early_warning/evaluate.py"])

    # Step 2: patch the report with real numbers
    run(["python3", "early_warning/patch_report_table.py"])

    # Step 3: compile the report (clean + full)
    for f in ["main.aux", "main.bbl", "main.blg", "main.fls",
              "main.fdb_latexmk", "main.log", "main.synctex.gz"]:
        try: (REPORT_DIR / f).unlink()
        except FileNotFoundError: pass
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=REPORT_DIR)
    shutil.copy(REPORT_DIR / "main.pdf", FINAL_PDF)
    print(f"\nCopied final PDF: {FINAL_PDF}")

    # Step 4: build the notebook
    run(["python3", "early_warning/build_notebook.py"])

    print("\nAll done. Deliverables:")
    for p in [FINAL_PDF, ROOT / "ID5030_Group7.ipynb"]:
        print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
