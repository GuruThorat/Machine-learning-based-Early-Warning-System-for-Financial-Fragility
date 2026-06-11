# `data/`

## What's here

- `cpi_india.csv` — monthly India Consumer Price Index, 2010-01 through 2013-12,
  sourced from RBI / MOSPI public releases. Used as the macroeconomic anchor in
  the simulator (`early_warning/simulator.py`).

## What's NOT here — and why

The **IHDS-II microdata** (ICPSR study 36151) is restricted-distribution and
cannot be republished. The terms of use explicitly forbid redistribution outside
authorised ICPSR users.

To run the pipeline you must obtain the data yourself and place it at the
**repository root** (not inside `data/`) in the directory structure ICPSR ships:

```
<repo-root>/
├── DS0001/
│   ├── 36151-0001-Data.dta          ← individual-level
│   ├── 36151-0001-Codebook.pdf
│   └── ...
├── DS0002/
│   └── 36151-0002-Data.dta          ← household-level (main file)
├── DS0003/
│   └── 36151-0003-Data.dta          ← eligible women (optional merge)
├── DS0004/ … DS0014/                ← other modules
└── 36151-Documentation-Dataguide.pdf
    36151-User_guide.pdf
    36151-manifest.txt
```

The scripts (`preprocess_ihds.py`, `build_ffi.py`, `early_warning/simulator.py`)
read from these relative paths.

## How to obtain IHDS-II

1. Create an ICPSR account at <https://www.icpsr.umich.edu/web/pages/>.
2. Go to study 36151:
   <https://www.icpsr.umich.edu/web/ICPSR/studies/36151>.
3. Agree to the terms of use, download the Stata `.dta` bundle.
4. Unpack into the layout shown above.

Once in place, the cross-sectional pipeline (`preprocess_ihds.py` → `build_ffi.py`
→ `training_full.py`) and the temporal early-warning pipeline
(`early_warning/finalize.py`) will run end-to-end.
