# ID5030 Report — Build Instructions

## Files
- `main.tex` — report source (sections currently outlined as `% TODO` comments)
- `references.bib` — starter bibliography
- `asmeconf.cls`, `asmeconf.bst` — ASME class + bib style (mandated template)

Figures are loaded from `../figures/` via `\graphicspath`, so there is no
need to duplicate them inside this folder.

## Build locally
```bash
cd "/Users/gururaj/Downloads/ID5030 ML project /report"
pdflatex main && bibtex main && pdflatex main && pdflatex main
# or, equivalently:
latexmk -pdf main.tex
```

## Build on Overleaf
1. Create a new project → **Upload Project** → select this entire `report/` folder
   zipped, OR upload `main.tex`, `references.bib`, `asmeconf.cls`, `asmeconf.bst`
   individually plus the figures from `../figures/` into a `figures/` subfolder.
2. Set the compiler to **pdfLaTeX**.
3. Set the main document to `main.tex`.

## Final filename for submission
Rename the compiled PDF to `Group7_Project_Report.pdf` before zipping (per
the course naming convention in the guidelines PDF).
