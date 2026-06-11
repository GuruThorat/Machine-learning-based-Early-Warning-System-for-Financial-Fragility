"""
Inventory IHDS-II DS0002 (household file) columns we need for the macro-anchored simulator.

We need:
  - Anchor variables (annual values to walk around): INCOME, COTOTAL, ASSETS, debt-total
  - Earner counts (drive income volatility): NWK* family
  - Household composition (drives expense level): NPERSONS, NCHILD*
  - Shock-history variables (empirical Poisson rates for shock injection):
      * Major morbidity / hospitalization (typically SM* / MM* / NMH* depending on IHDS module)
      * Crop loss / agricultural shocks
      * Job loss / employment shocks
  - Distress-borrowing variables (signals of past stress): DB1C, DB2C, DB5, DB6, DB6A

This script reads the variable labels (already cached at scratch/variable_labels.json
by extract_all_labels.py) and prints organised candidate lists for the simulator.

Output: writes a curated JSON manifest at early_warning/sim_variable_manifest.json
naming each role (anchor, earner_count, composition, shock, debt) and the IHDS code(s)
that fill it.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HH_FILE = ROOT / "DS0002" / "36151-0002-Data.dta"
LABELS_FILE = ROOT / "scratch" / "variable_labels.json"
OUT_MANIFEST = ROOT / "early_warning" / "sim_variable_manifest.json"


def load_labels() -> dict:
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            data = json.load(f)
        # Flatten if nested by dataset
        if any(isinstance(v, dict) for v in data.values()):
            flat = {}
            for ds, mapping in data.items():
                flat.update(mapping)
            return flat
        return data
    return {}


def list_columns_via_stata():
    """Read the .dta header to get the full column list without pulling the whole file into RAM."""
    with pd.read_stata(HH_FILE, convert_categoricals=False, iterator=True) as rdr:
        labels = rdr.variable_labels()
        cols = list(labels.keys())
    return cols, labels


def search(labels_text: dict, patterns: list[str]) -> list[tuple]:
    """Return (code, label) for any column whose label text contains any pattern."""
    out = []
    for code, lbl in labels_text.items():
        lbl_lo = (lbl or "").lower()
        if any(p.lower() in lbl_lo for p in patterns):
            out.append((code, lbl))
    return out


def by_prefix(cols: list[str], prefix: str) -> list[str]:
    return [c for c in cols if c.startswith(prefix)]


def main():
    cols, labels_in_dta = list_columns_via_stata()
    print(f"DS0002 columns: {len(cols)}")

    # Use the labels from the .dta directly (richer than the cached JSON for ad hoc use)
    labels = labels_in_dta

    # ----- (1) Anchor variables -----
    anchors = {
        "annual_income":       ("INCOME", labels.get("INCOME", "")),
        "annual_consumption":  ("COTOTAL", labels.get("COTOTAL", "")),
        "assets_stock":        ("ASSETS", labels.get("ASSETS", "")),
        "debt_hh":             ("DB5", labels.get("DB5", "")),
        "debt_shopkeeper":     ("DB6A", labels.get("DB6A", "")),
        "n_persons":           ("NPERSONS", labels.get("NPERSONS", "")),
        "urban":               ("URBAN2011", labels.get("URBAN2011", "")),
    }
    print("\n--- Anchor variables ---")
    for k, (c, lbl) in anchors.items():
        print(f"  {k:22s}  {c:10s}  {lbl}")

    # ----- (2) Earner counts -----
    earner_cols = [c for c in cols if c.startswith("NWK")]
    print(f"\n--- Earner-count columns (NWK*) [{len(earner_cols)}] ---")
    for c in earner_cols:
        print(f"  {c:10s}  {labels.get(c, '')}")

    # ----- (3) Composition -----
    comp_cols = [c for c in cols if c.startswith("NCH") or c.startswith("NEA") or c.startswith("NM") or c.startswith("NF") or c == "NPERSONS"]
    print(f"\n--- Composition columns [{len(comp_cols)}] ---")
    for c in comp_cols:
        print(f"  {c:10s}  {labels.get(c, '')}")

    # ----- (4) Shock history (text search across labels) -----
    print("\n--- Candidate shock-history variables ---")
    shock_patterns = [
        "hospital", "morbidity", "illness", "death of", "died",
        "crop loss", "crop failure", "drought", "flood",
        "lost job", "unemploy", "lay off", "wage loss",
        "shock", "disaster", "emergency",
    ]
    shock_hits = search(labels, shock_patterns)
    for c, lbl in shock_hits[:40]:
        print(f"  {c:10s}  {lbl}")
    print(f"  ... ({len(shock_hits)} total)")

    # ----- (5) Debt / distress borrowing (already used in build_ffi.py) -----
    debt_cols = [c for c in cols if c.startswith("DB")]
    print(f"\n--- Debt-module columns (DB*) [{len(debt_cols)}] ---")
    for c in debt_cols[:25]:
        print(f"  {c:10s}  {labels.get(c, '')}")

    # ----- (6) Save curated manifest for the simulator -----
    manifest = {
        "anchor_vars": {k: c for k, (c, _) in anchors.items()},
        "earner_count_vars": earner_cols,
        "composition_vars": comp_cols,
        "shock_candidate_vars": [c for c, _ in shock_hits],
        "debt_vars": ["DB5", "DB6", "DB6A", "DB1C", "DB2C"],
        "note": (
            "Used by simulator.py. Anchor variables provide the annual cross-sectional "
            "snapshot around which monthly trajectories are simulated. Earner counts drive "
            "income volatility (sole earner -> higher variance). Composition drives expense "
            "level. Shock candidates are screened by label text and then narrowed in the "
            "simulator to a small ordered list (medical, agricultural, employment) whose "
            "empirical incidence determines Poisson rates."
        ),
    }
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
