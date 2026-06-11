import pandas as pd
reader = pd.io.stata.StataReader('DS0002/36151-0002-Data.dta')
labels = reader.variable_labels()
debt_labels = {k: v for k, v in labels.items() if k.startswith('DB')}
for k, v in sorted(debt_labels.items()):
    print(f"{k}: {v}")
