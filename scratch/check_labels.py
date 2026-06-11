import pandas as pd

def get_stata_labels(file_path):
    try:
        reader = pd.io.stata.StataReader(file_path)
        labels = reader.variable_labels()
        return labels
    except Exception as e:
        return f"Error: {e}"

# Check Individual and Household datasets
ds1 = 'DS0001/36151-0001-Data.dta'
ds2 = 'DS0002/36151-0002-Data.dta'

print("Labels for DS0001 (Individual):")
labels1 = get_stata_labels(ds1)
if isinstance(labels1, dict):
    for i, (k, v) in enumerate(labels1.items()):
        if i < 20:
            print(f"{k}: {v}")
else:
    print(labels1)

print("\nLabels for DS0002 (Household):")
labels2 = get_stata_labels(ds2)
if isinstance(labels2, dict):
    for i, (k, v) in enumerate(labels2.items()):
        if i < 20:
            print(f"{k}: {v}")
else:
    print(labels2)
