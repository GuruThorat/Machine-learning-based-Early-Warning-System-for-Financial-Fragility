import pandas as pd
import json

def get_all_labels():
    files = {
        'Individual (DS0001)': 'DS0001/36151-0001-Data.dta',
        'Household (DS0002)': 'DS0002/36151-0002-Data.dta',
        'Eligible Women (DS0003)': 'DS0003/36151-0003-Data.dta'
    }
    
    all_mappings = {}
    
    for name, path in files.items():
        print(f"Processing {name}...")
        try:
            reader = pd.io.stata.StataReader(path)
            labels = reader.variable_labels()
            all_mappings[name] = labels
        except Exception as e:
            print(f"Could not read {path}: {e}")
            
    return all_mappings

if __name__ == "__main__":
    mappings = get_all_labels()
    with open('scratch/variable_labels.json', 'w') as f:
        json.dump(mappings, f, indent=4)
    print("Labels saved to scratch/variable_labels.json")
