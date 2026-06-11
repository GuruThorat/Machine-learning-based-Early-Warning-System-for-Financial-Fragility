import pandas as pd
import sys

try:
    # Individual data
    df1 = pd.read_stata('DS0001/36151-0001-Data.dta', iterator=True)
    chunk = df1.get_chunk(1)
    print("DS0001 Columns:", chunk.columns.tolist())
    
    # Household data
    df2 = pd.read_stata('DS0002/36151-0002-Data.dta', iterator=True)
    chunk2 = df2.get_chunk(1)
    print("DS0002 Columns:", chunk2.columns.tolist())
except Exception as e:
    print(f"Error: {e}")
