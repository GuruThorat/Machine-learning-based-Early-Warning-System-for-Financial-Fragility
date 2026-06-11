import pandas as pd
df1 = pd.read_stata('DS0001/36151-0001-Data.dta', iterator=True)
chunk = df1.get_chunk(1)
print("DS0001 Columns:", chunk.columns.tolist())
