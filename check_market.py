import pandas as pd

df = pd.read_csv(
    r'C:\RealEstateTools\weekend3-14\re-analytics\data\raw\redfin_market_conditions.tsv',
    sep='\t'
)

print(df.columns.tolist())
print()
print(df[df['REGION_NAME']=='Placer County, CA'].tail(3).to_string())
