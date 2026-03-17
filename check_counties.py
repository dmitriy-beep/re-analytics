import pandas as pd

df = pd.read_csv(
    r'C:\RealEstateTools\weekend3-14\re-analytics\weekly_housing_market_data_most_recent.tsv000',
    sep='\t',
    nrows=50000,
    usecols=['REGION_TYPE', 'REGION_NAME', 'REGION_ID']
)

print(df[df['REGION_TYPE']=='county']['REGION_NAME'].unique())
