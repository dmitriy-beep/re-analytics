import pandas as pd

df = pd.read_csv(
    r'C:\RealEstateTools\weekend3-14\re-analytics\weekly_housing_market_data_most_recent.tsv000',
    sep='\t',
    dtype=str
)

filtered = df[
    (df['REGION_TYPE'] == 'county') &
    (df['REGION_NAME'].isin(['Sacramento County, CA', 'Placer County, CA']))
]

print(f'Rows: {len(filtered)}')
print(filtered['PERIOD_BEGIN'].min(), 'to', filtered['PERIOD_BEGIN'].max())

filtered.to_csv(
    r'C:\RealEstateTools\weekend3-14\re-analytics\data\raw\redfin_market_conditions.tsv',
    sep='\t',
    index=False
)

print('Done')
