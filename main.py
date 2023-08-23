import pandas as pd

df = pd.read_csv('pokemon.csv')


df['Total'] = df.iloc[:, 4:10].sum(axis=1)

# rearranging the columns
cols = list(df.columns)
df = df[cols[0:4] + [cols[-1]] + cols[4:12]]


