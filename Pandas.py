import pandas as pd

df = pd.read_csv('pokemon.csv')

# print first 20 
print(df.head(20))

 # print the names of the columns
print(df.columns)

# print the name column
print(df['Name'])

# print the name column from 0 to 5
print(df['Name'][0:5])

# print just the column (integer location)
print(df.iloc[2])


# print the exact column
print(df.iloc[2,1])

# iterate though each row
for index, row in df.iterrows():
    print(index, row)

# iterate though each row
for index, row in df.iterrows():
    print(index, row['Name'])

# prints all the rows where Type 1 == Fire
print(df.loc[df['Type 1'] == 'Fire'])


# prints data info
print(df.describe())

# sorts in the alphabetical order
print(df.sort_values('Name', ascending=False))

# and multiple sorting conditions in different orders
print(df.sort_values(['Type 1', 'HP'], ascending=[1,0]))

# adding another column called Total which will be the sum of columns from 4 to 9. We add horizontally so we put axis=1
df['Total'] = df.iloc[:, 4:10].sum(axis=1)


# rearranging the columns
cols = list(df.columns)
df = df[cols[0:4] + [cols[-1]] + cols[4:12]]


# save to another file. False for no indexes
df.to_csv('modified.csv', index=False)

