import pandas as pd

df = pd.read_csv('top-rated-coffee.csv')
df.info() # Overview

print("This is the shape of the dataset (R x C): ", df.shape) # R x C

#print all the columns if the dataset
print("All the columns of the Dataset: ")
columns = df.columns.tolist()
datatype = df.dtypes
for column in columns:
    print(column)


# Looking for duplicates
print("Duplicates: ", df.duplicated().sum())

# Data Structuring
# Split price into numeric and unit

# df[['price_unit', 'price']] = df['est._price'].str.extract(r'(\d+)\s*(\w+)')
# df['price'] = pd.to_numeric(df['price'])

# Group and Count of missing (null) values in each column
print("Missing Values per column: \n", df.isnull().sum())

# Handle missing values on coffee_origin
#Filled missing values with "Not Given".
df.loc[df['coffee_origin'].isna(), 'coffee_origin'] = 'Not Given'




print(df.isnull().sum())  # Confirmed no key missing values on cofee_origin.
df.sample(5)

df.to_csv('cleaned_top_rated_coffee.csv', index=False)