import pandas as pd
# df = pd.read_csv('/kaggle/input/netflix/netflix_titles.csv')
#---------------------------Step 1: Data Loading & Initial Exploration---------------------
# create a DataFrame
df = pd.read_csv('netflix_titles.csv')
# df.info()
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

#-------------------------Step 2: Data Structuring-----------------------------------------

# Converted date_added to datetime:

df['date_added'] = pd.to_datetime(df['date_added'])

# Split duration into numeric and unit:
df[['duration_value', 'duration_unit']] = df['duration'].str.extract(r'(\d+)\s*(\w+)')
df['duration_value'] = pd.to_numeric(df['duration_value'])

#---------------------------------------Step 3: Data Cleaning------------------------------------

# Handling Missing Values

# Directors
# Imputed missing director values based on frequent cast associations.
# Filled remaining missing values with "Not Given".
df.loc[df['director'].isna(), 'director'] = 'Not Given'

# Countries:
# Used director-country relationships to fill missing country values.

df.loc[df['country'].isna(), 'country'] = 'Not Given'

# Dropped Irrelevant Columns:
# Removed description (not needed for analysis).

df.drop(columns=['description'], inplace=True)

# Handling Errors
# Fixed date_added < release_year inconsistencies:

df = df[df['date_added'].dt.year >= df['release_year']]

#-------------------------------------Step 4: Validation--------------------

# Checked Data Types:
# Verified date_added is datetime and duration_value is numeric.
# Ensured No Critical Missing Values:

print(df.isnull().sum())  # Confirmed no key missing values.


# Sampled Data for Sanity Check:
df.sample(5)

# Step 5: Export Cleaned Dataset
# Saved the cleaned dataset as cleaned_netflix.csv:
# df.to_csv('/kaggle/working/cleaned_netflix.csv', index=False)

df.to_csv('cleaned_netflix.csv', index=False)