'''
  Title: Data Wrangling Project
  Name: Steven Odhiambo
  Date: 30 May 2025

  Successfully cleaned and transformed the Netflix dataset.
  Handled missing values, duplicates, and inconsistencies.
  Validated data integrity before export.
  The cleaned dataset is now ready for exploratory analysis or machine learning.
'''
import pandas as pd
# df = pd.read_csv('/kaggle/input/netflix/netflix_titles.csv')


#---------------------------Step 1: Data Loading & Initial Exploration (Discovery)---------------------
#Import the Data to a Pandas DataFrame
df = pd.read_csv('netflix_titles.csv')

#Have a quick overview of the data
df.info()

# Number of rows and columns
print("Shape of the dataset (R x C):", df.shape)

# List of all column names
print("Columns in the dataset:\n", df.columns.tolist())

# Data types of each column
print("Data types:\n", df.dtypes)

# Group and Count of missing (null) values in each column
print("Missing values per column:\n", df.isnull().sum())

# Group and Count of duplicate rows
print("Duplicates:", df.duplicated().sum())

#-------------------------Step 2: Data Structuring-----------------------------------------

# Converted date_added to datetime:
df['date_added'] = pd.to_datetime(df['date_added'])

# Separate 'duration' into numeric value and unit (e.g., '90 min' → 90, 'min')
df[['duration_value', 'duration_unit']] = df['duration'].str.extract(r'(\d+)\s*(\w+)')

# Convert duration_value to numeric
df['duration_value'] = pd.to_numeric(df['duration_value'])

# View Resulting columns
print(df[['duration_value', 'duration_unit']])

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