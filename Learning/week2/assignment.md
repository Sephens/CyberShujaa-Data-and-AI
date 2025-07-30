# Assignment 2: Netflix Data Wrangling Project

## Overview
**Due Date:** Monday, 26 May 2025, 11:59 PM  
**Dataset:** [Netflix Movies and TV Shows on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)  
**Objective:** Practice data wrangling techniques to clean and prepare Netflix content data for analysis.

The objectives of the assignment are to:

- Load the Netflix dataset from a CSV file and explore its structure using pandas.
- Perform data discovery to assess data types, missing values, and quality issues.
- Clean the dataset by handling duplicates, missing values, and formatting inconsistencies.
- Transform and enrich the dataset using techniques like filtering, sorting, grouping, and feature extraction.
- Validate the final dataset by checking consistency, completeness, and logical accuracy.
- Export the final cleaned dataset to a .csv file ready for analysis or visualization.

## Complete Solution with Detailed Explanations

```python
'''
Title: Netflix Data Wrangling Project
Name: Paula Musuva
Date: 20 May 2025
Description: This script cleans and prepares Netflix content data for analysis.
'''

# Import pandas library for data manipulation
import pandas as pd

### 1. DATA LOADING ###
df = pd.read_csv('/kaggle/input/netflix/netflix_titles.csv')
'''
Explanation:
- pd.read_csv() loads the dataset from the specified file path
- Creates a pandas DataFrame object (df) containing all the data
- The dataset contains Netflix show information (title, director, cast, etc.)
'''

### 2. DATA DISCOVERY ###

# Get dataset overview
df.info()
'''
Explanation:
- Shows column names, non-null counts, and data types
- Helps identify missing values and memory usage
- First step in understanding data structure
'''

# Print shape (rows × columns)
print("Shape of the dataset (R x C):", df.shape)
'''
Explanation:
- df.shape returns a tuple (rows, columns)
- Helps understand dataset size before cleaning
'''

# List all column names
print("Columns in the dataset:\n", df.columns.tolist())
'''
Explanation:
- df.columns returns Index object of column names
- tolist() converts it to a Python list
- Useful for reference during data cleaning
'''

# Show data types
print("Data types:\n", df.dtypes)
'''
Explanation:
- Displays each column's data type (object, int64, float64, etc.)
- Helps identify columns needing type conversion
'''

# Count missing values
print("Missing values per column:\n", df.isnull().sum())
'''
Explanation:
- isnull() identifies null values
- sum() counts them per column
- Reveals which columns need imputation
'''

# Count duplicates
print("Number of duplicate rows:", df.duplicated().sum())
'''
Explanation:
- duplicated() flags duplicate rows
- sum() counts total duplicates
- Helps decide if deduplication is needed
'''

### 3. DATA STRUCTURING ###

# Convert date_added to datetime
df['date_added'] = pd.to_datetime(df['date_added'], format='mixed')
'''
Explanation:
- Converts string dates to datetime objects
- 'mixed' format handles various date formats
- Enables date-based operations and filtering
'''

# Split duration into value and unit
df[['duration_value', 'duration_unit']] = df['duration'].str.extract(r'(\d+)\s*(\w+)')
'''
Explanation:
- str.extract() uses regex to separate numbers and text
- (\d+) captures numeric duration
- (\w+) captures unit (min/Season/Seasons)
- Creates two new columns from one
'''

# Convert duration to numeric
df['duration_value'] = pd.to_numeric(df['duration_value'])
'''
Explanation:
- Ensures duration is numeric for calculations
- Required for statistical operations
'''

### 4. DATA CLEANING ###

# Remove duplicates
print("Duplicate rows before:", df.duplicated().sum())
df = df.drop_duplicates()
'''
Explanation:
- First checks duplicate count
- drop_duplicates() removes identical rows
- Preserves data integrity
'''

# Drop description column
df = df.drop(columns=['description'])
'''
Explanation:
- Removes text column not needed for analysis
- Reduces memory usage
'''

### 5. HANDLING MISSING VALUES ###

# Create director-cast pairs
df['dir_cast'] = df['director'] + '---' + df['cast']
'''
Explanation:
- Combines director and cast with separator
- Helps identify relationships between them
'''

# Find frequent director-cast pairs
counts = df['dir_cast'].value_counts()
filtered_counts = counts[counts >= 3]
filtered_values = filtered_counts.index
lst_dir_cast = list(filtered_values)
'''
Explanation:
- value_counts() shows pair frequencies
- Filters pairs appearing ≥3 times
- Converts to list for processing
'''

# Create director-cast dictionary
dict_direcast = dict()
for i in lst_dir_cast:
    director, cast = i.split('---')
    dict_direcast[director] = cast
'''
Explanation:
- Splits pairs into dictionary
- Key: director, Value: cast
- Enables director imputation
'''

# Impute missing directors
for i in range(len(dict_direcast)):
    df.loc[(df['director'].isna()) & 
           (df['cast'] == list(dict_direcast.items())[i][1]),
           'director'] = list(dict_direcast.items())[i][0]
'''
Explanation:
- For null directors with known cast:
  - Finds matching cast in dictionary
  - Imputes corresponding director
- Uses logical indexing with loc[]
'''

# Set remaining missing directors to 'Not Given'
df.loc[df['director'].isna(), 'director'] = 'Not Given'
'''
Explanation:
- Handles cases without cast-director relationships
- Maintains data completeness
'''

# Similar process for countries (abbreviated in this example)
# [... country imputation code ...]

### 6. ERROR CHECKING ###

# Verify date_added ≥ release_year
sum(df['date_added'].dt.year < df['release_year'])
'''
Explanation:
- Checks logical consistency
- Shows count of illogical date pairs
'''

### 7. FINAL VALIDATION ###

# Remove temporary columns
df.drop(columns=['dir_cast'], inplace=True)
'''
Explanation:
- Cleans up intermediate working columns
- inplace=True modifies DataFrame directly
'''

# Reset index
df_reset = df.reset_index(drop=True)
'''
Explanation:
- Recreates sequential index after deletions
- drop=True prevents old index becoming a column
'''

### 8. DATA EXPORT ###
df.to_csv('/kaggle/working/cleaned_netflix.csv', index=False)
'''
Explanation:
- Saves cleaned data to new CSV file
- index=False excludes pandas index column
- File saved in Kaggle working directory
'''
```

---