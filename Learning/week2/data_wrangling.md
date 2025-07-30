
# Data Wrangling

## Learning Outcomes
At the end of this session, the learner should be able to:
1. Explain data wrangling concepts and their importance in the data science workflow.
2. Load and inspect various datasets using Python.
3. Identify and handle missing values using techniques such as dropping, filling, and imputing.
4. Detect and correct inconsistencies in data.
5. Transform and reshape data using various techniques.
6. Apply the entire data wrangling process to a real-world dataset and present a clean, analysis-ready version in a Jupyter notebook.

## Data Wrangling Concepts

- The process of converting raw data into a desired and usable format for data science through cleaning, structuring, and enriching for better decision-making in less time.
- Happens in Stage 3 of CRISP-DM: Data Preparation or Stage 2 of OSEMN: Scrub.
- Increasingly important in the age of Big Data with increasing unstructured and semi-structured data of high volume, variety, velocity, veracity.
- Clean and well-structured data is critical for valid analysis, model training, and insights.
- Poor data leads to biased models, inaccurate conclusions, and poor decision-making.
- Good wrangling practices ensure transparency, reproducibility, and scalability in projects.
- Data wrangling is estimated to take 60-80% of a data scientist's time.

## Big Data

- Big Data refers to modern datasets that are very large or complex with diverse collections of structured, unstructured, and semi-structured data that continues to grow exponentially over time.
- Hard for traditional data management & processing tools to handle.

### Key Characteristics (The 5 Vs):
1. **Volume**: Massive amounts of data (terabytes, petabytes).
2. **Velocity**: Data is generated rapidly in real-time or near real-time.
3. **Variety**: Structured, semi-structured, and unstructured data from diverse sources.
4. **Veracity**: Data quality and uncertainty; noisy and inconsistent data.
5. **Value**: The potential insights and competitive advantage data can provide.

## Big Data Sources

1. **Machine-generated streaming data**: From the Internet of Things (IoT), sensors and other smart connected devices (e.g., wearables, smart cars, medical devices, industrial equipment).
2. **Human-generated data**: From interactions on social media (e.g., Facebook, YouTube, Instagram), email, mobile apps, etc. Often in unstructured or semi-structured forms.
3. **Transactional data**: From financial transactions (e.g., processing payments for banks and FinTech companies), ecommerce and CRM systems.
4. **Public/External data sources**: From open data sets like WHO, World Bank, US government's data.gov, CIA World Factbook or EU Open Data Portal.
5. **Other sources**: Data lakes, cloud data sources, suppliers and customers.

## Data Wrangling Tools

| Tool | Description |
|------|-------------|
| **Excel** | Good for small scale wrangling and visualization. Includes Power Query for data work. |
| **OpenRefine** | More sophisticated than Excel with automation for data cleaning. Good for big data jobs. |
| **Google DataPrep** | Tool for looking at, cleaning, and getting data ready for downstream work. |
| **Trifacta** | Online tool for managing, cleaning and changing data. Used for big and rough data jobs. |
| **Tabula** | Often referred to as the "all-in-one" data wrangling solution. Works best with data in neat formats. |
| **Data Wrangler** | Made by Stanford. Turns messy data into tidy data. |
| **Alteryx** | Expensive leading tool with intuitive drag-and-drop interface for cleansing and transforming data. |
| **Python** | With powerful libraries like NumPy, Pandas, Matplotlib. |
| **R** | With packages like dplyr and tidyverse. |

# Assignment 1: Data Wrangling in Python

You are required to use the Data Wrangling concepts in Python to clean up the Netflix dataset available on Kaggle:  
[https://www.kaggle.com/datasets/shivamb/netflix-shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)

## Steps to be taken:
1. Load and inspect a real-world dataset.
2. Follow the Data Wrangling steps one at a time.
3. Prepare a clean, analysis-ready version of the dataset.

## Starting Up

1. Login to your Kaggle account
2. Search for the Netflix Shows Dataset
3. Create a Notebook
4. Upload the dataset

```python
# Load the Netflix dataset from the attached Kaggle dataset
filepath = '/kaggle/input/netflix-shows/netflix_titles.csv'
df = pd.read_csv(filepath)
# If Excel format use pd.read_excel('netflix_titles.xlsx')
```

## Discovery - Understand the Data

```python
# Quick overview
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
print("Number of duplicate rows:", df.duplicated().sum())
```

### Initial Findings:
- All fields are object (string) types except release_year
- Consider making fields datetime and int for calculations
- Many Missing Values:
  - director: 2,634 missing
  - cast: 825 missing
  - country: 831 missing
  - date_added: 10 missing
  - rating: 4 missing
  - duration: 3 missing

## Structuring - Format and Standardize

```python
# Function to normalize column names
def clean_column_names(df):
    """Strips whitespace and converts all text to lowercase for the columns in the DataFrame."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

df = clean_column_names(df)

# Convert 'date_added' to datetime
df['date_added'] = pd.to_datetime(df['date_added'], format='mixed')

# Separate 'duration' into numeric value and unit (e.g. '90 min' → 90, 'min')
df[['duration_value', 'duration_unit']] = df['duration'].str.extract(r'(\d+)\s*(\w+)')

# Convert duration value to numeric
df['duration_value'] = pd.to_numeric(df['duration_value'])

print(df[['duration_value', 'duration_unit']])
```

## Cleaning

### Remove Duplicates
```python
print("Duplicate rows before:", df.duplicated().sum())
df = df.drop_duplicates()
```

### Remove Irrelevant Information
```python
# Drop description column because it will not be used
df = df.drop(columns=['description'])
```

## Handling Missing Values

### Director Imputation
```python
# Explore relationship between cast and director
df['dir_cast'] = df['director'] + '---' + df['cast']
counts = df['dir_cast'].value_counts()  # counts unique values
filtered_counts = counts[counts >= 3]  # checks if repeated 3 or more times
filtered_values = filtered_counts.index  # gets the values i.e. names
lst_dir_cast = list(filtered_values)  # convert to list

# Create dictionary of director-cast pairs
dict_direcast = dict()
for i in lst_dir_cast:
    director, cast = i.split('---')
    dict_direcast[director] = cast

# Fill in missing directors
for i in range(len(dict_direcast)):
    df.loc[(df['director'].isna()) & 
           (df['cast'] == list(dict_direcast.items())[i][1]),
           'director'] = list(dict_direcast.items())[i][0]

# Assign 'Not Given' to remaining missing directors
df.loc[df['director'].isna(), 'director'] = 'Not Given'
```

### Country Imputation
```python
# Use directors to fill missing countries
directors = df['director']
countries = df['country']
pairs = zip(directors, countries)
dir_cntry = dict(list(pairs))

# Fill missing countries
for i in range(len(dir_cntry)):
    df.loc[(df['country'].isna()) & 
           (df['director'] == list(dir_cntry.items())[i][0]),
           'country'] = list(dir_cntry.items())[i][1]

# Assign 'Not Given' to remaining missing countries
df.loc[df['country'].isna(), 'country'] = 'Not Given'
```

### Other Missing Values
```python
# Assign 'Not Given' to cast nulls
df.loc[df['cast'].isna(), 'cast'] = 'Not Given'

# Drop other missing values
df.drop(df[df['date_added'].isna()].index, axis=0, inplace=True)
df.drop(df[df['rating'].isna()].index, axis=0, inplace=True)
df.drop(df[df['duration'].isna()].index, axis=0, inplace=True)
```

## Cleaning Inconsistencies

```python
# Check for date_added before release_year
import datetime as dt
sum(df['date_added'].dt.year < df['release_year'])

# Check records before Netflix launch (1997)
sum(df['date_added'].dt.year < 1997)

# Check unique values
df['type'].unique()
df['rating'].unique()
```

## Transformation

```python
# Create separate columns for 'listed_in'
max(df['listed_in'].str.split(',').apply(lambda x: len(x)))  # Max 3 categories

# Split into 3 columns
df['listed_in_1'] = df['listed_in'].str.split(",", expand=True)[0]
df['listed_in_2'] = df['listed_in'].str.split(",", expand=True)[1]
df['listed_in_3'] = df['listed_in'].str.split(",", expand=True)[2]
```

## Validate

```python
# Remove temporary columns
df.drop(columns=['dir_cast'], inplace=True)

# Reset index
df_reset = df.reset_index(drop=True)

# Sample check
df.sample(5)
```

## Publishing

```python
# Save as CSV
df.to_csv('/kaggle/working/cleaned_netflix.csv', index=False)

# Save as Excel
df.to_excel('/kaggle/working/cleaned_netflix.xlsx', index=False)

# Save as JSON
df.to_json('/kaggle/working/cleaned_netflix.json', orient='records', lines=True)
```

# Assignment 2: Portfolio Website

This week you will work on building your portfolio website hosted on GitHub Pages. This portfolio will serve as your professional online presence and a space to showcase your projects.

## Requirements:
1. Prepare Your GitHub Account and setup github.io page
2. Host all your data science projects on GitHub together with the writeups and visualizations
3. Customize Your Website using themes and templates
4. Ensure your projects are well displayed
5. Submit the link to your live portfolio

---